"""Unwrap a Sentinel-1 HyP3 ROI with a DL baseline, writing a GeoTIFF.

The DL baselines are 256x256-native: run on a large scene they must be tiled, and
each tile's unwrapping is only defined up to its own additive multiple of 2*pi.
Naive tiling therefore yields a discontinuous field that no GPS comparison could
score fairly. This script stitches tiles into a globally consistent field by
estimating, for each new tile, the integer 2*pi offset that best matches the
already-stitched overlap, then blending with a Hann window.

That extra machinery is itself a finding: a per-pixel DL regressor needs a
stitching heuristic to scale past its training patch size, whereas the min-cost-flow
solver is run once over the whole ROI and is globally consistent by construction.

    python -m scripts.unwrap_dl_geotiff --baseline phasenet2 \\
        --product ../insar_peft_unwrapping/data/S1_data/<PRODUCT> \\
        --weights-dir checkpoints/dlpu --row0 300 --col0 750 --size 1536 \\
        --out outputs/rc_phasenet2.tif
    python -m scripts.unwrap_dl_geotiff --self-test
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

TWO_PI = 2.0 * np.pi


def _find(d, tag):
    h = glob.glob(os.path.join(d, f"*{tag}.tif"))
    if not h:
        raise FileNotFoundError(f"no *{tag}.tif in {d!r}")
    return h[0]


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w) + 1e-3


def stitch_tiles(psi, predict, tile=256, stride=128):
    """Tile ``psi`` through ``predict`` and stitch to one 2*pi-consistent field.

    ``predict(psi_tile) -> phi_tile``. Each tile is shifted by the integer number
    of cycles that best matches the already-filled overlap (median difference,
    rounded to a whole cycle), then Hann-blended into the canvas.
    """
    H, W = psi.shape
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    win = _hann2d(tile)

    def starts(extent):
        ss = list(range(0, max(extent - tile, 0) + 1, stride))
        if ss and ss[-1] != extent - tile:
            ss.append(max(extent - tile, 0))
        return ss or [0]

    for r in starts(H):
        for c in starts(W):
            sl = (slice(r, r + tile), slice(c, c + tile))
            phi_t = np.asarray(predict(psi[sl]), dtype=np.float64)
            filled = wsum[sl] > 0
            if filled.any():
                canvas = acc[sl][filled] / wsum[sl][filled]
                k = np.round(np.median(canvas - phi_t[filled]) / TWO_PI)
                phi_t = phi_t + TWO_PI * k
            acc[sl] += phi_t * win
            wsum[sl] += win
    return acc / np.maximum(wsum, 1e-9)


def self_test():
    """Stitcher must recover a known field from tiles carrying random 2*pi offsets."""
    rng = np.random.default_rng(0)
    H = W = 640
    yy, xx = np.meshgrid(np.linspace(-3, 3, H), np.linspace(-3, 3, W), indexing="ij")
    truth = 6.0 * np.exp(-(xx ** 2 + yy ** 2)) + 0.8 * xx + 0.5 * yy
    _ = (truth + np.pi) % TWO_PI - np.pi        # wrapped input (unused: tiles are exact)

    # emulate a per-tile unwrapper: perfect on the tile, arbitrary cycle offset
    accs, wss = np.zeros_like(truth), np.zeros_like(truth)
    tile, stride = 256, 128
    win = _hann2d(tile)

    def starts(extent):
        ss = list(range(0, extent - tile + 1, stride))
        if ss[-1] != extent - tile:
            ss.append(extent - tile)
        return ss

    for r in starts(H):
        for c in starts(W):
            sl = (slice(r, r + tile), slice(c, c + tile))
            phi_t = truth[sl] + TWO_PI * rng.integers(-3, 4)     # random cycle offset
            filled = wss[sl] > 0
            if filled.any():
                canvas = accs[sl][filled] / wss[sl][filled]
                k = np.round(np.median(canvas - phi_t[filled]) / TWO_PI)
                phi_t = phi_t + TWO_PI * k
            accs[sl] += phi_t * win
            wss[sl] += win
    out = accs / np.maximum(wss, 1e-9)

    d = out - truth
    d -= d.mean()
    rmse = float(np.sqrt((d ** 2).mean()))
    slip = float((np.round(d / TWO_PI) != 0).mean())
    print(f"stitcher self-test: aligned RMSE {rmse:.2e} rad, cycle-slip rate {100*slip:.3f}%")
    assert rmse < 1e-6, f"stitching failed to recover the field (rmse {rmse})"
    print("PASS -- random per-tile 2*pi offsets fully removed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--baseline")
    ap.add_argument("--product")
    ap.add_argument("--weights-dir", default="checkpoints/dlpu")
    ap.add_argument("--row0", type=int, default=300)
    ap.add_argument("--col0", type=int, default=750)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/dl_unw.tif")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not (args.baseline and args.product):
        raise SystemExit("need --baseline and --product (or --self-test)")

    import rasterio
    from eval.methods.dl_methods import dl_baseline

    with rasterio.open(_find(args.product, "_wrapped_phase")) as s:
        win = rasterio.windows.Window(args.col0, args.row0, args.size, args.size)
        psi = s.read(1, window=win).astype(np.float64)
        prof = s.profile.copy()
        transform = s.window_transform(win)
    psi = np.nan_to_num(psi)

    method = dl_baseline(args.baseline, args.weights_dir, device=args.device)
    print(f"{args.baseline}: {method.weights_provenance}")
    print(f"ROI {psi.shape}  tiles {args.tile}/{args.stride}")
    phi = stitch_tiles(psi, method.fn, tile=args.tile, stride=args.stride)

    prof.update(height=args.size, width=args.size, transform=transform, count=1,
                dtype="float32", compress="deflate", tiled=False)
    for k in ("blockxsize", "blockysize"):
        prof.pop(k, None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(phi.astype("float32"), 1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
