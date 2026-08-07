"""Quick montage of all NISAR GUNW granules' wrapped phase + coherence.

Renders a downsampled thumbnail of every granule in a directory so you can
eyeball which scenes carry a deformation signal (dense localised fringes in a
coherent area) before committing an expensive full-resolution unwrap. Uses the
coarse ``unwrappedPhase`` grid (wrapped for the fringe view) -- it's the clean
operational product, small, and shows deformation structure without noise.

    python -m scripts.preview_scenes \\
        --dir data/full_scenes/gunw_all --stride 8 --out outputs/scene_preview

Writes ``<out>_wrapped.png`` (fringe montage) and ``<out>_coherence.png``.
Read the two together: a deformation candidate has concentrated fringes AND
good coherence there.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/full_scenes/gunw_all")
    ap.add_argument("--stride", type=int, default=8, help="downsample factor for thumbnails")
    ap.add_argument("--out", default="outputs/scene_preview")
    args = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
    files = sorted(glob.glob(os.path.join(args.dir, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 granules in {args.dir!r}.")
    print(f"{len(files)} granules; rendering thumbnails (stride {args.stride})...")

    thumbs = []
    for i, f in enumerate(files):
        s = args.stride
        with h5py.File(f, "r") as h:
            uw = np.asarray(h[f"{G}/unwrappedPhase"][::s, ::s], dtype=np.float32)
            co = np.asarray(h[f"{G}/coherenceMagnitude"][::s, ::s], dtype=np.float32)
        wr = _wrap(np.nan_to_num(uw))
        wr = np.where(np.isfinite(uw), wr, np.nan)
        # short label: the two orbit/frame fields + acquisition date
        base = os.path.basename(f).split("_")
        tag = f"#{i}  {base[5]}_{base[6]}_{base[7]}  {base[11][:8]}"
        thumbs.append((tag, wr, co, float(np.nanmean(co))))
        print(f"  #{i:2d}  coh~{np.nanmean(co):.2f}  {os.path.basename(f)[:70]}")

    n = len(thumbs)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    for kind, idx, cmap, title in [("wrapped", 1, "twilight", "Wrapped phase (fringes = deformation/topo)"),
                                    ("coherence", 2, "gray", "Coherence")]:
        fig, ax = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
        for a, (tag, wr, co, cm) in zip(np.atleast_1d(ax).ravel(), thumbs):
            img = wr if kind == "wrapped" else co
            a.imshow(img, cmap=cmap, vmin=(-np.pi if kind == "wrapped" else 0),
                     vmax=(np.pi if kind == "wrapped" else 1))
            a.set_title(tag, fontsize=8)
            a.axis("off")
        for a in np.atleast_1d(ax).ravel()[n:]:
            a.axis("off")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        path = f"{args.out}_{kind}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
