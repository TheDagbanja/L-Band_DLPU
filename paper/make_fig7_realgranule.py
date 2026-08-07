"""Fig 7 -- sim-only DL model applied to a REAL granule (zero adaptation).

A deep network trained ONLY on the LB-DLPU synthetic corpus is applied, without
any fine-tuning, to a real interferogram from either sensor:

  --sensor nisar   : a NISAR L2 GUNW granule   (src.nisar_gunw.load_scene_wrapped)
  --sensor uavsar  : a UAVSAR .ann/.grd scene  (src.uavsar_io.load_scene)

Four panels: real wrapped phase, the network's unwrapped output, the input
residue map, and the output residue map.

The default baseline is PhaseNet2 (the strongest deployable network on the
benchmark). The network is 256x256-native, so over a large ROI it is tiled and
stitched into one 2*pi-consistent field by ``stitch_tiles`` (the same routine the
leaderboard uses for whole-scene DL inference). Unlike the min-cost-flow methods,
a per-pixel regressor is NOT residue-free by construction, so the output residue
panel reports the *actual* remaining count -- for a good transfer it is at or near
zero, but it is measured, not guaranteed by a solver theorem.

When the loader exposes a processor reference (UAVSAR's JPL ``.unw.grd``; NISAR
has none at matching resolution here), the script also prints an offset-aligned
agreement (RMSE and within-one-fringe fraction) against it.

    # NISAR
    python paper/make_fig7_realgranule.py --sensor nisar \
        --baseline phasenet2 --weights-dir checkpoints/dlpu \
        --granule data/full_scenes/gunw_all/NISAR_..._001.h5 \
        --row0 4000 --col0 4000 --size 768 \
        --out paper/figures/fig7_real_nisar.png

    # UAVSAR (--granule is the .ann; larger ROI at 6 m posting)
    python paper/make_fig7_realgranule.py --sensor uavsar \
        --baseline phasenet2 --weights-dir checkpoints/dlpu \
        --granule data/full_scenes/uavsar/B/SanAnd_..._01.ann \
        --row0 8000 --col0 8000 --size 1024 \
        --out paper/figures/fig7b_real_uavsar.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import mcf, nisar_gunw, uavsar_io
from scripts.unwrap_dl_geotiff import stitch_tiles
from eval.methods.dl_methods import dl_baseline
from eval import metrics as M

TWO_PI = 2.0 * np.pi

# Display names for the panel title (paper-facing, not the registry slug).
DISPLAY = {
    "phasenet2": "PhaseNet2",
    "dlpu_cnn": "DLPU-CNN",
    "attention_unet": "Attention U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "unetpp": "U-Net++",
    "gradient_net": "gradient net",
}


def load_roi(sensor, path, row0, col0, size):
    """Return (psi, coh, valid, ref) for a windowed real scene. ``ref`` may be None."""
    if sensor == "nisar":
        d = nisar_gunw.load_scene_wrapped(path, row0=row0, col0=col0, size=size)
        return d["psi"].astype(np.float64), d["coherence"], d["valid"], d.get("ref")
    if sensor == "uavsar":
        d = uavsar_io.load_scene(path, row0=row0, col0=col0, rows=size, cols=size)
        return d["psi"].astype(np.float64), d["coherence"], d["valid"], d.get("ref")
    raise SystemExit(f"unknown --sensor {sensor!r} (nisar|uavsar)")


def input_residues(psi):
    """(H-1, W-1) input branch-point charge in {-1,0,+1}: mcf.residues of the wrapped phase."""
    return mcf.residues(np.asarray(psi, dtype=np.float64))


def output_residue_defect(phi_hat, psi):
    """(H-1, W-1) residues REMAINING in the solution = loop defect curl(n) + R_psi,
    where n = round((grad(phi_hat) - W(grad(psi))) / 2pi) is the solution's implied
    per-edge integer ambiguity. Same edge-ambiguity counter the harness uses, so a
    DL regressor and an MCF solve are measured identically. NOT mcf.residues(phi_hat),
    which for a near-congruent output just returns the input residue field."""
    n_x, n_y = M.implied_edge_k(phi_hat, psi)
    return M._curl_edge(n_x, n_y) + mcf.residues(np.asarray(psi, dtype=np.float64))


def ref_agreement(phi_hat, ref, valid, coh):
    """Offset-aligned agreement vs a processor reference. Returns a dict or None."""
    if ref is None:
        return None
    m = valid & np.isfinite(ref)
    if m.sum() < 100:
        return None
    d = (phi_hat - ref)[m]
    d = d - np.median(d)                      # remove the global integration constant
    within = float(np.mean(np.abs(d) < np.pi))    # agreement to within one fringe
    rmse = float(np.sqrt(np.mean(d ** 2)))
    out = {"rmse": rmse, "within_1fringe": within, "n": int(m.sum())}
    hi = m & (coh > 0.7)
    if hi.sum() > 100:
        dh = (phi_hat - ref)[hi]; dh = dh - np.median(dh)
        out["rmse_hi_coh"] = float(np.sqrt(np.mean(dh ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=("nisar", "uavsar"), default="nisar")
    ap.add_argument("--baseline", default="phasenet2",
                    help="DL baseline slug (phasenet2, dlpu_cnn, ...)")
    ap.add_argument("--weights-dir", default="checkpoints/dlpu",
                    help="directory holding <baseline>_last.pt / _best.pt")
    ap.add_argument("--granule", required=True,
                    help="NISAR GUNW .h5, or (for --sensor uavsar) the UAVSAR .ann")
    ap.add_argument("--row0", type=int, default=4000)
    ap.add_argument("--col0", type=int, default=4000)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="paper/figures/fig7_real_nisar.png")
    args = ap.parse_args()

    import torch
    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"

    psi, coh, valid, ref = load_roi(args.sensor, args.granule, args.row0, args.col0, args.size)
    print(f"[{args.sensor}] ROI {psi.shape}  valid={valid.mean():.1%}  "
          f"mean coh={coh[valid].mean():.2f}  ref={'yes' if ref is not None else 'no'}")

    # A sim-only DL baseline, tiled over the ROI and stitched to one 2*pi-consistent
    # field (the network is 256-native and only defined up to a per-tile cycle).
    method = dl_baseline(args.baseline, args.weights_dir, device=device)
    print(f"{args.baseline}: {method.weights_provenance}")
    phi_hat = stitch_tiles(np.nan_to_num(psi), method.fn, tile=args.tile, stride=args.stride)

    R_in = input_residues(psi)
    R_out = output_residue_defect(phi_hat, psi)
    print(f"input residues: {(R_in != 0).sum()}   output residues: {(R_out != 0).sum()}")

    agree = ref_agreement(phi_hat, ref, valid, coh)
    if agree is not None:
        msg = (f"vs processor reference (offset-aligned, n={agree['n']}): "
               f"RMSE={agree['rmse']:.3f} rad, within-one-fringe={100*agree['within_1fringe']:.1f}%")
        if "rmse_hi_coh" in agree:
            msg += f", high-coh(>0.7) RMSE={agree['rmse_hi_coh']:.3f}"
        print("  " + msg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    vmask = valid
    pmask = vmask[:-1, :-1] & vmask[:-1, 1:] & vmask[1:, :-1] & vmask[1:, 1:]
    phi_a = phi_hat - np.nanmean(np.where(vmask, phi_hat, np.nan))

    n_in = int((R_in != 0)[pmask].sum())
    n_out = int((R_out != 0)[pmask].sum())
    disp = DISPLAY.get(args.baseline, args.baseline)
    frac_out = n_out / max(int(pmask.sum()), 1)
    out_note = "empty" if n_out == 0 else f"{100*frac_out:.3f}% of loops"
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 7.0))   # 2x2 fits a two-column layout
    (a0, a1), (a2, a3) = axes
    im0 = a0.imshow(np.where(vmask, psi, np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    a0.set_title("Real wrapped phase", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=a0, fraction=0.046)
    im1 = a1.imshow(np.where(vmask, phi_a, np.nan), cmap="jet")
    a1.set_title(f"{disp} unwrapped", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=a1, fraction=0.046)
    im2 = a2.imshow(np.where(pmask, R_in, np.nan), cmap="RdBu", vmin=-1, vmax=1)
    a2.set_title(f"Input residues (n = {n_in})", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=a2, fraction=0.046)
    im3 = a3.imshow(np.where(pmask, R_out, np.nan), cmap="RdBu", vmin=-1, vmax=1)
    a3.set_title(f"Output residues (n = {n_out}, {out_note})", fontsize=12, fontweight="bold")
    fig.colorbar(im3, ax=a3, fraction=0.046)

    for a in (a0, a1, a2, a3):
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print("Wrote", args.out, "(+ .pdf)")


if __name__ == "__main__":
    main()
