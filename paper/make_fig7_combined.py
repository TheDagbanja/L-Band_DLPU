"""Combined Fig 7 -- sim-only PhaseNet2 applied to BOTH real regimes in one figure.

Each sensor keeps its original 2x2 block (real wrapped phase / network unwrapped
output on top, input residues / output residues below), exactly as in the
single-sensor figure. The two blocks are stacked -- (a) NISAR on top, (b) UAVSAR
below -- separated by a single horizontal divider, matching the Fig. 1 style.

Runs the same tiled DL inference as ``make_fig7_realgranule.py`` for each sensor
(model loaded once), then composes the figure.

    python paper/make_fig7_combined.py \
        --weights-dir checkpoints/dlpu --baseline phasenet2 \
        --nisar-granule data/full_scenes/gunw_all/NISAR_..._001.h5 \
        --nisar-row0 4000 --nisar-col0 4000 --nisar-size 768 \
        --uavsar-ann data/full_scenes/uavsar/B/SanAnd_..._01.ann \
        --uavsar-row0 8000 --uavsar-col0 8000 --uavsar-size 1024 \
        --out paper/figures/fig7_real_transfer.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# repo root (for src/, scripts/, eval/) and paper/ (for the sibling helper module)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_fig7_realgranule import (load_roi, input_residues, output_residue_defect,
                                   ref_agreement, DISPLAY)
from scripts.unwrap_dl_geotiff import stitch_tiles
from eval.methods.dl_methods import dl_baseline


def run_sensor(sensor, path, row0, col0, size, method, tile, stride):
    psi, coh, valid, ref = load_roi(sensor, path, row0, col0, size)
    print(f"[{sensor}] ROI {psi.shape}  valid={valid.mean():.1%}  "
          f"mean coh={coh[valid].mean():.2f}  ref={'yes' if ref is not None else 'no'}")
    phi = stitch_tiles(np.nan_to_num(psi), method.fn, tile=tile, stride=stride)
    R_in = input_residues(psi)
    R_out = output_residue_defect(phi, psi)
    agree = ref_agreement(phi, ref, valid, coh)
    print(f"    input residues: {int((R_in != 0).sum())}   output residues: {int((R_out != 0).sum())}")
    if agree is not None:
        msg = (f"    vs processor reference (offset-aligned, n={agree['n']}): "
               f"RMSE={agree['rmse']:.3f} rad, within-one-fringe={100*agree['within_1fringe']:.1f}%")
        if "rmse_hi_coh" in agree:
            msg += f", high-coh(>0.7) RMSE={agree['rmse_hi_coh']:.3f}"
        print(msg)
    return dict(psi=psi, coh=coh, valid=valid, phi=phi, R_in=R_in, R_out=R_out,
                mean_coh=float(coh[valid].mean()), agree=agree)


def draw_block(fig, subgs, D, disp):
    """Render one sensor's original 2x2 block into a 2x2 sub-gridspec."""
    v = D["valid"]
    pmask = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
    phi_a = D["phi"] - np.nanmean(np.where(v, D["phi"], np.nan))
    n_in = int((D["R_in"] != 0)[pmask].sum())
    n_out = int((D["R_out"] != 0)[pmask].sum())

    a0 = fig.add_subplot(subgs[0, 0]); a1 = fig.add_subplot(subgs[0, 1])
    a2 = fig.add_subplot(subgs[1, 0]); a3 = fig.add_subplot(subgs[1, 1])
    im0 = a0.imshow(np.where(v, D["psi"], np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    im1 = a1.imshow(np.where(v, phi_a, np.nan), cmap="jet")
    im2 = a2.imshow(np.where(pmask, D["R_in"], np.nan), cmap="RdBu", vmin=-1, vmax=1)
    im3 = a3.imshow(np.where(pmask, D["R_out"], np.nan), cmap="RdBu", vmin=-1, vmax=1)
    for im, ax in ((im0, a0), (im1, a1), (im2, a2), (im3, a3)):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    a0.set_title("Real wrapped phase", fontsize=11, fontweight="bold")
    a1.set_title(f"{disp} unwrapped", fontsize=11, fontweight="bold")
    a2.set_title(f"Input residues (n = {n_in})", fontsize=11, fontweight="bold")
    note = "empty" if n_out == 0 else f"{100*n_out/max(int(pmask.sum()),1):.2f}% loops"
    a3.set_title(f"Output residues (n = {n_out}, {note})", fontsize=11, fontweight="bold")
    for a in (a0, a1, a2, a3):
        a.set_xticks([]); a.set_yticks([])
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="phasenet2")
    ap.add_argument("--weights-dir", default="checkpoints/dlpu")
    ap.add_argument("--nisar-granule", required=True)
    ap.add_argument("--nisar-row0", type=int, default=4000)
    ap.add_argument("--nisar-col0", type=int, default=4000)
    ap.add_argument("--nisar-size", type=int, default=768)
    ap.add_argument("--uavsar-ann", required=True)
    ap.add_argument("--uavsar-row0", type=int, default=8000)
    ap.add_argument("--uavsar-col0", type=int, default=8000)
    ap.add_argument("--uavsar-size", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="paper/figures/fig7_real_transfer.png")
    args = ap.parse_args()

    import torch
    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    method = dl_baseline(args.baseline, args.weights_dir, device=device)
    print(f"{args.baseline}: {method.weights_provenance}")
    disp = DISPLAY.get(args.baseline, args.baseline)

    N = run_sensor("nisar", args.nisar_granule, args.nisar_row0, args.nisar_col0,
                   args.nisar_size, method, args.tile, args.stride)
    U = run_sensor("uavsar", args.uavsar_ann, args.uavsar_row0, args.uavsar_col0,
                   args.uavsar_size, method, args.tile, args.stride)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "pdf.fonttype": 42})

    # Two stacked 2x2 blocks. Tall single-column figure (rendered at \columnwidth).
    fig = plt.figure(figsize=(7.16, 14.5))
    gsA = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.955, bottom=0.535,
                           hspace=0.16, wspace=0.22)
    gsB = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.465, bottom=0.045,
                           hspace=0.16, wspace=0.22)
    draw_block(fig, gsA, N, disp)
    draw_block(fig, gsB, U, disp)

    # (a)/(b) labels, read up the left margin, and a single divider between blocks.
    fig.text(0.030, 0.745, "(a) NISAR", rotation=90, fontsize=14, fontweight="bold",
             ha="center", va="center")
    fig.text(0.030, 0.255, "(b) UAVSAR", rotation=90, fontsize=14, fontweight="bold",
             ha="center", va="center")
    fig.add_artist(Line2D([0.04, 0.99], [0.50, 0.50], color="0.35", lw=1.0,
                          transform=fig.transFigure))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print("Wrote", args.out, "(+ .pdf)")


if __name__ == "__main__":
    main()
