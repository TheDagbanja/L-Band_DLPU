"""Generate the data-descriptor figures that can be built from local artifacts.

Produces (into paper/figures/):
  fig2_wrapped_grid.png      -- wrapped phase, 2 regimes x 3 difficulties
  fig3_wellposedness.png     -- rejection-sampling schematic + clean-oracle error hist
  fig4_dem_tiles.png         -- 29 GLO-30 tiles on a lon/lat map, colored by split
  fig5_calibration.png       -- synthetic vs real coherence + residue density
  fig6_rmse_by_difficulty.png-- per-method RMSE across smooth/mixed/dense, per regime

Figs 1 (real-DEM generation stages) and 7 (real granule + trained model) need the
server pipeline / weights and are produced there.

Run:  python paper/make_paper_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Darker, larger, bolder text so labels/legends stay legible when placed in the
# two-column document (and vector PDF export for high resolution).
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "0.15", "axes.linewidth": 1.0,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "LB_DLPU")


def _resolve_sim2():
    """Pick the COMPLETE benchmark copy (has index.jsonl + test split). The live
    `sim_2` may be mid-regeneration; the paper's numbers came from the complete
    copy (kept as `sim_2old` after a reorg)."""
    for name in ("sim_2", "sim_2old"):
        d = os.path.join(DATA, name)
        if os.path.exists(os.path.join(d, "index.jsonl")) and \
           os.path.isdir(os.path.join(d, "sim", "test")):
            return d
    return os.path.join(DATA, "sim_2")


SIM2 = _resolve_sim2()
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

PHASE_CMAP = "twilight"          # cyclic, good for wrapped phase
REGIMES = ["nisar", "uavsar"]
DIFFS = ["smooth", "mixed", "dense"]


def _save(fig, p):
    """Save both a high-res PNG (preview) and a vector PDF (for the document)."""
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(p)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return p


def _panel(ax, letter, dark=False):
    """IEEE-style (a)/(b) subpanel label in the top-left corner."""
    fg, bg = ("white", "black") if dark else ("black", "white")
    ax.text(0.04, 0.96, f"({letter})", transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold", color=fg,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, alpha=0.6))


def _load_index():
    rows = []
    with open(os.path.join(SIM2, "index.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _read_h5(path, keys):
    import h5py
    out = {}
    with h5py.File(path, "r") as fh:
        for k in keys:
            if k in fh:
                out[k] = np.asarray(fh[k][:])
        out["_attrs"] = dict(fh.attrs)
    return out


# ---------------------------------------------------------------------------
# Fig 2 -- wrapped phase, regimes x difficulties
# ---------------------------------------------------------------------------
def fig2_wrapped_grid(index):
    pick = {}
    for r in index:
        key = (r["sensor"], r["difficulty"])
        if r["split"] == "test" and key not in pick:
            pick[key] = r["id"]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.9))   # IEEE full text-width
    for i, reg in enumerate(REGIMES):
        for j, diff in enumerate(DIFFS):
            ax = axes[i, j]
            rid = pick.get((reg, diff))
            if rid is None:
                ax.set_axis_off(); continue
            d = _read_h5(os.path.join(SIM2, "sim", "test", f"{rid}.h5"), ["psi", "coherence"])
            ax.imshow(d["psi"], cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:                                     # difficulty = column header
                ax.set_title(diff.capitalize(), fontsize=12, fontweight="bold")
            if j == 0:                                     # regime = row label
                ax.set_ylabel(f"{reg.upper()}\n({'20 m, iono on' if reg=='nisar' else '6 m, iono off'})",
                              fontsize=11, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=PHASE_CMAP,
                               norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    cbar.set_label("wrapped phase (rad)")
    p = os.path.join(OUT, "fig2_wrapped_grid.png")
    return _save(fig, p)


# ---------------------------------------------------------------------------
# Fig 3 -- well-posedness: rejection loop schematic + clean-oracle error hist
# ---------------------------------------------------------------------------
def fig3_wellposedness(index):
    err = np.array([r.get("clean_oracle_rmse", 0.0) for r in index], dtype=float)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) accept/reject certificate flow -- clean vertical layout.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    axL.set_axis_off(); axL.set_xlim(0, 10); axL.set_ylim(0.2, 9.9)
    BLUE, ORANGE, GREEN, RED = "#dbe7f3", "#fde6cf", "#d7ecd9", "#f6dcd9"

    def box(cx, cy, w, h, text, fc, fs=8.0):
        axL.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                      boxstyle="round,pad=0.02,rounding_size=0.18",
                      fc=fc, ec="#333333", lw=1.2, zorder=2))
        axL.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)

    def arr(p0, p1, color="#333333", lw=1.5):
        axL.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                      color=color, lw=lw, shrinkA=0, shrinkB=0,
                      connectionstyle="arc3,rad=0", zorder=1))

    # boxes sized to fit their labels; stack dropped so the top box clears (a)
    box(5.0, 8.7, 7.0, 1.2, "Sample scene  ($\\phi,\\ \\gamma$, sensor)", BLUE)
    box(5.0, 6.7, 7.0, 1.2, "Noiseless-oracle MCF unwrap", ORANGE)
    box(5.0, 4.7, 7.0, 1.2, "Clean-oracle RMSE  $\\rho$", BLUE)
    box(3.1, 1.3, 3.2, 1.25, "ACCEPT\n(store labels)", GREEN)
    box(6.9, 1.3, 3.2, 1.25, "REJECT\n(resample)", RED)
    arr((5.0, 8.10), (5.0, 7.30))            # sample -> oracle
    arr((5.0, 6.10), (5.0, 5.30))            # oracle -> rmse
    arr((4.3, 4.10), (3.35, 1.925), color="#2e7d32")   # rho < tau -> accept
    arr((5.7, 4.10), (6.65, 1.925), color="#c62828")   # rho >= tau -> reject
    axL.text(3.35, 3.0, r"$\rho<\tau$", color="#2e7d32", fontsize=9, ha="right")
    axL.text(6.65, 3.0, r"$\rho\geq\tau$", color="#c62828", fontsize=9, ha="left")
    axL.text(0.05, 1.0, "(a)", transform=axL.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")

    # (b) distribution of clean-oracle error over the retained corpus.
    axR.hist(err, bins=np.linspace(0, max(0.05, err.max() * 1.05), 40),
             color="#4a7fb5", edgecolor="white")
    axR.axvline(0.3, color="firebrick", ls="--", lw=1.5)
    axR.set_yscale("log")
    axR.text(0.288, 40, r"accept threshold $\tau=0.3$", rotation=90, va="center",
             ha="right", fontsize=8, color="firebrick")               # mid-height, clear of x-axis
    axR.text(0.17, 2e3, f"max = {err.max():.4f} rad\n(n = {len(err):,}; 0 rejected retained)",
             fontsize=8, ha="center")
    axR.set_xlabel("clean-oracle RMSE (rad)")
    axR.set_ylabel("number of retained scenes")
    axR.text(0.035, 0.975, "(b)", transform=axR.transAxes, va="top", ha="left",
             fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.55", lw=0.8))
    p = os.path.join(OUT, "fig3_wellposedness.png")
    return _save(fig, p)


# ---------------------------------------------------------------------------
# Fig 4 -- DEM tile coverage on a lon/lat map, colored by split (illustrative)
# ---------------------------------------------------------------------------
def _parse_tile_lonlat(fname):
    m = re.search(r"_([NS])(\d+)_00_([EW])(\d+)_00_DEM", fname)
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return lon + 0.5, lat + 0.5


def _dem_split_from_manifest():
    """Real {tile_filename: split} map from dem_manifest.csv (tile,split,...)."""
    import csv
    for cand in (os.path.join(DATA, "dem_tiles", "dem_manifest.csv"),
                 os.path.join(DATA, "dem_manifest.csv")):
        if os.path.exists(cand):
            rows = [l for l in open(cand, encoding="utf-8") if not l.lstrip().startswith("#")]
            m = {f"Copernicus_DSM_COG_10_{r['tile']}_DEM.tif": r["split"]
                 for r in csv.DictReader(rows)}
            return m or None
    return None


def _dem_split_from_disk():
    """Real {tile: split} map from dem_tiles/{train,val,test}/*.tif, if populated."""
    m = {}
    for sp in ("train", "val", "test"):
        for f in glob.glob(os.path.join(DATA, "dem_tiles", sp, "*.tif")):
            m[os.path.basename(f)] = sp
    return m or None


def _draw_basemap(ax):
    """Fill world land polygons (Natural Earth 110m) as an equirectangular basemap."""
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    path = os.path.join(ROOT, "paper", "assets", "ne_110m_land.geojson")
    if not os.path.exists(path):
        return False
    gj = json.load(open(path, encoding="utf-8"))
    patches = []
    for feat in gj["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            patches.append(MplPoly(np.asarray(poly[0]), closed=True))   # exterior ring
    ax.add_collection(PatchCollection(patches, facecolor="#e8eaed", edgecolor="#b3b8bf",
                                      linewidth=0.4, zorder=1))
    return True


def fig5_dem_tiles(split_map=None):
    tiles = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "dem_tiles", "*.tif")))
    pts = [(t, _parse_tile_lonlat(t)) for t in tiles]
    pts = [(t, ll) for t, ll in pts if ll is not None]
    # Real split from disk if available; else a deterministic 17/5/7 split (correct
    # counts, illustrative geography) unless a {tile: split} mapping is supplied.
    if split_map is None:
        split_map = _dem_split_from_manifest() or _dem_split_from_disk()   # real split
    if split_map is None:
        # fall back to a deterministic split with the TRUE counts (17 train / 5 val / 7 test)
        rng = np.random.default_rng(0)
        order = list(range(len(pts))); rng.shuffle(order)
        n_tr, n_va = 17, 5                            # 29 tiles: 17 / 5 / 7
        split_map = {}
        for rank, idx in enumerate(order):
            split_map[pts[idx][0]] = ("train" if rank < n_tr else
                                      "val" if rank < n_tr + n_va else "test")
        illustrative = True
    else:
        illustrative = False

    colors = {"train": "#2c7fb8", "val": "#f0a202", "test": "#d7191c"}
    fig, ax = plt.subplots(figsize=(7.16, 3.7))
    _draw_basemap(ax)
    for sp in ["train", "val", "test"]:
        xs = [ll[0] for t, ll in pts if split_map.get(t) == sp]
        ys = [ll[1] for t, ll in pts if split_map.get(t) == sp]
        ax.scatter(xs, ys, s=55, c=colors[sp], edgecolor="black", lw=0.7,
                   label=f"{sp} ({len(xs)})", zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 84); ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 90, 30))
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.grid(True, ls=":", alpha=0.35, zorder=0)
    leg_title = "split (illustrative)" if illustrative else "split"
    ax.legend(loc="lower left", fontsize=8.5, title=leg_title,
              framealpha=0.9, borderpad=0.6).set_zorder(5)
    p = os.path.join(OUT, "fig5_dem_tiles.png")
    return _save(fig, p)


# ---------------------------------------------------------------------------
# Fig 4 -- true phase-gradient distribution by difficulty stratum (Nyquist)
# ---------------------------------------------------------------------------
def fig4_phase_gradient(index, per_stratum=80):
    """|∇φ| (π/pixel) histogram per difficulty; dashed Nyquist line at 1 π/px."""
    diffs = ["smooth", "mixed", "dense"]
    colors = {"smooth": "#2c7fb8", "mixed": "#e08214", "dense": "#d7191c"}
    vals = {d: [] for d in diffs}
    counts = {d: 0 for d in diffs}
    for r in index:
        d = r.get("difficulty")
        if d not in vals or counts[d] >= per_stratum:
            continue
        path = os.path.join(SIM2, "sim", r.get("split", "train"), f"{r['id']}.h5")
        if not os.path.exists(path):
            continue
        phi = _read_h5(path, ["phi"]).get("phi")
        if phi is None:
            continue
        phi = phi.astype(np.float64)
        gx = np.abs(phi[:, 1:] - phi[:, :-1]).ravel()
        gy = np.abs(phi[1:, :] - phi[:-1, :]).ravel()
        vals[d].append(np.concatenate([gx, gy]) / np.pi)   # units of π / pixel
        counts[d] += 1

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.linspace(0, 2.6, 140)
    for d in diffs:
        if not vals[d]:
            continue
        v = np.concatenate(vals[d])
        ax.hist(v, bins=bins, histtype="step", lw=1.8, density=True,
                color=colors[d], label=f"{d} (n={counts[d]})")
    ax.axvline(1.0, ls="--", color="0.4", lw=1.4)
    ax.text(1.03, 0.72, "Nyquist", transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="left", fontsize=9, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("true phase gradient  |∇φ|  (π / pixel)")
    ax.set_ylabel("density (log)")
    ax.set_xlim(0, 2.6)
    ax.legend(frameon=False, loc="upper right")
    p = os.path.join(OUT, "fig4_phase_gradient.png")
    return _save(fig, p)


# ---------------------------------------------------------------------------
# Fig 2 -- calibration: synthetic vs real coherence + residue density
# ---------------------------------------------------------------------------
def fig2_calibration(index):
    from scipy.stats import beta as beta_dist
    calib = json.load(open(os.path.join(DATA, "calib", "calibration_params.json")))
    prof = {r: json.load(open(os.path.join(DATA, "calib", f"profile_{r}.json"))) for r in REGIMES}
    real_beta = {"nisar": (calib["coherence"]["beta_a"], calib["coherence"]["beta_b"]),
                 "uavsar": (prof["uavsar"]["coherence"]["beta_a"], prof["uavsar"]["coherence"]["beta_b"])}
    # synthetic coherence pixels from a sample of test patches per regime
    ids = {reg: [r["id"] for r in index if r["split"] == "test" and r["sensor"] == reg][:40]
           for reg in REGIMES}
    syn_coh = {}
    for reg in REGIMES:
        vals = []
        for rid in ids[reg]:
            d = _read_h5(os.path.join(SIM2, "sim", "test", f"{rid}.h5"), ["coherence"])
            vals.append(d["coherence"].ravel())
        syn_coh[reg] = np.concatenate(vals) if vals else np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.9), layout="constrained")  # 2x2, two-column
    (ax_n, ax_u), (axR, ax_off) = axes
    ax_off.axis("off")
    x = np.linspace(0, 1, 200)
    for ax, reg, lett in [(ax_n, "nisar", "a"), (ax_u, "uavsar", "b")]:
        a, b = real_beta[reg]
        ax.hist(syn_coh[reg], bins=40, density=True, color="#4a7fb5", alpha=0.65,
                label="synthetic")
        ax.plot(x, beta_dist.pdf(x, a, b), color="firebrick", lw=2, label="real fit")
        ax.set_xlabel(f"coherence γ — {reg.upper()}"); ax.set_ylabel("density")
        # short 2-line legend in the empty band above the curve: NISAR peaks
        # right so the upper-left (below the (a) label) is clear; UAVSAR peaks
        # left so the upper-right is clear.
        loc = "upper left" if reg == "nisar" else "upper right"
        anchor = (0.03, 0.82) if reg == "nisar" else (0.97, 0.97)
        ax.legend(fontsize=10, loc=loc, bbox_to_anchor=anchor, frameon=False)
        _panel(ax, lett)

    # (c) residue density: synthetic distribution vs real value per regime
    real_resid = {"nisar": 18100.0, "uavsar": 28794.0}
    syn = {reg: np.array([r["residues_per_mp"] for r in index if r["sensor"] == reg])
           for reg in REGIMES}
    positions = [1, 2]
    vp = axR.violinplot([syn["nisar"], syn["uavsar"]], positions=positions, showmeans=True)
    for b in vp["bodies"]:
        b.set_facecolor("#4a7fb5"); b.set_alpha(0.6)
    for k, reg in enumerate(REGIMES):
        axR.hlines(real_resid[reg], positions[k] - 0.3, positions[k] + 0.3,
                   color="firebrick", lw=2.5, label="real sensor" if k == 0 else None)
    axR.set_xticks(positions); axR.set_xticklabels(["NISAR", "UAVSAR"])
    axR.set_ylabel("residues per Mpixel")
    axR.set_ylim(0, 45000)                     # clip outlier tail so the bulk + real lines show
    axR.legend(fontsize=10, loc="lower right", frameon=False)   # bottom: clear of the violins
    _panel(axR, "c")
    p = os.path.join(OUT, "fig2_calibration.png")
    return _save(fig, p)


# ---------------------------------------------------------------------------
# Fig 6 -- RMSE by difficulty stratum, per method and regime
# ---------------------------------------------------------------------------
def fig6_rmse_by_difficulty():
    import csv
    # prefer the canonical run output; fall back to a loose results/leaderboard.csv
    candidates = [os.path.join(ROOT, "results", "sim_2_leaderboard", "leaderboard.csv"),
                  os.path.join(ROOT, "results", "leaderboard.csv")]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    # derive the method list from the leaderboard (robust to registry changes),
    # in a preferred order with the ceiling/learned rows first.
    present = {r["method"] for r in rows}
    # The learned-cost MCF is released with its own method paper (AP-GARSS 2026),
    # not in this repo; its leaderboard result is reported in the LB-DLPU paper. The
    # oracle cost (gt_grad_cost/oracle_cost) remains as the certified difficulty ceiling.
    EXCLUDE = {"ours_learned", "learned_cost"}
    present -= EXCLUDE
    preferred = ["oracle_cost", "gt_grad_cost", "unit_mcf",
                 "snaphu_cost", "phasenet2", "dlpu_cnn", "gradient_net", "attention_unet",
                 "deeplabv3plus", "unetpp", "denet", "quality_guided", "weighted_ls", "goldstein"]
    methods = [m for m in preferred if m in present] + sorted(present - set(preferred))
    HILITE = {"oracle_cost", "gt_grad_cost"}
    DISPLAY = {"oracle_cost": "oracle cost", "gt_grad_cost": "oracle cost"}
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.2), sharey=True)
    handles = labels = None
    for k, reg in enumerate(REGIMES):
        ax = axes[k]
        for mi, m in enumerate(methods):
            ys = []
            for diff in DIFFS:
                cell = [r for r in rows if r["regime"] == reg and r["difficulty"] == diff
                        and r["method"] == m]
                ys.append(float(cell[0]["rmse_mean"]) if cell else np.nan)
            style = "-o" if m in HILITE else "--o"
            lw = 2.6 if m in HILITE else 1.4
            ax.plot(DIFFS, ys, style, lw=lw, ms=5, color=cmap(mi % 20), label=DISPLAY.get(m, m))
        ax.set_xlabel("difficulty stratum"); ax.grid(True, ls=":", alpha=0.4)
        if k == 0:
            ax.set_ylabel("RMSE (rad)")
            handles, labels = ax.get_legend_handles_labels()
        _panel(ax, "ab"[k])
    # Shared legend below both panels, pushed clear of the x-labels.
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    p = os.path.join(OUT, "fig6_rmse_by_difficulty.png")
    return _save(fig, p)


def main():
    index = _load_index()
    made = [
        fig2_calibration(index),          # Fig 2: coherence + residue calibration
        fig3_wellposedness(index),        # Fig 3: rejection loop + clean-oracle error
        fig4_phase_gradient(index),       # Fig 4: phase-gradient dist by difficulty
        fig5_dem_tiles(),                 # Fig 5: DEM tiles by split
        fig6_rmse_by_difficulty(),        # Fig 6: RMSE by difficulty
    ]                                     # (Fig 1 = make_fig1_pipeline.py; Fig 7 = make_fig7 on server)
    print("Wrote:")
    for p in made:
        print("  ", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
