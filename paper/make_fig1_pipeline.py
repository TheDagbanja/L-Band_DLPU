"""Fig 1 -- generation pipeline for the two regimes, from a REAL GLO-30 DEM crop.

Left to right: real DEM crop, topographic phase (from the DEM + InSAR geometry),
absolute phase after deformation + atmosphere (+ ionosphere for NISAR), the
wrapped phase, and the noisy wrapped phase at the regime coherence. Top row
NISAR (20 m, ionosphere on), bottom row UAVSAR (6 m, ionosphere off).

Deformation and atmosphere come from the shared physical engine
(src.synthetic_engine); the ionospheric screen from src.synth_lband; only the
topographic stage is swapped to use a real Copernicus GLO-30 tile so the leading
panel is a genuine DEM crop. InSAR geometry (ambiguity height) is per-regime and
representative.

Run:  python paper/make_fig1_pipeline.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.synthetic_engine import SyntheticConfig, SyntheticInterferogramGenerator
from src.synth_lband import ionospheric_field, NISAR_WAVELENGTH_M

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "text.color": "black", "savefig.dpi": 300, "pdf.fonttype": 42,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMDIR = os.path.join(ROOT, "data", "LB_DLPU", "dem_tiles")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
TWO_PI = 2 * np.pi
PHASE_CMAP = "twilight"

# Per-regime pipeline settings: (DEM tile, ambiguity height [m/fringe],
# mean coherence, looks, ionosphere on, deformation source scale, seed).
# h_amb (m/fringe) and iono RMS are chosen so the pipeline STAGES stay legible
# (a handful of fringes), not to maximise difficulty; geometry is representative.
REGIMES = {
    "NISAR (20 m, iono on)": dict(
        tile="Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif",   # Everest region
        h_amb=2600.0, coh=0.60, looks=20, iono=True, iono_rms=3.0, seed=7),
    "UAVSAR (6 m, iono off)": dict(
        tile="Copernicus_DSM_COG_10_N37_00_W119_00_DEM.tif",   # Sierra Nevada
        h_amb=1200.0, coh=0.40, looks=12, iono=False, iono_rms=0.0, seed=3),
}


def read_dem(tile, n=256):
    path = os.path.join(DEMDIR, tile)
    with rasterio.open(path) as ds:
        h = ds.read(1, out_shape=(n, n), resampling=Resampling.bilinear).astype(np.float64)
        nod = ds.nodata
    if nod is not None:
        h[h == nod] = np.nan
    h[~np.isfinite(h)] = np.nanmedian(h)
    return h


def build_stages(cfg):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(cfg["seed"])
    dem = read_dem(cfg["tile"])
    # Topographic phase from the DEM: one 2pi fringe per ambiguity height. Smooth
    # the DEM lightly first -- the sensor posting (20/6 m) does not resolve every
    # 30 m drainage pixel, so this keeps the topo fringes physical, not aliased.
    dem_s = gaussian_filter(dem, 1.6)
    topo = TWO_PI * (dem_s - dem_s.mean()) / cfg["h_amb"]

    syn = SyntheticConfig(img_size=256, wavelength_m=NISAR_WAVELENGTH_M, looks=cfg["looks"],
                          topo_max_rad=(5.0, 30.0), atm_std_rad=(0.0, 1.2),
                          flat_max_rad=(0.0, 4.0), difficulty=0.6)
    gen = SyntheticInterferogramGenerator(syn)
    defo = gen._deformation_phase(rng)
    atm = gen._atm_phase(rng)
    iono = (ionospheric_field(256, 256, rng, psd_exponent=-2.39, rms=cfg["iono_rms"])
            if cfg["iono"] else np.zeros((256, 256)))

    absolute = topo + defo + atm + iono
    wrapped = np.angle(np.exp(1j * absolute))
    gamma = np.clip(rng.beta(3.58, 2.37, size=(256, 256)) if cfg["coh"] > 0.5
                    else rng.beta(1.91, 2.85, size=(256, 256)), 0.05, 0.995)
    sigma = np.sqrt((1.0 - gamma ** 2) / (2.0 * cfg["looks"] * gamma ** 2))
    noisy = np.angle(np.exp(1j * (absolute + sigma * rng.standard_normal((256, 256)))))
    return dict(dem=dem, topo=topo, absolute=absolute, coherence=gamma,
                wrapped=wrapped, noisy=noisy)


def _regime_panels(s):
    return [
        ("DEM (m)", s["dem"], "terrain", None, None),
        ("Topographic phase", s["topo"], PHASE_CMAP, None, None),
        ("Absolute phase", s["absolute"], "viridis", None, None),
        ("Coherence", s["coherence"], "gray", 0.0, 1.0),
        ("Wrapped phase", s["wrapped"], PHASE_CMAP, -np.pi, np.pi),
        ("Noisy wrapped phase", s["noisy"], PHASE_CMAP, -np.pi, np.pi),
    ]


def main():
    """One figure, two regime blocks in a single frame split by one divider line."""
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    order = ["NISAR (20 m, iono on)", "UAVSAR (6 m, iono off)"]
    fig = plt.figure(figsize=(7.16, 6.9))
    subfigs = fig.subfigures(2, 1, hspace=0.0)
    for sf, name, letter in zip(subfigs, order, "ab"):
        s = build_stages(REGIMES[name])
        axes = sf.subplots(2, 3)
        # margins leave room INSIDE the frame for titles, colorbars and the (a)/(b) tag
        sf.subplots_adjust(left=0.05, right=0.93, top=0.86, bottom=0.06,
                           wspace=0.42, hspace=0.24)
        for ax, (title, img, cmap, vmin, vmax) in zip(axes.ravel(), _regime_panels(s)):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            cb = sf.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6.5)
        sf.text(0.016, 0.965, f"({letter})", fontsize=13, fontweight="bold",
                va="top", ha="left")
    # ONE outer frame around both blocks + a SINGLE divider line between them
    fig.add_artist(Rectangle((0.008, 0.008), 0.984, 0.984, transform=fig.transFigure,
                             fill=False, edgecolor="0.2", linewidth=1.8, zorder=30))
    fig.add_artist(Line2D([0.008, 0.992], [0.5, 0.5], transform=fig.transFigure,
                          color="0.2", linewidth=1.1, zorder=30))
    p = os.path.join(OUT, "fig1_pipeline.png")
    fig.savefig(p, dpi=300)                        # no tight bbox: frame stays at fig coords
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    print("Wrote", os.path.relpath(p, ROOT), "(+ .pdf)")


if __name__ == "__main__":
    main()
