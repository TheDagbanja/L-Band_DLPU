---
license: cc-by-4.0
language:
  - en
pretty_name: "LB-DLPU: L-band / NISAR Phase-Unwrapping Benchmark"
tags:
  - insar
  - sar
  - phase-unwrapping
  - remote-sensing
  - geoscience
  - nisar
  - uavsar
  - l-band
task_categories:
  - image-to-image
size_categories:
  - 10K<n<100K
---

# LB-DLPU — An L-band / NISAR Benchmark for InSAR Phase Unwrapping

![LB-DLPU: example wrapped phase (input) and ground-truth unwrapped phase (target) across regimes and difficulty strata](assets/preview.png)

LB-DLPU is a physically-simulated benchmark for **interferometric SAR (InSAR)
phase unwrapping** at **L-band**, calibrated to the **NISAR** (spaceborne, 20 m)
and **UAVSAR** (airborne, 6 m) regimes. Each of the 10,000 patches ships with the
wrapped phase, the ground-truth absolute phase, coherence, per-edge integer
ambiguity labels, residues, and a validity mask — everything needed to train,
validate, and test learning-based *and* classical unwrappers.

Two properties set it apart from existing (C-band, RMSE-only) PU datasets:

- **Well-posedness certificate.** Every scene is *provably* recoverable: a
  noiseless-oracle minimum-cost-flow (MCF) unwrapper reconstructs each patch to
  within **0.035 rad** given the correct per-edge costs (100% of 10,000 scenes
  pass; max clean-oracle RMSE = 0.0347 rad). Any error a method incurs is
  attributable to the method, not to an unsolvable target.
- **L-band-specific difficulty.** Calibrated coherence (Beta fits to real
  granules), Cramér–Rao phase noise, an ionospheric screen (NISAR), and steep
  near-fault gradients that push the true per-edge ambiguity into the
  five-arc range {−2,…,+2}.

## Dataset at a glance

| | |
|---|---|
| Patches | 10,000 (256 × 256) |
| Splits | train 8,000 / val 1,000 / test 1,000 |
| Regimes | NISAR (20 m, ionosphere on), UAVSAR (6 m, ionosphere off) |
| Difficulty strata | smooth (3,090) / mixed (4,015) / dense (2,895) |
| Labels | absolute phase, per-edge ambiguity (kx, ky), residues |
| Quicklooks | wrapped-phase PNG per patch |
| Topography | 29 Copernicus GLO-30 DEM tiles, whole-tile split (no terrain leakage) |

Per-regime statistics (from `datasheet.md`):

| sensor | n | mean coherence | posting | residues/Mpix (sim) | residues/Mpix (real) |
|---|---|---|---|---|---|
| nisar | 5,940 | 0.57 | 20 m | 19,535 | 18,100 |
| uavsar | 4,060 | 0.40 | 6 m | 33,948 | 28,794 |

## Directory layout

```
LB_DLPU/
├── README.md              # this card
├── datasheet.md           # auto-generated statistics + well-posedness certificate
├── dem_manifest.csv       # DEM tile → split assignment (tile, split, region, regime)
├── index.jsonl            # one JSON record per patch (metadata, no arrays)
├── assets/                # figures used in this card
│   ├── preview.png
│   └── dem_tiles_map.png
├── sim/
│   ├── train/  000000.h5 … 007999.h5      (8,000)
│   ├── val/    008000.h5 … 008999.h5      (1,000)
│   └── test/   009000.h5 … 009999.h5      (1,000)
└── sim_wrapped_png/       # wrapped-phase quicklooks, mirroring sim/
    ├── train/  000000.png … 007999.png
    ├── val/    008000.png …
    └── test/   009000.png …
```

Patch ids are shared across `sim/<split>/<id>.h5` and
`sim_wrapped_png/<split>/<id>.png`.

## Per-patch HDF5 schema

Each `.h5` file (≈0.7 MB) contains:

| dataset | shape | dtype | description |
|---|---|---|---|
| `psi` | (256, 256) | float32 | **wrapped phase** (network input), radians in (−π, π]; noisy |
| `phi` | (256, 256) | float32 | **ground-truth absolute (unwrapped) phase**, radians |
| `coherence` | (256, 256) | float32 | interferometric coherence, [0, 1] |
| `kx` | (256, 255) | int8 | horizontal per-edge integer ambiguity, {−2,…,+2} |
| `ky` | (255, 256) | int8 | vertical per-edge integer ambiguity, {−2,…,+2} |
| `residues` | (255, 255) | int8 | Goldstein loop residues of `psi`, {−1, 0, +1} |
| `water_mask` | (256, 256) | bool | invalid / no-signal pixels (excluded from metrics) |

`psi` is the noisy wrapped observation and `phi` the clean target; they are
**not** exactly congruent (that is the noise the unwrapper must survive). The
per-edge labels satisfy `Δφ_e = W(Δψ)_e + 2π·k_e`, where `W` wraps to (−π, π].

**Per-patch attributes** (HDF5 `.attrs`): `sensor` (nisar|uavsar), `difficulty`
(smooth|mixed|dense), `mean_coherence`, `px_m` (pixel spacing), `NL` (looks),
`residue_count`, `residues_per_mp`, `max_grad_rad_per_px`, `frac_edges_k1`,
`frac_edges_k2`, `label_clip_frac`, `water_frac`, `clean_oracle_rmse`, `seed`,
and `components` (JSON: topography / deformation / atmosphere / ionosphere
provenance).

## `index.jsonl`

One record per patch with the same metadata as the HDF5 attributes plus `id` and
`split`, for fast filtering without opening every file:

```json
{"id": "000000", "split": "train", "sensor": "nisar", "difficulty": "smooth",
 "mean_coherence": 0.56, "NL": 8, "px_m": 20.0, "residue_count": 2182,
 "residues_per_mp": 33294.7, "clean_oracle_rmse": 0.0, "seed": 939529293,
 "components": {"topo_src": "dem", "topo": {"B_perp_m": 32.97}, ...}}
```

## Splits

Train / val / test are **disjoint by whole DEM tile**: the 29 Copernicus GLO-30
tiles (worldwide tectonic, volcanic, and glacial terrain) are partitioned
17 / 5 / 7, so no terrain is shared across splits and the test set measures
generalization to unseen geography. The exact tile → split assignment is in
`dem_manifest.csv`.

![Global distribution of the 29 GLO-30 DEM tiles, colored by split (17 train / 5 validation / 7 test)](assets/dem_tiles_map.png)

## Loading

```python
import h5py, glob

def load_patch(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f}, dict(f.attrs)

for p in sorted(glob.glob("sim/test/*.h5"))[:1]:
    arrays, attrs = load_patch(p)
    psi, phi = arrays["psi"], arrays["phi"]          # input, target
    print(attrs["sensor"], attrs["difficulty"], psi.shape)
```

Quicklooks are plain PNGs:

```python
from PIL import Image
Image.open("sim_wrapped_png/test/009000.png")        # wrapped-phase preview
```

## Intended use

Training and benchmarking L-band phase-unwrapping methods — deep networks
(wrap-count regression/classification, gradient estimation) and classical /
minimum-cost-flow solvers — with per-regime × difficulty evaluation. The
well-posedness certificate makes the test split a fair ceiling reference; the
per-edge labels support both pixel-wise and edge-wise supervision.

## Reference baselines

A method-blind evaluation harness scores every unwrapper on the identical test
split, per regime × difficulty, on **RMSE, MAE, PSNR, SSIM**, a cycle-slip
(jump) rate, residue count, and five-arc |k|≥2 edge accuracy. Reference findings
across classical, minimum-cost-flow, and in-domain-trained deep baselines:

- Deep networks train cleanly on this data and **outperform classical and
  statistical solvers** (e.g. SNAPHU) — especially on the dense/high-gradient
  stratum, which the benchmark is designed to stress.
- A **learned-cost MCF** approaches the certified oracle ceiling while remaining
  residue-free by construction; gradient-domain methods leave residues.
- The benchmark is **not saturated**: a gap to the oracle remains for every
  deployable method, and the strata form a genuine difficulty gradient
  (error widens smooth → mixed → dense).

The full 13-method leaderboard, metric definitions, and significance tests are in
the accompanying code/paper.

## Limitations and considerations

- **Synthetic ground truth.** `phi` is physically modelled (topography from real
  Copernicus DEMs; coherence, noise, and ionosphere calibrated to real NISAR /
  UAVSAR granules) but is not field-validated absolute truth. It is intended for
  supervised training and controlled benchmarking; real-scene generalization
  should be assessed separately.
- **Two regimes.** Sensor characteristics are approximated for NISAR-like and
  UAVSAR-like acquisitions; other L-band sensors may differ.
- **No decorrelation-only patches.** Every scene is certified recoverable given
  correct costs; the benchmark isolates cost/prior estimation, not
  irrecoverable-noise regimes.

## Provenance and licensing

- **Topography:** Copernicus GLO-30 DEM (© ESA / Copernicus; free and open,
  attribution required).
- **Noise / coherence / ionosphere models:** calibrated to real NISAR L2 GUNW
  and UAVSAR granules.
- **Simulated phase, labels, and quicklooks:** this release.

**License: CC-BY-4.0.** Free to use, share, and adapt with attribution.
The Copernicus GLO-30 DEM attribution above must be retained. If you intend a
different license, update both this line and the `license:` field in the card
metadata before publishing.

## Citation

```bibtex
@misc{lb_dlpu,
  title  = {LB-DLPU:  Well-Posedness-Certified, Sensor-Calibrated L-Band Benchmark Dataset for Deep-Learning InSAR Phase Unwrapping},
  author = {Dagbanja S., Qian J.},
  year   = {2026},
  note   = {\url{https://huggingface.co/datasets/TheDagbanja/LB-DLPU}}
}
```
