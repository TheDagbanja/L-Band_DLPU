"""L-band synthetic interferogram generator (NEW, this work) -- proposal §8.2.

Reuses the GRSL letter's physical deformation/topography/atmosphere engine
(:mod:`src.synthetic_engine`, reused unchanged from our GRSL letter's engine) for the
absolute phase ``phi``, then replaces the noise model with the L-band
derivation of proposal §2.3:

* **Coherence**: drawn from ``Beta(3.58, 2.37)`` (mean ~0.60), calibrated from
  real NISAR L2 GUNW granules (``data/full_scenes/gunw_all``), instead of the
  C-band engine's power-law coherence field.
* **Phase noise**: the Cramer-Rao bound (Eq. 7 / proposal Listing 2) with
  ``n_looks=20`` (NISAR's higher-look GUNW product vs. Sentinel-1's 8).
* **Ionospheric screen**: an additive, spatially-correlated phase-ramp field
  with power spectral density ``S(f) ~ f^-2.39`` and RMS in ``[1.7, 19.5]``
  rad (proposal §2.3.3, Listing 3) -- the long-wavelength streak noise that is
  ``~4.3x`` more prominent at L-band than C-band for the same TEC gradient.
* **Wider deformation gradient** (``max_grad`` up to ``4*pi``) so the true
  ambiguity spans ``{-2,...,+2}``, exercising the five-arc MCF (Pillar 3)
  rather than only ``{-1,0,+1}``.

Matches the on-disk schema of the pre-generated release
(``data/LB_DLPU/sim/{train,val,test}/*.h5``: ``psi, phi, coherence, kx, ky,
residues, water_mask``) so this module can extend or regenerate that dataset
identically -- see :func:`generate_dataset`.

Run ``python -m src.synth_lband`` for a physics self-test (mirrors
:mod:`src.synthetic_engine`'s own self-test conventions).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import mcf
from .synthetic_engine import SyntheticConfig, SyntheticInterferogramGenerator

PI = math.pi
TWO_PI = 2.0 * PI

#: NISAR L-band center frequency 1257.5 MHz -> wavelength (proposal: "lambda ~ 24 cm").
NISAR_WAVELENGTH_M = 299_792_458.0 / 1.2575e9   # ~0.23836 m


@dataclass
class LBandConfig:
    """Parameters for the L-band noise/ionosphere overlay (proposal §2.3, Listing 2-3)."""

    img_size: int = 256
    n_looks: int = 20                    # NISAR GUNW equivalent looks
    coherence_beta_a: float = 3.58        # Beta(a, b) coherence fit to real GUNW granules
    coherence_beta_b: float = 2.37        # mean ~= a/(a+b) ~= 0.60
    iono_psd_exponent: float = -2.39      # ionospheric screen spectral slope
    iono_rms_range: tuple = (1.7, 19.5)   # rad
    water_prob: float = 0.06              # fraction of scenes with a water patch
    water_patches: int = 2
    max_grad: float = 4.0 * PI            # allow |k|<=2 (five-arc) deformation gradients


def ionospheric_field(
    H: int, W: int, rng: np.random.Generator, psd_exponent: float = -2.39, rms: float = 1.0,
) -> np.ndarray:
    """Spatially-correlated ionospheric phase screen, PSD ``~ f^psd_exponent``.

    Mirrors proposal Listing 3's ``ionospheric_field``: a zero-mean Gaussian
    random field shaped by a radial power-law spectrum, rescaled to the
    requested RMS (rather than a fixed peak-to-peak scale) so the sampled
    ``[1.7, 19.5]`` rad range (Table 2 caption) is exact.
    """
    fx = np.fft.fftfreq(W)[None, :]
    fy = np.fft.fftfreq(H)[:, None]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1e-6
    psd = f ** psd_exponent
    spectrum = psd * (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    field = np.real(np.fft.ifft2(spectrum))
    field = field - field.mean()
    std = field.std()
    return field * (rms / std) if std > 0 else field


def _water_mask(H: int, W: int, rng: np.random.Generator, n_patches: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for _ in range(n_patches):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        r = rng.uniform(0.04, 0.15) * min(H, W)
        yy, xx = np.ogrid[:H, :W]
        mask |= ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    return mask


def generate_lband_pair(
    H: int = 256, W: int = 256, seed: Optional[int] = None,
    n_looks: int = 20, difficulty: float = 0.8,
    cfg: Optional[LBandConfig] = None,
) -> Dict[str, np.ndarray]:
    """Generate one synthetic L-band ``(psi, phi, coherence, kx, ky, residues,
    water_mask)`` pair -- NISAR-calibrated, matching ``data/LB_DLPU/sim`` schema.

    Mirrors proposal Listing 2 (``synth_lband.py``): deformation phase from
    the shared physical engine (Okada/Mogi/subsidence + topography +
    atmosphere), coherence from ``Beta(3.58, 2.37)``, phase noise via the
    Cramer-Rao bound at ``n_looks`` looks, and an added ionospheric screen.
    """
    cfg = cfg or LBandConfig(img_size=H)
    rng = np.random.default_rng(seed)

    # 1) True deformation + topo + atmosphere phase (allow |grad phi| up to
    # cfg.max_grad, i.e. up to 2*(cfg.max_grad/2pi) cycles/pixel -> |k| up to 2).
    syn_cfg = SyntheticConfig(
        img_size=H, wavelength_m=NISAR_WAVELENGTH_M, looks=n_looks,
        topo_max_rad=(5.0, cfg.max_grad * 6.0), atm_std_rad=(0.0, cfg.max_grad),
        flat_max_rad=(0.0, cfg.max_grad * 3.0), difficulty=difficulty,
    )
    gen = SyntheticInterferogramGenerator(syn_cfg)
    phi = gen._deformation_phase(rng) + gen._topo_phase(rng) + gen._atm_phase(rng) + gen._flat_phase(rng)
    phi = float(difficulty) * (phi - phi.mean())

    # 2) NISAR-calibrated coherence map (Beta fit to real GUNW granules).
    gamma = rng.beta(cfg.coherence_beta_a, cfg.coherence_beta_b, size=(H, W))
    water_mask = np.zeros((H, W), dtype=bool)
    if rng.random() < cfg.water_prob:
        water_mask = _water_mask(H, W, rng, cfg.water_patches)
        gamma = np.where(water_mask, rng.uniform(0.02, 0.15, size=(H, W)), gamma)
    gamma = np.clip(gamma, 0.02, 0.995)

    # 3) Phase noise from the Cramer-Rao bound (Eq. 7).
    sigma = np.sqrt((1.0 - gamma ** 2) / (2.0 * n_looks * gamma ** 2))
    noise = sigma * rng.standard_normal((H, W))

    # 4) Ionospheric screen: f^-2.39 PSD, RMS sampled in [1.7, 19.5] rad.
    rms = rng.uniform(*cfg.iono_rms_range)
    iono = ionospheric_field(H, W, rng, psd_exponent=cfg.iono_psd_exponent, rms=rms)

    # 5) Compose and wrap.
    phi_total = phi + noise + iono
    psi = np.angle(np.exp(1j * phi_total)).astype(np.float32)
    phi = phi_total.astype(np.float32)

    # 6) Per-edge ambiguity labels + plaquette residues (mcf.py convention:
    # kx (H,W-1), ky (H-1,W), residues (H-1,W-1); clamped to five-arc range).
    wgx, wgy = mcf.wrapped_gradients(psi.astype(np.float64))
    dphix = phi[:, 1:] - phi[:, :-1]
    dphiy = phi[1:, :] - phi[:-1, :]
    kx = np.clip(np.round((dphix - wgx) / TWO_PI), -2, 2).astype(np.int8)
    ky = np.clip(np.round((dphiy - wgy) / TWO_PI), -2, 2).astype(np.int8)
    residues = mcf.residues(psi.astype(np.float64)).astype(np.int8)

    return {
        "psi": psi, "phi": phi, "coherence": gamma.astype(np.float32),
        "kx": kx, "ky": ky, "residues": residues, "water_mask": water_mask,
    }


def generate_dataset(
    out_dir: str, n: int, seed0: int = 0, H: int = 256, W: int = 256,
    n_looks: int = 20, difficulty: float = 0.8, start_index: int = 0,
) -> None:
    """Write ``n`` L-band pairs as individually-named ``.h5`` files.

    Filenames zero-padded to 6 digits (``{start_index:06d}.h5``, ...),
    matching ``data/LB_DLPU/sim/{train,val,test}``'s existing convention, so
    this can either populate a fresh split directory or append more samples
    to an existing one (pass ``start_index`` past the current max).
    """
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        idx = start_index + i
        d = generate_lband_pair(H, W, seed=seed0 + idx, n_looks=n_looks, difficulty=difficulty)
        with h5py.File(os.path.join(out_dir, f"{idx:06d}.h5"), "w") as f:
            for k, v in d.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# Self-test (mirrors src.synthetic_engine's physics self-test conventions)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1) Ionospheric field: requested RMS is honoured, zero mean.
    f = ionospheric_field(128, 128, rng, psd_exponent=-2.39, rms=5.0)
    assert abs(f.mean()) < 1e-6
    assert abs(f.std() - 5.0) < 0.5
    print(f"[self-test] ionospheric field: mean={f.mean():.2e} std={f.std():.2f} (target 5.0)")

    # 2) Cramer-Rao noise: n_looks=20 gives tighter phase noise than n_looks=8
    # at matched coherence (more looks -> lower variance).
    from .synthetic_engine import coherence_phase_std
    s20 = coherence_phase_std(np.array([0.6]), 20)[0]
    s8 = coherence_phase_std(np.array([0.6]), 8)[0]
    assert s20 < s8, "more looks must reduce phase noise"
    print(f"[self-test] phase std @ gamma=0.6: n_looks=20 -> {s20:.3f} rad, n_looks=8 -> {s8:.3f} rad")

    # 3) Full pair generation: shapes, ranges, five-arc ambiguity, reproducibility.
    a = generate_lband_pair(128, 128, seed=42, difficulty=0.9)
    b = generate_lband_pair(128, 128, seed=42, difficulty=0.9)
    assert a["psi"].shape == (128, 128) and a["kx"].shape == (128, 127) and a["ky"].shape == (127, 128)
    assert -PI - 1e-3 <= a["psi"].min() and a["psi"].max() <= PI + 1e-3
    assert (0 < a["coherence"]).all() and (a["coherence"] <= 1).all()
    assert np.allclose(a["phi"], b["phi"]), "seeded generation must be reproducible"
    assert np.array_equal(a["kx"], b["kx"]) and np.array_equal(a["ky"], b["ky"])
    # Residue-consistency: the labels must exactly cancel the wrapped gradients'
    # loop curl (same identity Theorem 2.1 relies on).
    residues_from_labels = np.round(
        (a["kx"][:-1, :].astype(np.int64) + a["ky"][:, 1:].astype(np.int64)
         - a["kx"][1:, :].astype(np.int64) - a["ky"][:, :-1].astype(np.int64))
    )
    assert np.array_equal(-residues_from_labels, a["residues"].astype(np.int64)), \
        "ambiguity labels inconsistent with plaquette residues"
    k_range = (int(a["kx"].min()), int(a["kx"].max()), int(a["ky"].min()), int(a["ky"].max()))
    print(f"[self-test] pair: psi range OK, coherence mean={a['coherence'].mean():.2f}, "
          f"k range={k_range}, residue-consistent labels: OK")
    print("[self-test] PASSED")


if __name__ == "__main__":
    _self_test()
