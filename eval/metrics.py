"""Metrics -- the SINGLE SOURCE OF TRUTH (BASELINE_HARNESS_SPEC.md s3).

Every method's output field ``phi_hat`` goes through the *same* functions here,
computed on the *same* valid-pixel mask. No per-method post-processing: the
differentiation "MCF leaves 0 residues, DL leaves residues" is only defensible
if the residue counter cannot tell which method produced the field.

--------------------------------------------------------------------------
Residue count -- the discriminating definition (deviation from spec s3.1)
--------------------------------------------------------------------------
The spec's literal pseudo-code counts loops of the *node-wise* implied
ambiguity ``k_hat = round((phi_hat - psi) / 2pi)`` differenced across edges.
That is degenerate: the discrete curl of any node field's own gradient
telescopes to *exactly* zero, so ``count_nonzero(curl + R_psi)`` collapses to
``count_nonzero(R_psi)`` -- the **input** residue count -- for *every* method,
MCF and DL alike (verified numerically). It would make the headline column
identical for all rows.

The counter that actually measures integrability of a method's output uses the
per-edge *implied ambiguity relative to the wrapped gradient*:

    n_x = round( ( (phi_hat[:,1:]-phi_hat[:,:-1]) - W(psi[:,1:]-psi[:,:-1]) ) / 2pi )
    n_y = round( ( (phi_hat[1:,:]-phi_hat[:-1,:]) - W(psi[1:,:]-psi[:-1,:]) ) / 2pi )

The method's *corrected* gradient field is ``g = W(grad psi) + 2pi n``; it is
curl-free (a valid single-valued unwrapping) iff

    curl(n) + R_psi == 0   at every loop,     R_psi = mcf.residues(psi)

because ``curl(g)/2pi = curl(n) + R_psi``. MCF methods enforce exactly this
flow constraint, so they return 0 by construction; a DL/LS regression whose
rounded per-edge jumps are not loop-consistent returns >0. This is the same
``curl + R_psi`` structure the spec intends, with the correct (edge, not node)
ambiguity -- and it reduces to the MCF solver's own feasibility identity, so
MCF rows are provably 0. See :func:`residue_count`; the self-test in
``eval_baselines.py`` verifies MCF->0, a seeded discontinuity->known count, and
a smooth ramp->0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src import mcf

TWO_PI = mcf.TWO_PI


# ---------------------------------------------------------------------------
# Edge helpers (one definition, shared by residue count and |k| accuracy)
# ---------------------------------------------------------------------------
def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % TWO_PI - np.pi


def implied_edge_k(phi_hat: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-edge integer ambiguity a method's output implies vs the wrapped input.

    ``n_x`` shape ``(H, W-1)`` (horizontal edges), ``n_y`` shape ``(H-1, W)``
    (vertical) -- the same layout as the stored ``kx``/``ky`` labels and
    :mod:`src.utils`. Method-blind: derived identically from ``phi_hat`` for
    every method, MCF or DL.
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    wgx = wrap(psi[:, 1:] - psi[:, :-1])
    wgy = wrap(psi[1:, :] - psi[:-1, :])
    n_x = np.round(((phi_hat[:, 1:] - phi_hat[:, :-1]) - wgx) / TWO_PI).astype(np.int64)
    n_y = np.round(((phi_hat[1:, :] - phi_hat[:-1, :]) - wgy) / TWO_PI).astype(np.int64)
    return n_x, n_y


def _curl_edge(n_x: np.ndarray, n_y: np.ndarray) -> np.ndarray:
    """Loop curl of an edge field, shape ``(H-1, W-1)`` -- matches mcf.residues layout."""
    return n_x[:-1, :] + n_y[:, 1:] - n_x[1:, :] - n_y[:, :-1]


def valid_plaquettes(valid: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Boolean ``(H-1, W-1)`` mask: a loop is valid iff all 4 corner pixels are valid."""
    H, W = shape
    if valid is None:
        return np.ones((H - 1, W - 1), dtype=bool)
    v = np.asarray(valid, dtype=bool)
    return v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]


# ---------------------------------------------------------------------------
# 3.1 Residue count (PRIMARY -- the differentiator)
# ---------------------------------------------------------------------------
def residue_count(
    phi_hat: np.ndarray,
    psi: np.ndarray,
    valid: Optional[np.ndarray] = None,
    R_psi: Optional[np.ndarray] = None,
) -> int:
    """Number of loops where ``phi_hat`` is not integrable-consistent with ``psi``.

    ``R_psi`` (from :func:`src.mcf.residues`) may be passed to avoid recomputing
    it per method. Counts only loops whose four corner pixels are all valid.
    MCF methods return 0 by construction; DL/LS regressions return >0.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if R_psi is None:
        R_psi = mcf.residues(psi)
    n_x, n_y = implied_edge_k(phi_hat, psi)
    defect = _curl_edge(n_x, n_y) + R_psi
    vmask = valid_plaquettes(valid, psi.shape)
    return int(np.count_nonzero((defect != 0) & vmask))


# ---------------------------------------------------------------------------
# 3.2 RMSE vs ground truth (reference-offset removed)
# ---------------------------------------------------------------------------
def aligned_rmse(
    phi_hat: np.ndarray, phi: np.ndarray, valid: Optional[np.ndarray] = None
) -> float:
    """RMSE (radians) after removing a global constant (incl. any 2pi.c) offset.

    ``off = median((phi_hat - phi)[valid])``; ``rmse = sqrt(mean(((phi_hat-off)-phi)^2))``.
    Adding ``2pi.c`` to ``phi_hat`` does not change the result (offset invariance).
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    d = phi_hat - phi
    m = np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not m.any():
        return float("nan")
    off = np.median(d[m])
    r = (d[m] - off)
    return float(np.sqrt(np.mean(r * r)))


# ---------------------------------------------------------------------------
# 3.2a Reconstruction-quality suite (MAE, PSNR, SSIM) -- the DL-PU standard set
# ---------------------------------------------------------------------------
def reconstruction_quality(phi_hat: np.ndarray, phi: np.ndarray,
                           valid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """MAE (rad), PSNR (dB) and SSIM after the same reference-offset removal as RMSE.

    These are the standard image-reconstruction metrics reported by deep-learning
    phase-unwrapping papers, so a benchmark that includes them is directly
    comparable to that literature. PSNR/SSIM use the per-patch ground-truth range
    (max-min over valid pixels) as the data range; invalid pixels are filled with
    the ground truth for SSIM so they inject no error.
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    d = phi_hat - phi
    m = np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not m.any():
        return {"mae": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
    off = np.median(d[m])
    aligned = phi_hat - off
    res = (aligned - phi)[m]
    mae = float(np.mean(np.abs(res)))
    mse = float(np.mean(res * res))
    rng = float(phi[m].max() - phi[m].min())
    if mse == 0:
        psnr = float("inf")
    elif rng > 0:
        psnr = float(10.0 * np.log10(rng * rng / mse))
    else:
        psnr = float("nan")
    ssim = float("nan")
    try:
        from skimage.metrics import structural_similarity as _ssim
        if rng > 0:
            filled = np.where(m, aligned, phi)          # invalid -> GT (zero error)
            ssim = float(_ssim(phi, filled, data_range=rng))
    except Exception:
        pass
    return {"mae": mae, "psnr": psnr, "ssim": ssim}


# ---------------------------------------------------------------------------
# 3.2b Cycle-slip rate (the honest MCF-vs-DL differentiator)
# ---------------------------------------------------------------------------
def jump_rate(phi_hat: np.ndarray, phi: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Fraction of valid pixels off by >=1 whole 2pi cycle after offset alignment.

    ``round(((phi_hat - off) - phi) / 2pi) != 0``. Unlike residue count, this
    separates the *accurate* congruent methods (MCF / ours -> ~0) from the
    *cycle-slipping* congruent methods (per-pixel-K DL -> >0): both are
    residue-free, but only the former places the right integer cycle. Offset is
    the same median used by :func:`aligned_rmse`, so it is 2pi-shift invariant.
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    d = phi_hat - phi
    m = np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not m.any():
        return float("nan")
    off = np.median(d[m])
    return float(np.mean(np.round((d[m] - off) / TWO_PI) != 0))


# ---------------------------------------------------------------------------
# 3.3 Five-arc |k| edge accuracy (where per-edge k is available)
# ---------------------------------------------------------------------------
def k_edge_accuracy(
    n_x: np.ndarray,
    n_y: np.ndarray,
    kx_true: np.ndarray,
    ky_true: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Edge-ambiguity accuracy vs stored labels, overall and on the ``|k|>=2`` subset.

    The ``|k|>=2`` accuracy is the five-arc discriminator (the hard edges that
    justify the k_max=2 extension). ``n_x``/``n_y`` are the method's implied
    per-edge ambiguities from :func:`implied_edge_k`; ``kx_true``/``ky_true`` the
    stored integer labels (same shapes).
    """
    kx_true = np.asarray(kx_true).astype(np.int64)
    ky_true = np.asarray(ky_true).astype(np.int64)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        vx = v[:, :-1] & v[:, 1:]
        vy = v[:-1, :] & v[1:, :]
    else:
        vx = np.ones_like(kx_true, dtype=bool)
        vy = np.ones_like(ky_true, dtype=bool)

    correct = np.concatenate([(n_x == kx_true)[vx].ravel(), (n_y == ky_true)[vy].ravel()])
    acc = float(correct.mean()) if correct.size else float("nan")

    hard_x = vx & (np.abs(kx_true) >= 2)
    hard_y = vy & (np.abs(ky_true) >= 2)
    hard = np.concatenate([(n_x == kx_true)[hard_x].ravel(), (n_y == ky_true)[hard_y].ravel()])
    acc2 = float(hard.mean()) if hard.size else float("nan")
    return {"k_acc": acc, "k2_acc": acc2, "n_hard_edges": int(hard.size)}


# ---------------------------------------------------------------------------
# 3.4 Coherence-stratified RMSE (optional column)
# ---------------------------------------------------------------------------
def coherence_stratified_rmse(
    phi_hat: np.ndarray,
    phi: np.ndarray,
    coherence: np.ndarray,
    valid: Optional[np.ndarray] = None,
    thresh: float = 0.5,
) -> Dict[str, float]:
    """RMSE restricted to high- vs low-coherence pixels (gamma > ``thresh``).

    Uses the *same* global offset (from all valid pixels) for both strata so the
    split does not re-reference each subset. NaN where a stratum is empty.
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    coh = np.asarray(coherence, dtype=np.float64)
    d = phi_hat - phi
    m = np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not m.any():
        return {"rmse_hi_coh": float("nan"), "rmse_lo_coh": float("nan")}
    off = np.median(d[m])
    r = d - off

    def _rmse(sel):
        s = sel & m
        return float(np.sqrt(np.mean(r[s] ** 2))) if s.any() else float("nan")

    return {"rmse_hi_coh": _rmse(coh > thresh), "rmse_lo_coh": _rmse(coh <= thresh)}
