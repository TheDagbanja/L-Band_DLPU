"""Raw ``sim_2`` patch loader for the harness (BASELINE_HARNESS_SPEC.md s4).

Reads each ``.h5`` patch directly -- ``psi, phi, coherence, water_mask`` and the
stored per-edge labels ``kx, ky`` -- plus the ``.attrs`` needed to place the
patch in its report cell (``sensor`` in {nisar, uavsar}, ``difficulty`` in
{smooth, mixed, dense}). Deliberately independent of
:class:`src.lband_dataset.LBandSimDataset` (which recomputes labels and drops
``.attrs``): the harness must read the *generation* conventions verbatim so the
residue definition and the regime/difficulty cells are exactly the ones the
datasheet reports.

Determinism (spec s1.5): patches are returned in sorted-id order, so every
method sees the identical sequence and paired significance tests are valid.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np

_SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}
_REAL_SENSOR_DIRS = {"nisar": "real_nisar", "uavsar": "real_uavsar"}
_TWO_PI = 2.0 * np.pi


@dataclass
class Patch:
    """One evaluation patch, everything a method or metric needs."""

    id: str
    path: str
    regime: str                       # sensor: nisar | uavsar
    difficulty: str                   # smooth | mixed | dense
    psi: np.ndarray                   # (H, W) wrapped phase
    phi: Optional[np.ndarray]         # (H, W) ground-truth absolute (sim only)
    coherence: Optional[np.ndarray]   # (H, W) in [0, 1]
    valid: np.ndarray                 # (H, W) bool valid-pixel mask
    kx_true: Optional[np.ndarray]     # (H, W-1) stored horizontal edge ambiguity
    ky_true: Optional[np.ndarray]     # (H-1, W) stored vertical edge ambiguity
    attrs: Dict = field(default_factory=dict)


def _read(path: str) -> Patch:
    import h5py

    with h5py.File(path, "r") as f:
        attrs = {k: _to_py(v) for k, v in f.attrs.items()}
        psi = np.asarray(f["psi"][:], dtype=np.float64)
        phi = np.asarray(f["phi"][:], dtype=np.float64) if "phi" in f else None
        coh = np.asarray(f["coherence"][:], dtype=np.float64) if "coherence" in f else None
        water = np.asarray(f["water_mask"][:], dtype=bool) if "water_mask" in f else None
        kx = np.asarray(f["kx"][:], dtype=np.int64) if "kx" in f else None
        ky = np.asarray(f["ky"][:], dtype=np.int64) if "ky" in f else None

    valid = np.isfinite(psi)
    if phi is not None:
        valid &= np.isfinite(phi)
    if water is not None:
        valid &= ~water

    pid = os.path.splitext(os.path.basename(path))[0]
    return Patch(
        id=pid, path=path,
        regime=str(attrs.get("sensor", "unknown")),
        difficulty=str(attrs.get("difficulty", "unknown")),
        psi=psi, phi=phi, coherence=coh, valid=valid,
        kx_true=kx, ky_true=ky, attrs=attrs,
    )


def _to_py(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def list_patches(data_root: str, split: str = "test") -> List[str]:
    """Sorted list of patch ``.h5`` paths for ``<data_root>/sim/<split>``."""
    if split not in _SPLIT_DIRS:
        raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got {split!r}.")
    d = os.path.join(data_root, "sim", _SPLIT_DIRS[split])
    if not os.path.isdir(d):
        raise FileNotFoundError(f"split dir not found: {d!r}")
    files = sorted(glob.glob(os.path.join(d, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 patches in {d!r}")
    return files


def iter_patches(
    data_root: str,
    split: str = "test",
    max_per_cell: Optional[int] = None,
) -> Iterator[Patch]:
    """Yield patches in deterministic id order.

    ``max_per_cell`` caps how many patches are taken per (regime, difficulty)
    cell -- a smoke-test knob (spec s8 ``--max N``) that keeps every cell
    represented rather than truncating the whole split.
    """
    counts: Dict = {}
    for path in list_patches(data_root, split):
        p = _read(path)
        if max_per_cell is not None:
            key = (p.regime, p.difficulty)
            c = counts.get(key, 0)
            if c >= max_per_cell:
                continue
            counts[key] = c + 1
        yield p


# ---------------------------------------------------------------------------
# Real L-band patches (real/real_{nisar,uavsar}) -- for the real-data test.
# ---------------------------------------------------------------------------
def _edge_labels(psi, phi):
    """GT per-edge integer ambiguity from (psi, phi), clipped to the five-arc range."""
    def wrap(x):
        return (x + np.pi) % _TWO_PI - np.pi
    wgx = wrap(psi[:, 1:] - psi[:, :-1]); wgy = wrap(psi[1:, :] - psi[:-1, :])
    kx = np.clip(np.round(((phi[:, 1:] - phi[:, :-1]) - wgx) / _TWO_PI), -2, 2).astype(np.int64)
    ky = np.clip(np.round(((phi[1:, :] - phi[:-1, :]) - wgy) / _TWO_PI), -2, 2).astype(np.int64)
    return kx, ky


def _read_real(path: str, sensor: str) -> Patch:
    import h5py

    with h5py.File(path, "r") as f:
        psi = np.asarray(f["psi"][:], dtype=np.float64)
        ref = np.asarray(f["ref"][:], dtype=np.float64) if "ref" in f else None
        coh = np.asarray(f["coherence"][:], dtype=np.float64) if "coherence" in f else None
        conncomp = np.asarray(f["conncomp"][:]) if "conncomp" in f else None

    # Real patches store psi = wrap(ref) (the provider product re-wrapped): GT is
    # ref, but there is NO interferometric noise -- this tests generalisation to
    # real fringe/coherence STRUCTURE, not to real noise (see Fig. 7 for noise).
    valid = np.isfinite(psi)
    if ref is not None:
        valid &= np.isfinite(ref)
    if conncomp is not None:
        valid &= conncomp != 0

    # Fill invalid pixels so the solvers don't choke; metrics use `valid` to
    # exclude them. MCF stays globally residue-free regardless of the fill.
    psi_f = np.nan_to_num(psi, nan=0.0)
    phi_f = np.nan_to_num(ref, nan=0.0) if ref is not None else None
    coh_f = np.clip(np.nan_to_num(coh, nan=0.0), 0.0, 1.0) if coh is not None else None
    kx = ky = None
    if phi_f is not None:
        kx, ky = _edge_labels(psi_f, phi_f)

    pid = os.path.splitext(os.path.basename(path))[0]
    return Patch(id=pid, path=path, regime=sensor, difficulty="real",
                 psi=psi_f, phi=phi_f, coherence=coh_f, valid=valid,
                 kx_true=kx, ky_true=ky, attrs={"sensor": sensor, "valid_frac": float(valid.mean())})


def iter_real_patches(data_root: str, sensors=("nisar", "uavsar"),
                      max_per_cell: Optional[int] = None,
                      min_valid: float = 0.0) -> Iterator[Patch]:
    """Yield real L-band patches (psi=wrap(ref)) in sorted order, per sensor.

    ``min_valid`` skips patches whose valid fraction is below the threshold
    (NISAR patches often have nodata / disconnected regions).
    """
    # real/ sits at the LB_DLPU root; accept either that root or a sim_* subdir.
    roots = [data_root, os.path.dirname(data_root.rstrip("/\\"))]
    base = next((r for r in roots if os.path.isdir(os.path.join(r, "real"))), data_root)
    for sensor in sensors:
        d = os.path.join(base, "real", _REAL_SENSOR_DIRS[sensor])
        if not os.path.isdir(d):
            continue
        n = 0
        for path in sorted(glob.glob(os.path.join(d, "*.h5"))):
            p = _read_real(path, sensor)
            if p.attrs["valid_frac"] < min_valid:
                continue
            if max_per_cell is not None and n >= max_per_cell:
                break
            n += 1
            yield p
