"""InSAR-DLPU (.mat) adapter for the evaluation harness.

Yields the same :class:`eval.data.Patch` objects as the L-band ``.h5`` loader,
so the entire method registry (:mod:`eval.methods`) and the method-blind scorer
(:mod:`eval.metrics`) score InSAR-DLPU (C-band) exactly as they score the L-band
benchmark. This gives one consistent leaderboard across every method family on
the conference-paper's C-band data, with no duplicated scoring logic.

InSAR-DLPU has no coherence band, so ``coherence`` is ``None`` here (the
coherence-fused cost falls back to the learned-only cost, matching the C-band
setup). Splits map to the released folder pairs; ``.mat`` variables are
``input`` (wrapped) / ``output`` (absolute), read via
:func:`src.dataset._load_mat_array` (handles both v5 and v7.3/HDF5).
"""

from __future__ import annotations

import glob
import os
from typing import Iterator, Optional

import numpy as np

from src.dataset import _DLPU_SPLIT_DIRS, _load_mat_array, resolve_dlpu_root

from .data import Patch

_TWO_PI = 2.0 * np.pi


def _edge_labels(psi: np.ndarray, phi: np.ndarray, kmax: int = 1):
    """GT per-edge integer ambiguity from (psi, phi), clipped to the ``kmax`` range.

    ``kmax=1`` reproduces the conference-paper 3-class ambiguity ``{-1,0,+1}``.
    """
    def wrap(x):
        return (x + np.pi) % _TWO_PI - np.pi

    wgx = wrap(psi[:, 1:] - psi[:, :-1])
    wgy = wrap(psi[1:, :] - psi[:-1, :])
    kx = np.clip(np.round(((phi[:, 1:] - phi[:, :-1]) - wgx) / _TWO_PI), -kmax, kmax).astype(np.int64)
    ky = np.clip(np.round(((phi[1:, :] - phi[:-1, :]) - wgy) / _TWO_PI), -kmax, kmax).astype(np.int64)
    return kx, ky


def iter_patches_dlpu(
    dlpu_root: str,
    split: str = "test",
    max_per_cell: Optional[int] = None,
    kmax: int = 1,
) -> Iterator[Patch]:
    """Yield InSAR-DLPU patches in deterministic filename order.

    ``dlpu_root`` may point anywhere above the nested ``*_wrapped`` split folders
    (e.g. ``../InSAR-DLPU/data``); :func:`resolve_dlpu_root` walks down to them.
    ``split`` is one of ``train`` / ``test`` / ``test_real``. ``max_per_cell``
    caps the patch count (InSAR-DLPU is a single regime, so it caps the split).
    """
    if split not in _DLPU_SPLIT_DIRS:
        raise ValueError(f"DLPU split must be one of {list(_DLPU_SPLIT_DIRS)}, got {split!r}.")
    root = resolve_dlpu_root(dlpu_root)
    wdir, adir = _DLPU_SPLIT_DIRS[split]
    wfiles = sorted(glob.glob(os.path.join(root, wdir, "*.mat")))
    if not wfiles:
        raise FileNotFoundError(f"No .mat files in {os.path.join(root, wdir)!r}.")

    n = 0
    for wpath in wfiles:
        if max_per_cell is not None and n >= max_per_cell:
            break
        n += 1
        base = os.path.basename(wpath)
        psi = np.asarray(_load_mat_array(wpath, "input"), dtype=np.float64)
        apath = os.path.join(root, adir, base)
        phi = (np.asarray(_load_mat_array(apath, "output"), dtype=np.float64)
               if os.path.exists(apath) else None)

        valid = np.isfinite(psi)
        if phi is not None:
            valid &= np.isfinite(phi)

        kx = ky = None
        if phi is not None:
            kx, ky = _edge_labels(psi, phi, kmax=kmax)

        yield Patch(
            id=os.path.splitext(base)[0], path=wpath,
            regime="dlpu", difficulty="all",
            psi=psi, phi=phi, coherence=None, valid=valid,
            kx_true=kx, ky_true=ky,
            attrs={"sensor": "dlpu", "difficulty": "all", "band": "c"},
        )
