"""NISAR L2 GUNW product I/O (NEW, this work).

Reads the NASA/JPL NISAR L2 GUNW HDF5 granules in ``data/full_scenes/gunw_all``
(delivered as interferometric products via ASF DAAC; no SLC processing). Each
granule stores the unwrapped-interferogram group at the standard 80 m posting
(``science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/...``) --
``unwrappedPhase``, ``coherenceMagnitude``, ``connectedComponents`` and the
``ionospherePhaseScreen`` the L-band cost derivation (proposal §2.3.3) is
built on.

Two grids, two very different tests:

* :func:`load_scene` reads the **unwrapped-interferogram grid** (4194x4284)
  and sets ``psi = wrap(ref)``. This lines every field up pixel-for-pixel and
  matches how the pre-generated ``data/LB_DLPU/real/real_nisar`` patches were
  built -- but ``psi`` is then a *noise-free re-wrap* of the operational
  product (per-pixel floor 0, a degenerate unwrapping test; the operational
  reference is trivially recoverable and SNAPHU-class methods reproduce their
  own kind of output). Use it for schema/consistency, not for a real test.

* :func:`load_scene_wrapped` reads the **fine complex ``wrappedInterferogram``
  grid** (16776x17136, 4x finer) and takes its *phase* -- the ACTUAL noisy
  observed interferogram (~10^4 residues/megapixel from genuine decorrelation,
  vs ~10^2 for the re-wrap). This is the real unwrapping test. Its only
  reference is the 4x-coarser ``unwrappedPhase``, so evaluation is
  residue-freeness + *approximate* agreement with the operational product
  after 4x downsampling (a qualitative/agreement claim, not RMSE-vs-truth).

Real NISAR grids routinely contain NaN outside the valid swath (nodata) or
over water/layover; callers should always check ``valid``.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

_GROUP = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
_WGROUP = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"


def list_scenes(root: str) -> list:
    """List NISAR L2 GUNW ``.h5`` granules under ``root``."""
    return sorted(glob.glob(os.path.join(root, "*.h5")))


def _read_window(dset, row0: int, col0: int, rows: Optional[int], cols: Optional[int]) -> np.ndarray:
    H, W = dset.shape
    r1 = H if rows is None else min(H, row0 + rows)
    c1 = W if cols is None else min(W, col0 + cols)
    return np.asarray(dset[row0:r1, col0:c1])


def scene_shape(path: str) -> Tuple[int, int]:
    """``(rows, cols)`` of a granule's unwrapped-interferogram grid."""
    import h5py

    with h5py.File(path, "r") as f:
        return tuple(f[f"{_GROUP}/HH/unwrappedPhase"].shape)


def load_scene(
    path: str,
    row0: int = 0, col0: int = 0,
    rows: Optional[int] = None, cols: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load a (windowed) NISAR scene: wrapped phase, coherence, reference unwrap.

    Returns ``{psi, coherence, ref, iono, conncomp, valid}`` matching the
    ``LBandRealDataset`` sample schema (:mod:`src.lband_dataset`). ``ref`` is
    the product's own ``unwrappedPhase`` (an ISCE3/SNAPHU-class reference, not
    independently validated ground truth); ``psi = wrap(ref)`` since no
    separate wrapped-phase grid exists at matching resolution/posting.
    """
    import h5py

    with h5py.File(path, "r") as f:
        g = f[f"{_GROUP}/HH"]
        ref = _read_window(g["unwrappedPhase"], row0, col0, rows, cols).astype(np.float32)
        coh = _read_window(g["coherenceMagnitude"], row0, col0, rows, cols).astype(np.float32)
        conncomp = _read_window(g["connectedComponents"], row0, col0, rows, cols).astype(np.int32)
        iono = _read_window(g["ionospherePhaseScreen"], row0, col0, rows, cols).astype(np.float32)

    psi = np.angle(np.exp(1j * ref)).astype(np.float32)
    valid = np.isfinite(ref) & np.isfinite(coh) & (conncomp != 0)
    return {
        "psi": psi, "coherence": np.nan_to_num(coh, nan=0.0),
        "ref": ref, "iono": iono, "conncomp": conncomp, "valid": valid,
    }


def wrapped_shape(path: str) -> Tuple[int, int]:
    """``(rows, cols)`` of a granule's FINE wrapped-interferogram grid."""
    import h5py

    with h5py.File(path, "r") as f:
        return tuple(f[f"{_WGROUP}/HH/wrappedInterferogram"].shape)


def load_scene_wrapped(
    path: str, row0: int = 0, col0: int = 0, size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Load a (windowed) REAL noisy wrapped interferogram + coarse reference.

    Reads the fine complex ``wrappedInterferogram`` grid (the genuine noisy
    observation) and returns its phase for a real unwrapping test. The window
    ``[row0:row0+size, col0:col0+size]`` is in **fine-grid coordinates**. The
    only available reference (``unwrappedPhase``) is 4x coarser, so the
    matching coarse region ``[row0//4 : (row0+size)//4, ...]`` is returned as
    ``ref_coarse`` for approximate agreement scoring after the caller
    downsamples its unwrapped result 4x (see ``scripts/unwrap_nisar.py``).

    Returns ``{psi, coherence, valid, ref_coarse, coh_coarse, valid_coarse}``.
    ``psi``/``coherence``/``valid`` are fine-grid ``(size, size)``;
    ``ref_coarse``/``coh_coarse``/``valid_coarse`` are ``(size//4, size//4)``.
    """
    import h5py

    with h5py.File(path, "r") as f:
        wi = f[f"{_WGROUP}/HH/wrappedInterferogram"]
        wc = f[f"{_WGROUP}/HH/coherenceMagnitude"]
        uw = f[f"{_GROUP}/HH/unwrappedPhase"]
        uc = f[f"{_GROUP}/HH/coherenceMagnitude"]
        figram = np.asarray(wi[row0:row0 + size, col0:col0 + size])
        coh = np.asarray(wc[row0:row0 + size, col0:col0 + size], dtype=np.float32)
        cr0, cc0, csz = row0 // 4, col0 // 4, size // 4
        ref = np.asarray(uw[cr0:cr0 + csz, cc0:cc0 + csz], dtype=np.float32)
        coh_c = np.asarray(uc[cr0:cr0 + csz, cc0:cc0 + csz], dtype=np.float32)

    psi = np.angle(figram).astype(np.float32)
    valid = np.isfinite(psi) & (figram != 0) & np.isfinite(coh)
    valid_c = np.isfinite(ref) & np.isfinite(coh_c)
    return {
        "psi": psi, "coherence": np.nan_to_num(coh, nan=0.0), "valid": valid,
        "ref_coarse": ref, "coh_coarse": np.nan_to_num(coh_c, nan=0.0), "valid_coarse": valid_c,
    }


def downsample_phase(phi: np.ndarray, factor: int = 4) -> np.ndarray:
    """Block-average an unwrapped phase field by ``factor`` (fine -> coarse grid).

    Unwrapped phase is smooth, so simple block averaging is an adequate
    resampler to the coarse reference grid for an approximate agreement score.
    Crops to a multiple of ``factor`` first.
    """
    h = (phi.shape[0] // factor) * factor
    w = (phi.shape[1] // factor) * factor
    p = phi[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return p.mean(axis=(1, 3))


def iter_patches(
    path: str, patch: int = 256, stride: int = 256, min_valid: float = 0.9,
) -> Iterator[Tuple[int, int, Dict[str, np.ndarray]]]:
    """Yield ``(row0, col0, scene_dict)`` patches with enough valid coverage.

    Useful for extending the training set with fresh NISAR patches beyond the
    pre-generated release in ``data/LB_DLPU/real/real_nisar``.
    """
    H, W = scene_shape(path)
    for r in range(0, max(H - patch, 0) + 1, stride):
        for c in range(0, max(W - patch, 0) + 1, stride):
            d = load_scene(path, row0=r, col0=c, rows=patch, cols=patch)
            if d["valid"].mean() >= min_valid:
                yield r, c, d
