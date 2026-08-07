"""Sentinel-1 HyP3 geotiff adapter for the evaluation harness.

Turns each real coseismic interferogram into 256x256 ``Patch`` tiles so the whole
method registry (ours + DL baselines + classical) is scored on real seismic phase
in one run. There is no ground truth for real data, so the bundled SNAPHU
``_unw_phase`` product is used as the reference ``phi``: every ``rmse`` here is
therefore *agreement with SNAPHU*, not accuracy, and the coherence-stratified
metric is the high-coherence SNAPHU agreement. Coherence comes from ``_corr``.

DL baselines are 256x256-native and do not tile to full scenes (per-tile 2pi
offsets), so the comparison is on 256x256 tiles at their native resolution.
"""

from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional

import numpy as np

from src import mcf

from .data import Patch


def _difficulty(psi_tile: np.ndarray) -> str:
    """Fringe difficulty of a tile from its residue density (deformation + decorrelation
    both raise it); lets the harness report a `dense` column where DL cycle-slips."""
    r = int(np.abs(mcf.residues(np.nan_to_num(psi_tile))).sum())
    return "smooth" if r < 100 else ("mixed" if r < 1000 else "dense")


def _find(product_dir: str, tag: str) -> str:
    hits = glob.glob(os.path.join(product_dir, f"*{tag}.tif"))
    if not hits:
        raise FileNotFoundError(f"no *{tag}.tif in {product_dir!r}")
    return hits[0]


def _product_dirs(root: str) -> List[str]:
    """`root` is either a single HyP3 product dir or a parent of several."""
    if glob.glob(os.path.join(root, "*_wrapped_phase.tif")):
        return [root]
    subs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "*_wrapped_phase.tif")))
    if not subs:
        raise FileNotFoundError(f"no HyP3 products (with *_wrapped_phase.tif) under {root!r}")
    return subs


def iter_patches_s1(
    root: str,
    split: str = "test",
    max_per_cell: Optional[int] = None,
    tile: int = 256,
    stride: int = 256,
    min_valid: float = 0.98,
) -> Iterator[Patch]:
    """Yield 256x256 valid tiles from each S1 product; ``phi`` = SNAPHU reference."""
    import rasterio

    for pdir in _product_dirs(root):
        with rasterio.open(_find(pdir, "_wrapped_phase")) as s:
            psi_full = s.read(1).astype(np.float64)
        with rasterio.open(_find(pdir, "_corr")) as s:
            coh_full = s.read(1).astype(np.float64)
        with rasterio.open(_find(pdir, "_unw_phase")) as s:
            unw_full = s.read(1).astype(np.float64)

        event = os.path.basename(pdir.rstrip("/\\"))[:24]
        H, W = psi_full.shape
        n = 0
        for r in range(0, H - tile + 1, stride):
            for c in range(0, W - tile + 1, stride):
                if max_per_cell is not None and n >= max_per_cell:
                    break
                p = psi_full[r:r + tile, c:c + tile]
                k = coh_full[r:r + tile, c:c + tile]
                u = unw_full[r:r + tile, c:c + tile]
                valid = np.isfinite(p) & np.isfinite(u) & (k > 0) & (u != 0)
                if valid.mean() < min_valid:
                    continue
                n += 1
                diff = _difficulty(p)
                yield Patch(
                    id=f"{event}_{r}_{c}", path=pdir, regime=event, difficulty=diff,
                    psi=np.nan_to_num(p), phi=np.nan_to_num(u),
                    coherence=np.clip(np.nan_to_num(k), 0.0, 1.0), valid=valid,
                    kx_true=None, ky_true=None,
                    attrs={"sensor": event, "difficulty": diff, "band": "c",
                           "reference": "snaphu_unw_phase"},
                )
            if max_per_cell is not None and n >= max_per_cell:
                break
