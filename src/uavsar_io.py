"""UAVSAR ground-range product I/O (NEW, this work).

UAVSAR (JPL) ships raw flat binary ``.grd`` files described by a companion
``.ann`` metadata file (not GeoTIFF/HDF5), so it needs its own reader. This
covers the "fine-posting near-fault |k|>=2 (five-arc) validation" data
(``data/full_scenes/uavsar/{A,B,C}``): San Andreas ground-range interferograms
at ~6 m posting, dense enough that near-fault creep can require the widened
``{-2,...,+2}`` ambiguity range.

Format (confirmed against the ``.ann`` file's own byte counts):

* ``*.int.grd``  -- complex64, interferogram (real, imag interleaved).
* ``*.unw.grd``, ``*.cor.grd``, ``*.amp{1,2}.grd``, ``*.hgt.grd`` -- float32.
* Row-major, ``rows = Ground Range Data Latitude Lines``,
  ``cols = Ground Range Data Longitude Samples`` (both in the ``.ann``).
* Regular lat/lon grid: ``Ground Range Data Starting Latitude/Longitude`` is
  the *center of the upper-left pixel*; ``.../Spacing`` is the per-pixel step
  (latitude spacing is negative -- rows go north to south).

Large files (up to ~2 GB for a single band) are read via ``numpy.memmap`` so a
256x256 patch or a bounded ROI can be pulled out without loading the whole
scene into memory.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

_ANN_LINE = re.compile(
    r"^\s*([^()=]+?)\s*\(([^)]*)\)\s*=\s*([^;]+?)\s*(?:;.*)?$"
)


def parse_ann(path: str) -> Dict[str, str]:
    """Parse a UAVSAR ``.ann`` annotation file into ``{key: value_string}``.

    Keys are the left-hand label with the unit stripped and whitespace
    collapsed, e.g. ``"Ground Range Data Latitude Lines"``. Values are left as
    strings (callers cast to float/int as needed) since the file mixes
    numbers, filenames and free text.
    """
    out: Dict[str, str] = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            m = _ANN_LINE.match(line)
            if not m:
                continue
            key = re.sub(r"\s+", " ", m.group(1)).strip()
            out[key] = m.group(3).strip()
    return out


def find_grd_siblings(ann_path: str) -> Dict[str, str]:
    """Locate the ground-range ``.grd`` files sitting next to an ``.ann`` file."""
    d = os.path.dirname(ann_path)
    out = {}
    for tag, ext in (("int", ".int.grd"), ("unw", ".unw.grd"),
                     ("cor", ".cor.grd"), ("amp1", ".amp1.grd"),
                     ("amp2", ".amp2.grd"), ("hgt", ".hgt.grd")):
        hits = glob.glob(os.path.join(d, f"*{ext}"))
        if hits:
            out[tag] = hits[0]
    return out


def grid_shape(ann: Dict[str, str]) -> Tuple[int, int]:
    """Ground-range ``(rows, cols)`` from a parsed ``.ann`` dict."""
    rows = int(float(ann["Ground Range Data Latitude Lines"]))
    cols = int(float(ann["Ground Range Data Longitude Samples"]))
    return rows, cols


def geo_transform(ann: Dict[str, str]) -> Dict[str, float]:
    """Affine geolocation of the ground-range grid (center of upper-left pixel)."""
    return {
        "lat0": float(ann["Ground Range Data Starting Latitude"]),
        "lon0": float(ann["Ground Range Data Starting Longitude"]),
        "dlat": float(ann["Ground Range Data Latitude Spacing"]),
        "dlon": float(ann["Ground Range Data Longitude Spacing"]),
    }


def _memmap(path: str, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def read_band(
    grd_path: str, ann: Dict[str, str], complex_valued: bool = False,
    row0: int = 0, col0: int = 0, rows: Optional[int] = None, cols: Optional[int] = None,
) -> np.ndarray:
    """Read a (windowed) band from a ``.grd`` file via memmap.

    ``complex_valued=True`` for ``*.int.grd`` (complex64); float32 otherwise.
    Window defaults to the full scene if ``rows``/``cols`` are omitted --
    beware this can be several GB for the full UAVSAR product.
    """
    H, W = grid_shape(ann)
    dtype = np.complex64 if complex_valued else np.float32
    mm = _memmap(grd_path, (H, W), dtype)
    r1 = H if rows is None else min(H, row0 + rows)
    c1 = W if cols is None else min(W, col0 + cols)
    return np.array(mm[row0:r1, col0:c1])   # copy out of the memmap


def load_scene(
    ann_path: str,
    row0: int = 0, col0: int = 0,
    rows: Optional[int] = None, cols: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load a (windowed) UAVSAR scene: wrapped phase, coherence, reference unwrap.

    Returns ``{psi, coherence, ref, valid}`` matching the L-band sample schema
    (``ref`` is the JPL processor's own ``.unw.grd`` product, used as an
    evaluation reference -- see :mod:`src.lband_dataset` for the same
    epistemic caveat as the NISAR ``ref`` field). ``psi`` is derived from the
    complex interferogram's phase (native slant-derived wrap), independent of
    ``ref`` (unlike the NISAR reader, which has no separate wrapped-phase grid
    at matching resolution).
    """
    ann = parse_ann(ann_path)
    siblings = find_grd_siblings(ann_path)
    if "int" not in siblings or "cor" not in siblings:
        raise FileNotFoundError(f"Missing .int.grd/.cor.grd next to {ann_path!r}.")
    igram = read_band(siblings["int"], ann, complex_valued=True,
                      row0=row0, col0=col0, rows=rows, cols=cols)
    coh = read_band(siblings["cor"], ann, row0=row0, col0=col0, rows=rows, cols=cols)
    psi = np.angle(igram).astype(np.float32)
    out = {"psi": psi, "coherence": np.nan_to_num(coh, nan=0.0).astype(np.float32)}
    if "unw" in siblings:
        ref = read_band(siblings["unw"], ann, row0=row0, col0=col0, rows=rows, cols=cols)
        out["ref"] = ref
    valid = np.isfinite(psi) & (igram != 0) & np.isfinite(coh)
    out["valid"] = valid
    out["ann"] = ann
    return out


def iter_patches(
    ann_path: str, patch: int = 256, stride: int = 256, min_valid: float = 0.9,
) -> Iterator[Tuple[int, int, Dict[str, np.ndarray]]]:
    """Yield ``(row0, col0, scene_dict)`` patches with enough valid coverage.

    Useful for extending the training set with fresh UAVSAR patches beyond
    the pre-generated release in ``data/LB_DLPU/real/real_uavsar``.
    """
    ann = parse_ann(ann_path)
    H, W = grid_shape(ann)
    for r in range(0, max(H - patch, 0) + 1, stride):
        for c in range(0, max(W - patch, 0) + 1, stride):
            d = load_scene(ann_path, row0=r, col0=c, rows=patch, cols=patch)
            if d["valid"].mean() >= min_valid:
                yield r, c, d
