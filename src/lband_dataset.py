"""L-band datasets (NEW, this work) -- proposal §4, Table 2.

Loaders for ``data/LB_DLPU`` (the pre-generated patch release used by this
extension), matching the C-band :class:`src.dataset.InSARDLPUDataset` sample
schema so training/eval code is band-agnostic. Two families:

* :class:`LBandSimDataset` -- ``sim/{train,val,test}/*.h5``, synthetic
  NISAR-calibrated pairs with ground-truth absolute phase (Cramer-Rao noise +
  ionospheric screen; see :mod:`src.synth_lband`). Each ``.h5`` already
  stores ``psi, phi, coherence, kx, ky, water_mask``; ``kx``/``ky`` span
  ``{-2,...,+2}`` (this release was generated for the five-arc extension),
  but we **recompute** the class labels from ``(psi, phi)`` via
  :func:`src.dataset.build_sample` rather than trust the stored ``kx``/``ky``
  directly, so label conventions (padding, clamping, ``k_max``) stay in one
  place. ``water_mask`` marks pixels with no coherent signal (excluded via
  ``valid_mask``).

* :class:`LBandRealDataset` -- ``real/real_nisar/*.h5`` (NISAR L2 GUNW
  patches) and ``real/real_uavsar/*.h5`` (UAVSAR ground-range patches). These
  carry a ``ref`` field -- the *provider's own* unwrapped-phase product
  (ISCE3/SNAPHU-class for NISAR, the JPL UAVSAR processor for UAVSAR), **not**
  independently validated ground truth (the same epistemic status as the
  GRSL letter's TanDEM-X ``test_real`` split: useful for RMSE-vs-reference
  and oracle-floor diagnostics, not a proof of absolute correctness). NISAR
  patches routinely contain NaN pixels (nodata outside the valid swath /
  over water); both loaders derive a per-pixel ``valid_mask`` from
  finiteness (+ ``conncomp`` for NISAR) so invalid pixels are excluded from
  edge weights, losses and metrics rather than silently zeroed.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .dataset import build_sample

_SIM_SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}
_REAL_SENSOR_DIRS = {"nisar": "real_nisar", "uavsar": "real_uavsar"}


def _read_h5(path: str, keys: List[str]) -> Dict[str, np.ndarray]:
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                out[k] = np.asarray(f[k][:])
    return out


class LBandSimDataset(Dataset):
    """L-band synthetic benchmark, ``data/LB_DLPU/sim/{train,val,test}``.

    NISAR-calibrated (Beta(3.58,2.37) coherence, Cramer-Rao phase noise at
    ``n_looks=20``, ionospheric screen with PSD ``f^-2.39``); see
    :mod:`src.synth_lband` for the generator that produced this data.
    """

    def __init__(
        self,
        root: str = "./data/LB_DLPU",
        split: str = "train",
        return_labels: bool = True,
        clamp_k: bool = True,
        limit: Optional[int] = None,
        input_grad: bool = True,
        k_max: int = 2,
        band_token: bool = False,
        sim_dir: str = "sim",
        augment: Optional[Callable[..., Any]] = None,
    ) -> None:
        if split not in _SIM_SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_SIM_SPLIT_DIRS)}, got {split!r}.")
        self.dir = os.path.join(root, sim_dir, _SIM_SPLIT_DIRS[split])
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"L-band sim split not found: {self.dir!r}.")
        self.return_labels = return_labels
        self.clamp_k = clamp_k
        self.input_grad = input_grad
        self.k_max = k_max
        self.band_token = band_token
        # Raw-array augmentor (wrapped, absolute, coherence) -> (...), applied
        # BEFORE build_sample so labels are recomputed exactly -- same contract
        # as InSARDLPUDataset.augment (src/dataset.py), NOT a dict transform.
        self.augment = augment

        files = sorted(glob.glob(os.path.join(self.dir, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files in {self.dir!r}.")
        if limit is not None:
            files = files[:limit]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        keys = ["psi", "phi", "coherence", "water_mask"] if self.return_labels else ["psi", "coherence", "water_mask"]
        d = _read_h5(path, keys)
        wrapped, absolute, coherence = d["psi"], d.get("phi"), d.get("coherence")

        # water_mask has no geometry-aware counterpart in PhaseAugmentor's API
        # (wrapped/absolute/coherence only), so skip mask-based validity when
        # augmenting to avoid a silent post-D4/scale misalignment; the un-
        # augmented (eval/val/test) path always uses the exact water mask.
        valid_mask = None
        if self.augment is not None:
            wrapped, absolute, coherence = self.augment(wrapped, absolute, coherence)
        elif "water_mask" in d:
            valid_mask = ~d["water_mask"].astype(bool)   # water = no coherent signal

        sample = build_sample(
            wrapped=wrapped,
            absolute=absolute if self.return_labels else None,
            coherence=coherence,
            name=name,
            clamp_k=self.clamp_k,
            input_grad=self.input_grad,
            k_max=self.k_max,
            band="l",
            band_token=self.band_token,
            valid_mask=valid_mask,
        )
        return sample


class LBandRealDataset(Dataset):
    """Real L-band patches: ``real/real_nisar`` (NISAR L2 GUNW) or
    ``real/real_uavsar`` (UAVSAR ground-range), no trustworthy ground truth.

    ``sample["absolute"]`` is populated from the ``ref`` field for RMSE-vs-
    reference / oracle-floor diagnostics -- see the module docstring for its
    epistemic status. Invalid (NaN) pixels are masked via ``valid_mask``, not
    silently treated as k=0.
    """

    def __init__(
        self,
        root: str = "./data/LB_DLPU",
        sensor: str = "nisar",
        return_labels: bool = True,
        clamp_k: bool = True,
        limit: Optional[int] = None,
        input_grad: bool = True,
        k_max: int = 2,
        band_token: bool = False,
    ) -> None:
        if sensor not in _REAL_SENSOR_DIRS:
            raise ValueError(f"sensor must be one of {list(_REAL_SENSOR_DIRS)}, got {sensor!r}.")
        self.sensor = sensor
        self.dir = os.path.join(root, "real", _REAL_SENSOR_DIRS[sensor])
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"L-band real split not found: {self.dir!r}.")
        self.return_labels = return_labels
        self.clamp_k = clamp_k
        self.input_grad = input_grad
        self.k_max = k_max
        self.band_token = band_token

        files = sorted(glob.glob(os.path.join(self.dir, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files in {self.dir!r}.")
        if limit is not None:
            files = files[:limit]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        keys = ["psi", "coherence", "ref", "iono", "conncomp"]
        d = _read_h5(path, keys)

        valid = np.isfinite(d["psi"]) & np.isfinite(d.get("coherence", d["psi"]))
        if "conncomp" in d:
            valid &= d["conncomp"] != 0
        if self.return_labels and "ref" in d:
            valid &= np.isfinite(d["ref"])

        sample = build_sample(
            wrapped=d["psi"],
            absolute=d.get("ref") if self.return_labels else None,
            coherence=d.get("coherence"),
            name=name,
            clamp_k=self.clamp_k,
            input_grad=self.input_grad,
            k_max=self.k_max,
            band="l",
            band_token=self.band_token,
            valid_mask=valid,
        )
        if "iono" in d:
            import torch
            sample["iono"] = torch.from_numpy(np.nan_to_num(d["iono"], nan=0.0).astype(np.float32)).unsqueeze(0)
        sample["sensor"] = self.sensor
        return sample
