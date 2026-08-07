"""PyTorch datasets and dataloaders for InSAR phase unwrapping (C- and L-band).

Dataset classes provided:

* :class:`InSARDLPUDataset` -- the public InSAR-DLPU C-band benchmark, stored
  as ``.mat`` pairs (REUSED unchanged from the GRSL letter; C-band pretraining
  data lives in ``../InSAR-DLPU/data``).
* :class:`GeoTIFFInSARDataset` -- real Sentinel-1 (Stream C) multi-band
  GeoTIFF / ENVI files (REUSED unchanged).
* L-band sources (NEW, this work; see :mod:`src.lband_dataset`):
  ``lband_sim`` (synthetic, ``data/LB_DLPU/sim``), ``lband_real_nisar`` and
  ``lband_real_uavsar`` (real patches, ``data/LB_DLPU/real``).

Every sample is a ``dict`` of tensors with a shared schema (see
:func:`build_sample`), so the training and adaptation code is agnostic to
which source/band a batch came from. The one addition over the GRSL schema is
``k_max`` (the ambiguity half-range labels are clamped to -- 1 for the
3-class GRSL head, 2 for the TGRS five-arc head) and an optional
band-conditioning channel (proposal §7, Exp. D joint C+L training).

Note on labels: for a labelled source (sim/DLPU) the wrapped field carries
decorrelation noise (``psi = W(phi + n)``) while ``phi``/``output`` is the
clean absolute phase. Per-edge ambiguity labels are derived exactly as in
Eq. (3), ``k_e = round((Delta phi - W(Delta psi)) / 2pi)`` and clamped to
``{-k_max,...,+k_max}``; the noise is what produces the sparse non-zero cuts.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import utils

# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------
# Each sample dict contains (shapes for a single item, batched by DataLoader):
#   "input"      : (2, H, W) or (4, H, W) [+1 band channel] float
#   "wrapped"    : (1, H, W) float  -- raw wrapped phase psi (losses / eval)
#   "coherence"  : (1, H, W) float  -- pixel coherence in [0, 1]
#   "weights_x"  : (H, W)    float  -- horizontal edge weights w_e
#   "weights_y"  : (H, W)    float  -- vertical   edge weights w_e
#   "valid_x"    : (H, W)    bool   -- horizontal edge validity (last col False)
#   "valid_y"    : (H, W)    bool   -- vertical   edge validity (last row False)
#   "name"       : str              -- source filename (no extension)
#   "band"       : str              -- "c" or "l" (informational; band token is
#                                       baked into "input" when enabled)
#   "absolute"   : (1, H, W) float  -- clean absolute phase (if available)
#   "k_x"        : (H, W)    long   -- horizontal ambiguity class (if available)
#   "k_y"        : (H, W)    long   -- vertical   ambiguity class (if available)
# ---------------------------------------------------------------------------

_DLPU_SPLIT_DIRS = {
    "train": ("train_wrapped", "train_absolute"),
    "test": ("test_wrapped", "test_absolute"),
    "test_real": ("test_wrapped_real", "test_absolute_real"),
}

_BAND_ID = {"c": 0.0, "l": 1.0}


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------
def _load_mat_array(path: str, key: str) -> np.ndarray:
    """Load a single 2-D array from a MATLAB ``.mat`` file.

    Handles both classic (v5, ``scipy.io.loadmat``) and v7.3/HDF5 (``h5py``)
    files transparently. ``key`` is the variable name (``input`` or ``output``
    in InSAR-DLPU).
    """
    try:
        import scipy.io as sio

        mat = sio.loadmat(path)
        return np.asarray(mat[key], dtype=np.float32)
    except (NotImplementedError, ValueError):
        # v7.3 files are HDF5 and raise here; fall back to h5py.
        import h5py

        with h5py.File(path, "r") as f:
            arr = np.asarray(f[key], dtype=np.float32)
        # h5py returns Fortran-order (transposed) relative to MATLAB.
        return arr.T if arr.ndim == 2 else arr


# ---------------------------------------------------------------------------
# Shared sample builder
# ---------------------------------------------------------------------------
def build_sample(
    wrapped: np.ndarray,
    absolute: Optional[np.ndarray] = None,
    coherence: Optional[np.ndarray] = None,
    *,
    name: str = "",
    weight_reduce: str = "mean",
    clamp_k: bool = True,
    input_grad: bool = True,
    k_max: int = 1,
    band: Optional[str] = None,
    band_token: bool = False,
    valid_mask: Optional[np.ndarray] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Assemble the canonical sample dict from raw numpy arrays.

    Parameters
    ----------
    wrapped:
        Wrapped phase ``(H, W)`` in ``(-pi, pi]``.
    absolute:
        Optional clean/reference absolute phase ``(H, W)``. When provided,
        per-edge ambiguity labels are computed.
    coherence:
        Optional pixel coherence ``(H, W)`` in ``[0, 1]``. When provided, edge
        weights are derived from it; otherwise uniform weights are used.
    name:
        Identifier carried through for logging / visualisation.
    weight_reduce:
        Endpoint combiner for coherence -> edge weights (``mean|min|geo``).
    clamp_k:
        Clamp ambiguity to ``{-k_max, ..., +k_max}``.
    input_grad:
        Append wrapped-gradient channels to the network input, giving the
        classifier the local cut cue directly (see :func:`utils.encode_input`).
    k_max:
        Ambiguity half-range for label computation: 1 (GRSL 3-class, C-band
        default) or 2 (TGRS five-arc, L-band default).
    band:
        ``"c"`` or ``"l"`` (informational, carried through as ``sample["band"]``).
    band_token:
        If ``True`` (and ``band`` given), append a constant band-id channel
        (0.0=C, 1.0=L) to ``sample["input"]`` -- proposal §7 Exp. D joint
        C+L training with a 1-bit band-conditioning token.
    valid_mask:
        Optional boolean ``(H, W)`` array marking pixels with real data
        (``False`` = no-data/NaN, e.g. NISAR swath edges or water). When
        given, invalid pixels are zeroed in every field and also returned as
        ``sample["valid_pixels"]`` so losses/metrics can exclude them.
    """
    wrapped = np.nan_to_num(np.asarray(wrapped, dtype=np.float32), nan=0.0)
    psi = torch.from_numpy(np.ascontiguousarray(wrapped)).to(dtype)
    h, w = psi.shape[-2], psi.shape[-1]

    inp = utils.encode_input(psi, include_grad=input_grad)  # (2 or 4, H, W)
    if band_token:
        if band not in _BAND_ID:
            raise ValueError(f"band_token=True needs band in {list(_BAND_ID)}, got {band!r}.")
        tok = torch.full((1, h, w), _BAND_ID[band], dtype=dtype)
        inp = torch.cat((inp, tok), dim=0)

    sample: Dict[str, Any] = {
        "input": inp,
        "wrapped": psi.unsqueeze(0),                  # (1, H, W)
        "name": name,
        "band": band or "c",
    }

    # --- coherence + edge weights ---
    if coherence is not None:
        coh = np.nan_to_num(np.asarray(coherence, dtype=np.float32), nan=0.0)
        coh = torch.from_numpy(np.ascontiguousarray(coh)).to(dtype).clamp(0.0, 1.0)
        wx, wy = utils.coherence_to_edge_weights(coh, reduce=weight_reduce, pad=True)
    else:
        coh = torch.ones((h, w), dtype=dtype)
        wx, wy = utils.default_edge_weights((h, w), dtype=dtype)
    sample["coherence"] = coh.unsqueeze(0)            # (1, H, W)
    sample["weights_x"] = wx
    sample["weights_y"] = wy

    # --- validity masks (edge padding + optional no-data pixel mask) ---
    # "valid_pixels" is always present (all-True when no valid_mask is given)
    # so batches mixing sources with/without a no-data mask (e.g. C-band +
    # L-band in Exp. D's joint training, src.dataset._JointDataset) still
    # collate: every sample dict must carry the same key set.
    mask_x, mask_y = utils.edge_valid_masks(h, w)
    if valid_mask is not None:
        vp = torch.from_numpy(np.ascontiguousarray(valid_mask)).bool()
        # An edge is only valid if both endpoints have real data.
        mask_x = mask_x.bool() & vp & torch.roll(vp, shifts=-1, dims=-1)
        mask_x[:, -1] = False
        mask_y = mask_y.bool() & vp & torch.roll(vp, shifts=-1, dims=-2)
        mask_y[-1, :] = False
        sample["weights_x"] = sample["weights_x"] * mask_x.to(dtype)
        sample["weights_y"] = sample["weights_y"] * mask_y.to(dtype)
    else:
        vp = torch.ones((h, w), dtype=torch.bool)
    sample["valid_pixels"] = vp
    sample["valid_x"] = mask_x
    sample["valid_y"] = mask_y

    # --- labels (only when a ground-truth/reference absolute phase is available) ---
    if absolute is not None:
        absolute = np.nan_to_num(np.asarray(absolute, dtype=np.float32), nan=0.0)
        phi = torch.from_numpy(np.ascontiguousarray(absolute)).to(dtype)
        sample["absolute"] = phi.unsqueeze(0)         # (1, H, W)
        cls_x, cls_y = utils.compute_ambiguity_labels(psi, phi, pad=True, clamp=clamp_k, k_max=k_max)
        sample["k_x"] = cls_x                          # (H, W) long
        sample["k_y"] = cls_y                          # (H, W) long
        sample["k_pixel"] = torch.round((phi - psi) / utils.TWO_PI).unsqueeze(0)  # (1,H,W) float

    return sample


# ---------------------------------------------------------------------------
# InSAR-DLPU dataset (C-band, REUSED unchanged from the GRSL letter)
# ---------------------------------------------------------------------------
def resolve_dlpu_root(root: str) -> str:
    """Resolve the directory that actually holds the ``*_wrapped`` split folders.

    The released archive nests the data several levels deep
    (``.../InSAR-DLPU/InSAR-DLPU/InSAR-DLPU/``). This walks down through any
    chain of single ``InSAR-DLPU`` subfolders until it finds one containing a
    recognised split directory, so configs can simply point at ``data/``.
    """
    candidates = [root]
    for _ in range(6):
        new_candidates = []
        for c in candidates:
            if not os.path.isdir(c):
                continue
            if any(os.path.isdir(os.path.join(c, d[0])) for d in _DLPU_SPLIT_DIRS.values()):
                return c
            for entry in os.listdir(c):
                p = os.path.join(c, entry)
                if os.path.isdir(p) and "insar" in entry.lower():
                    new_candidates.append(p)
        candidates = new_candidates
        if not candidates:
            break
    raise FileNotFoundError(
        f"Could not locate InSAR-DLPU split folders under {root!r}. "
        f"Expected a directory containing e.g. 'train_wrapped'."
    )


class InSARDLPUDataset(Dataset):
    """InSAR-DLPU C-band benchmark of wrapped/absolute ``.mat`` pairs."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        return_labels: bool = True,
        clamp_k: bool = True,
        limit: Optional[int] = None,
        input_grad: bool = True,
        k_max: int = 1,
        band_token: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        augment: Optional[Callable[..., Any]] = None,
    ) -> None:
        if split not in _DLPU_SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_DLPU_SPLIT_DIRS)}, got {split!r}.")
        self.root = resolve_dlpu_root(root)
        self.split = split
        self.return_labels = return_labels
        self.clamp_k = clamp_k
        self.input_grad = input_grad
        self.k_max = k_max
        self.band_token = band_token
        self.transform = transform
        self.augment = augment

        wrapped_dir, absolute_dir = _DLPU_SPLIT_DIRS[split]
        self.wrapped_dir = os.path.join(self.root, wrapped_dir)
        self.absolute_dir = os.path.join(self.root, absolute_dir)

        files = sorted(glob.glob(os.path.join(self.wrapped_dir, "*.mat")))
        if not files:
            raise FileNotFoundError(f"No .mat files in {self.wrapped_dir!r}.")
        if limit is not None:
            files = files[:limit]
        self.wrapped_files = files

    def __len__(self) -> int:
        return len(self.wrapped_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wpath = self.wrapped_files[idx]
        base = os.path.basename(wpath)
        name = os.path.splitext(base)[0]

        wrapped = _load_mat_array(wpath, "input")

        absolute = None
        if self.return_labels:
            apath = os.path.join(self.absolute_dir, base)
            if os.path.exists(apath):
                absolute = _load_mat_array(apath, "output")

        if self.augment is not None:
            wrapped, absolute, _ = self.augment(wrapped, absolute, None)

        sample = build_sample(
            wrapped=wrapped,
            absolute=absolute,
            coherence=None,           # InSAR-DLPU has no coherence band
            name=name,
            clamp_k=self.clamp_k,
            input_grad=self.input_grad,
            k_max=self.k_max,
            band="c",
            band_token=self.band_token,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Real Sentinel-1 / GeoTIFF dataset (Stream C, REUSED unchanged)
# ---------------------------------------------------------------------------
class GeoTIFFInSARDataset(Dataset):
    """Multi-band GeoTIFF / ENVI interferograms (real Sentinel-1, Stream C).

    Each file must contain at least two bands: wrapped phase and coherence.
    There is no reliable unwrapped ground truth, so no ambiguity labels are
    produced; the coherence band becomes the edge weights ``w_e`` used by the
    integrator, mirroring SNAPHU's physical prior.
    """

    def __init__(
        self,
        files: Sequence[str] | str,
        wrapped_band: int = 1,
        coherence_band: int = 2,
        absolute_band: Optional[int] = None,
        weight_reduce: str = "mean",
        input_grad: bool = True,
        band_token: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.files = self._resolve_files(files)
        if not self.files:
            raise FileNotFoundError(f"No GeoTIFF/ENVI files found for {files!r}.")
        self.wrapped_band = wrapped_band
        self.coherence_band = coherence_band
        self.absolute_band = absolute_band
        self.weight_reduce = weight_reduce
        self.input_grad = input_grad
        self.band_token = band_token
        self.transform = transform

    @staticmethod
    def _resolve_files(files: Sequence[str] | str) -> List[str]:
        if isinstance(files, str):
            if os.path.isdir(files):
                patterns = ("*.tif", "*.tiff", "*.img", "*.dat")
                out: List[str] = []
                for pat in patterns:
                    out.extend(glob.glob(os.path.join(files, pat)))
                return sorted(out)
            return sorted(glob.glob(files))
        return list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import rasterio  # lazy import: only needed for Stream C

        path = self.files[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        with rasterio.open(path) as src:
            wrapped = src.read(self.wrapped_band).astype(np.float32)
            coherence = src.read(self.coherence_band).astype(np.float32)
            absolute = (
                src.read(self.absolute_band).astype(np.float32)
                if self.absolute_band is not None
                else None
            )

        sample = build_sample(
            wrapped=wrapped,
            absolute=absolute,           # usually None for real data
            coherence=coherence,
            name=name,
            weight_reduce=self.weight_reduce,
            input_grad=self.input_grad,
            band="c",
            band_token=self.band_token,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Config-driven factory helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from either an attribute-style (OmegaConf) or dict config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataset(cfg: Any, split: str, difficulty: Optional[float] = None) -> Dataset:
    """Construct a dataset for ``split`` from a (Hydra/OmegaConf or dict) config.

    Recognised ``cfg`` (the ``data`` group) keys::

        source        : "dlpu" (C-band) | "geotiff" (C-band) | "synthetic" (C-band)
                      | "lband_sim" | "lband_real_nisar" | "lband_real_uavsar"
        root          : dataset root path
        return_labels : bool
        clamp_k       : bool
        k_max         : ambiguity half-range for labels (1 = 3-class, 2 = five-arc)
        band_token    : append a constant band-id channel to the network input
        limit         : Optional[int]            # per-split sample cap
        # geotiff-only:
        files         : list/dir/glob
        wrapped_band, coherence_band, absolute_band, weight_reduce
        # synthetic-only (see SyntheticConfig):
        synthetic     : dict of generator params
        epoch_size, val_size : per-split sample counts
        # lband_* only: see src.lband_dataset
    """
    source = _cfg_get(cfg, "source", "dlpu")
    input_grad = bool(_cfg_get(cfg, "input_grad", True))
    k_max = int(_cfg_get(cfg, "k_max", 1))
    band_token = bool(_cfg_get(cfg, "band_token", False))
    # Train-time augmentation (sim->real robustness) is applied to the train
    # split only; eval/val splits are always seen clean.
    augment = None
    if split == "train":
        from .augment import build_augmentor

        augment = build_augmentor(_cfg_get(cfg, "augment", None), seed=_cfg_get(cfg, "seed", None))
    if source == "dlpu":
        return InSARDLPUDataset(
            root=_cfg_get(cfg, "root", "./data"),
            split=split,
            return_labels=bool(_cfg_get(cfg, "return_labels", True)),
            clamp_k=bool(_cfg_get(cfg, "clamp_k", True)),
            limit=_cfg_get(cfg, "limit", None),
            input_grad=input_grad,
            k_max=k_max,
            band_token=band_token,
            augment=augment,
        )
    if source == "geotiff":
        return GeoTIFFInSARDataset(
            files=_cfg_get(cfg, "files", _cfg_get(cfg, "root", "./data")),
            wrapped_band=int(_cfg_get(cfg, "wrapped_band", 1)),
            coherence_band=int(_cfg_get(cfg, "coherence_band", 2)),
            absolute_band=_cfg_get(cfg, "absolute_band", None),
            weight_reduce=_cfg_get(cfg, "weight_reduce", "mean"),
            input_grad=input_grad,
            band_token=band_token,
        )
    if source == "synthetic":
        from .synthetic_engine import build_synthetic_dataset

        syn_cfg = _cfg_get(cfg, "synthetic", {})
        is_train = split == "train"
        length = int(_cfg_get(cfg, "epoch_size", 30000)) if is_train else int(_cfg_get(cfg, "val_size", 1000))
        base_seed = 0 if is_train else 10_000_000
        return build_synthetic_dataset(
            syn_cfg, length=length, base_seed=base_seed,
            deterministic=(None if is_train else True), input_grad=input_grad,
            difficulty=difficulty,
        )
    if source in ("lband_sim", "lband_real_nisar", "lband_real_uavsar"):
        from . import lband_dataset as lb

        root = _cfg_get(cfg, "root", "./data/LB_DLPU")
        limit = _cfg_get(cfg, "limit", None)
        common = dict(
            input_grad=input_grad, k_max=k_max, band_token=band_token,
            clamp_k=bool(_cfg_get(cfg, "clamp_k", True)), limit=limit,
        )
        if source == "lband_sim":
            sim_split = {"train": "train", "test": "test", "test_real": "val", "val": "val"}.get(split, split)
            sim_dir = _cfg_get(cfg, "sim_dir", "sim")
            return lb.LBandSimDataset(root=root, split=sim_split, sim_dir=sim_dir,
                                      augment=augment, **common)
        if source == "lband_real_nisar":
            return lb.LBandRealDataset(root=root, sensor="nisar", **common)
        return lb.LBandRealDataset(root=root, sensor="uavsar", **common)
    raise ValueError(
        f"Unknown data.source={source!r}; use 'dlpu', 'geotiff', 'synthetic', "
        f"'lband_sim', 'lband_real_nisar' or 'lband_real_uavsar'."
    )


def input_channels(cfg: Any) -> int:
    """Number of network input channels implied by the data config.

    2 for ``(cos, sin)``; +2 when wrapped-gradient channels are appended
    (``data.input_grad``, default true); +1 more when ``data.band_token`` is
    set (proposal §7 Exp. D joint C+L training). Used to keep the model's
    ``in_chans`` in sync with the data without hardcoding.
    """
    n = 4 if bool(_cfg_get(cfg, "input_grad", True)) else 2
    if bool(_cfg_get(cfg, "band_token", False)):
        n += 1
    return n


def sync_model_data_cfg(model_cfg: Any, data_cfg: Any) -> None:
    """Sync the two config groups' shared fields in place before building the model.

    Mirrors of ``in_chans = input_channels(data)`` for the ambiguity range:
    the edge head's output shape (``2*(2*edge_k_max+1)`` channels) and the
    dataset's ambiguity-label clamp range (``data.k_max``) MUST agree, or the
    loss's per-class weight vector silently breaks on the first batch whose
    labels exceed the model's class count (e.g. data.k_max=2 five-arc labels
    fed to an edge_k_max=1 model). Call this right before constructing
    the network model -- exactly where ``in_chans`` is already synced.
    """
    model_cfg.backbone.in_chans = input_channels(data_cfg)
    head_type = str(_cfg_get(model_cfg.decoder, "head_type", "pixel"))
    if head_type == "edge":
        edge_k = int(_cfg_get(model_cfg.decoder, "edge_k_max", 1))
        data_k = _cfg_get(data_cfg, "k_max", None)
        # The sync used to overwrite data.k_max silently, so `data.k_max=1` on the
        # command line was discarded whenever the model config still carried
        # edge_k_max=2 -- the run trained five-arc while the caller believed it was
        # three-class. Because the shipped configs keep the two in agreement, any
        # disagreement here means exactly one of them was overridden, which is
        # never intentional. Fail loudly instead.
        if data_k is not None and int(data_k) != edge_k:
            raise ValueError(
                f"Ambiguity range mismatch: data.k_max={int(data_k)} but "
                f"model.decoder.edge_k_max={edge_k}. These must agree -- pass both, "
                f"e.g. `data.k_max={int(data_k)} model.decoder.edge_k_max={int(data_k)}`. "
                f"(Previously data.k_max was silently overwritten by edge_k_max, so a "
                f"run could train a different ambiguity range than the one requested.)"
            )
        data_cfg.k_max = edge_k


def build_dataloader(
    cfg: Any,
    split: str,
    shuffle: Optional[bool] = None,
    device: Optional[torch.device] = None,
    difficulty: Optional[float] = None,
) -> DataLoader:
    """Build a ``DataLoader`` for ``split`` from a config.

    ``cfg`` is the ``data`` config group. Relevant keys: ``batch_size``,
    ``num_workers``, ``pin_memory``, ``drop_last``. ``pin_memory`` defaults to
    ``True`` only when training on CUDA. ``shuffle`` defaults to ``True`` for
    the ``train`` split and ``False`` otherwise.
    """
    dataset = build_dataset(cfg, split, difficulty=difficulty)
    if shuffle is None:
        shuffle = split == "train"

    use_cuda = device is not None and device.type == "cuda"
    pin_memory = bool(_cfg_get(cfg, "pin_memory", use_cuda))
    num_workers = int(_cfg_get(cfg, "num_workers", 4))

    return DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "batch_size", 8)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(_cfg_get(cfg, "drop_last", split == "train")),
        persistent_workers=num_workers > 0,
    )
