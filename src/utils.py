"""Core utilities: reproducibility, phase wrapping, edge gradients, and
per-edge integer-ambiguity labels.

This module is the single source of truth for the *edge conventions* used
throughout the codebase, carried over unchanged from the GRSL letter
(``src/utils.py``). The one generalisation for the TGRS
extension (proposal Pillar 3, five-arc MCF) is that the ambiguity range is no
longer hardcoded to ``{-1, 0, +1}``: every function that touches the
per-edge class encoding now takes a ``k_max`` parameter so the same code
serves the GRSL 3-class case (``k_max=1``, the default -- byte-identical
behaviour to the letter) and the TGRS 5-class case (``k_max=2``, ambiguity
range ``{-2,...,+2}``).

--------------------------------------------------------------------------
Tensor / edge conventions
--------------------------------------------------------------------------
Phase fields are ``torch.Tensor`` of shape ``(..., H, W)`` (any number of
leading batch/channel dims). Two edge directions are defined:

* **Horizontal** edge ``(i, j)`` connects pixel ``(i, j)`` to ``(i, j+1)``.
  A full-resolution horizontal edge map has shape ``(..., H, W)``; the **last
  column** ``j = W-1`` has no right neighbour and is *padding* (invalid).
* **Vertical** edge ``(i, j)`` connects pixel ``(i, j)`` to ``(i+1, j)``.
  A full-resolution vertical edge map has shape ``(..., H, W)``; the **last
  row** ``i = H-1`` is *padding* (invalid).

The decoder emits per-edge logits of shape ``(B, 2*(2*k_max+1), H, W)`` where
the first half of channels are the ``2*k_max+1`` ambiguity classes for
**horizontal** edges and the second half for **vertical**, matching the
ordering returned here. ``k_max=1`` gives the GRSL 6-channel layout.

--------------------------------------------------------------------------
Ambiguity class encoding
--------------------------------------------------------------------------
The integer ambiguity ``k_e`` lives in ``{-k_max, ..., +k_max}``. It is
stored as a class index via ``class = k + k_max``, so ``class = k_max``
always means ``k = 0`` (the "no cut" / Itoh-satisfied edge).
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI: float = float(np.pi)
TWO_PI: float = 2.0 * PI

#: Default ambiguity half-range (GRSL letter: 3-class {-1,0,+1}). The TGRS
#: five-arc extension uses k_max=2 ({-2,...,+2}); pass k_max explicitly
#: wherever the letter's default would be wrong.
DEFAULT_K_MAX: int = 1

#: Real-valued ambiguity per class index for the GRSL default (k_max=1),
#: kept for backward compatibility with code that imports this directly.
AMBIGUITY_VALUES: Tuple[int, int, int] = (-1, 0, 1)

#: Class index assigned to k = 0 for the GRSL default (k_max=1).
ZERO_AMBIGUITY_CLASS: int = 1


def ambiguity_values(k_max: int = DEFAULT_K_MAX) -> Tuple[int, ...]:
    """Integer ambiguity values in class order: ``(-k_max, ..., 0, ..., +k_max)``."""
    return tuple(range(-k_max, k_max + 1))


def zero_ambiguity_class(k_max: int = DEFAULT_K_MAX) -> int:
    """Class index of ``k=0`` for a given ``k_max`` (always ``k_max`` itself)."""
    return int(k_max)


def num_classes(k_max: int = DEFAULT_K_MAX) -> int:
    """Number of ambiguity classes ``2*k_max + 1``."""
    return 2 * int(k_max) + 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Parameters
    ----------
    seed:
        The base seed applied to every RNG.
    deterministic:
        If ``True``, force deterministic cuDNN behaviour
        (``cudnn.deterministic = True``, ``cudnn.benchmark = False``). This
        trades a little speed for run-to-run reproducibility, which is what we
        want when measuring adaptation deltas.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the best available device without hardcoding ``.cuda()``.

    ``prefer`` may be ``"cuda"``, ``"cpu"`` or an explicit device string such as
    ``"cuda:1"``. Falls back to CPU whenever CUDA is unavailable, so the same
    config runs unchanged on a laptop and on a GPU server.
    """
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Phase wrapping
# ---------------------------------------------------------------------------
def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    r"""Wrap phase values into ``(-pi, pi]`` (the operator ``W(.)`` of Eq. 1).

    Implemented as ``((x + pi) mod 2pi) - pi``. This returns values in
    ``[-pi, pi)``; the single boundary point ``x = pi`` maps to ``-pi``, which
    is immaterial for the noisy floating-point phases we handle. The operation
    is elementwise and shape-preserving.
    """
    return torch.remainder(x + PI, TWO_PI) - PI


def encode_wrapped(psi: torch.Tensor) -> torch.Tensor:
    r"""Circular ``(cos psi, sin psi)`` encoding (proposal §3.2).

    Parameters
    ----------
    psi:
        Wrapped phase of shape ``(..., H, W)``.

    Returns
    -------
    torch.Tensor
        Encoded tensor with a new channel axis inserted *before* the spatial
        dims: ``(..., 2, H, W)``. Channel 0 is ``cos psi``, channel 1 is
        ``sin psi``. For a bare ``(H, W)`` input this yields ``(2, H, W)``.
    """
    cos = torch.cos(psi)
    sin = torch.sin(psi)
    return torch.stack((cos, sin), dim=-3)


def encode_input(psi: torch.Tensor, include_grad: bool = True) -> torch.Tensor:
    r"""Network input encoding: ``(cos psi, sin psi)`` plus optional wrapped gradients.

    The circular ``(cos, sin)`` channels (proposal §3.2) are augmented with the
    wrapped phase gradients ``W(Delta psi)`` along x and y (scaled to ``[-1, 1]``
    by ``1/pi``). A per-edge cut sits where the wrapped gradient is large, so
    handing this physics cue to the network directly — rather than making it
    re-derive the local difference from ``(cos, sin)`` — substantially eases cut
    localisation. Returns ``(..., 2, H, W)`` if ``include_grad`` is false, else
    ``(..., 4, H, W)`` with channels ``[cos, sin, W(dpsi)_x/pi, W(dpsi)_y/pi]``.
    """
    cos = torch.cos(psi)
    sin = torch.sin(psi)
    if not include_grad:
        return torch.stack((cos, sin), dim=-3)
    gx, gy = wrapped_gradients(psi, pad=True)            # (..., H, W), invalid edges = 0
    return torch.stack((cos, sin, gx / PI, gy / PI), dim=-3)


# ---------------------------------------------------------------------------
# Edge differences (valid-sized, no padding)
# ---------------------------------------------------------------------------
def edge_diff_x(field: torch.Tensor, wrap: bool = False) -> torch.Tensor:
    """Horizontal forward difference ``field[..., :, 1:] - field[..., :, :-1]``.

    Shape ``(..., H, W)`` -> ``(..., H, W-1)``. With ``wrap=True`` the result is
    wrapped to ``(-pi, pi]`` (i.e. ``W(Delta psi)`` along x).
    """
    d = field[..., :, 1:] - field[..., :, :-1]
    return wrap_to_pi(d) if wrap else d


def edge_diff_y(field: torch.Tensor, wrap: bool = False) -> torch.Tensor:
    """Vertical forward difference ``field[..., 1:, :] - field[..., :-1, :]``.

    Shape ``(..., H, W)`` -> ``(..., H-1, W)``. With ``wrap=True`` the result is
    wrapped to ``(-pi, pi]`` (i.e. ``W(Delta psi)`` along y).
    """
    d = field[..., 1:, :] - field[..., :-1, :]
    return wrap_to_pi(d) if wrap else d


def pad_edge_x(edge_x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Pad a valid-sized horizontal edge map ``(..., H, W-1)`` to ``(..., H, W)``.

    The padding column (last column) represents the non-existent edge past the
    right border and is filled with ``value``.
    """
    pad = edge_x.new_full(edge_x.shape[:-1] + (1,), value)
    return torch.cat((edge_x, pad), dim=-1)


def pad_edge_y(edge_y: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Pad a valid-sized vertical edge map ``(..., H-1, W)`` to ``(..., H, W)``.

    The padding row (last row) represents the non-existent edge past the bottom
    border and is filled with ``value``.
    """
    pad = edge_y.new_full(edge_y.shape[:-2] + (1, edge_y.shape[-1]), value)
    return torch.cat((edge_y, pad), dim=-2)


def edge_valid_masks(
    height: int, width: int, *, device=None, dtype=torch.bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Boolean validity masks for full-resolution edge maps.

    Returns ``(mask_x, mask_y)`` each of shape ``(H, W)``. ``mask_x`` is ``True``
    everywhere except the last column; ``mask_y`` is ``True`` everywhere except
    the last row. Use these to exclude padding edges from losses and metrics.
    """
    mask_x = torch.ones((height, width), device=device, dtype=dtype)
    mask_x[:, -1] = False
    mask_y = torch.ones((height, width), device=device, dtype=dtype)
    mask_y[-1, :] = False
    return mask_x, mask_y


# ---------------------------------------------------------------------------
# Wrapped gradients of the observation (used to form corrected gradients)
# ---------------------------------------------------------------------------
def wrapped_gradients(
    psi: torch.Tensor, pad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Wrapped phase gradients ``W(Delta psi)`` along x and y.

    Parameters
    ----------
    psi:
        Wrapped phase ``(..., H, W)``.
    pad:
        If ``True`` (default) returns full-resolution maps ``(..., H, W)`` with
        the invalid last column / last row set to 0. If ``False`` returns the
        valid-sized maps ``(..., H, W-1)`` and ``(..., H-1, W)``.

    Returns
    -------
    (gx, gy):
        ``gx = W(psi[...,:,1:] - psi[...,:,:-1])`` and
        ``gy = W(psi[...,1:,:] - psi[...,:-1,:])``.
    """
    gx = edge_diff_x(psi, wrap=True)
    gy = edge_diff_y(psi, wrap=True)
    if pad:
        gx = pad_edge_x(gx, 0.0)
        gy = pad_edge_y(gy, 0.0)
    return gx, gy


# ---------------------------------------------------------------------------
# Per-edge ambiguity labels
# ---------------------------------------------------------------------------
def compute_ambiguity_labels(
    psi: torch.Tensor,
    phi: torch.Tensor,
    pad: bool = True,
    clamp: bool = True,
    k_max: int = DEFAULT_K_MAX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Ground-truth per-edge ambiguity classes ``k_e in {-k_max,...,+k_max}`` (Eq. 3).

    For each edge the true phase difference satisfies
    ``Delta phi_e = W(Delta psi)_e + 2 pi k_e``, so

        k_e = round( (Delta phi_e - W(Delta psi)_e) / (2 pi) ).

    Parameters
    ----------
    psi:
        Wrapped phase ``(..., H, W)``.
    phi:
        Ground-truth absolute phase ``(..., H, W)`` (same shape as ``psi``).
    pad:
        If ``True`` returns full-resolution class maps ``(..., H, W)`` with the
        invalid last column / row filled with the zero-ambiguity class.
        If ``False`` returns valid-sized maps ``(..., H, W-1)`` / ``(..., H-1, W)``.
    clamp:
        If ``True`` (default) clamp ``k_e`` to ``{-k_max, ..., +k_max}``. The
        GRSL letter argues corrections at native sampling are almost always in
        ``{-1,0,+1}``; the TGRS five-arc extension widens this to ``k_max=2``
        for near-fault / volcanic scenes where ``|grad phi| > 2*pi``.
    k_max:
        Ambiguity half-range (1 for the GRSL 3-class default, 2 for five-arc).

    Returns
    -------
    (cls_x, cls_y):
        ``torch.long`` class-index maps where ``class = k + k_max``.
    """
    # Wrapped observed gradients.
    wgx = edge_diff_x(psi, wrap=True)          # (..., H, W-1)
    wgy = edge_diff_y(psi, wrap=True)          # (..., H-1, W)
    # True (unwrapped) gradients from the absolute field.
    dphix = edge_diff_x(phi, wrap=False)       # (..., H, W-1)
    dphiy = edge_diff_y(phi, wrap=False)       # (..., H-1, W)

    kx = torch.round((dphix - wgx) / TWO_PI)
    ky = torch.round((dphiy - wgy) / TWO_PI)
    if clamp:
        kx = kx.clamp_(-k_max, k_max)
        ky = ky.clamp_(-k_max, k_max)

    zero_cls = zero_ambiguity_class(k_max)
    cls_x = (kx + k_max).long()
    cls_y = (ky + k_max).long()
    if pad:
        cls_x = pad_edge_x(cls_x, float(zero_cls)).long()
        cls_y = pad_edge_y(cls_y, float(zero_cls)).long()
    return cls_x, cls_y


def expected_ambiguity(probs: torch.Tensor, k_max: int = DEFAULT_K_MAX) -> torch.Tensor:
    r"""Differentiable expected ambiguity ``E[k_e] = sum_c c * p_{e,c}``.

    Parameters
    ----------
    probs:
        Class probabilities with the class axis of size ``2*k_max+1`` as the
        channel dimension, shape ``(..., 2*k_max+1, H, W)`` (softmax already
        applied).
    k_max:
        Ambiguity half-range matching ``probs``' channel count.

    Returns
    -------
    torch.Tensor
        ``E[k_e]`` of shape ``(..., H, W)``.
    """
    n = num_classes(k_max)
    if probs.shape[-3] != n:
        raise ValueError(f"probs has {probs.shape[-3]} classes, expected {n} for k_max={k_max}.")
    values = torch.tensor(
        ambiguity_values(k_max), dtype=probs.dtype, device=probs.device
    ).view(*([1] * (probs.dim() - 3)), n, 1, 1)
    return (probs * values).sum(dim=-3)


# ---------------------------------------------------------------------------
# Coherence -> edge weights (the weights w_e in Eq. 4)
# ---------------------------------------------------------------------------
def coherence_to_edge_weights(
    coherence: torch.Tensor,
    reduce: str = "mean",
    pad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Derive per-edge weights ``w_e in [0, 1]`` from a pixel coherence field.

    Each edge weight combines the coherence of its two endpoint pixels. This is
    the physical prior SNAPHU uses and is what the weighted least-squares solver
    (Eq. 4) consumes; on real GeoTIFF/NISAR/UAVSAR products ``coherence`` is a
    dedicated band/dataset.

    Parameters
    ----------
    coherence:
        Pixel coherence ``(..., H, W)`` in ``[0, 1]``.
    reduce:
        How to combine the two endpoint coherences: ``"mean"`` (arithmetic),
        ``"min"`` (conservative), or ``"geo"`` (geometric mean).
    pad:
        If ``True`` return full-resolution maps with invalid edges weighted 0.

    Returns
    -------
    (wx, wy):
        Horizontal and vertical edge weights.
    """
    a_x, b_x = coherence[..., :, :-1], coherence[..., :, 1:]
    a_y, b_y = coherence[..., :-1, :], coherence[..., 1:, :]

    if reduce == "min":
        wx = torch.minimum(a_x, b_x)
        wy = torch.minimum(a_y, b_y)
    elif reduce == "geo":
        wx = torch.sqrt(torch.clamp(a_x * b_x, min=0.0))
        wy = torch.sqrt(torch.clamp(a_y * b_y, min=0.0))
    elif reduce == "mean":
        wx = 0.5 * (a_x + b_x)
        wy = 0.5 * (a_y + b_y)
    else:
        raise ValueError(f"Unknown reduce={reduce!r}; use mean|min|geo.")

    if pad:
        wx = pad_edge_x(wx, 0.0)
        wy = pad_edge_y(wy, 0.0)
    return wx, wy


def default_edge_weights(
    shape: Tuple[int, ...], *, device=None, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform edge weights (all 1) with invalid edges zeroed.

    Used for datasets without a coherence band, so that the weighted
    least-squares solver degrades gracefully to the unweighted case.
    ``shape`` is the full field shape ``(..., H, W)``.
    """
    h, w = shape[-2], shape[-1]
    wx = torch.ones(shape, device=device, dtype=dtype)
    wy = torch.ones(shape, device=device, dtype=dtype)
    wx[..., :, -1] = 0.0
    wy[..., -1, :] = 0.0
    return wx, wy


# ---------------------------------------------------------------------------
# Reference fixing / alignment (for evaluation)
# ---------------------------------------------------------------------------
def circular_mean(angle: torch.Tensor) -> torch.Tensor:
    """Per-sample circular mean of an angle field, shape ``(..., H, W) -> (..., 1, 1)``.

    Computed as ``atan2(mean sin, mean cos)`` over the last two dims (kept for
    broadcasting). Used to estimate the unidentifiable global phase reference.
    """
    s = torch.sin(angle).mean(dim=(-2, -1), keepdim=True)
    c = torch.cos(angle).mean(dim=(-2, -1), keepdim=True)
    return torch.atan2(s, c)


def align_reference(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""Shift ``phi`` by the (detached) constant that best aligns it to ``psi`` mod 2π.

    The weighted least-squares solution is defined only up to one additive
    constant (the unwrapping reference, proposal §2.2), which the integrator
    cannot recover. The congruence loss ``1 - cos(phi - psi)`` is *not* invariant
    to that constant, so before comparing we add the circular mean of
    ``psi - phi``. The offset is detached: the global reference is genuinely
    unidentifiable from the data, so no gradient should push the network to
    change it.
    """
    offset = circular_mean(psi - phi).detach()
    return phi + offset


def move_batch_to_device(batch: dict, device: torch.device, non_blocking: bool = True) -> dict:
    """Move every tensor in a sample/batch dict to ``device`` (leaves non-tensors)."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
    return out


def zero_mean(
    field: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Remove the additive reference constant by subtracting the (masked) mean.

    The weighted least-squares solution is defined only up to one additive
    constant (proposal §2.2). For fair RMSE/MAE we align predictions and
    targets by zero-meaning over a (coherent) region. ``mask`` selects the
    region; if ``None`` the whole field is used. Reduces over the last two dims.
    """
    if mask is None:
        m = field.mean(dim=(-2, -1), keepdim=True)
    else:
        mask = mask.to(field.dtype)
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        m = (field * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return field - m
