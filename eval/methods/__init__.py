"""Method registry (BASELINE_HARNESS_SPEC.md s2).

Each unwrapper is a :class:`Method` -- a callable with declared metadata. The
driver runs every registered method on the identical patch sequence and scores
its output through the single :mod:`eval.metrics` module.

``fn`` contract::

    fn(psi, coherence=None, labels=None) -> phi_hat   # (H, W) float64

``labels`` is a dict ``{"phi", "kx", "ky", "valid"}`` (ground truth; only the
gt_grad_cost method consumes it). DL weights are bound at registry-build time
(closured into ``fn``), so a method whose weights are missing is reported as
skipped rather than silently mixed into the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


class MethodUnavailable(Exception):
    """Raised when a method cannot be built (e.g. weights/snaphu missing)."""


@dataclass
class Method:
    name: str
    kind: str                                  # "classical" | "mcf" | "dl"
    residue_free_by_construction: bool
    needs: Tuple[str, ...]                     # subset of (coherence, labels, weights)
    fn: Callable
    weights_provenance: str = ""               # dataset + checkpoint hash (spec s5)
    exposes_edge_k: bool = False               # True -> report |k| accuracy


# Canonical leaderboard row order (spec s2 table).
ALL_METHODS: Tuple[str, ...] = (
    "goldstein", "quality_guided", "weighted_ls",
    "unit_mcf", "snaphu_cost", "gt_grad_cost",
    "phasenet2", "dlpu_cnn", "gradient_net",
    "attention_unet", "deeplabv3plus", "unetpp",
)


def build_registry(
    names: List[str],
    weights_dir: Optional[str] = None,
    snaphu_cfg: Optional[Dict] = None,
    device: str = "cpu",
) -> Tuple[List[Method], Dict[str, str]]:
    """Instantiate the requested methods. Returns (methods, skipped{name: reason}).

    A method that cannot be built (missing weights, snaphu not installed) is
    placed in ``skipped`` with a human-readable reason instead of aborting the
    whole run -- the classical/MCF floor still produces a table.
    """
    from . import classical, mcf_methods, dl_methods

    builders: Dict[str, Callable[[], Method]] = {
        "goldstein": classical.goldstein,
        "quality_guided": classical.quality_guided,
        "weighted_ls": classical.weighted_ls,
        "unit_mcf": mcf_methods.unit_mcf,
        "snaphu_cost": lambda: mcf_methods.snaphu_cost(snaphu_cfg or {}),
        "gt_grad_cost": mcf_methods.gt_grad_cost,
        "phasenet2": lambda: dl_methods.dl_baseline("phasenet2", weights_dir, device),
        "dlpu_cnn": lambda: dl_methods.dl_baseline("dlpu_cnn", weights_dir, device),
        "gradient_net": lambda: dl_methods.dl_baseline("gradient_net", weights_dir, device),
        "attention_unet": lambda: dl_methods.dl_baseline("attention_unet", weights_dir, device),
        "deeplabv3plus": lambda: dl_methods.dl_baseline("deeplabv3plus", weights_dir, device),
        "unetpp": lambda: dl_methods.dl_baseline("unetpp", weights_dir, device),
    }

    methods: List[Method] = []
    skipped: Dict[str, str] = {}
    for name in names:
        if name not in builders:
            skipped[name] = f"unknown method (not in {list(builders)})"
            continue
        try:
            methods.append(builders[name]())
        except MethodUnavailable as e:
            skipped[name] = str(e)
    return methods, skipped
