"""MCF-based unwrappers (spec s2): unit / snaphu / gt_grad_cost.

All three reuse the *same* residue-free solver (:func:`src.mcf.unwrap_mcf`) --
only the per-edge **cost** differs (spec s2: "do not re-implement the solver;
swap the cost"):

* ``unit_mcf``   -- flat unit cost (Costantini L1); MCF floor, no prior.
* ``snaphu_cost``-- the real SNAPHU statistical cost (Chen & Zebker), the
  classical gold standard; needs the ``snaphu`` package + a coherence layer.
* ``gt_grad_cost``-- cost derived from the ground-truth cuts; a reference,
  NOT a ceiling.

Every one is residue-free by construction (Theorem 2.1), so their residue
count is 0 on every patch -- the property the leaderboard's headline column
proves DL methods cannot match.
"""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Dict, Optional

import numpy as np

from src import baselines, mcf

from . import Method, MethodUnavailable


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------
def unit_mcf() -> Method:
    def call(psi, coherence=None, labels=None):
        return mcf.unwrap_mcf(psi, k_max=1)

    return Method(name="unit_mcf", kind="mcf", residue_free_by_construction=True,
                  needs=(), fn=call, exposes_edge_k=True)


# ---------------------------------------------------------------------------
# snaphu
# ---------------------------------------------------------------------------
# Per-regime equivalent number of looks (spec s8): match the generator's
# wrapped-phase looks so SNAPHU's statistical cost is correctly specified.
# A single ``nlooks`` in cfg overrides this map (applied to every regime).
_ENL_BY_REGIME = {"nisar": 8.0, "uavsar": 24.0}


def snaphu_cost(cfg: Dict) -> Method:
    if not baselines.snaphu_available():
        raise MethodUnavailable("snaphu package not installed (pip install snaphu)")
    fixed_nlooks = cfg.get("nlooks", None)         # None -> per-regime ENL
    cost = str(cfg.get("cost", "defo"))
    use_real_coh = bool(cfg.get("real_coherence", True))

    def call(psi, coherence=None, labels=None):
        regime = (labels or {}).get("regime", "nisar")
        nlooks = float(fixed_nlooks) if fixed_nlooks is not None else _ENL_BY_REGIME.get(regime, 8.0)
        coh = coherence if use_real_coh else None
        return baselines.snaphu_unwrap(psi, coherence=coh, nlooks=nlooks, cost=cost)

    nl = f"{fixed_nlooks:g}" if fixed_nlooks is not None else "per-regime(nisar 8/uavsar 24)"
    prov = f"snaphu:cost={cost},nlooks={nl},coh={'real' if use_real_coh else 'pseudo'}"
    return Method(name="snaphu_cost", kind="mcf", residue_free_by_construction=True,
                  needs=("coherence",), fn=call, weights_provenance=prov, exposes_edge_k=True)


# ---------------------------------------------------------------------------
# GT-gradient reference cost (NOT a ceiling; see the gt_grad_cost docstring)
# ---------------------------------------------------------------------------
def _gt_gradient_costs(psi: np.ndarray, phi: np.ndarray):
    """Cut-location prior from GT: cheap (0.01) where GT has a nonzero cut, else 1.0.

    NOT a ceiling: k*-matching is not the RMSE-optimal residue-free unwrapping
    under noise (the per-pixel GT-K floor is). This row can legitimately be beaten.
    """
    dxw = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    dyw = mcf.wrap(psi[1:, :] - psi[:-1, :])
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - dxw) / mcf.TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - dyw) / mcf.TWO_PI)
    return np.where(kx != 0, 0.01, 1.0), np.where(ky != 0, 0.01, 1.0)


def gt_grad_cost() -> Method:
    def call(psi, coherence=None, labels=None):
        if labels is None or labels.get("phi") is None:
            raise MethodUnavailable("gt_grad_cost needs ground-truth phi labels")
        wx, wy = _gt_gradient_costs(psi, np.asarray(labels["phi"], dtype=np.float64))
        return mcf.unwrap_mcf(psi, wx=wx, wy=wy, k_max=1)

    return Method(name="gt_grad_cost", kind="mcf", residue_free_by_construction=True,
                  needs=("labels",), fn=call, weights_provenance="reference:GT-gradient-cuts",
                  exposes_edge_k=True)
