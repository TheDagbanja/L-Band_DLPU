"""Classical reference unwrappers (spec s2): goldstein, quality-guided, weighted LS.

Thin wrappers around :mod:`src.baselines` so the harness reuses the exact same
implementations the rest of the codebase uses. None is residue-free *by
construction* in the MCF sense, though a congruent path-following method may
still happen to leave zero residues on a given patch -- the harness measures
that identically for every method rather than asserting it.
"""

from __future__ import annotations

import numpy as np

from src import baselines

from . import Method


def _wrap(name, fn, needs=()):
    def call(psi, coherence=None, labels=None):
        return np.asarray(fn(psi), dtype=np.float64)

    return Method(name=name, kind="classical", residue_free_by_construction=False,
                  needs=needs, fn=call)


def goldstein() -> Method:
    return _wrap("goldstein", baselines.goldstein_unwrap)


def quality_guided() -> Method:
    return _wrap("quality_guided", baselines.quality_guided_unwrap, needs=("coherence",))


def weighted_ls() -> Method:
    # Least-squares (Ghiglia-Romero) DCT integration of the wrapped gradients.
    return _wrap("weighted_ls", baselines.ls_unwrap, needs=("coherence",))
