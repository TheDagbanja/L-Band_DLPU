"""Baseline evaluation harness (BASELINE_HARNESS_SPEC.md).

One command turns the frozen ``sim_2`` split into the data-paper leaderboard:
every baseline unwrapper run on the identical test split, scored per
regime x difficulty on **RMSE** *and* **residue count** through a single,
method-blind :mod:`eval.metrics` module.

Layout (spec s7):
  eval_baselines.py   -- driver: load split, run registry, aggregate, write out
  metrics.py          -- residue counter, RMSE, |k| accuracy (SINGLE SOURCE OF TRUTH)
  data.py             -- raw sim_2 .h5 loader (psi, phi, coherence, labels, attrs)
  methods/            -- classical / mcf / dl method registry
  report.py           -- leaderboard.md/.csv, results.jsonl, significance.md
"""
