"""Aggregation & outputs (BASELINE_HARNESS_SPEC.md s6).

Turns the per-(patch, method) records into the four deliverables:

* ``results.jsonl``  -- one line per (patch, method); raw substrate.
* ``leaderboard.csv``-- long format, one row per (regime, difficulty, method) cell.
* ``leaderboard.md`` -- the paper table: per regime, a block per difficulty,
  methods sorted by RMSE, with the ``% residue-free`` column bolded for the MCF
  rows (that column is the contribution).
* ``significance.md``-- paired Wilcoxon signed-rank per regime.

Invariant (spec s1.2): cells are never aggregated across regime or difficulty.
The per-regime "all" block is a labelled convenience only.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

DIFFICULTY_ORDER = ("smooth", "mixed", "dense")


# ---------------------------------------------------------------------------
# raw records
# ---------------------------------------------------------------------------
def write_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_json_safe(r)) + "\n")


def _json_safe(r: Dict) -> Dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _nanstd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.std(xs)) if xs else float("nan")


def aggregate(records: List[Dict], difficulty: Optional[str] = None) -> List[Dict]:
    """Aggregate per (regime, difficulty, method). ``difficulty=None`` -> per
    (regime, method) collapsing difficulty (the labelled "all" convenience)."""
    cells: Dict = {}
    for r in records:
        if difficulty is not None and r["difficulty"] != difficulty:
            continue
        key = (r["regime"], difficulty or "all", r["method"])
        cells.setdefault(key, []).append(r)

    rows = []
    for (regime, diff, method), rs in cells.items():
        rows.append({
            "regime": regime, "difficulty": diff, "method": method,
            "kind": rs[0]["kind"],
            "residue_free_by_construction": rs[0]["residue_free_by_construction"],
            "n": len(rs),
            "rmse_mean": _nanmean([r.get("rmse") for r in rs]),
            "rmse_std": _nanstd([r.get("rmse") for r in rs]),
            "mae_mean": _nanmean([r.get("mae") for r in rs]),
            "psnr_mean": _nanmean([r.get("psnr") for r in rs]),
            "ssim_mean": _nanmean([r.get("ssim") for r in rs]),
            "residues_mean": _nanmean([r.get("residues") for r in rs]),
            "frac_resfree": _nanmean([1.0 if r.get("residues") == 0 else 0.0 for r in rs]),
            "jump_rate": _nanmean([r.get("jump_rate") for r in rs]),
            "k_acc": _nanmean([r.get("k_acc") for r in rs]),
            "k2_acc": _nanmean([r.get("k2_acc") for r in rs]),
            "rmse_hi_coh": _nanmean([r.get("rmse_hi_coh") for r in rs]),
            "rmse_lo_coh": _nanmean([r.get("rmse_lo_coh") for r in rs]),
            "weights_provenance": rs[0].get("weights_provenance", ""),
        })
    return rows


def write_csv(records: List[Dict], path: str) -> List[Dict]:
    """Long-format CSV: one row per (regime, difficulty, method) cell + the
    'all'-difficulty aggregate. Returns the aggregated rows."""
    regimes = sorted({r["regime"] for r in records})
    rows: List[Dict] = []
    for regime in regimes:
        for diff in (*DIFFICULTY_ORDER, None):
            rows.extend([c for c in aggregate(records, diff) if c["regime"] == regime])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["regime", "difficulty", "method", "kind", "n", "rmse_mean", "rmse_std",
            "mae_mean", "psnr_mean", "ssim_mean", "jump_rate", "residues_mean",
            "frac_resfree", "k_acc", "k2_acc", "rmse_hi_coh", "rmse_lo_coh",
            "residue_free_by_construction", "weights_provenance"]
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return rows


# ---------------------------------------------------------------------------
# leaderboard.md
# ---------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _cell_table(rows: List[Dict]) -> str:
    """One markdown table, methods sorted by RMSE (ascending). Bold %resfree for MCF."""
    rows = sorted(rows, key=lambda r: (math.inf if math.isnan(r["rmse_mean"]) else r["rmse_mean"]))
    lines = ["| method | kind | RMSE (rad) | MAE (rad) | PSNR (dB) | SSIM | jump-rate | "
             "residues (mean) | % residue-free | |k|>=2 acc | n |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rmse = f"{_fmt(r['rmse_mean'])} +/- {_fmt(r['rmse_std'])}"
        resfree = f"{100*r['frac_resfree']:.0f}%"
        if r["residue_free_by_construction"]:
            resfree = f"**{resfree}**"
        jr = f"{100*r['jump_rate']:.1f}%" if not math.isnan(r.get("jump_rate", float('nan'))) else "--"
        k2 = _fmt(r["k2_acc"]) if not math.isnan(r["k2_acc"]) else "--"
        mae = _fmt(r.get("mae_mean", float("nan")))
        psnr = _fmt(r.get("psnr_mean", float("nan")), 1)
        ssim = _fmt(r.get("ssim_mean", float("nan")))
        lines.append(f"| {r['method']} | {r['kind']} | {rmse} | {mae} | {psnr} | {ssim} | {jr} | "
                     f"{_fmt(r['residues_mean'],1)} | {resfree} | {k2} | {r['n']} |")
    return "\n".join(lines)


def write_leaderboard_md(records: List[Dict], path: str, header: str = "") -> None:
    regimes = sorted({r["regime"] for r in records})
    out = ["# Baseline leaderboard\n", header, ""]
    out.append("> `% residue-free` is bold for methods that are residue-free **by "
               "construction** (MCF family). That column is the contribution: no DL "
               "baseline can match it structurally.\n")
    for regime in regimes:
        out.append(f"\n## Regime: {regime}\n")
        all_rows = [c for c in aggregate(records, None) if c["regime"] == regime]
        out.append("### All difficulties (convenience aggregate)\n")
        out.append(_cell_table(all_rows))
        for diff in DIFFICULTY_ORDER:
            rows = [c for c in aggregate(records, diff) if c["regime"] == regime]
            if not rows:
                continue
            out.append(f"\n### Difficulty: {diff}\n")
            out.append(_cell_table(rows))
        out.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------------------
# significance.md (paired Wilcoxon per regime)
# ---------------------------------------------------------------------------
def _paired(records, regime, method_a, method_b, field):
    by_id_a, by_id_b = {}, {}
    for r in records:
        if r["regime"] != regime:
            continue
        if r["method"] == method_a:
            by_id_a[r["id"]] = r.get(field)
        elif r["method"] == method_b:
            by_id_b[r["id"]] = r.get(field)
    a, b = [], []
    for pid in sorted(set(by_id_a) & set(by_id_b)):
        va, vb = by_id_a[pid], by_id_b[pid]
        if va is None or vb is None:
            continue
        if isinstance(va, float) and math.isnan(va):
            continue
        if isinstance(vb, float) and math.isnan(vb):
            continue
        a.append(va); b.append(vb)
    return np.array(a, dtype=float), np.array(b, dtype=float)


def _wilcoxon_line(records, regime, a, b, field, present):
    if a not in present or b not in present:
        return f"- **{a} vs {b}** ({field}): skipped (a method is absent)."
    xa, xb = _paired(records, regime, a, b, field)
    if xa.size == 0:
        return f"- **{a} vs {b}** ({field}): no paired patches."
    diff = xa - xb
    med = float(np.median(diff))
    win = float(np.mean(xa < xb)) if field == "rmse" else float(np.mean(xa < xb))
    try:
        from scipy.stats import wilcoxon
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided").pvalue)
        pstr = f"p={p:.2e}"
    except Exception as e:
        pstr = f"p=NA ({type(e).__name__})"
    return (f"- **{a} vs {b}** ({field}): {pstr}, median({a}-{b})={med:+.4f}, "
            f"win-rate({a}<{b})={100*win:.0f}%, n={xa.size}")


def write_significance_md(records: List[Dict], path: str,
                          dl_methods=("phasenet2", "dlpu_cnn", "gradient_net",
                                      "attention_unet", "deeplabv3plus", "unetpp")) -> None:
    present = sorted({r["method"] for r in records})
    regimes = sorted({r["regime"] for r in records})
    out = ["# Significance (paired Wilcoxon signed-rank)\n",
           "Paired over patches, per regime. Accuracy claim on RMSE; "
           "differentiation claim on residue count.\n"]
    for regime in regimes:
        out.append(f"\n## Regime: {regime}\n")
        out.append("**Accuracy (RMSE):**")
        out.append(_wilcoxon_line(records, regime, "gt_grad_cost", "snaphu_cost", "rmse", present))
        out.append("\n**Differentiation (residue count):**")
        for dl in dl_methods:
            out.append(_wilcoxon_line(records, regime, "gt_grad_cost", dl, "residues", present))
        out.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
