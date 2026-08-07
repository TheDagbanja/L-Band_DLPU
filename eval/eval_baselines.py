"""Driver (BASELINE_HARNESS_SPEC.md s8): load split -> run registry -> aggregate.

    python -m eval.eval_baselines \
        --data  data/LB_DLPU/sim_2 \
        --split test \
        --methods all \
        --weights-dir checkpoints \
        --out   results/sim_2_leaderboard \
        [--max N]            # smoke: cap patches per (regime x difficulty) cell

    python -m eval.eval_baselines --data data/LB_DLPU/sim_2 --self-test

Every method sees the identical sorted-id patch sequence (spec s1.5), and every
output is scored through :mod:`eval.metrics` -- the single, method-blind source
of truth. Methods whose weights (or the ``snaphu`` package) are missing are
reported as skipped rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from src import mcf

from . import data as data_mod
from . import metrics as M
from . import report as R
from .methods import ALL_METHODS, MethodUnavailable, build_registry


# ---------------------------------------------------------------------------
# per-patch scoring
# ---------------------------------------------------------------------------
def _score_patch(method, patch) -> Dict:
    """Run one method on one patch, return a metrics record (or an error stub)."""
    labels = {"phi": patch.phi, "kx": patch.kx_true, "ky": patch.ky_true,
              "valid": patch.valid, "regime": patch.regime}
    rec = {
        "id": patch.id, "regime": patch.regime, "difficulty": patch.difficulty,
        "method": method.name, "kind": method.kind,
        "residue_free_by_construction": method.residue_free_by_construction,
        "weights_provenance": method.weights_provenance,
    }
    try:
        phi_hat = method.fn(patch.psi, coherence=patch.coherence, labels=labels)
    except MethodUnavailable as e:
        rec.update(error=str(e), rmse=None, residues=None)
        return rec
    except Exception as e:                      # keep the run alive; flag the patch
        rec.update(error=f"{type(e).__name__}: {e}", rmse=None, residues=None)
        return rec

    R_psi = mcf.residues(patch.psi)
    rec["residues"] = M.residue_count(phi_hat, patch.psi, valid=patch.valid, R_psi=R_psi)
    rec["rmse"] = (M.aligned_rmse(phi_hat, patch.phi, valid=patch.valid)
                   if patch.phi is not None else None)
    rec["jump_rate"] = (M.jump_rate(phi_hat, patch.phi, valid=patch.valid)
                        if patch.phi is not None else None)
    if patch.phi is not None:
        rec.update(M.reconstruction_quality(phi_hat, patch.phi, valid=patch.valid))  # mae, psnr, ssim
    if method.exposes_edge_k and patch.kx_true is not None and patch.ky_true is not None:
        nx, ny = M.implied_edge_k(phi_hat, patch.psi)
        rec.update(M.k_edge_accuracy(nx, ny, patch.kx_true, patch.ky_true, valid=patch.valid))
    if patch.phi is not None and patch.coherence is not None:
        rec.update(M.coherence_stratified_rmse(phi_hat, patch.phi, patch.coherence, valid=patch.valid))
    return rec


def run(args) -> None:
    names = list(ALL_METHODS) if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    snaphu_cfg = {"cost": args.snaphu_cost, "real_coherence": not args.snaphu_pseudo_coherence}
    if args.snaphu_nlooks is not None:      # else per-regime ENL (nisar 8 / uavsar 24)
        snaphu_cfg["nlooks"] = args.snaphu_nlooks
    methods, skipped = build_registry(names, weights_dir=args.weights_dir,
                                      snaphu_cfg=snaphu_cfg, device=args.device)
    print(f"Methods to run ({len(methods)}): {[m.name for m in methods]}")
    if skipped:
        print("Skipped:")
        for n, why in skipped.items():
            print(f"  - {n}: {why}")
    if not methods:
        raise SystemExit("No runnable methods. Provide --weights-dir or install snaphu.")

    if args.source == "s1":
        from . import data_s1
        patch_iter = data_s1.iter_patches_s1(args.data, args.split, max_per_cell=args.max)
        print(f"Sentinel-1 real seismic: 256x256 tiles, phi=SNAPHU _unw_phase "
              f"(rmse = AGREEMENT with SNAPHU, not accuracy; no ground truth).")
    elif args.source == "dlpu":
        from . import data_dlpu
        patch_iter = data_dlpu.iter_patches_dlpu(args.data, args.split, max_per_cell=args.max)
        print(f"InSAR-DLPU (C-band) test: split=`{args.split}`, no coherence band "
              f"(coherence-fused cost falls back to learned-only).")
    elif args.real:
        patch_iter = data_mod.iter_real_patches(args.data, max_per_cell=args.max,
                                                min_valid=args.min_valid)
        print(f"REAL-DATA test: real/real_{{nisar,uavsar}} patches "
              f"(psi=wrap(ref); noise-free, tests structure generalisation). "
              f"min_valid={args.min_valid}")
    else:
        patch_iter = data_mod.iter_patches(args.data, args.split, max_per_cell=args.max)

    records: List[Dict] = []
    fail_counts: Dict[str, int] = {}
    t0 = time.time()
    n_patch = 0
    for patch in patch_iter:
        n_patch += 1
        for method in methods:
            rec = _score_patch(method, patch)
            if "error" in rec:
                fail_counts[method.name] = fail_counts.get(method.name, 0) + 1
                if fail_counts[method.name] <= 2:
                    print(f"  [warn] {method.name} on {patch.id}: {rec['error']}")
            records.append(rec)
        if n_patch % 20 == 0:
            dt = time.time() - t0
            print(f"  ...{n_patch} patches  ({dt:.0f}s, {dt/n_patch:.2f}s/patch)")
    print(f"Scored {n_patch} patches x {len(methods)} methods in {time.time()-t0:.0f}s.")

    # Drop methods that errored on *every* patch (e.g. oracle without labels).
    good_methods = {m.name for m in methods if fail_counts.get(m.name, 0) < n_patch}
    records = [r for r in records if r["method"] in good_methods or "error" not in r]
    for m in methods:
        if fail_counts.get(m.name, 0) >= n_patch:
            skipped[m.name] = f"errored on all {n_patch} patches"

    os.makedirs(args.out, exist_ok=True)
    header = _run_header(args, methods, skipped, n_patch)
    R.write_jsonl(records, os.path.join(args.out, "results.jsonl"))
    R.write_csv(records, os.path.join(args.out, "leaderboard.csv"))
    R.write_leaderboard_md(records, os.path.join(args.out, "leaderboard.md"), header=header)
    R.write_significance_md(records, os.path.join(args.out, "significance.md"))
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(_config_dict(args, methods, skipped, n_patch), f, indent=2)

    print(f"\nWrote leaderboard to {args.out}/")
    _print_console_summary(records)


def _run_header(args, methods, skipped, n_patch) -> str:
    prov = "; ".join(f"{m.name}={m.weights_provenance}" for m in methods if m.weights_provenance)
    lines = [
        f"Data: `{args.data}`  split=`{args.split}`  patches={n_patch}"
        + (f"  (max {args.max}/cell)" if args.max else ""),
        f"SNAPHU cfg: cost={args.snaphu_cost}, "
        f"nlooks={args.snaphu_nlooks if args.snaphu_nlooks is not None else 'per-regime(nisar 8/uavsar 24)'}, "
        f"coherence={'pseudo' if args.snaphu_pseudo_coherence else 'real'}.",
        f"Provenance: {prov}" if prov else "",
        f"Skipped: {', '.join(f'{k} ({v})' for k, v in skipped.items())}" if skipped else "",
    ]
    return "\n".join(x for x in lines if x)


def _config_dict(args, methods, skipped, n_patch) -> Dict:
    return {
        "data": args.data, "split": args.split, "n_patches": n_patch,
        "max_per_cell": args.max, "device": args.device,
        "snaphu": {"cost": args.snaphu_cost,
                   "nlooks": args.snaphu_nlooks if args.snaphu_nlooks is not None
                   else "per-regime(nisar 8/uavsar 24)",
                   "real_coherence": not args.snaphu_pseudo_coherence},
        "methods_run": {m.name: m.weights_provenance for m in methods},
        "skipped": skipped,
    }


def _print_console_summary(records: List[Dict]) -> None:
    for regime in sorted({r["regime"] for r in records}):
        rows = [c for c in R.aggregate(records, None) if c["regime"] == regime]
        rows.sort(key=lambda r: (float("inf") if np.isnan(r["rmse_mean"]) else r["rmse_mean"]))
        print(f"\n=== {regime} (all difficulties) ===")
        print(f"{'method':16s} {'RMSE':>8s} {'jump%':>7s} {'resid':>8s} {'%resfree':>9s} {'|k|>=2':>7s}  n")
        for r in rows:
            rmse = "--" if np.isnan(r["rmse_mean"]) else f"{r['rmse_mean']:.3f}"
            jr = "--" if np.isnan(r.get("jump_rate", float("nan"))) else f"{100*r['jump_rate']:.1f}"
            k2 = "--" if np.isnan(r["k2_acc"]) else f"{r['k2_acc']:.2f}"
            print(f"{r['method']:16s} {rmse:>8s} {jr:>7s} {r['residues_mean']:>8.1f} "
                  f"{100*r['frac_resfree']:>8.0f}% {k2:>7s}  {r['n']}")


# ---------------------------------------------------------------------------
# self-test (spec s9) -- run before trusting any table
# ---------------------------------------------------------------------------
def _brute_residue_count(phi_hat, psi, valid=None) -> int:
    """Naive O(HW) reference for the residue counter -- guards the vectorized one."""
    phi_hat = np.asarray(phi_hat, float); psi = np.asarray(psi, float)
    H, W = psi.shape
    def wg(a, b):  # wrapped gradient of psi
        return M.wrap(b - a)
    n = 0
    for i in range(H - 1):
        for j in range(W - 1):
            if valid is not None and not (valid[i, j] and valid[i, j+1]
                                          and valid[i+1, j] and valid[i+1, j+1]):
                continue
            nx_t = round(((phi_hat[i, j+1]-phi_hat[i, j]) - wg(psi[i, j], psi[i, j+1])) / M.TWO_PI)
            nx_b = round(((phi_hat[i+1, j+1]-phi_hat[i+1, j]) - wg(psi[i+1, j], psi[i+1, j+1])) / M.TWO_PI)
            ny_l = round(((phi_hat[i+1, j]-phi_hat[i, j]) - wg(psi[i, j], psi[i+1, j])) / M.TWO_PI)
            ny_r = round(((phi_hat[i+1, j+1]-phi_hat[i, j+1]) - wg(psi[i, j+1], psi[i+1, j+1])) / M.TWO_PI)
            curl = nx_t + ny_r - nx_b - ny_l
            Rp = round((wg(psi[i, j], psi[i, j+1]) + wg(psi[i, j+1], psi[i+1, j+1])
                        - wg(psi[i+1, j], psi[i+1, j+1]) - wg(psi[i, j], psi[i+1, j])) / M.TWO_PI)
            if curl + Rp != 0:
                n += 1
    return n


def self_test(args) -> None:
    from src import baselines
    print("== Correctness self-test (spec s9) ==")
    rng = np.random.default_rng(0)

    # (3) Residue-counter: smooth ramp -> 0; and vectorized == brute-force reference.
    H, W = 40, 44
    yy, xx = np.meshgrid(np.linspace(0, 2, H), np.linspace(0, 2, W), indexing="ij")
    ramp = 0.3 * xx + 0.2 * yy
    psi_ramp = M.wrap(ramp)
    assert M.residue_count(ramp, psi_ramp) == 0, "smooth ramp must have 0 residues"
    print("[3a] smooth ramp -> residue_count = 0  OK")

    # A field with genuine residues (steep vortex) unwrapped by LS (leaves residues).
    yy2, xx2 = np.meshgrid(np.linspace(-6, 6, H), np.linspace(-6, 6, W), indexing="ij")
    phi_v = 3.0 * np.arctan2(yy2, xx2) + 0.1 * (xx2**2 + yy2**2)
    psi_v = M.wrap(phi_v + 0.4 * rng.standard_normal((H, W)))
    ls = baselines.ls_unwrap(psi_v)
    vec = M.residue_count(ls, psi_v)
    brute = _brute_residue_count(ls, psi_v)
    assert vec == brute, f"vectorized ({vec}) != brute-force ({brute}) residue counter"
    assert vec > 0, "LS on a residue-rich field must leave >0 residues"
    print(f"[3b] LS field: vectorized == brute-force reference = {vec} (>0)  OK")

    # (4) Offset invariance: adding 2*pi*k must not change RMSE.
    phi_hat = phi_v + 0.05 * rng.standard_normal((H, W))
    r0 = M.aligned_rmse(phi_hat, phi_v)
    r1 = M.aligned_rmse(phi_hat + M.TWO_PI * 7, phi_v)
    r2 = M.aligned_rmse(phi_hat + 3.1234, phi_v)
    assert abs(r0 - r1) < 1e-9 and abs(r0 - r2) < 1e-9, f"offset variance: {r0},{r1},{r2}"
    print(f"[4] offset invariance: rmse={r0:.4f} unchanged by +2pi*7 and +const  OK")

    # (1,2) Oracle/unit sanity on real patches (needs data).
    if os.path.isdir(os.path.join(args.data, "sim")):
        from .methods import mcf_methods
        unit = mcf_methods.unit_mcf()
        oracle = mcf_methods.oracle_cost()
        n_ok = 0
        worst_o = 0
        rmses_u, rmses_o, rmses_floor = [], [], []
        for k, patch in enumerate(data_mod.iter_patches(args.data, args.split, max_per_cell=2)):
            if k >= 12:
                break
            lab = {"phi": patch.phi}
            phi_u = unit.fn(patch.psi)
            phi_o = oracle.fn(patch.psi, labels=lab)
            ru = M.residue_count(phi_u, patch.psi, valid=patch.valid)
            ro = M.residue_count(phi_o, patch.psi, valid=patch.valid)
            assert ru == 0, f"unit_mcf left {ru} residues on {patch.id}"
            assert ro == 0, f"oracle_cost left {ro} residues on {patch.id}"
            worst_o = max(worst_o, ro)
            if patch.phi is not None:
                ru_rmse = M.aligned_rmse(phi_u, patch.phi, patch.valid)
                ro_rmse = M.aligned_rmse(phi_o, patch.phi, patch.valid)
                kf = np.round((patch.phi - patch.psi) / M.TWO_PI)
                floor = M.aligned_rmse(patch.psi + M.TWO_PI * kf, patch.phi, patch.valid)
                rmses_u.append(ru_rmse); rmses_o.append(ro_rmse); rmses_floor.append(floor)
            n_ok += 1
        print(f"[1] oracle_cost residue-free on all {n_ok} patches (max {worst_o})  OK")
        print(f"[2] unit_mcf residue-free on all {n_ok} patches  OK")
        if rmses_u:
            mu, mo, mf = np.mean(rmses_u), np.mean(rmses_o), np.mean(rmses_floor)
            print(f"    RMSE: oracle={mo:.3f}  unit={mu:.3f}  floor(GT-K)={mf:.3f}  "
                  f"(expect oracle<=unit, oracle~floor)")
            assert mo <= mu + 1e-6, "oracle RMSE should not exceed unit RMSE"

    # (5) Method-blindness: metrics depend only on the array, not the method label.
    a = M.residue_count(ls, psi_v)
    b = M.residue_count(ls.copy(), psi_v.copy())
    assert a == b, "residue counter is not method/label invariant"
    print("[5] metrics are method-blind (depend only on the field)  OK")
    print("== self-test PASSED ==")


# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline evaluation harness for sim_2.")
    p.add_argument("--data", required=True, help="dataset root (…/LB_DLPU/sim_2, or InSAR-DLPU root for --source dlpu)")
    p.add_argument("--source", default="lband_sim", choices=["lband_sim", "dlpu", "s1"],
                   help="benchmark family: L-band .h5 (default), InSAR-DLPU C-band .mat, "
                        "or Sentinel-1 HyP3 geotiffs (agreement-vs-SNAPHU, phi=_unw_phase)")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "test_real"])
    p.add_argument("--methods", default="all",
                   help="'all' or comma list, e.g. unit_mcf,oracle_cost,snaphu_cost")
    p.add_argument("--weights-dir", default=None, help="dir with DL / edge-head checkpoints")
    p.add_argument("--out", default="results/sim_2_leaderboard")
    p.add_argument("--max", type=int, default=None, help="cap patches per (regime x difficulty) cell")
    p.add_argument("--device", default="cpu")
    p.add_argument("--snaphu-nlooks", type=float, default=None,
                   help="fixed ENL for SNAPHU; default None -> per-regime (nisar 8, uavsar 24)")
    p.add_argument("--snaphu-cost", default="defo")
    p.add_argument("--snaphu-pseudo-coherence", action="store_true",
                   help="use phase-derivative pseudo-coherence for SNAPHU (default: real band)")
    p.add_argument("--real", action="store_true",
                   help="score the REAL L-band patches (real/real_{nisar,uavsar}) instead of the sim split")
    p.add_argument("--min-valid", type=float, default=0.9,
                   help="(--real) skip patches below this valid-pixel fraction")
    p.add_argument("--self-test", action="store_true", help="run spec s9 checks and exit")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.self_test:
        self_test(args)
        return
    run(args)


if __name__ == "__main__":
    main()
