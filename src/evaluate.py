"""Per-output evaluation metrics for the LB-DLPU baselines.

Reports, after reference alignment between prediction and ground truth:

* **RMSE / MAE** in radians.
* **2pi-jump rate** -- fraction of pixels off by at least one whole cycle
  (``round((phi_hat - phi_gt)/2pi) != 0`` after alignment).
* **residues** -- number of Goldstein residues remaining in the predicted
  solution, measured on the *corrected* gradient (wrapped input + the
  solution's implied integer jumps), so a residue-free solver scores exactly
  0 and a direct regression scores >0. This is the discriminating
  edge-ambiguity definition shared with :func:`eval.metrics.residue_count`;
  counting the raw curl of ``phi_hat`` would telescope to 0 for every method.
  The wrapped-input residue count is reported for reference.

These functions consume a network's ``outputs`` dict and are reused by the
baseline training loop (:mod:`scripts.train_baseline`). They are pure array
ops -- no model is imported here.
"""

from __future__ import annotations

from typing import Dict

import torch

from . import utils


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def phase_metrics(
    phi_hat: torch.Tensor, phi_gt: torch.Tensor
) -> Dict[str, float]:
    """RMSE, MAE and 2pi-jump rate after removing the per-sample reference.

    ``phi_hat`` / ``phi_gt`` are ``(B, 1, H, W)`` or ``(B, H, W)``. Both are
    zero-meaned (the integrator's additive constant is unidentifiable), then
    compared. Returned values are averaged over the batch.
    """
    if phi_hat.dim() == 4:
        phi_hat = phi_hat.squeeze(1)
    if phi_gt.dim() == 4:
        phi_gt = phi_gt.squeeze(1)
    pred = utils.zero_mean(phi_hat)
    gt = utils.zero_mean(phi_gt)
    resid = pred - gt
    rmse = torch.sqrt((resid ** 2).mean(dim=(-2, -1)))
    mae = resid.abs().mean(dim=(-2, -1))
    jumps = (torch.round(resid / utils.TWO_PI) != 0).float().mean(dim=(-2, -1))
    return {
        "rmse": rmse.mean().item(),
        "mae": mae.mean().item(),
        "jump_rate": jumps.mean().item(),
    }


@torch.no_grad()
def residue_charge(g_x: torch.Tensor, g_y: torch.Tensor) -> torch.Tensor:
    r"""Goldstein residue charge per 2x2 loop from a gradient field.

    ``g_x``/``g_y`` are full-resolution padded edge maps ``(..., H, W)`` (the
    horizontal/vertical gradients). The loop curl around cell ``(i, j)`` is

        curl = g_x[i,j] + g_y[i,j+1] - g_x[i+1,j] - g_y[i,j]

    and the charge is ``round(curl / 2pi) in {..., -1, 0, +1, ...}``. Returns a
    ``(..., H-1, W-1)`` integer-valued tensor.
    """
    curl = (
        g_x[..., :-1, :-1]
        + g_y[..., :-1, 1:]
        - g_x[..., 1:, :-1]
        - g_y[..., :-1, :-1]
    )
    return torch.round(curl / utils.TWO_PI)


@torch.no_grad()
def count_residues(g_x: torch.Tensor, g_y: torch.Tensor) -> float:
    """Mean number of non-zero residues per image in a gradient field."""
    charge = residue_charge(g_x, g_y)
    per_image = (charge != 0).flatten(start_dim=-2).sum(dim=-1).float()
    return per_image.mean().item()


# ---------------------------------------------------------------------------
# Per-output evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_outputs(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
    zero_class: int = utils.ZERO_AMBIGUITY_CLASS,
) -> Dict[str, float]:
    """Compute all metrics for a single forward pass (requires ``absolute``).

    ``zero_class`` is the class index of ``k=0`` for the edge head in use
    (``utils.zero_ambiguity_class(k_max)``; defaults to the GRSL 3-class value).
    """
    metrics: Dict[str, float] = {}
    if "absolute" in batch:
        metrics.update(phase_metrics(outputs["phi_hat"], batch["absolute"]))
    # residues_pred: residues of the CORRECTED gradient (the wrapped input plus
    # the solution's implied integer jumps), NOT of phi_hat's raw gradients. The
    # raw curl of any single-valued field telescopes to exactly 0, so counting it
    # reports 0 for EVERY method (MCF and DL alike) and cannot discriminate a
    # residue-free solver from a regression. Building
    #   g = W(grad psi) + 2pi * round((grad phi_hat - W(grad psi)) / 2pi)
    # makes MCF outputs exactly 0 (flow feasibility) and leaves DL regressions
    # >0 -- the same edge-ambiguity definition as eval.metrics.residue_count.
    psi = batch["wrapped"].squeeze(1)
    wgx, wgy = utils.wrapped_gradients(psi, pad=True)
    metrics["residues_input"] = count_residues(wgx, wgy)
    n_x = torch.round((outputs["g_x"] - wgx) / utils.TWO_PI)
    n_y = torch.round((outputs["g_y"] - wgy) / utils.TWO_PI)
    metrics["residues_pred"] = count_residues(wgx + utils.TWO_PI * n_x,
                                              wgy + utils.TWO_PI * n_y)

    # --- Per-pixel wrap-count head: report K accuracy, skip edge-cut metrics ---
    if "k_reg" in outputs:
        if "k_pixel" in batch:
            kp = outputs["k_pred"]
            kg = batch["k_pixel"]
            # offset-invariant: K is defined up to a global integer constant.
            off = torch.round((kg - kp).mean(dim=(-2, -1), keepdim=True))
            metrics["k_accuracy"] = (kp + off == kg).float().mean().item()
        return metrics

    # Applied-cut rate: fraction of edges where a non-zero correction was actually
    # integrated (i.e. after any cut_threshold gating). Derived from the corrected
    # gradients so it reflects what the integrator really used, not raw argmax.
    applied_h = torch.round((outputs["g_x"] - wgx) / utils.TWO_PI)
    applied_v = torch.round((outputs["g_y"] - wgy) / utils.TWO_PI)
    metrics["applied_cut_rate"] = 0.5 * (
        (applied_h[..., :, :-1] != 0).float().mean()
        + (applied_v[..., :-1, :] != 0).float().mean()
    ).item()
    if "k_x" in batch and "k_y" in batch:
        gt_nz = 0.5 * (
            (batch["k_x"][..., :, :-1] != zero_class).float().mean()
            + (batch["k_y"][..., :-1, :] != zero_class).float().mean()
        )
        metrics["gt_nonzero_rate"] = gt_nz.item()

    # Cut-detection precision/recall/F1 (binary cut vs no-cut on the raw
    # argmax, over valid edges). Needs per-class probabilities (baselines
    # without a class distribution, e.g. a continuous gradient correction,
    # don't -- skip gracefully rather than KeyError).
    if "k_x" in batch and "k_y" in batch and "probs_h" in outputs and "probs_v" in outputs:
        tp = fp = fn = 0.0
        for probs, kc, valid in (
            (outputs["probs_h"], batch["k_x"], batch.get("valid_x")),
            (outputs["probs_v"], batch["k_y"], batch.get("valid_y")),
        ):
            pred = probs.argmax(1) != zero_class
            gt = kc != zero_class
            if valid is not None:
                m = valid.bool()
                pred, gt = pred[m], gt[m]
            tp += float((pred & gt).sum())
            fp += float((pred & ~gt).sum())
            fn += float((~pred & gt).sum())
        metrics["cut_precision"] = tp / (tp + fp + 1e-9)
        metrics["cut_recall"] = tp / (tp + fn + 1e-9)
        metrics["cut_f1"] = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Render a metrics dict as a small aligned text table."""
    width = max((len(k) for k in metrics), default=10)
    lines = [f"=== {title} ==="]
    for k, v in metrics.items():
        lines.append(f"  {k:<{width}} : {v:.6f}")
    return "\n".join(lines)
