"""Train/evaluate a deep-learning baseline on identical splits.

Shared entry point for the three reproduced baselines
(:mod:`src.baselines_dl`), so they are trained on the *same* InSAR-DLPU-style
training set and scored on the *same* test sets as every other method in
the harness -- the fair-comparison requirement of a shared, frozen protocol.

    python -m scripts.train_baseline baseline=dlpu_cnn data.source=lband_sim \\
        train.epochs=25 logging.backend=none
    python -m scripts.train_baseline baseline=phasenet2 data.source=lband_sim \\
        model.decoder.k_max=16 logging.backend=none
    python -m scripts.train_baseline baseline=gradient_net data.source=lband_sim \\
        data.k_max=2 logging.backend=none

Adds one config group, ``baseline`` (default ``dlpu_cnn``), on top of the
usual ``data``/``model``/``train`` groups; everything else (Hydra overrides,
checkpoint dir, logging backend) works exactly like ``src.train``.
"""

from __future__ import annotations

import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import dataset as dataset_module
from src import utils
from src.evaluate import evaluate_outputs, format_metrics_table
from src.baselines_dl.dlpu_cnn import DLPUCNNBaseline, dlpu_loss
from src.baselines_dl.phasenet2 import PhaseNet2Baseline, phasenet2_loss
from src.baselines_dl.gradient_net import GradientNetBaseline, loss_from_k_max
from src.baselines_dl.attention_unet import AttentionUNetBaseline, attention_unet_loss
from src.baselines_dl.deeplabv3plus import DeepLabV3PlusBaseline, deeplabv3plus_loss
from src.baselines_dl.unetpp import UNetPPBaseline, unetpp_loss


def _kcls(cfg):
    return int(cfg.model.decoder.get("k_max", 16))


_BUILDERS = {
    "dlpu_cnn": lambda cfg, in_chans: DLPUCNNBaseline(in_chans=in_chans),
    "phasenet2": lambda cfg, in_chans: PhaseNet2Baseline(in_chans=in_chans, k_max=_kcls(cfg)),
    "gradient_net": lambda cfg, in_chans: GradientNetBaseline(in_chans=in_chans),
    "attention_unet": lambda cfg, in_chans: AttentionUNetBaseline(in_chans=in_chans),
    "deeplabv3plus": lambda cfg, in_chans: DeepLabV3PlusBaseline(in_chans=in_chans, k_max=_kcls(cfg)),
    "unetpp": lambda cfg, in_chans: UNetPPBaseline(in_chans=in_chans),
}


def _init_wandb(cfg: DictConfig, name: str):
    """Return a wandb run (or None). Mirrors src.train.Logger's wandb path."""
    if str(cfg.logging.get("backend", "none")).lower() != "wandb":
        return None
    try:
        import wandb
        wandb.init(
            project=cfg.logging.get("project", "insar-peft-unwrap-lband"),
            name=cfg.logging.get("run_name", None) or f"baseline-{name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.logging.get("log_dir", "./outputs"),
            reinit=True,
        )
        return wandb
    except Exception as exc:
        print(f"[wandb] unavailable ({exc}); training without logging.")
        return None


def _loss(name: str, cfg: DictConfig, outputs, batch):
    if name == "dlpu_cnn":
        return dlpu_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", 0.01)))
    if name == "phasenet2":
        return phasenet2_loss(outputs, batch, k_max=int(cfg.model.decoder.get("k_max", 16)))
    if name == "gradient_net":
        return loss_from_k_max(outputs, batch, k_max=int(cfg.data.get("k_max", 1)),
                               lambda_cong=float(cfg.train.get("lambda_cong_gn", 0.1)))
    if name == "attention_unet":
        return attention_unet_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", 0.01)))
    if name == "deeplabv3plus":
        return deeplabv3plus_loss(outputs, batch, k_max=_kcls(cfg))
    if name == "unetpp":
        return unetpp_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", 0.01)))
    raise ValueError(f"Unknown baseline={name!r}; choose from {list(_BUILDERS)}.")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    name = str(cfg.get("baseline", "dlpu_cnn"))
    if name not in _BUILDERS:
        raise SystemExit(f"baseline must be one of {list(_BUILDERS)}, got {name!r}.")
    utils.set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = utils.get_device(cfg.device)
    in_chans = dataset_module.input_channels(cfg.data)
    model = _BUILDERS[name](cfg, in_chans).to(device)
    print(f"Baseline={name}  params={model.count_parameters()['total']:,}  device={device}")

    # Optional Weights & Biases logging (mirrors src.train's Logger). No-op unless
    # logging.backend=wandb; run_name defaults to the baseline name so the three
    # baselines show up as distinct runs in the same project.
    wb = _init_wandb(cfg, name)

    if cfg.checkpoint.resume and os.path.exists(cfg.checkpoint.resume):
        model.load_state_dict(torch.load(cfg.checkpoint.resume, map_location=device))
        print(f"Loaded {cfg.checkpoint.resume}")

    if str(cfg.regime) == "pretrain":
        loader = dataset_module.build_dataloader(cfg.data, split="train", device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        step = 0
        for epoch in range(int(cfg.train.epochs)):
            model.train()
            bar = tqdm(loader, desc=f"[{name}] epoch {epoch + 1}/{cfg.train.epochs}", dynamic_ncols=True)
            for batch in bar:
                batch = utils.move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                terms = _loss(name, cfg, outputs, batch)
                terms["total"].backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                bar.set_postfix(loss=f"{terms['total'].item():.4f}")
                if wb is not None and step % int(cfg.logging.get("log_every", 50)) == 0:
                    wb.log({f"train/{k}": (v.detach().item() if torch.is_tensor(v) else float(v))
                            for k, v in terms.items()} | {"epoch": epoch + 1}, step=step)
                step += 1
            os.makedirs(cfg.checkpoint.dir, exist_ok=True)
            path = os.path.join(cfg.checkpoint.dir, f"{name}_last.pt")
            torch.save(model.state_dict(), path)
        print(f"Training complete. Final checkpoint: {path}")

    # --- Evaluation on the configured eval split (same protocol as eval.eval_baselines) ---
    split = cfg.eval.get("split", "test")
    eval_loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    model.eval()
    sums, total = {}, 0
    zero_class = int(cfg.data.get("k_max", 1)) if name == "gradient_net" else 0
    with torch.no_grad():
        bar = tqdm(eval_loader, desc=f"[{name}] eval [{split}]", dynamic_ncols=True)
        for batch in bar:
            batch = utils.move_batch_to_device(batch, device)
            outputs = model(batch, hard=True)
            m = evaluate_outputs(outputs, batch, zero_class=zero_class)
            bs = batch["wrapped"].shape[0]
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v * bs
            total += bs
    metrics = {k: v / max(total, 1) for k, v in sums.items()}
    print(format_metrics_table(metrics, title=f"{name} on '{split}' ({total} scenes)"))
    if wb is not None:
        wb.log({f"eval/{k}": float(v) for k, v in metrics.items()})
        wb.finish()


if __name__ == "__main__":
    main()
