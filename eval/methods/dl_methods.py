"""Deep-learning baselines (spec s2, s5): phasenet2, dlpu_cnn, gradient_net.

Loads an in-domain-trained checkpoint (``<weights_dir>/<name>_last.pt``, a bare
``state_dict`` as saved by ``scripts.train_baseline``) and runs inference. These
are direct-regression / classification unwrappers -- **not** residue-free by
construction (spec s10): even trained on ``sim_2/train`` their rounded output is
not guaranteed loop-consistent, so they leave residues where the MCF methods do
not. That contrast is the leaderboard's contribution.

Fairness protocol (spec s5): the leaderboard trains these *in-domain* on
``sim_2/train``. Each row records a ``weights_provenance`` string so a
cross-band checkpoint can never be silently mixed into the in-domain table.
"""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Optional

import numpy as np

from . import Method, MethodUnavailable

def _phasenet2(in_chans):
    from src.baselines_dl.phasenet2 import PhaseNet2Baseline
    return PhaseNet2Baseline(in_chans=in_chans, k_max=16)


def _dlpu(in_chans):
    from src.baselines_dl.dlpu_cnn import DLPUCNNBaseline
    return DLPUCNNBaseline(in_chans=in_chans)


def _gradnet(in_chans):
    from src.baselines_dl.gradient_net import GradientNetBaseline
    return GradientNetBaseline(in_chans=in_chans)


def _attn_unet(in_chans):
    from src.baselines_dl.attention_unet import AttentionUNetBaseline
    return AttentionUNetBaseline(in_chans=in_chans)


def _deeplab(in_chans):
    from src.baselines_dl.deeplabv3plus import DeepLabV3PlusBaseline
    return DeepLabV3PlusBaseline(in_chans=in_chans, k_max=16)


def _unetpp(in_chans):
    from src.baselines_dl.unetpp import UNetPPBaseline
    return UNetPPBaseline(in_chans=in_chans)


_BUILDERS = {
    "phasenet2": _phasenet2,
    "dlpu_cnn": _dlpu,
    "gradient_net": _gradnet,
    "attention_unet": _attn_unet,
    "deeplabv3plus": _deeplab,
    "unetpp": _unetpp,
}


def _sha1(path: str, cap: int = 8 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(cap))
    return h.hexdigest()[:12]


def _find_weights(weights_dir: str, name: str) -> str:
    for pat in (f"{name}_last.pt", f"{name}_best.pt", f"{name}*.pt"):
        hits = sorted(glob.glob(os.path.join(weights_dir, pat)))
        if hits:
            return hits[0]
    raise MethodUnavailable(f"no {name} checkpoint in {weights_dir!r} (train via scripts.train_baseline)")


def dl_baseline(name: str, weights_dir: Optional[str], device: str = "cpu") -> Method:
    if name not in _BUILDERS:
        raise MethodUnavailable(f"unknown DL baseline {name!r}")
    if not weights_dir or not os.path.isdir(weights_dir):
        raise MethodUnavailable(f"{name} needs --weights-dir with a trained checkpoint")
    import torch
    from src import utils

    path = _find_weights(weights_dir, name)
    model = _BUILDERS[name](4).to(device)
    try:
        state = torch.load(path, map_location=device)
    except Exception as e:
        raise MethodUnavailable(f"cannot load {path!r}: {type(e).__name__}: {e}")
    state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise MethodUnavailable(f"{name} checkpoint/arch mismatch "
                                f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    def call(psi, coherence=None, labels=None):
        psi_t = torch.from_numpy(np.asarray(psi, dtype=np.float32))
        batch = {
            "input": utils.encode_input(psi_t, include_grad=True).unsqueeze(0),
            "wrapped": psi_t.unsqueeze(0).unsqueeze(0),
        }
        batch = utils.move_batch_to_device(batch, torch.device(device))
        with torch.no_grad():
            out = model(batch, hard=True)
        return np.asarray(out["phi_hat"][0, 0].detach().cpu().numpy(), dtype=np.float64)

    prov = f"{os.path.basename(path)}@{_sha1(path)} (weights: {weights_dir})"
    return Method(name=name, kind="dl", residue_free_by_construction=False,
                  needs=("weights",), fn=call, weights_provenance=prov, exposes_edge_k=False)
