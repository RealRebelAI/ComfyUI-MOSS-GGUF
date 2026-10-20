"""Prepare Comfy-Org MOSS-Image BF16 weights for the official MOSS loader.

The Comfy-Org BF16 repack stores per-layer QKV as one fused weight,
while the official OpenMOSS ComfyUI model currently defines three separate
q/k/v Linear modules. Splitting along output channels is an exact *view*,
not a weight conversion or a second checkpoint allocation.
"""
from __future__ import annotations

import re

import torch

_FUSED = re.compile(r"^(?P<stem>.*?layers\.(?P<index>\d+)\.proj)\.qkv\.weight$")
_FFN = re.compile(r"^(?P<stem>.*?layers\.(?P<index>\d+)\.proj)\.ffn_up_gate\.weight$")


def normalize_bf16_state(state):
    """Convert fused QKV to q/k/v views in place; return (state, split_count).

    Reject missing or mixed projection layouts instead of letting ComfyUI
    instantiate unresolved meta weights. Does not alter numerical values.
    """
    if not isinstance(state, dict):
        raise TypeError("MOSS BF16 expected a checkpoint state dictionary")

    for key in ("video_embedder.bias", "final_linear_video.weight", "layers.0.proj.ffn_up_gate.weight"):
        if key not in state:
            raise ValueError(f"MOSS BF16 missing required tensor: {key}")

    for key in ("video_embedder.bias", "layers.0.proj.ffn_up_gate.weight"):
        tensor = state[key]
        if not torch.is_tensor(tensor) or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"MOSS BF16 requires floating-point weights; {key} is "
                f"{getattr(tensor, 'dtype', type(tensor).__name__)}. "
                "INT8 ConvRot safetensors are not BF16 checkpoints."
            )

    fused_keys = [key for key in state if _FUSED.fullmatch(key)]
    expected_stems = {m.group("stem") for key in state if (m := _FFN.fullmatch(key))}
    if not expected_stems:
        raise ValueError("MOSS BF16 contains no transformer projection blocks")

    for key in fused_keys:
        match = _FUSED.fullmatch(key)
        stem = match.group("stem")
        weight = state[key]
        if not torch.is_tensor(weight) or weight.ndim != 2 or weight.shape[0] != weight.shape[1] * 3:
            raise ValueError(f"MOSS BF16 fused QKV has invalid shape: {key}: {getattr(weight, 'shape', None)}")
        targets = tuple(f"{stem}.{letter}.weight" for letter in "qkv")
        if any(target in state for target in targets):
            raise ValueError(f"MOSS BF16 contains both fused and split QKV: {key}")
        # torch.chunk returns zero-copy views; avoids doubling RAM for 10GB model.
        q, k, v = weight.chunk(3, dim=0)
        del state[key]
        state[targets[0]] = q
        state[targets[1]] = k
        state[targets[2]] = v

    for stem in expected_stems:
        missing = [f"{stem}.{letter}.weight" for letter in "qkv" if f"{stem}.{letter}.weight" not in state]
        if missing:
            raise ValueError(f"MOSS BF16 has incomplete attention projections: {missing}")

    return state, len(fused_keys)
