"""LoRA bridge for Rebel MOSS GGUF and BF16 transformer models.

ComfyUI's standard LoRA loader does not split an adapter trained on a fused
QKV projection into the official OpenMOSS separate q/k/v modules. Convert
only that representation mismatch, then delegate actual patching to ComfyUI.
"""
from __future__ import annotations

import logging
import re

import torch

_LOG = logging.getLogger("MOSS-Rebel-LoRA")
_FUSED_A = re.compile(
    r"^(?P<stem>(?:(?:diffusion_model|transformer|base_model\.model|model)\.)?layers\.\d+\.proj)\.qkv\.(?P<a>lora_A\.weight|lora_down\.weight)$"
)
_FORMATS = {
    "lora_A.weight": ("lora_A.weight", "lora_B.weight"),
    "lora_down.weight": ("lora_down.weight", "lora_up.weight"),
}


def split_fused_qkv_loras(lora, model):
    """Return (normalized_lora, fused_layer_count), without modifying input.

    A fused QKV LoRA computes B @ A, where B's output dimension is the
    concatenation [Q, K, V]. Splitting B by the *actual model projection
    dimensions* and reusing A gives identical deltas for each projection.
    Tensor slices are views, so no large merged-weight allocation is needed.
    """
    if not isinstance(lora, dict):
        raise TypeError("Expected a LoRA state dictionary")
    state = model.model.state_dict()
    pairs = []
    for key in lora:
        match = _FUSED_A.fullmatch(key)
        if match:
            stem = match.group("stem")
            a_suffix, b_suffix = _FORMATS[match.group("a")]
            pairs.append((stem, a_suffix, b_suffix, key, f"{stem}.qkv.{b_suffix}"))

    if not pairs:
        return lora, 0

    converted = dict(lora)
    for stem, a_suffix, b_suffix, a_key, b_key in pairs:
        if b_key not in lora:
            raise ValueError(f"MOSS fused QKV LoRA missing matching B/up tensor: {b_key}")
        a, b = lora[a_key], lora[b_key]
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.ndim == 2 and b.ndim == 2):
            raise ValueError(f"MOSS fused QKV LoRA must use 2D A/B tensors: {stem}")
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"MOSS fused QKV LoRA rank mismatch: {stem}: {tuple(a.shape)}, {tuple(b.shape)}")

        # Accept the trainer's prefix, but always patch the real Comfy model.
        unwrapped = re.sub(r"^(?:diffusion_model|transformer|base_model\.model|model)\.", "", stem)
        base = f"diffusion_model.{unwrapped}"
        names = [f"{base}.{letter}.weight" for letter in "qkv"]
        if not all(name in state for name in names):
            raise ValueError(
                f"MOSS model has no separate Q/K/V projections for {stem}. "
                "This LoRA requires a Rebel MOSS GGUF/BF16 transformer."
            )
        dims = [state[name].shape[0] for name in names]
        if sum(dims) != b.shape[0] or any(state[name].shape[1] != a.shape[1] for name in names):
            raise ValueError(
                f"MOSS QKV LoRA shape mismatch at {stem}: "
                f"B={tuple(b.shape)}, A={tuple(a.shape)}, model QKV={dims}"
            )

        extras = (".alpha", ".dora_scale")
        known = {a_key, b_key, *(f"{stem}.qkv{suffix}" for suffix in extras)}
        unexpected = [
            key for key in lora if key.startswith(f"{stem}.qkv.")
            and key not in known
        ]
        if unexpected:
            raise ValueError(f"Unsupported fused MOSS LoRA tensors at {stem}: {unexpected}")

        del converted[a_key]
        del converted[b_key]
        alpha_key = f"{stem}.qkv.alpha"
        dora_key = f"{stem}.qkv.dora_scale"
        alpha = converted.pop(alpha_key, None)
        dora = converted.pop(dora_key, None)
        if dora is not None and (not torch.is_tensor(dora) or dora.shape[0] != sum(dims)):
            raise ValueError(f"Invalid MOSS fused DoRA scale: {dora_key}")
        for i, letter in enumerate("qkv"):
            target = f"{base}.{letter}"
            target_a = f"{target}.{a_suffix}"
            target_b = f"{target}.{b_suffix}"
            if target_a in converted or target_b in converted:
                raise ValueError(f"Conflicting fused and separate LoRA adapters: {target}")
            start = sum(dims[:i])
            converted[target_a] = a
            converted[target_b] = b.narrow(0, start, dims[i])
            if alpha is not None:
                converted[f"{target}.alpha"] = alpha
            if dora is not None:
                converted[f"{target}.dora_scale"] = dora.narrow(0, start, dims[i])

    _LOG.info(
        "MOSS LoRA: mapped %d fused QKV adapters to %d Q/K/V projection adapters",
        len(pairs), len(pairs) * 3,
    )
    return converted, len(pairs)


def apply_moss_lora(model, clip, lora, strength_model, strength_clip, metadata=None):
    """Use ComfyUI's normal LoRA patcher, after MOSS-only key normalization."""
    import comfy.sd

    converted, count = split_fused_qkv_loras(lora, model)
    before = len(model.patches) if model is not None else 0
    patched_model, patched_clip = comfy.sd.load_lora_for_models(
        model, clip, converted, strength_model, strength_clip,
        lora_metadata=metadata,
    )
    after = len(patched_model.patches) if patched_model is not None else 0
    if strength_model != 0 and count and after <= before:
        raise RuntimeError(
            "MOSS fused QKV LoRA was converted but no model patches were installed. "
            "Check that the MODEL input comes from Rebel MOSS GGUF/BF16."
        )
    if strength_model != 0 and count:
        _LOG.info("MOSS LoRA: %d fused QKV blocks patched successfully", count)
    return patched_model, patched_clip
