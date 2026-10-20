"""Comfy-native INT8 ConvRot / W4A8 ConvRot MOSS checkpoint adapter.

Normalizes transformer-layer fused QKV keys to the official MOSS q/k/v
layout. Quantized weights remain packed; ComfyUI/comfy-kitchen handles all
dequantization, kernels, LoRA patching, and CPU/GPU offloading.
"""
from __future__ import annotations

import json
import re

import torch

FUSED = re.compile(r"^(?P<stem>layers\.(?P<index>\d+)\.proj)\.qkv\.weight$")
FFN = re.compile(r"^layers\.(\d+)\.proj\.ffn_up_gate\.weight$")
SUPPORTED = {"int8_tensorwise", "asym_w4a8_int8"}
SPLIT_ROWS = {"bias", "weight_scale", "weight_s_rel", "weight_s_channel"}
SHARE = {"comfy_quant", "weight_codebook", "input_scale", "pre_quant_scale"}


def _config(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint8 JSON tensor")
    try:
        config = json.loads(bytes(tensor.cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid Comfy quantization JSON at {name}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Expected object in {name}")
    return config


def normalize_quantized_state(state):
    """Return (normalized_state, split_count, format_counts).

    The original input mapping is not modified on failure. Tensor slices are
    views; packed int4 rows remain packed, and their scales stay row-aligned.
    """
    if not isinstance(state, dict):
        raise TypeError("MOSS ConvRot loader expects a state dictionary")
    for key in ("video_embedder.bias", "final_linear_video.weight", "layers.0.proj.ffn_up_gate.weight"):
        if key not in state:
            raise ValueError(f"Incomplete MOSS checkpoint, missing {key}")
    dim = state["video_embedder.bias"].numel()
    if not dim:
        raise ValueError("Invalid MOSS hidden dimension")

    output = dict(state)
    formats = {}
    for key, tensor in state.items():
        if not key.endswith(".comfy_quant"):
            continue
        config = _config(tensor, key)
        fmt = config.get("format")
        if fmt not in SUPPORTED:
            raise ValueError(
                f"Unsupported quantized format {fmt!r} in {key}. "
                "This experimental loader accepts INT8 ConvRot and W4A8 ConvRot only."
            )
        if fmt == "int8_tensorwise":
            params = config.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError(f"Invalid quantization params in {key}")
            if not config.get("convrot", params.get("convrot", False)):
                raise ValueError(f"{key} is INT8 without ConvRot; unsupported by this experimental path")
        formats[fmt] = formats.get(fmt, 0) + 1

    if not formats:
        raise ValueError("No per-module .comfy_quant metadata; not a Comfy ConvRot checkpoint")

    for key, tensor in state.items():
        if not key.endswith(".weight") or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dtype in (torch.int8, torch.uint8) and key[:-len("weight")] + "comfy_quant" not in state:
            raise ValueError(f"Integer weight has no .comfy_quant metadata: {key}")

    fused = [key for key in state if FUSED.fullmatch(key)]
    expected_layers = {int(FFN.fullmatch(k).group(1)) for k in state if FFN.fullmatch(k)}
    if not expected_layers:
        raise ValueError("No MOSS transformer blocks found")

    for key in fused:
        stem = FUSED.fullmatch(key).group("stem")
        weight = state[key]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != dim * 3:
            raise ValueError(f"Invalid fused MOSS QKV tensor {key}: {getattr(weight, 'shape', None)}")
        prefix = stem + ".qkv."
        parts = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        config_tensor = parts.get("comfy_quant")
        if config_tensor is not None:
            fmt = _config(config_tensor, prefix + "comfy_quant")["format"]
            if fmt == "int8_tensorwise":
                if weight.dtype != torch.int8 or weight.shape[1] != dim:
                    raise ValueError(f"Invalid INT8 ConvRot QKV storage shape/dtype at {key}")
                if "weight_scale" not in parts:
                    raise ValueError(f"Missing INT8 scale at {stem}")
            elif fmt == "asym_w4a8_int8":
                if weight.dtype != torch.int8 or weight.shape[1] * 2 != dim:
                    raise ValueError(f"Invalid packed W4A8 ConvRot QKV storage shape/dtype at {key}")
                if "weight_s_rel" not in parts or "weight_s_channel" not in parts:
                    raise ValueError(f"Missing W4A8 group/channel scales at {stem}")
        elif weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"QKV {key} is neither floating point nor Comfy-quantized")

        allowed = {"weight"} | SPLIT_ROWS | SHARE
        unknown = sorted(set(parts) - allowed)
        if unknown:
            raise ValueError(f"Unsupported fused QKV companion tensors for {stem}: {unknown}")
        for letter in "qkv":
            if any(k.startswith(f"{stem}.{letter}.") for k in state):
                raise ValueError(f"Mixed fused and split QKV keys at {stem}.{letter}")

        for field, tensor in parts.items():
            if field == "weight" or field in SPLIT_ROWS:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected tensor for {prefix}{field}")
                if tensor.ndim > 0 and tensor.shape[0] == dim * 3:
                    chunks = tensor.split(dim, dim=0)
                elif field == "weight_scale" and tensor.numel() == 1:
                    chunks = (tensor, tensor, tensor)
                else:
                    raise ValueError(f"Unexpected {field} shape at {stem}: {tuple(tensor.shape)}")
            elif field in SHARE:
                chunks = (tensor, tensor, tensor)
            else:
                raise AssertionError(field)
            del output[prefix + field]
            for letter, chunk in zip("qkv", chunks):
                output[f"{stem}.{letter}.{field}"] = chunk

    for layer in expected_layers:
        stem = f"layers.{layer}.proj"
        missing = [f"{stem}.{c}.weight" for c in "qkv" if f"{stem}.{c}.weight" not in output]
        if missing:
            raise ValueError(f"MOSS ConvRot missing split projections: {missing}")
    return output, len(fused), formats
