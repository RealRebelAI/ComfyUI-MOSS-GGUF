"""Integrate MOSS's *native* model loader with city96 GGUF on-demand ops.

This module intentionally modifies no official plugin files. The dtype
adaptation is bound only to the loaded GGUF model instance.
"""
import ast
import inspect
import logging
from pathlib import Path
import sys
import textwrap
import types

import torch

logger = logging.getLogger("MOSS-Rebel-Integration")


def _loaded_plugin_file(plugin_folder, filename, expected):
    """Locate modules by the actual on-disk plugin folder; no suffix guessing."""
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        p = Path(path)
        if p.name.casefold() != filename.casefold():
            continue
        dirname = p.parent.name.casefold()
        if plugin_folder == "gguf":
            is_match = "comfyui-gguf" in dirname
        else:
            is_match = dirname in {"moss-image", "moss_image"}
        if is_match and hasattr(module, expected):
            return module
    raise RuntimeError(
        f"Required plugin module missing: {plugin_folder}/{filename} ({expected}). "
        "Keep the official OpenMOSS 'moss-image' and city96 'ComfyUI-GGUF' "
        "plugins installed and enabled, then restart ComfyUI."
    )


def _native_moss_loaders():
    return _loaded_plugin_file("moss", "loaders.py", "load_model_state")


def _gguf_modules():
    return (
        _loaded_plugin_file("gguf", "loader.py", "gguf_sd_loader"),
        _loaded_plugin_file("gguf", "nodes.py", "GGUFModelPatcher"),
        _loaded_plugin_file("gguf", "ops.py", "GGMLOps"),
    )


def _patched_forward(original, compute_dtype):
    """Adapt official MOSS forward methods without changing plugin globals.

    GGUF RMSNorm can return fp32 q/k, while projection v can be bf16.
    torch SDPA rejects mixed q/k/v. Apply the dtype correction at the
    private _attention call site for every transformer/text block, AND keep
    the existing GGUF integer-weight dtype correction.
    """
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)

    class Transform(ast.NodeTransformer):
        weight_dtype_replacements = 0
        attention_replacements = 0

        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if (node.attr == "dtype" and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "weight"):
                self.weight_dtype_replacements += 1
                return ast.copy_location(
                    ast.Call(func=ast.Name(id="_moss_effective_dtype", ctx=ast.Load()),
                             args=[node.value.value], keywords=[]), node)
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_attention":
                self.attention_replacements += 1
                node.func = ast.copy_location(
                    ast.Name(id="_moss_safe_attention", ctx=ast.Load()), node.func
                )
            return node

    visitor = Transform()
    tree = visitor.visit(tree)
    if not (visitor.weight_dtype_replacements or visitor.attention_replacements):
        raise RuntimeError(
            f"No GGUF dtype or attention sites found in MOSS {original.__qualname__}; "
            "the official code may have changed. Refusing to patch blindly."
        )
    ast.fix_missing_locations(tree)

    def effective_dtype(layer):
        weight = layer.weight
        dtype = weight.dtype
        # A quantized GGMLTensor stores uint8 bytes, not activation values.
        return compute_dtype if not dtype.is_floating_point else dtype

    native_attention = original.__globals__.get("_attention")
    if visitor.attention_replacements and native_attention is None:
        raise RuntimeError(f"Cannot find official MOSS _attention in {original.__qualname__}")

    def safe_attention(q, k, v, rope, heads):
        # The traceback showed q/k fp32 and v bf16 on the RTX 3070.
        # Standard SDPA requires ALL three to use the same dtype.
        # Convert the RoPE phase too: fp32 sin/cos would promote q/k
        # right back to fp32 inside MOSS's _rotate operation.
        dtype = compute_dtype
        if not dtype.is_floating_point:
            raise TypeError(f"MOSS GGUF invalid attention compute dtype: {dtype}")
        if not (q.dtype == dtype and k.dtype == dtype and v.dtype == dtype and rope.dtype == dtype):
            q, k, v, rope = (t.to(dtype=dtype) for t in (q, k, v, rope))
        return native_attention(q, k, v, rope, heads)

    g = dict(original.__globals__)
    g["_moss_effective_dtype"] = effective_dtype
    g["_moss_safe_attention"] = safe_attention
    loc = {}
    compiled_name = f"<MOSS-GGUF safe attention {original.__qualname__}>"
    exec(compile(tree, compiled_name, "exec"), g, loc)
    fn = loc[original.__name__]
    logger.info(
        "MOSS-GGUF: %s: %d GGUF weight casts, %d attention dtype guards",
        original.__qualname__, visitor.weight_dtype_replacements,
        visitor.attention_replacements,
    )
    return fn


def _patch_model_instance(patcher, compute_dtype):
    base = getattr(patcher, "model", None)
    transformer = getattr(base, "diffusion_model", None)
    if transformer is None:
        raise RuntimeError("MOSS patcher contains no 'model.diffusion_model' transformer")
    text_projection = getattr(transformer, "text_embedder", None)
    blocks = getattr(transformer, "layers", None)
    text_blocks = getattr(text_projection, "blocks", None)
    if text_projection is None or blocks is None or not blocks or text_blocks is None or not text_blocks:
        raise RuntimeError("MOSS transformer/text block structure changed; cannot install attention adapter")
    # Per-model instance only. Do not edit the installed moss-image model.py.
    transformer.forward = types.MethodType(
        _patched_forward(type(transformer).forward, compute_dtype), transformer
    )
    text_projection.forward = types.MethodType(
        _patched_forward(type(text_projection).forward, compute_dtype), text_projection
    )
    transformer_block_fn = _patched_forward(type(blocks[0]).forward, compute_dtype)
    text_block_fn = _patched_forward(type(text_blocks[0]).forward, compute_dtype)
    for block in blocks:
        block.forward = types.MethodType(transformer_block_fn, block)
    for block in text_blocks:
        block.forward = types.MethodType(text_block_fn, block)
    logger.info(
        "MOSS-GGUF safe attention installed for %d transformer blocks and %d text blocks (compute=%s)",
        len(blocks), len(text_blocks), compute_dtype,
    )


def load_gguf(path, precision):
    loader, nodes, operations = _gguf_modules()
    moss = _native_moss_loaders()
    # city96 explicitly checks architecture allowlist. Register this one only.
    loader.IMG_ARCH_LIST.add("moss_image")
    state, extra = loader.gguf_sd_loader(path, handle_prefix=None)
    if extra.get("arch_str") != "moss_image":
        raise ValueError(f"Expected MOSS GGUF (moss_image), found {extra.get('arch_str')!r}")
    required = ("video_embedder.bias", "layers.0.proj.q.weight",
                "layers.0.proj.ffn_up_gate.weight", "final_linear_video.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"MOSS GGUF is missing essential transformer tensors: {missing}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    # Instance-isolated GGUF operations: never change city96 global Linear dtype.
    class MossGGUFLinear(operations.GGMLOps.Linear):
        dequant_dtype = dtype

    class MossGGUFOps(operations.GGMLOps):
        Linear = MossGGUFLinear

    ops = MossGGUFOps()
    # MOSS's load_model_state is REQUIRED; generic Comfy model detection
    # cannot recognize MOSS's architecture.
    tensor_count = len(state)
    patcher = moss.load_model_state(
        state, dtype, disable_dynamic=True,
        metadata=extra.get("metadata"), model_options={"custom_operations": ops}
    )
    # Crucial: standard ModelPatcher doesn't manage quantized GGMLTensor
    # memory / patching correctly. Same patcher as city96's native loader.
    patcher = nodes.GGUFModelPatcher.clone(patcher)
    patcher.patch_on_device = False
    _patch_model_instance(patcher, dtype)
    logger.info("MOSS-GGUF loaded %d tensors from %s", tensor_count, Path(path).name)
    return patcher


def load_encoder(path, device):
    """Actual official Qwen3.5 loader; supports Comfy-native quant metadata."""
    moss = _native_moss_loaders()
    try:
        clip = moss.load_clip(path, device)
    except Exception as e:
        raise RuntimeError(
            f"Official MOSS Qwen3.5 loader rejected '{Path(path).name}': {e}. "
            "If this is INT8 ConvRot, it must be in a ComfyUI-supported "
            "quantization format with the required metadata."
        ) from e
    if not hasattr(clip, "patcher") or not hasattr(clip, "tokenize"):
        raise TypeError("Official MOSS text loader did not return a functional CLIP object")
    return clip


def load_vae(path):
    import comfy.sd
    import comfy.utils
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    vae = comfy.sd.VAE(sd=sd, dtype=torch.float32)
    vae.throw_exception_if_invalid()
    return vae
