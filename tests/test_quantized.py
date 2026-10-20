"""CPU-only regression tests for Comfy-native MOSS ConvRot QKV splitting.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_quantized.py' -v
"""
import importlib.util
import json
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "rebel_moss_quantized", Path(__file__).resolve().parents[1] / "quantized.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
normalize_quantized_state = module.normalize_quantized_state

D = 256


def cfg(fmt, **extra):
    return torch.tensor(list(json.dumps({"format": fmt, **extra}).encode()), dtype=torch.uint8)


def state(fmt="int8_tensorwise", fused=True):
    data = {
        "video_embedder.bias": torch.zeros(D),
        "final_linear_video.weight": torch.zeros(16, D),
        "layers.0.proj.ffn_up_gate.weight": torch.zeros(2 * D, D),
    }
    stem = "layers.0.proj"
    width = D if fmt == "int8_tensorwise" else D // 2
    qkv = torch.arange(3 * D * width, dtype=torch.int32).reshape(3 * D, width).remainder(127).to(torch.int8)
    if fused:
        data[stem + ".qkv.weight"] = qkv
        data[stem + ".qkv.comfy_quant"] = cfg(fmt, convrot=True, convrot_groupsize=256)
        if fmt == "int8_tensorwise":
            data[stem + ".qkv.weight_scale"] = torch.arange(3 * D).reshape(3 * D, 1).float()
        else:
            data[stem + ".qkv.weight_s_rel"] = torch.ones(3 * D, D // 16)
            data[stem + ".qkv.weight_s_channel"] = torch.arange(3 * D).float()
            data[stem + ".qkv.weight_codebook"] = torch.arange(16).float()
    else:
        for letter in "qkv":
            data[stem + f".{letter}.weight"] = qkv[:D]
            data[stem + f".{letter}.comfy_quant"] = cfg(fmt, convrot=True)
    return data


class QuantTests(unittest.TestCase):
    def test_int8(self):
        src = state()
        result, count, fmts = normalize_quantized_state(src)
        self.assertEqual(count, 1)
        self.assertEqual(fmts, {"int8_tensorwise": 1})
        self.assertNotIn("layers.0.proj.qkv.weight", result)
        self.assertEqual(result["layers.0.proj.k.weight"].shape, (D, D))
        self.assertEqual(result["layers.0.proj.v.weight_scale"][0, 0].item(), 2 * D)
        self.assertIn("layers.0.proj.qkv.weight", src)
        self.assertEqual(
            result["layers.0.proj.q.weight"].untyped_storage().data_ptr(),
            src["layers.0.proj.qkv.weight"].untyped_storage().data_ptr(),
        )

    def test_w4a8(self):
        src = state("asym_w4a8_int8")
        result, count, fmts = normalize_quantized_state(src)
        self.assertEqual(count, 1)
        self.assertEqual(fmts, {"asym_w4a8_int8": 1})
        self.assertEqual(result["layers.0.proj.q.weight"].shape, (D, D // 2))
        self.assertEqual(result["layers.0.proj.k.weight_s_rel"].shape, (D, D // 16))
        self.assertEqual(result["layers.0.proj.v.weight_s_channel"][0].item(), 2 * D)
        self.assertIs(result["layers.0.proj.q.weight_codebook"], result["layers.0.proj.k.weight_codebook"])

    def test_already_split(self):
        result, count, fmts = normalize_quantized_state(state(fused=False))
        self.assertEqual(count, 0)
        self.assertEqual(fmts["int8_tensorwise"], 3)

    def test_missing_metadata(self):
        src = state()
        del src["layers.0.proj.qkv.comfy_quant"]
        with self.assertRaisesRegex(ValueError, "No per-module .comfy_quant metadata"):
            normalize_quantized_state(src)

    def test_unknown_format(self):
        src = state()
        src["layers.0.proj.qkv.comfy_quant"] = cfg("not_supported")
        with self.assertRaisesRegex(ValueError, "Unsupported quantized format"):
            normalize_quantized_state(src)

    def test_unknown_companion(self):
        src = state()
        src["layers.0.proj.qkv.weight_correction"] = torch.ones(3 * D)
        with self.assertRaisesRegex(ValueError, "Unsupported fused QKV companion"):
            normalize_quantized_state(src)

    def test_mixed_split_fused(self):
        src = state()
        src["layers.0.proj.q.weight"] = torch.zeros(D, D)
        with self.assertRaisesRegex(ValueError, "Mixed fused and split"):
            normalize_quantized_state(src)

    def test_int8_scalar_scale(self):
        src = state()
        src["layers.0.proj.qkv.weight_scale"] = torch.tensor(0.02)
        result, count, _ = normalize_quantized_state(src)
        self.assertEqual(count, 1)
        for letter in "qkv":
            self.assertIs(result[f"layers.0.proj.{letter}.weight_scale"], src["layers.0.proj.qkv.weight_scale"])

    def test_missing_w4a8_scales(self):
        src = state("asym_w4a8_int8")
        del src["layers.0.proj.qkv.weight_s_rel"]
        with self.assertRaisesRegex(ValueError, "Missing W4A8 group/channel scales"):
            normalize_quantized_state(src)

    def test_non_convrot_int8(self):
        src = state()
        src["layers.0.proj.qkv.comfy_quant"] = cfg("int8_tensorwise", convrot=False)
        with self.assertRaisesRegex(ValueError, "INT8 without ConvRot"):
            normalize_quantized_state(src)

    def test_bare_integer_weight(self):
        src = state()
        src["layers.0.proj.extra.weight"] = torch.ones(16, 16, dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, "no .comfy_quant metadata"):
            normalize_quantized_state(src)


if __name__ == "__main__":
    unittest.main()
