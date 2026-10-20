"""CPU-only regression tests for fused-QKV LoRA mapping.

Run: python -m pytest -q tests/test_moss_lora.py
Requires torch and pytest, not a running ComfyUI instance.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import torch

module_path = Path(__file__).resolve().parents[1] / "moss_lora.py"
spec = importlib.util.spec_from_file_location("rebel_moss_lora_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
split = module.split_fused_qkv_loras


def model(blocks=1, dim=6):
    state = {
        f"diffusion_model.layers.{n}.proj.{letter}.weight": torch.zeros(dim, dim)
        for n in range(blocks) for letter in "qkv"
    }
    return SimpleNamespace(model=SimpleNamespace(state_dict=lambda: state))


def test_fused_math_and_metadata():
    a = torch.randn(2, 6)
    b = torch.randn(18, 2)
    prefix = "diffusion_model.layers.0.proj.qkv"
    original = {
        f"{prefix}.lora_A.weight": a,
        f"{prefix}.lora_B.weight": b,
        f"{prefix}.alpha": torch.tensor(2.0),
        f"{prefix}.dora_scale": torch.arange(18),
        "other.weight": torch.ones(1),
    }
    converted, count = split(original, model())
    assert count == 1
    assert "other.weight" in converted
    assert f"{prefix}.lora_A.weight" in original  # input untouched
    delta = torch.cat([
        converted[f"diffusion_model.layers.0.proj.{letter}.lora_B.weight"]
        @ converted[f"diffusion_model.layers.0.proj.{letter}.lora_A.weight"]
        for letter in "qkv"
    ], dim=0)
    torch.testing.assert_close(delta, b @ a)
    for i, letter in enumerate("qkv"):
        stem = f"diffusion_model.layers.0.proj.{letter}"
        assert converted[f"{stem}.lora_A.weight"] is a
        assert converted[f"{stem}.alpha"].item() == 2.0
        torch.testing.assert_close(
            converted[f"{stem}.dora_scale"], torch.arange(i * 6, (i + 1) * 6)
        )


def test_42_layer_moss_lora():
    lora = {}
    for n in range(42):
        stem = f"diffusion_model.layers.{n}.proj.qkv"
        lora[f"{stem}.lora_A.weight"] = torch.randn(2, 6)
        lora[f"{stem}.lora_B.weight"] = torch.randn(18, 2)
    converted, count = split(lora, model(blocks=42))
    assert count == 42
    assert len(converted) == 42 * 3 * 2
    assert not any(".qkv." in key for key in converted)


def test_no_fused_noop():
    lora = {"other": torch.ones(1)}
    converted, count = split(lora, model())
    assert converted is lora and count == 0


def test_missing_b_fails():
    with pytest.raises(ValueError, match="missing matching B"):
        split({"diffusion_model.layers.0.proj.qkv.lora_A.weight": torch.ones(2, 6)}, model())


def test_bad_shape_fails():
    stem = "diffusion_model.layers.0.proj.qkv"
    with pytest.raises(ValueError, match="shape mismatch"):
        split({
            f"{stem}.lora_A.weight": torch.ones(2, 6),
            f"{stem}.lora_B.weight": torch.ones(15, 2),
        }, model())


def test_down_up_format():
    stem = "layers.0.proj.qkv"
    converted, count = split({
        f"{stem}.lora_down.weight": torch.ones(2, 6),
        f"{stem}.lora_up.weight": torch.ones(18, 2),
    }, model())
    assert count == 1
    assert "diffusion_model.layers.0.proj.q.lora_up.weight" in converted


def test_conflicting_adapters_fail():
    stem = "diffusion_model.layers.0.proj.qkv"
    with pytest.raises(ValueError, match="Conflicting"):
        split({
            f"{stem}.lora_A.weight": torch.ones(2, 6),
            f"{stem}.lora_B.weight": torch.ones(18, 2),
            "diffusion_model.layers.0.proj.q.lora_A.weight": torch.ones(2, 6),
        }, model())
