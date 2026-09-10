from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "architectures"


class _FakeField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _FakeTensor:
    def __init__(self, name: str):
        self.name = name


class _FakeReader:
    def __init__(self, mapping, tensor_names=()):
        self._mapping = mapping
        self.tensors = [_FakeTensor(name) for name in tensor_names]

    def get_field(self, key):
        value = self._mapping.get(key)
        return None if value is None else _FakeField(value)


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_name", ["gemma", "phi3"])
def test_architecture_fixture_config_and_tensor_map(fixture_name: str):
    fixture = _load_fixture(fixture_name)
    arch = fixture["architecture"]
    config = core.build_config(_FakeReader(fixture["metadata"]), arch)

    for key, expected in fixture["expected_config"].items():
        assert config[key] == expected
    for gguf_name, expected in fixture["tensor_map"].items():
        assert core._map_tensor_name(gguf_name, arch) == expected


def test_gemma_fixture_restores_shifted_norm_weights():
    shifted = np.array([1.25, 0.5], dtype=np.float16)

    restored = core._restore_architecture_tensor(
        "blk.0.attn_norm.weight", shifted, "gemma"
    )

    np.testing.assert_array_equal(restored, np.array([0.25, -0.5], dtype=np.float16))


def test_phi3_fixture_fuses_split_qkv_weights():
    pending_qkv = {}
    mla_dims = {"num_heads": 4, "qk_nope_head_dim": 4, "v_head_dim": 4}
    pending_kv_b = {}

    assert core._plan_tensor_emit(
        "blk.0.attn_q.weight",
        np.full((4, 2), 1, dtype=np.float16),
        "phi3",
        mla_dims,
        pending_kv_b,
        pending_qkv,
    ) == []
    assert core._plan_tensor_emit(
        "blk.0.attn_k.weight",
        np.full((2, 2), 2, dtype=np.float16),
        "phi3",
        mla_dims,
        pending_kv_b,
        pending_qkv,
    ) == []
    emitted = core._plan_tensor_emit(
        "blk.0.attn_v.weight",
        np.full((2, 2), 3, dtype=np.float16),
        "phi3",
        mla_dims,
        pending_kv_b,
        pending_qkv,
    )

    assert emitted[0][0] == "model.layers.0.self_attn.qkv_proj.weight"
    np.testing.assert_array_equal(
        emitted[0][1][:, 0],
        np.array([1, 1, 1, 1, 2, 2, 3, 3], dtype=np.float16),
    )
    assert pending_qkv == {}


def test_new_adapters_reject_unknown_tensor_names():
    with pytest.raises(ValueError, match="Unsupported gemma tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "gemma")
    with pytest.raises(ValueError, match="Unsupported phi3 tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "phi3")


def test_phi3_longrope_variant_fails_closed():
    reader = _FakeReader({}, tensor_names=["rope_factors_long", "rope_factors_short"])

    error = core.validate_architecture_variant(reader, "phi3")

    assert error is not None
    assert "LongRoPE" in error


def test_python_name_detection_prefers_specific_gemma_variants(monkeypatch):
    monkeypatch.setattr(core, "rust_detect_architecture", lambda arch, name: None)

    assert core.detect_architecture(_FakeReader({"general.name": "Gemma-2-9B"})) == "gemma"
    assert core.detect_architecture(_FakeReader({"general.name": "gemma2-9b"})) == "gemma2"
    assert core.detect_architecture(_FakeReader({"general.name": "gemma3-4b"})) == "gemma3"
    assert core.detect_architecture(_FakeReader({"general.name": "phi2"})) == "phi2"
