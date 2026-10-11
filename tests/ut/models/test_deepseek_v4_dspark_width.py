# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DSpark draft MoE width (vllm_ascend/models/deepseek_v4/dspark_width.py).

Loaded by path: the helpers import nothing from vLLM, so this runs without a device.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v4/dspark_width.py"
_spec = importlib.util.spec_from_file_location("dspark_width", _PATH)
dspark_width = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dspark_width)


def _target():
    return SimpleNamespace(
        n_routed_experts=256,
        num_experts_per_tok=6,
        hidden_size=4096,
        rope_parameters={"original_max_position_embeddings": 65536},
    )


def test_a_draft_without_the_fields_keeps_the_target_config():
    target = _target()
    assert dspark_width.dspark_draft_layer_config(target, SimpleNamespace()) is None
    same = SimpleNamespace(dspark_n_routed_experts=256, dspark_num_experts_per_tok=6)
    assert dspark_width.dspark_draft_layer_config(target, same) is None


def test_the_draft_width_overrides_a_copy():
    target = _target()
    draft = SimpleNamespace(dspark_n_routed_experts=128, dspark_num_experts_per_tok=3)
    cfg = dspark_width.dspark_draft_layer_config(target, draft)
    assert cfg is not target
    assert (cfg.n_routed_experts, cfg.num_experts_per_tok) == (128, 3)
    assert cfg.hidden_size == 4096
    assert cfg.rope_parameters is not target.rope_parameters  # deep copy
    assert (target.n_routed_experts, target.num_experts_per_tok) == (256, 6)  # untouched


def test_one_field_alone():
    cfg = dspark_width.dspark_draft_layer_config(_target(), SimpleNamespace(dspark_n_routed_experts=128))
    assert (cfg.n_routed_experts, cfg.num_experts_per_tok) == (128, 6)
    cfg = dspark_width.dspark_draft_layer_config(_target(), SimpleNamespace(dspark_num_experts_per_tok=3))
    assert (cfg.n_routed_experts, cfg.num_experts_per_tok) == (256, 3)


def test_top_k_larger_than_the_experts_is_refused():
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        dspark_width.dspark_draft_layer_config(
            _target(), SimpleNamespace(dspark_n_routed_experts=4, dspark_num_experts_per_tok=6)
        )


def test_expert_mapping_count():
    assert dspark_width.dspark_draft_num_experts(SimpleNamespace(n_routed_experts=256)) == 256
    assert (
        dspark_width.dspark_draft_num_experts(SimpleNamespace(n_routed_experts=256, dspark_n_routed_experts=128)) == 128
    )
