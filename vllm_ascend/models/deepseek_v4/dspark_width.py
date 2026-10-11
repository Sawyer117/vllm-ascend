# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE width of the DSpark draft layers, when it differs from the target's.

The DSpark draft layers of DeepSeek-V4 are built from the target's config, so they get the
target's ``n_routed_experts`` and ``num_experts_per_tok``. A draft may use fewer: the
DeepSeek-V4.1-Flash config carries ``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok``
(128 / 3 against a 384 / 6 target). These helpers read the two fields from the draft config;
a draft config without them (or with the target's values) is built exactly as before.

No vLLM imports, so the logic can be unit-tested without a device.
"""

import copy
from typing import Any


def dspark_draft_layer_config(target_config: Any, draft_config: Any) -> Any | None:
    """The config to build the draft decoder layers from, or ``None`` to keep the target's.

    A deep copy of ``target_config`` with ``n_routed_experts`` / ``num_experts_per_tok`` taken
    from the draft config's ``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok``,
    returned only when at least one of them differs from the target's.
    """
    overrides = {}
    for draft_key, key in (
        ("dspark_n_routed_experts", "n_routed_experts"),
        ("dspark_num_experts_per_tok", "num_experts_per_tok"),
    ):
        value = getattr(draft_config, draft_key, None)
        if value is not None and int(value) != int(getattr(target_config, key)):
            overrides[key] = int(value)
    if not overrides:
        return None
    n_experts = overrides.get("n_routed_experts", int(target_config.n_routed_experts))
    top_k = overrides.get("num_experts_per_tok", int(target_config.num_experts_per_tok))
    if not 1 <= top_k <= n_experts:
        raise ValueError(f"DSpark draft: num_experts_per_tok={top_k} must be in 1..n_routed_experts={n_experts}")
    layer_config = copy.deepcopy(target_config)
    for key, value in overrides.items():
        setattr(layer_config, key, value)
    return layer_config


def dspark_draft_num_experts(draft_config: Any) -> int:
    """Routed experts per draft layer: ``dspark_n_routed_experts`` if set, else ``n_routed_experts``."""
    value = getattr(draft_config, "dspark_n_routed_experts", None)
    return int(value) if value is not None else int(draft_config.n_routed_experts)
