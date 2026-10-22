# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the QwenImage full-parameter rollout adapter."""

from __future__ import annotations

import pytest

from vllm_omni.rl_rollout import QwenImageParameterPolicy, QwenImageRLAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def init_weight_transfer_engine(self, init_info):
        self.calls.append(("init", init_info))

    async def start_weight_update(self) -> None:
        self.calls.append(("start", None))

    async def update_weights(self, update_info) -> None:
        self.calls.append(("update", update_info))

    async def finish_weight_update(self, weight_version=None) -> None:
        self.calls.append(("finish", weight_version))


@pytest.mark.asyncio
async def test_qwen_image_publishes_full_transformer_and_freezes_auxiliary_modules() -> None:
    client = _FakeClient()
    adapter = QwenImageRLAdapter(client, chunk_size=2)
    parameters = {
        "transformer.blocks.0.attn.weight": "attn-0",
        "transformer.blocks.0.mlp.weight": "mlp-0",
        "transformer.blocks.1.attn.weight": "attn-1",
        "vae.decoder.conv_in.weight": "vae",
        "text_encoder.layers.0.weight": "text",
    }

    await adapter.initialize({"backend": "ipc", "packed": False})
    await adapter.publish(parameters)

    assert adapter.current_version == 1
    assert adapter.trainable_parameter_names(parameters) == (
        "transformer.blocks.0.attn.weight",
        "transformer.blocks.0.mlp.weight",
        "transformer.blocks.1.attn.weight",
    )
    assert [name for name, _ in client.calls] == ["init", "start", "update", "update", "finish"]
    assert client.calls[-1] == ("finish", "1")
    assert client.calls[2][1] == {
        "transformer.blocks.0.attn.weight": "attn-0",
        "transformer.blocks.0.mlp.weight": "mlp-0",
    }
    assert client.calls[3][1] == {"transformer.blocks.1.attn.weight": "attn-1"}


def test_qwen_image_policy_can_be_explicitly_extended_for_a_recipe() -> None:
    policy = QwenImageParameterPolicy(
        trainable_prefixes=("transformer.", "vae."),
        frozen_prefixes=("text_encoder.",),
    )
    parameters = {
        "transformer.block.weight": 1,
        "vae.decoder.weight": 2,
        "text_encoder.layer.weight": 3,
    }

    assert policy.select_names(parameters) == ("transformer.block.weight", "vae.decoder.weight")


def test_qwen_image_adapter_rejects_snapshot_without_transformer_parameters() -> None:
    adapter = QwenImageRLAdapter(_FakeClient())

    with pytest.raises(ValueError, match="no QwenImage trainable parameters"):
        adapter.build_publication({"vae.decoder.weight": "frozen"})
