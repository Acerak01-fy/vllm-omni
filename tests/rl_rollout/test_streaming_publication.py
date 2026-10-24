# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming WTE failure and manifest checks without CUDA."""

import pytest

from tests.rl_rollout.test_qwen_image_adapter import _FakeClient
from vllm_omni.rl_rollout import RLPublicationError, RLWeightTransferAdapter, WeightUpdateChunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "version", "transport", "finish"])
async def test_streaming_failure_cannot_publish_or_retry(failure):
    class Client(_FakeClient):
        async def update_weights(self, update_info):
            if failure == "transport":
                raise RuntimeError("receiver failed")
            await super().update_weights(update_info)

        async def finish_weight_update(self, weight_version=None):
            if failure == "finish":
                raise RuntimeError("finish failed")
            await super().finish_weight_update(weight_version)

    client = Client()
    adapter = RLWeightTransferAdapter(client)
    await adapter.initialize({"packed": True})
    await adapter.start_publication(version=1, parameter_names=("a", "b"))
    with pytest.raises((RLPublicationError, RuntimeError)):
        await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=("a",), payload={}))
        if failure == "duplicate":
            await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=("a",), payload={}))
        elif failure == "unknown":
            await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=("c",), payload={}))
        elif failure == "version":
            await adapter.update_chunk(WeightUpdateChunk(version=2, parameter_names=("b",), payload={}))
        elif failure == "finish":
            await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=("b",), payload={}))
        await adapter.finish_publication()
    assert adapter.current_version == 0
    assert "finish" not in [name for name, _ in client.calls]
    with pytest.raises(RLPublicationError, match="failed"):
        await adapter.start_publication(version=1, parameter_names=("a", "b"))


@pytest.mark.asyncio
async def test_streaming_requires_start_and_does_not_overlap():
    adapter = RLWeightTransferAdapter(_FakeClient())
    await adapter.initialize({"packed": True})
    with pytest.raises(RLPublicationError, match="start_publication"):
        await adapter.finish_publication()
    await adapter.start_publication(version=1, parameter_names=("a",))
    with pytest.raises(RLPublicationError, match="already active"):
        await adapter.start_publication(version=1, parameter_names=("a",))
