# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the mock HunyuanImage3 RL rollout boundary."""

from __future__ import annotations

import pytest
import torch

from tests.rl_rollout.mock_hunyuan_image3 import (
    MockARRolloutEngine,
    MockDiffusionRolloutEngine,
    MockHunyuanImage3Trainer,
    MockWeightTransferError,
)
from vllm_omni.rl_rollout import (
    ARRolloutSample,
    DiffusionRolloutSample,
    RLPublicationError,
    RLWeightTransferAdapter,
    WeightPublication,
    WeightUpdateChunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeVLLMWeightTransferClient:
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
async def test_production_adapter_reuses_native_wte_lifecycle() -> None:
    client = _FakeVLLMWeightTransferClient()
    adapter = RLWeightTransferAdapter(client)
    payload_a = {"model.layers.0.weight": "ipc-payload-a"}
    payload_b = {"model.layers.1.weight": "ipc-payload-b"}
    publication = WeightPublication(
        version=1,
        parameter_names=("model.layers.0.weight", "model.layers.1.weight"),
        chunks=(
            WeightUpdateChunk(version=1, parameter_names=("model.layers.0.weight",), payload=payload_a),
            WeightUpdateChunk(version=1, parameter_names=("model.layers.1.weight",), payload=payload_b),
        ),
    )

    await adapter.initialize({"backend": "ipc", "packed": False})
    await adapter.publish(publication)

    assert adapter.current_version == 1
    assert client.calls == [
        ("init", {"backend": "ipc", "packed": False}),
        ("start", None),
        ("update", payload_a),
        ("update", payload_b),
        ("finish", "1"),
    ]


@pytest.mark.asyncio
async def test_production_adapter_rejects_incomplete_manifest_before_transport_calls() -> None:
    client = _FakeVLLMWeightTransferClient()
    adapter = RLWeightTransferAdapter(client)
    await adapter.initialize({"backend": "ipc"})
    publication = WeightPublication(
        version=1,
        parameter_names=("model.layers.0.weight", "model.layers.1.weight"),
        chunks=(
            WeightUpdateChunk(
                version=1,
                parameter_names=("model.layers.0.weight",),
                payload={"model.layers.0.weight": "ipc-payload-a"},
            ),
        ),
    )

    with pytest.raises(RLPublicationError, match="manifest mismatch"):
        await adapter.publish(publication)
    assert [name for name, _ in client.calls] == ["init"]


@pytest.mark.asyncio
async def test_ar_rollout_exposes_token_logprob_and_response_boundaries() -> None:
    engine = MockARRolloutEngine()
    trainer = MockHunyuanImage3Trainer(engine)

    result = await trainer.run_round("a red kite", seed=7)
    sample = result.rollout
    assert isinstance(sample, ARRolloutSample)

    assert result.mode == "ar"
    assert sample.response_start == len(sample.prompt_token_ids)
    assert sample.response_end == len(sample.token_ids)
    assert len(sample.response_token_ids) == len(sample.response_token_logprobs)
    assert result.published_parameter_names == engine.trainable_parameter_names
    assert engine.current_version == 1
    assert result.published_version == 1
    assert engine.weight_transfer.published_versions == [0, 1]


@pytest.mark.asyncio
async def test_diffusion_rollout_exposes_image_trajectory_and_keeps_vae_frozen() -> None:
    engine = MockDiffusionRolloutEngine(num_inference_steps=4)
    vae_before = engine.parameter_snapshot()["vae.decoder.weight"]
    trainer = MockHunyuanImage3Trainer(engine)

    result = await trainer.run_round("a red kite", seed=7)
    sample = result.rollout
    assert isinstance(sample, DiffusionRolloutSample)

    assert result.mode == "diffusion"
    assert sample.image.shape == (1, 3, 2, 2)
    assert len(sample.latent_trajectory) == 4
    assert len(sample.sigmas) == len(sample.timesteps) == 4
    assert sample.sde_step_indices == (0, 2)
    assert sample.conditioning["prompt"] == "a red kite"
    assert engine.current_version == 1
    assert torch.equal(engine.parameter_snapshot()["vae.decoder.weight"], vae_before)
    assert set(result.published_parameter_names) == set(engine.trainable_parameter_names)


@pytest.mark.asyncio
async def test_weight_update_is_atomic_until_finish() -> None:
    engine = MockARRolloutEngine()
    await engine.init_weight_transfer_engine({"backend": "mock", "packed": False})
    before = await engine.generate("a red kite", seed=3)
    exported = engine.export_trainable_weights()
    updated = {name: value + 1.0 for name, value in exported.items()}

    await engine.start_weight_update(version=1, parameter_names=engine.trainable_parameter_names)
    await engine.update_weights(WeightUpdateChunk(version=1, parameter_names=tuple(updated), payload=updated))
    during = await engine.generate("a red kite", seed=3)
    assert during.weight_version == 0
    assert engine.current_version == 0

    await engine.finish_weight_update()
    after = await engine.generate("a red kite", seed=3)
    assert after.weight_version == 1
    assert engine.current_version == 1
    assert before.response_token_logprobs != after.response_token_logprobs


@pytest.mark.asyncio
async def test_weight_update_state_machine_rejects_invalid_order_and_incomplete_manifest() -> None:
    engine = MockARRolloutEngine()
    update = WeightUpdateChunk(
        version=1,
        parameter_names=engine.trainable_parameter_names,
        payload=engine.export_trainable_weights(),
    )

    with pytest.raises(MockWeightTransferError, match="init_weight_transfer_engine"):
        await engine.start_weight_update(version=1, parameter_names=engine.trainable_parameter_names)

    await engine.init_weight_transfer_engine({"backend": "mock"})
    with pytest.raises(MockWeightTransferError, match="active weight update"):
        await engine.update_weights(update)

    await engine.start_weight_update(version=1, parameter_names=engine.trainable_parameter_names)
    first_name = engine.trainable_parameter_names[0]
    await engine.update_weights(
        WeightUpdateChunk(
            version=1,
            parameter_names=(first_name,),
            payload={first_name: update.payload[first_name]},
        )
    )
    with pytest.raises(MockWeightTransferError, match="incomplete"):
        await engine.finish_weight_update()

    with pytest.raises(MockWeightTransferError, match="already"):
        await engine.start_weight_update(version=1, parameter_names=engine.trainable_parameter_names)


@pytest.mark.asyncio
async def test_duplicate_publication_version_is_rejected() -> None:
    engine = MockDiffusionRolloutEngine()
    trainer = MockHunyuanImage3Trainer(engine)
    await trainer.run_round("a red kite")

    with pytest.raises(MockWeightTransferError, match="must advance"):
        await engine.start_weight_update(version=1, parameter_names=engine.trainable_parameter_names)


@pytest.mark.asyncio
async def test_mock_session_rejects_wrong_shape_payload() -> None:
    engine = MockARRolloutEngine()
    await engine.init_weight_transfer_engine({"backend": "mock"})
    name = engine.trainable_parameter_names[0]
    await engine.start_weight_update(version=1, parameter_names=(name,))

    with pytest.raises(MockWeightTransferError, match="shape"):
        await engine.update_weights(
            WeightUpdateChunk(version=1, parameter_names=(name,), payload={name: torch.zeros(3)})
        )
