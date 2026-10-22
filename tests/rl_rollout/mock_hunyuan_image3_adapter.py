# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mock HunyuanImage3 AR/DiT adapter for rollout contract tests.

The real HunyuanImage3 trainer is outside vLLM-Omni.  This module therefore
only models the rollout-side boundary: stage-specific parameter selection,
sample extraction, a deterministic reward placeholder, and publication of a
complete trainable snapshot through :class:`RLWeightTransferAdapter`.  It is
deliberately CPU-only and does not stand in for UniRL's optimizer or loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from tests.rl_rollout.mock_hunyuan_image3 import (
    MockARRolloutEngine,
    MockDiffusionRolloutEngine,
    MockWeightTransferError,
)
from vllm_omni.rl_rollout import (
    ARRolloutSample,
    DiffusionRolloutSample,
    RLUpdateResult,
    RLWeightTransferAdapter,
    WeightPublication,
    WeightUpdateChunk,
)

HunyuanImage3Mode = Literal["ar", "diffusion"]
ParameterItems = Mapping[str, Any]


class HunyuanImage3ParameterPolicy:
    """Select the shared HunyuanImage3 backbone for one training stage.

    HunyuanImage3 exposes one shared backbone to its AR and diffusion stages.
    The stages nevertheless publish independently: a trainer can run an AR
    round or a DiT round without updating VAE/vision auxiliaries.  The policy
    mirrors that rollout-side manifest and makes the stage choice explicit.
    """

    _PREFIXES: dict[HunyuanImage3Mode, tuple[str, ...]] = {
        "ar": ("model.", "lm_head."),
        "diffusion": ("model.", "patch_embed.", "time_embed.", "final_layer."),
    }
    _FROZEN_PREFIXES = ("vae.", "vision_model.")

    def __init__(self, mode: HunyuanImage3Mode) -> None:
        self.mode = mode

    @property
    def trainable_prefixes(self) -> tuple[str, ...]:
        return self._PREFIXES[self.mode]

    def select(self, parameters: ParameterItems) -> tuple[tuple[str, Any], ...]:
        names = tuple(parameters)
        if len(names) != len(set(names)):
            raise ValueError("HunyuanImage3 parameter names must be unique")
        selected = tuple((name, parameters[name]) for name in names if name.startswith(self.trainable_prefixes))
        if not selected:
            raise ValueError(f"no HunyuanImage3 {self.mode} parameters matched the stage manifest")
        frozen = tuple(name for name, _ in selected if name.startswith(self._FROZEN_PREFIXES))
        if frozen:
            raise ValueError(f"HunyuanImage3 stage manifest includes frozen parameters: {frozen}")
        return selected


class _MockNativeWTEClient:
    """Bridge the mock engine's metadata-rich calls to the native WTE shape.

    The production adapter intentionally sends only backend payloads to
    ``start/update/finish``.  The in-memory engine used by these tests needs a
    manifest to validate tensor names, so the bridge derives it from the first
    mapping payload.  A real vLLM client receives that metadata in its native
    backend payload and does not use this class.
    """

    def __init__(self, engine: MockARRolloutEngine | MockDiffusionRolloutEngine) -> None:
        self.engine = engine
        self._started = False
        self._pending_updates: list[Mapping[str, Any]] = []

    async def init_weight_transfer_engine(self, init_info: Mapping[str, Any]) -> None:
        await self.engine.init_weight_transfer_engine(init_info)

    async def start_weight_update(self) -> None:
        if self._started:
            raise MockWeightTransferError("a mock native update is already in progress")
        self._started = True
        self._pending_updates = []

    async def update_weights(self, update_info: Any) -> None:
        if not self._started:
            raise MockWeightTransferError("mock native update requires start_weight_update")
        if not isinstance(update_info, Mapping) or not update_info:
            raise MockWeightTransferError("mock native payload must be a non-empty parameter mapping")
        self._pending_updates.append(dict(update_info))

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        del weight_version
        if not self._started or not self._pending_updates:
            raise MockWeightTransferError("mock native finish requires an update payload")
        version = self.engine.current_version + 1
        names = tuple(name for update in self._pending_updates for name in update)
        if len(names) != len(set(names)):
            raise MockWeightTransferError("mock native update contains duplicate parameter names")
        await self.engine.start_weight_update(version=version, parameter_names=names)
        for update in self._pending_updates:
            await self.engine.update_weights(
                WeightUpdateChunk(version=version, parameter_names=tuple(update), payload=dict(update))
            )
        await self.engine.finish_weight_update()
        self._started = False
        self._pending_updates = []


class MockHunyuanImage3RLAdapter:
    """CPU mock of one independent HunyuanImage3 AR or DiT RL recipe."""

    def __init__(
        self,
        engine: MockARRolloutEngine | MockDiffusionRolloutEngine,
        *,
        mode: HunyuanImage3Mode,
        learning_rate: float = 0.01,
        chunk_size: int = 64,
    ) -> None:
        if engine.mode != mode:
            raise ValueError(f"engine mode {engine.mode!r} does not match adapter mode {mode!r}")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.engine = engine
        self.mode = mode
        self.policy = HunyuanImage3ParameterPolicy(mode)
        self.learning_rate = learning_rate
        self._transfer = RLWeightTransferAdapter(
            _MockNativeWTEClient(engine),
            current_version=engine.current_version,
        )
        self._chunk_size = chunk_size

    @property
    def current_version(self) -> int:
        return self._transfer.current_version

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.policy.select(self.engine.export_trainable_weights()))

    def build_publication(self, parameters: ParameterItems, *, version: int | None = None) -> WeightPublication:
        selected = self.policy.select(parameters)
        publication_version = self.current_version + 1 if version is None else version
        names = tuple(name for name, _ in selected)
        chunks = tuple(
            WeightUpdateChunk(
                version=publication_version,
                parameter_names=tuple(name for name, _ in selected[offset : offset + self._chunk_size]),
                payload=dict(selected[offset : offset + self._chunk_size]),
            )
            for offset in range(0, len(selected), self._chunk_size)
        )
        return WeightPublication(
            version=publication_version,
            parameter_names=names,
            chunks=chunks,
        )

    async def initialize(self) -> None:
        await self._transfer.initialize({"backend": "mock", "stage": self.mode})

    async def publish(self, parameters: ParameterItems) -> None:
        await self._transfer.publish(self.build_publication(parameters))

    async def run_round(self, prompt: str, *, seed: int = 0) -> RLUpdateResult:
        """Generate a sample, apply a placeholder update, and publish it."""
        if not self._transfer.initialized:
            await self.initialize()
        rollout = await self.engine.generate(prompt, seed=seed)
        reward = self._reward(rollout)
        if isinstance(rollout, ARRolloutSample):
            loss = -reward * sum(rollout.response_token_logprobs) / len(rollout.response_token_logprobs)
        else:
            loss = -reward * rollout.trajectory_logprob

        updated = {
            name: value + self.learning_rate * reward
            for name, value in self.policy.select(self.engine.export_trainable_weights())
        }
        await self.publish(updated)
        return RLUpdateResult(
            mode=self.mode,
            rollout=rollout,
            reward=reward,
            advantage=reward,
            loss=loss,
            published_version=self.current_version,
            published_parameter_names=tuple(updated),
        )

    @staticmethod
    def _reward(rollout: ARRolloutSample | DiffusionRolloutSample) -> float:
        if isinstance(rollout, ARRolloutSample):
            return 1.0 + float(sum(rollout.response_token_ids) % 1000) / 1000.0
        return float(rollout.image.mean().item()) + 0.01 * len(rollout.sde_step_indices)


__all__ = [
    "HunyuanImage3ParameterPolicy",
    "MockHunyuanImage3RLAdapter",
    "_MockNativeWTEClient",
]
