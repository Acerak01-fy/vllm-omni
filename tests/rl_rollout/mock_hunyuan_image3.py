# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU mock of the HunyuanImage3 AR/diffusion RL rollout boundary.

This module is a contract harness, not a model implementation.  It keeps a
small tensor registry in place of the 80B HunyuanImage3 checkpoint and models
the same direction of data flow as a real trainer:

``generate -> reward/loss -> full trainable snapshot -> init/start/update/finish``

The updater commits only after ``finish_weight_update``.  That makes it useful
for checking publication/version semantics without requiring a real vLLM
worker, CUDA IPC handles, or a training backend.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm_omni.rl_rollout import (
    ARRolloutSample,
    DiffusionRolloutSample,
    RLUpdateResult,
    WeightUpdateChunk,
)


class MockWeightTransferError(RuntimeError):
    """Raised when a mock weight-transfer lifecycle call is invalid."""


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _stable_bytes(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


class _MockParameterStore:
    """Small CPU parameter registry shared by the rollout and mock trainer."""

    def __init__(self, parameters: Mapping[str, torch.Tensor], trainable_names: Sequence[str]) -> None:
        self._parameters = {name: _clone(value) for name, value in parameters.items()}
        self._specs = {
            name: _ParameterSpec(name=name, shape=tuple(value.shape), dtype=value.dtype)
            for name, value in self._parameters.items()
        }
        self._trainable_names = tuple(trainable_names)
        if not self._trainable_names:
            raise ValueError("the mock model must expose at least one trainable parameter")
        unknown = set(self._trainable_names) - set(self._parameters)
        if unknown:
            raise ValueError(f"trainable parameters are not registered: {sorted(unknown)}")

    @property
    def all_names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self._trainable_names

    @property
    def frozen_names(self) -> tuple[str, ...]:
        trainable = set(self._trainable_names)
        return tuple(name for name in self._parameters if name not in trainable)

    def snapshot(self, names: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        selected = self._parameters.keys() if names is None else names
        return {name: _clone(self._parameters[name]) for name in selected}

    def validate_value(self, name: str, value: Any) -> None:
        if name not in self._specs:
            raise MockWeightTransferError(f"unknown parameter {name!r}")
        if not isinstance(value, torch.Tensor):
            raise MockWeightTransferError(f"parameter {name!r} must be a torch.Tensor in the mock transport")
        spec = self._specs[name]
        if tuple(value.shape) != spec.shape:
            raise MockWeightTransferError(f"parameter {name!r} has shape {tuple(value.shape)}, expected {spec.shape}")
        if value.dtype != spec.dtype:
            raise MockWeightTransferError(f"parameter {name!r} has dtype {value.dtype}, expected {spec.dtype}")
        if value.device.type != "cpu":
            raise MockWeightTransferError("mock weight updates must be CPU tensors")

    def commit(self, values: Mapping[str, torch.Tensor]) -> None:
        for name, value in values.items():
            self.validate_value(name, value)
        self._parameters.update({name: _clone(value) for name, value in values.items()})

    def signal(self, names: Sequence[str] | None = None) -> float:
        selected = self._parameters.keys() if names is None else names
        return sum(float(self._parameters[name].sum().item()) for name in selected)


class MockWeightTransferSession:
    """In-memory implementation of Omni's four weight-transfer calls.

    The session accepts chunks in any order, rejects duplicates, and publishes
    a version only when every name declared at ``start`` has arrived.  This is
    deliberately stricter than a loose dictionary assignment because it
    catches incomplete full-parameter exports before they affect rollout.
    """

    def __init__(self, store: _MockParameterStore) -> None:
        self._store = store
        self._initialized = False
        self._active_version: int | None = None
        self._expected_names: tuple[str, ...] = ()
        self._pending: dict[str, torch.Tensor] = {}
        self._current_version = 0
        self.init_info: dict[str, Any] | None = None
        self.published_versions: list[int] = [0]

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def update_in_progress(self) -> bool:
        return self._active_version is not None

    async def init_weight_transfer_engine(self, init_info: Mapping[str, Any]) -> None:
        if self._initialized:
            raise MockWeightTransferError("weight transfer engine is already initialized")
        if self._active_version is not None:
            raise MockWeightTransferError("cannot initialize during an active weight update")
        self.init_info = dict(init_info)
        self._initialized = True

    async def start_weight_update(self, *, version: int, parameter_names: Sequence[str]) -> None:
        if not self._initialized:
            raise MockWeightTransferError("init_weight_transfer_engine must run before start_weight_update")
        if self._active_version is not None:
            raise MockWeightTransferError("a weight update is already in progress")
        if version != self._current_version + 1:
            raise MockWeightTransferError(
                f"weight version must advance from {self._current_version} to "
                f"{self._current_version + 1}, got {version}"
            )
        names = tuple(parameter_names)
        if not names or len(set(names)) != len(names):
            raise MockWeightTransferError("start_weight_update requires unique, non-empty parameter names")
        if set(names) - set(self._store.trainable_names):
            raise MockWeightTransferError("weight update manifest contains a frozen or unknown parameter")
        self._active_version = version
        self._expected_names = names
        self._pending = {}

    async def update_weights(self, update: WeightUpdateChunk) -> None:
        if self._active_version is None:
            raise MockWeightTransferError("update_weights requires an active weight update")
        if update.version != self._active_version:
            raise MockWeightTransferError(
                f"update version {update.version} does not match active version {self._active_version}"
            )
        if not update.parameter_names:
            raise MockWeightTransferError("update_weights requires at least one parameter")
        expected = set(self._expected_names)
        payload = update.payload
        if not isinstance(payload, Mapping):
            raise MockWeightTransferError("mock update payload must be a parameter mapping")
        if set(update.parameter_names) != set(payload):
            raise MockWeightTransferError("chunk parameter names do not match its payload")
        for name in update.parameter_names:
            value = payload[name]
            if name not in expected:
                raise MockWeightTransferError(f"parameter {name!r} is not in the active update manifest")
            if name in self._pending:
                raise MockWeightTransferError(f"parameter {name!r} was sent more than once")
            self._store.validate_value(name, value)
            self._pending[name] = _clone(value)

    async def finish_weight_update(self) -> None:
        if self._active_version is None:
            raise MockWeightTransferError("finish_weight_update requires an active weight update")
        missing = set(self._expected_names) - set(self._pending)
        if missing:
            raise MockWeightTransferError(f"cannot publish incomplete update; missing {sorted(missing)}")
        version = self._active_version
        self._store.commit(self._pending)
        self._current_version = version
        self.published_versions.append(version)
        self._active_version = None
        self._expected_names = ()
        self._pending = {}


class _BaseMockRolloutEngine:
    mode: Literal["ar", "diffusion"]

    def __init__(self, parameters: Mapping[str, torch.Tensor], trainable_names: Sequence[str]) -> None:
        self._store = _MockParameterStore(parameters, trainable_names)
        self._weight_transfer = MockWeightTransferSession(self._store)

    @property
    def current_version(self) -> int:
        return self._weight_transfer.current_version

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._store.trainable_names

    @property
    def frozen_parameter_names(self) -> tuple[str, ...]:
        return self._store.frozen_names

    @property
    def weight_transfer(self) -> MockWeightTransferSession:
        return self._weight_transfer

    def export_trainable_weights(self) -> dict[str, torch.Tensor]:
        """Return a detached full snapshot for the mock trainer."""
        return self._store.snapshot(self.trainable_parameter_names)

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        """Return all parameters, including frozen VAE/vision components."""
        return self._store.snapshot()

    def parameter_signal(self) -> float:
        return self._store.signal(self.trainable_parameter_names)

    async def init_weight_transfer_engine(self, init_info: Mapping[str, Any]) -> None:
        await self._weight_transfer.init_weight_transfer_engine(init_info)

    async def start_weight_update(self, *, version: int, parameter_names: Sequence[str]) -> None:
        await self._weight_transfer.start_weight_update(version=version, parameter_names=parameter_names)

    async def update_weights(self, update: WeightUpdateChunk) -> None:
        await self._weight_transfer.update_weights(update)

    async def finish_weight_update(self) -> None:
        await self._weight_transfer.finish_weight_update()


def _default_ar_parameters() -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    parameters = {
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 0.10),
        "model.layers.0.mlp.down_proj.weight": torch.full((2, 2), 0.20),
        "lm_head.weight": torch.full((2, 2), 0.30),
        # This represents an auxiliary component that the AR recipe freezes.
        "vision_model.encoder.weight": torch.full((2, 2), 0.40),
    }
    return parameters, tuple(name for name in parameters if not name.startswith("vision_model."))


def _default_diffusion_parameters() -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    parameters = {
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 0.10),
        "model.layers.0.mlp.down_proj.weight": torch.full((2, 2), 0.20),
        "patch_embed.proj.weight": torch.full((2, 2), 0.30),
        "time_embed.weight": torch.full((2, 2), 0.40),
        "final_layer.linear.weight": torch.full((2, 2), 0.50),
        # VAE is intentionally frozen; it is used to decode the generated image.
        "vae.decoder.weight": torch.full((2, 2), 0.60),
        "vision_model.encoder.weight": torch.full((2, 2), 0.70),
    }
    trainable = tuple(name for name in parameters if not name.startswith(("vae.", "vision_model.")))
    return parameters, trainable


class MockARRolloutEngine(_BaseMockRolloutEngine):
    """Small AR engine exposing token/logprob and boundary metadata."""

    mode: Literal["ar"] = "ar"

    def __init__(self) -> None:
        parameters, trainable = _default_ar_parameters()
        super().__init__(parameters, trainable)

    async def generate(self, prompt: str, *, seed: int = 0) -> ARRolloutSample:
        digest = _stable_bytes(f"ar:{prompt}:{seed}:{self.current_version}")
        prompt_token_ids = tuple(100 + byte % 200 for byte in digest[:4])
        response_token_ids = tuple(500 + byte % 500 for byte in digest[4:8])
        signal = self.parameter_signal()
        response_token_logprobs = tuple(-0.25 - 0.01 * index - 0.001 * abs(signal) for index in range(4))
        return ARRolloutSample(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_token_logprobs=response_token_logprobs,
            response_start=len(prompt_token_ids),
            response_end=len(prompt_token_ids) + len(response_token_ids),
            weight_version=self.current_version,
        )


class MockDiffusionRolloutEngine(_BaseMockRolloutEngine):
    """Small diffusion engine exposing image and denoising trajectory metadata."""

    mode: Literal["diffusion"] = "diffusion"

    def __init__(self, num_inference_steps: int = 4) -> None:
        if num_inference_steps < 2:
            raise ValueError("the mock diffusion rollout needs at least two denoising steps")
        parameters, trainable = _default_diffusion_parameters()
        super().__init__(parameters, trainable)
        self.num_inference_steps = num_inference_steps

    async def generate(self, prompt: str, *, seed: int = 0) -> DiffusionRolloutSample:
        digest = _stable_bytes(f"diffusion:{prompt}:{seed}")
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big"))
        latent = torch.randn((1, 4, 2, 2), generator=generator)
        signal = self.parameter_signal() * 0.01 + self.current_version * 0.05
        trajectory: list[torch.Tensor] = []
        sigmas: list[float] = []
        timesteps: list[int] = []
        for index in range(self.num_inference_steps):
            sigma = 1.0 - index / self.num_inference_steps
            latent = torch.tanh(latent * (1.0 + signal) + sigma * 0.1)
            trajectory.append(latent.clone())
            sigmas.append(sigma)
            timesteps.append(self.num_inference_steps - index - 1)
        image = torch.sigmoid(trajectory[-1][:, :3])
        return DiffusionRolloutSample(
            prompt=prompt,
            image=image,
            latent_trajectory=tuple(trajectory),
            sigmas=tuple(sigmas),
            timesteps=tuple(timesteps),
            sde_step_indices=tuple(range(0, self.num_inference_steps, 2)),
            conditioning={"prompt": prompt, "prompt_hash": digest.hex()[:16]},
            trajectory_logprob=-float(sum(sigmas)) * (1.0 + abs(signal)),
            weight_version=self.current_version,
        )


class MockHunyuanImage3Trainer:
    """Trainer-side stand-in for one independent AR or diffusion RL recipe.

    It computes a deterministic reward and loss placeholder, updates every
    trainable parameter in its local snapshot, then publishes that complete
    snapshot in multiple chunks.  The reward/loss math is intentionally simple;
    the value of this class is exercising the interface and publication order.
    """

    def __init__(
        self,
        rollout_engine: MockARRolloutEngine | MockDiffusionRolloutEngine,
        *,
        transfer_client: Any | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.rollout_engine = rollout_engine
        self.transfer_client = transfer_client or rollout_engine
        self.learning_rate = learning_rate
        self._trainable_parameters = rollout_engine.export_trainable_weights()
        self._baseline = 0.0
        self._initialized = False
        # Publication identity belongs to the trainer.  A real HTTP client may
        # talk to a different process, so deriving the next version from a
        # local rollout object's state would be the wrong ownership boundary.
        self._published_version = rollout_engine.current_version

    async def initialize(self) -> None:
        await self.transfer_client.init_weight_transfer_engine({"backend": "mock", "packed": False})
        self._initialized = True

    async def run_round(self, prompt: str, *, seed: int = 0) -> RLUpdateResult:
        if not self._initialized:
            await self.initialize()
        rollout = await self.rollout_engine.generate(prompt, seed=seed)
        reward = self._reward(rollout)
        advantage = reward - self._baseline
        self._baseline = 0.5 * self._baseline + 0.5 * reward
        if isinstance(rollout, ARRolloutSample):
            mean_logprob = sum(rollout.response_token_logprobs) / len(rollout.response_token_logprobs)
            loss = -advantage * mean_logprob
        else:
            loss = -advantage * rollout.trajectory_logprob

        delta = self.learning_rate * advantage
        for name, value in self._trainable_parameters.items():
            self._trainable_parameters[name] = value + delta

        version = self._published_version + 1
        names = self.rollout_engine.trainable_parameter_names
        await self.transfer_client.start_weight_update(version=version, parameter_names=names)
        items = list(self._trainable_parameters.items())
        for offset in range(0, len(items), 2):
            await self.transfer_client.update_weights(
                WeightUpdateChunk(
                    version=version,
                    parameter_names=tuple(name for name, _ in items[offset : offset + 2]),
                    payload=dict(items[offset : offset + 2]),
                )
            )
        await self.transfer_client.finish_weight_update()
        self._published_version = version
        return RLUpdateResult(
            mode=self.rollout_engine.mode,
            rollout=rollout,
            reward=reward,
            advantage=advantage,
            loss=loss,
            published_version=version,
            published_parameter_names=names,
        )

    @staticmethod
    def _reward(rollout: ARRolloutSample | DiffusionRolloutSample) -> float:
        if isinstance(rollout, ARRolloutSample):
            return float(sum(rollout.response_token_ids) % 1000) / 1000.0 + 1.0
        return float(rollout.image.mean().item()) + float(len(rollout.sde_step_indices)) * 0.01
