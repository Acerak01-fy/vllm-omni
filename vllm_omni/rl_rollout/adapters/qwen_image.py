# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""QwenImage full-parameter rollout adapter.

QwenImage RL normally updates the DiT transformer while keeping the VAE and
conditioning components fixed.  This module makes that model boundary
explicit and packages every parameter under ``transformer.`` into the
model-neutral weight publication contract.  The publication is then handed to
the existing vLLM/Omni weight-transfer lifecycle; this module does not know
whether the configured transport is IPC, HTTP, or another upstream backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vllm_omni.rl_rollout.adapter import RLWeightTransferAdapter
from vllm_omni.rl_rollout.contracts import (
    VLLMWeightTransferClient,
    WeightPublication,
    WeightUpdateChunk,
)

ParameterItems = Mapping[str, Any] | Iterable[tuple[str, Any]]


def _items(parameters: ParameterItems) -> tuple[tuple[str, Any], ...]:
    """Materialize a parameter source and reject ambiguous manifests."""
    materialized = tuple(parameters.items()) if isinstance(parameters, Mapping) else tuple(parameters)
    names = [name for name, _ in materialized]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("parameter names must be unique")
    return materialized


class QwenImageParameterPolicy:
    """Select the full QwenImage RL parameter set.

    ``full parameter`` refers to all parameters in the trainable DiT
    transformer.  The VAE, text encoders, and other conditioning modules are
    intentionally excluded by default because they are inference-time
    components in the QwenImage RL recipe.  A caller can provide a different
    prefix policy when a future recipe deliberately trains an auxiliary
    component.
    """

    DEFAULT_TRAINABLE_PREFIXES = ("transformer.",)
    DEFAULT_FROZEN_PREFIXES = (
        "vae.",
        "text_encoder.",
        "text_encoder_2.",
        "vision_model.",
    )

    def __init__(
        self,
        *,
        trainable_prefixes: Sequence[str] = DEFAULT_TRAINABLE_PREFIXES,
        frozen_prefixes: Sequence[str] = DEFAULT_FROZEN_PREFIXES,
    ) -> None:
        self.trainable_prefixes = tuple(trainable_prefixes)
        self.frozen_prefixes = tuple(frozen_prefixes)
        if not self.trainable_prefixes:
            raise ValueError("at least one trainable prefix is required")
        if any(not prefix for prefix in (*self.trainable_prefixes, *self.frozen_prefixes)):
            raise ValueError("parameter prefixes must be non-empty")

    def select_names(self, parameters: ParameterItems) -> tuple[str, ...]:
        """Return the ordered trainable manifest for a pipeline snapshot."""
        selected = tuple(name for name, _ in _items(parameters) if name.startswith(self.trainable_prefixes))
        if not selected:
            prefixes = ", ".join(repr(prefix) for prefix in self.trainable_prefixes)
            raise ValueError(f"no QwenImage trainable parameters matched prefixes {prefixes}")
        frozen = tuple(name for name in selected if name.startswith(self.frozen_prefixes))
        if frozen:
            raise ValueError(f"trainable manifest includes frozen QwenImage parameters: {frozen}")
        return selected

    def select(self, parameters: ParameterItems) -> tuple[tuple[str, Any], ...]:
        """Return parameter values for the selected trainable manifest."""
        materialized = _items(parameters)
        selected_names = set(self.select_names(materialized))
        return tuple((name, value) for name, value in materialized if name in selected_names)


class QwenImageRLAdapter:
    """Build and publish QwenImage full-parameter snapshots.

    The adapter is intentionally small: model-specific work is limited to
    selecting the transformer parameters and chunking their snapshot.  Native
    vLLM/Omni WTE calls remain in :class:`RLWeightTransferAdapter`.
    """

    def __init__(
        self,
        client: VLLMWeightTransferClient | RLWeightTransferAdapter,
        *,
        policy: QwenImageParameterPolicy | None = None,
        current_version: int = 0,
        chunk_size: int = 64,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.policy = policy or QwenImageParameterPolicy()
        self._transfer = (
            client
            if isinstance(client, RLWeightTransferAdapter)
            else RLWeightTransferAdapter(client, current_version=current_version)
        )
        self.chunk_size = chunk_size

    @property
    def current_version(self) -> int:
        return self._transfer.current_version

    @property
    def initialized(self) -> bool:
        return self._transfer.initialized

    def trainable_parameter_names(self, parameters: ParameterItems) -> tuple[str, ...]:
        """Return the exact manifest that will be published."""
        return self.policy.select_names(parameters)

    def build_publication(self, parameters: ParameterItems, *, version: int | None = None) -> WeightPublication:
        """Build a versioned full transformer snapshot from pipeline parameters."""
        selected = self.policy.select(parameters)
        publication_version = self.current_version + 1 if version is None else version
        names = tuple(name for name, _ in selected)
        chunks = tuple(
            WeightUpdateChunk(
                version=publication_version,
                parameter_names=tuple(name for name, _ in selected[offset : offset + self.chunk_size]),
                payload=dict(selected[offset : offset + self.chunk_size]),
            )
            for offset in range(0, len(selected), self.chunk_size)
        )
        return WeightPublication(
            version=publication_version,
            parameter_names=names,
            chunks=chunks,
        )

    async def initialize(self, init_info: Mapping[str, Any]) -> None:
        """Initialize the configured native WTE transport."""
        await self._transfer.initialize(init_info)

    async def publish(self, parameters: ParameterItems) -> None:
        """Publish one complete QwenImage transformer snapshot."""
        await self._transfer.publish(self.build_publication(parameters))
