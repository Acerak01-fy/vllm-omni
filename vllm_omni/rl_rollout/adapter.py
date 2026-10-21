# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Trainer-facing adapter over vLLM-Omni's existing WTE lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.rl_rollout.contracts import VLLMWeightTransferClient, WeightPublication


class RLPublicationError(ValueError):
    """Raised when a trainer publication cannot be sent safely."""


class RLWeightTransferAdapter:
    """Validate RL publication metadata and call the existing Omni WTE API.

    The adapter owns trainer-side version and manifest bookkeeping.  It does
    not implement IPC, HTTP, NCCL, layerwise reload, or model parameter
    application; those remain in the upstream WTE and Omni workers.
    """

    def __init__(self, client: VLLMWeightTransferClient, *, current_version: int = 0) -> None:
        if current_version < 0:
            raise ValueError("current_version must be non-negative")
        self._client = client
        self._current_version = current_version
        self._initialized = False

    @property
    def current_version(self) -> int:
        """Highest version successfully handed to the rollout client."""
        return self._current_version

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self, init_info: Mapping[str, Any]) -> None:
        """Initialize the configured upstream transport exactly once."""
        if self._initialized:
            raise RLPublicationError("RL weight transfer is already initialized")
        await self._client.init_weight_transfer_engine(init_info)
        self._initialized = True

    async def publish(self, publication: WeightPublication) -> None:
        """Publish one complete version through Omni's native lifecycle."""
        if not self._initialized:
            raise RLPublicationError("initialize must run before publish")
        expected_version = self._current_version + 1
        if publication.version != expected_version:
            raise RLPublicationError(
                f"publication version must advance from {self._current_version} to {expected_version}, "
                f"got {publication.version}"
            )

        expected_names = set(publication.parameter_names)
        received_names: set[str] = set()
        for chunk in publication.chunks:
            if chunk.version != publication.version:
                raise RLPublicationError("chunk version does not match publication version")
            chunk_names = set(chunk.parameter_names)
            if chunk_names - expected_names:
                raise RLPublicationError("chunk contains a parameter outside the publication manifest")
            if received_names.intersection(chunk_names):
                raise RLPublicationError("a parameter appears in more than one publication chunk")
            received_names.update(chunk_names)
        if received_names != expected_names:
            missing = sorted(expected_names - received_names)
            extra = sorted(received_names - expected_names)
            raise RLPublicationError(f"publication manifest mismatch; missing={missing}, extra={extra}")

        await self._client.start_weight_update()
        for chunk in publication.chunks:
            await self._client.update_weights(chunk.payload)
        await self._client.finish_weight_update(str(publication.version))
        self._current_version = publication.version
