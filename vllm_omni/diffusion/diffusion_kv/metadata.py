# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-to-Worker allocation payloads for diffusion KV.

Like native vLLM's SchedulerOutput DTOs, these classes carry trusted internal
Scheduler results. Cache geometry and block-table validation stay with the
native cache builders and Worker installation path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiffusionKVContextMetadata:
    """Worker allocation for an independently managed K/V context."""

    context_id: str
    cache_role: str
    num_tokens: int
    block_ids: tuple[list[int], ...]


@dataclass
class DiffusionKVSequenceMetadata:
    """Worker allocation for one Scheduler-owned diffusion sequence."""

    sequence_id: int
    prefix_len: int
    target_len: int
    seq_len: int
    block_ids: tuple[list[int], ...]
    contexts: tuple[DiffusionKVContextMetadata, ...] = ()


@dataclass
class DiffusionKVMetadata:
    """Scheduler allocation sent with a newly scheduled public request."""

    request_id: str
    allocation_generation: int
    sequences: tuple[DiffusionKVSequenceMetadata, ...]
