# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import Enum


class DiffusionKVCacheMode(str, Enum):
    """Migration mode for diffusion KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    # Reserved for the experimental Worker-owned paging design. Production
    # configuration rejects it until that engine is migrated to the common
    # Scheduler-owned path.
    PAGED_WORKER_LOCAL = "paged_worker_local"
    PAGED_SCHEDULER = "paged_scheduler"
