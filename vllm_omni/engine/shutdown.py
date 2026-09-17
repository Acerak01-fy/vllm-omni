# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Nested process cleanup budgets shared by AR and diffusion stages."""

import vllm.envs as vllm_envs


def worker_shutdown_grace_s() -> float:
    """Honor the native worker setting without shortening diffusion cleanup."""
    return max(15.0, float(vllm_envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS))


def stage_shutdown_grace_s() -> float:
    # A worker returning from shutdown is not yet an exited process: Python
    # finalizers and the device runtime still have to release their resources.
    # Leave room for those and for the worker's bounded termination fallback.
    return worker_shutdown_grace_s() + 30.0
