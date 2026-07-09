# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in profiling ranges shared by torch profiler and Nsight Systems."""

from __future__ import annotations

import os
from types import TracebackType

import torch

_PROFILE_RANGES_ENV = "VLLM_OMNI_PROFILE_RANGES"
_ENABLED_VALUES = frozenset({"", "1", "true", "yes", "on", "enabled", "enable"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})


def _env_enabled() -> bool:
    value = os.environ.get(_PROFILE_RANGES_ENV)
    if value is None:
        return False
    value = value.strip().lower()
    if value in _DISABLED_VALUES:
        return False
    return value in _ENABLED_VALUES


_PROFILE_RANGES_ENABLED = _env_enabled()


class ProfileRange:
    """Context manager emitting torch profiler and NVTX ranges when enabled."""

    __slots__ = ("_name", "_record_ctx", "_nvtx_pushed")

    def __init__(self, name: str) -> None:
        self._name = name
        self._record_ctx = None
        self._nvtx_pushed = False

    def __enter__(self) -> None:
        if not _PROFILE_RANGES_ENABLED:
            return None
        self._record_ctx = torch.profiler.record_function(self._name)
        self._record_ctx.__enter__()
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self._name)
            self._nvtx_pushed = True
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._nvtx_pushed:
            torch.cuda.nvtx.range_pop()
            self._nvtx_pushed = False
        if self._record_ctx is not None:
            self._record_ctx.__exit__(exc_type, exc, tb)
            self._record_ctx = None
        return None


def profile_range(name: str) -> ProfileRange:
    return ProfileRange(name)
