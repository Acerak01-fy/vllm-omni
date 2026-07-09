# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

import vllm_omni.profiler.ranges as ranges_mod


class _FakeRecordFunction:
    def __init__(self, events: list[tuple[str, str]], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self) -> None:
        self._events.append(("record_enter", self._name))
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._events.append(("record_exit", self._name))
        return None


def _reload_ranges(monkeypatch, env_value: str | None):
    if env_value is None:
        monkeypatch.delenv("VLLM_OMNI_PROFILE_RANGES", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_PROFILE_RANGES", env_value)
    return importlib.reload(ranges_mod)


def test_profile_range_is_disabled_by_default(monkeypatch):
    ranges = _reload_ranges(monkeypatch, None)
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        ranges.torch.profiler,
        "record_function",
        lambda name: _FakeRecordFunction(events, name),
    )
    monkeypatch.setattr(ranges.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_push", lambda name: events.append(("nvtx_push", name)))
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_pop", lambda: events.append(("nvtx_pop", "")))

    with ranges.profile_range("hy3.disabled"):
        pass

    assert events == []


def test_profile_range_emits_record_function_and_nvtx_when_enabled(monkeypatch):
    ranges = _reload_ranges(monkeypatch, "1")
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        ranges.torch.profiler,
        "record_function",
        lambda name: _FakeRecordFunction(events, name),
    )
    monkeypatch.setattr(ranges.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_push", lambda name: events.append(("nvtx_push", name)))
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_pop", lambda: events.append(("nvtx_pop", "")))

    with ranges.profile_range("hy3.enabled"):
        events.append(("body", "hy3.enabled"))

    assert events == [
        ("record_enter", "hy3.enabled"),
        ("nvtx_push", "hy3.enabled"),
        ("body", "hy3.enabled"),
        ("nvtx_pop", ""),
        ("record_exit", "hy3.enabled"),
    ]


def test_profile_range_skips_nvtx_when_cuda_is_unavailable(monkeypatch):
    ranges = _reload_ranges(monkeypatch, "yes")
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        ranges.torch.profiler,
        "record_function",
        lambda name: _FakeRecordFunction(events, name),
    )
    monkeypatch.setattr(ranges.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_push", lambda name: events.append(("nvtx_push", name)))
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_pop", lambda: events.append(("nvtx_pop", "")))

    with ranges.profile_range("hy3.cpu"):
        pass

    assert events == [
        ("record_enter", "hy3.cpu"),
        ("record_exit", "hy3.cpu"),
    ]
