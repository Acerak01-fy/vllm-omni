# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the experimental cache-pool execution paths.

Covers three flavors of the cache backend:

* Cache-DiT stepwise (single-request lifecycle)
* Cache-DiT batched forward + batched stepwise runner
* TeaCache stepwise (hook state restoration across requests)

The fakes in this module implement just enough of the pipeline contract
(``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` / ``post_decode``)
and of the ``CacheStateDriver`` interface to drive ``DiffusionModelRunner``
end-to-end on CPU.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.experimental as diffusion_experimental
import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.cache.cache_dit_batch import (
    _forward_batched_345,
    _forward_batched_base,
    clear_batch_contexts,
    set_batch_contexts,
)
from vllm_omni.diffusion.cache.cache_dit_driver import CacheDiTStateDriver
from vllm_omni.diffusion.cache.cache_manager import CacheManager, CacheStateDriver
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.driver import TeaCacheStateDriver
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import CacheBackendSlot

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


@pytest.fixture(autouse=True)
def _cachepool_test_env(monkeypatch):
    """Enable the experimental cache pool and stub ``set_forward_context``.

    Individual tests may still override ``EXPERIMENT_CACHEPOOL`` — later
    monkeypatches chain on top of this one.
    """
    monkeypatch.setattr(diffusion_experimental, "EXPERIMENT_CACHEPOOL", True)
    monkeypatch.setattr(
        model_runner_module, "set_forward_context", _noop_forward_context
    )


def _make_request(req_id: str, num_inference_steps: int = 3):
    return SimpleNamespace(
        prompts=[f"prompt-{req_id}"],
        request_ids=[req_id],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
            lora_request=None,
            cache_plan=(),
        ),
    )


def _make_scheduler_output(
    *,
    new_reqs: Iterable = (),
    cached_req_ids: Iterable[str] = (),
    step_id: int = 0,
    finished_req_ids: Iterable[str] | None = None,
) -> DiffusionSchedulerOutput:
    """Build a ``DiffusionSchedulerOutput`` for the runner under test."""
    new_reqs = list(new_reqs)
    cached_ids = list(cached_req_ids)
    cached = (
        CachedRequestData.make_empty()
        if not cached_ids
        else CachedRequestData(sched_req_ids=cached_ids)
    )
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[
            NewRequestData(sched_req_id=req.request_ids[0], req=req)
            for req in new_reqs
        ],
        scheduled_cached_reqs=cached,
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=len(new_reqs) + len(cached_ids),
        num_waiting_reqs=0,
    )


def _make_cache_state(
    req_id: str,
    *,
    num_inference_steps: int,
    rows: int = 1,
    slot: CacheBackendSlot | None = None,
):
    return SimpleNamespace(
        req_id=req_id,
        sampling=SimpleNamespace(num_inference_steps=num_inference_steps),
        latents=torch.zeros((rows, 2), dtype=torch.float32),
        cache_slot=slot,
    )


def _attach_fresh_slots(driver, states) -> None:
    """Create + initialize a fresh cache slot on each state."""
    for state in states:
        state.cache_slot = driver.create_empty_slot()
        driver.initialize_fresh_slot(
            state.cache_slot, state.sampling.num_inference_steps
        )


def _build_runner(pipeline, driver, *, backend_name: str = "cache_dit"):
    """Assemble a minimal ``DiffusionModelRunner`` backed by test doubles."""
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=backend_name,
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = SimpleNamespace(is_enabled=lambda: True)
    runner.cache_manager = CacheManager(driver)
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    pipeline.runner = runner
    return runner


# ---------------------------------------------------------------------------
# Shared pipeline base
# ---------------------------------------------------------------------------


class _BasePipeline:
    """Shared ``prepare_encode``/``step_scheduler``/``post_decode`` for fakes."""

    supports_step_execution = True

    def __init__(self):
        self.runner = None
        self.prepare_calls: list[str] = []
        self.decode_calls: list[str] = []

    def _make_latents(self) -> torch.Tensor:
        return torch.tensor([0.0])

    def _extend_prepared_state(self, state) -> None:
        """Hook for subclasses to populate extra state during ``prepare_encode``."""

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls.append(state.req_id)
        num_steps = state.sampling.num_inference_steps
        state.timesteps = [torch.tensor(num_steps - idx) for idx in range(num_steps)]
        state.latents = self._make_latents()
        state.prompt_embeds = torch.zeros((1, 2, 4), dtype=torch.float32)
        state.prompt_embeds_mask = torch.tensor([[True, True]])
        self._extend_prepared_state(state)
        return state

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls.append(state.req_id)
        return DiffusionOutput(
            output=torch.tensor([state.step_index], dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# Cache-DiT single-request fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CacheDiTSnapshot:
    req_id: str
    step_index: int
    active_req_id: str | None
    waiting_req_ids: tuple[str, ...]
    resident_req_ids: tuple[str, ...]
    slot_id: int
    trace_before: tuple[str, ...]


class _FakeCacheDiTDriver(CacheStateDriver):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._next_slot_id = 0
        self.initialize_history: list[tuple[int, int]] = []
        self.deactivate_history: list[int] = []
        self.clear_history: list[int] = []

    @property
    def backend_name(self) -> str:
        return "cache_dit"

    def create_empty_slot(self) -> CacheBackendSlot:
        self._next_slot_id += 1
        return CacheBackendSlot(
            backend_name=self.backend_name,
            payload={"slot_id": self._next_slot_id, "trace": []},
        )

    def install_slot(self, slot: CacheBackendSlot) -> None:
        self.pipeline.live_cache_slot = slot

    def initialize_fresh_slot(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> None:
        slot.payload["trace"].clear()
        slot.payload["initialized_for_steps"] = num_inference_steps
        self.initialize_history.append(
            (slot.payload["slot_id"], num_inference_steps)
        )

    def is_slot_compatible(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> bool:
        return slot.metadata.get("num_inference_steps") == num_inference_steps

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        if slot is not None:
            self.deactivate_history.append(slot.payload["slot_id"])
        self.pipeline.live_cache_slot = None

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        self.clear_history.append(slot.payload["slot_id"])
        slot.payload["trace"].clear()
        slot.payload["cleared"] = True
        slot.metadata.clear()
        slot.resident_bytes = 0
        if self.pipeline.live_cache_slot is slot:
            self.pipeline.live_cache_slot = None

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        return 1024 + 64 * len(slot.payload["trace"])


class _CacheDiTPipeline(_BasePipeline):
    def __init__(self):
        super().__init__()
        self.live_cache_slot = None
        self.snapshots: list[_CacheDiTSnapshot] = []
        self.none_req_ids: set[str] = set()
        self.fail_req_ids: set[str] = set()

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        assert self.runner is not None
        active_req_id = self.runner.cache_manager._active_req_id
        if active_req_id is None:
            raise AssertionError(
                "cache manager must activate one request before denoise_step"
            )
        state = self.runner.state_cache[active_req_id]
        assert self.live_cache_slot is state.cache_slot

        resident_req_ids = tuple(
            sorted(
                req_id
                for req_id, cached_state in self.runner.state_cache.items()
                if cached_state.cache_slot is not None
            )
        )
        waiting_req_ids = tuple(
            rid for rid in resident_req_ids if rid != active_req_id
        )
        self.snapshots.append(
            _CacheDiTSnapshot(
                req_id=state.req_id,
                step_index=state.step_index,
                active_req_id=active_req_id,
                waiting_req_ids=waiting_req_ids,
                resident_req_ids=resident_req_ids,
                slot_id=state.cache_slot.payload["slot_id"],
                trace_before=tuple(state.cache_slot.payload["trace"]),
            )
        )

        if state.req_id in self.fail_req_ids:
            raise RuntimeError(f"boom-{state.req_id}")
        if state.req_id in self.none_req_ids:
            return None

        state.cache_slot.payload["trace"].append(
            f"{state.req_id}:{state.step_index}"
        )
        return torch.tensor([float(state.step_index + 1)])


def _make_cache_dit_runner():
    pipeline = _CacheDiTPipeline()
    return _build_runner(pipeline, _FakeCacheDiTDriver(pipeline))


# ---------------------------------------------------------------------------
# Cache-DiT batched-forward fakes
# ---------------------------------------------------------------------------


@dataclass
class _BatchCacheContext:
    name: str
    cache_decision: bool = False
    warmup_blocks_cache: bool = False
    block_due_to_continuous_limit: bool = False
    block_due_to_accumulated_diff: bool = False
    is_l1_enabled: bool = False
    cache_residual: bool = False
    encoder_cache_residual: bool = False
    buffers: dict[str, torch.Tensor] | None = None
    mark_step_calls: int = 0
    cached_step_calls: int = 0
    last_check_tensor: torch.Tensor | None = None
    last_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.buffers is None:
            self.buffers = {}


class _BatchContextManager:
    def __init__(
        self,
        *,
        is_l1_enabled: bool = False,
        cache_residual: bool = False,
        encoder_cache_residual: bool = False,
        bn_compute_blocks: int = 1,
    ):
        self._batch_contexts = None
        self._batch_row_offsets = None
        self._batch_row_counts = None
        self._current_context: _BatchCacheContext | None = None
        self._is_l1_enabled = is_l1_enabled
        self._cache_residual = cache_residual
        self._encoder_cache_residual = encoder_cache_residual
        self._bn_compute_blocks = bn_compute_blocks

    def mark_step_begin(self):
        assert self._current_context is not None
        self._current_context.mark_step_calls += 1

    def can_cache(self, states_tensor, parallelized=False, prefix="Fn"):
        del parallelized
        ctx = self._current_context
        assert ctx is not None
        ctx.last_check_tensor = states_tensor.detach().clone()
        ctx.last_prefix = prefix
        if ctx.warmup_blocks_cache:
            return False
        if ctx.block_due_to_continuous_limit:
            return False
        if ctx.block_due_to_accumulated_diff:
            return False
        return ctx.cache_decision

    def add_cached_step(self):
        assert self._current_context is not None
        self._current_context.cached_step_calls += 1

    def set_Fn_buffer(self, buffer, prefix="Fn"):
        assert self._current_context is not None
        self._current_context.buffers[prefix] = buffer.detach().clone()

    def set_Bn_buffer(self, buffer, prefix="Bn"):
        assert self._current_context is not None
        self._current_context.buffers[prefix] = buffer.detach().clone()

    def set_Bn_encoder_buffer(self, buffer, prefix="Bn"):
        assert self._current_context is not None
        if buffer is None:
            return
        self._current_context.buffers[f"{prefix}_encoder"] = buffer.detach().clone()

    def apply_cache(
        self, hidden_states, encoder_hidden_states, prefix="Bn", encoder_prefix="Bn"
    ):
        ctx = self._current_context
        assert ctx is not None
        cached_hs = ctx.buffers[prefix]
        cached_hs = (
            hidden_states + cached_hs if self.is_cache_residual() else cached_hs.clone()
        )

        cached_enc = None
        encoder_key = f"{encoder_prefix}_encoder"
        if encoder_hidden_states is not None and encoder_key in ctx.buffers:
            cached_enc = ctx.buffers[encoder_key]
            cached_enc = (
                encoder_hidden_states + cached_enc
                if self.is_encoder_cache_residual()
                else cached_enc.clone()
            )

        return cached_hs, cached_enc

    def is_l1_diff_enabled(self):
        if self._current_context is not None:
            return self._current_context.is_l1_enabled
        return self._is_l1_enabled

    def is_cache_residual(self):
        if self._current_context is not None:
            return self._current_context.cache_residual
        return self._cache_residual

    def is_encoder_cache_residual(self):
        if self._current_context is not None:
            return self._current_context.encoder_cache_residual
        return self._encoder_cache_residual

    def Bn_compute_blocks(self):
        return self._bn_compute_blocks


class _FakeBatchedPatternBase:
    def __init__(self, context_manager: _BatchContextManager):
        self.context_manager = context_manager
        self.cache_context = "ctx"
        self.cache_prefix = "fake"
        self.mn_call_shapes: list[tuple[int, ...]] = []
        self.bn_call_shapes: list[tuple[int, ...]] = []
        self.fn_call_shapes: list[tuple[int, ...]] = []

    def _check_cache_params(self):
        return None

    def _is_parallelized(self):
        return False

    def call_Fn_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        del args, kwargs
        self.fn_call_shapes.append(tuple(hidden_states.shape))
        new_enc = None if encoder_hidden_states is None else encoder_hidden_states + 10
        return hidden_states + 1, new_enc

    def _get_Fn_residual(self, original_hidden_states, hidden_states):
        return hidden_states - original_hidden_states

    def call_Mn_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        del args, kwargs
        self.mn_call_shapes.append(tuple(hidden_states.shape))
        new_hs = hidden_states + 2
        new_enc = None if encoder_hidden_states is None else encoder_hidden_states + 20
        hs_residual = torch.full_like(hidden_states, 5)
        enc_residual = (
            None
            if encoder_hidden_states is None
            else torch.full_like(encoder_hidden_states, 7)
        )
        return new_hs, new_enc, hs_residual, enc_residual

    def call_Bn_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        del args, kwargs
        self.bn_call_shapes.append(tuple(hidden_states.shape))
        new_enc = None if encoder_hidden_states is None else encoder_hidden_states + 30
        return hidden_states + 3, new_enc

    def _process_forward_outputs(self, hidden_states, encoder_hidden_states):
        return hidden_states, encoder_hidden_states


class _FakeBatchedPattern345:
    def __init__(self, context_manager: _BatchContextManager):
        self.context_manager = context_manager
        self.cache_context = "ctx"
        self.cache_prefix = "fake"
        self.mn_call_shapes: list[tuple[int, ...]] = []
        self.bn_call_shapes: list[tuple[int, ...]] = []

    def _check_cache_params(self):
        return None

    def _is_parallelized(self):
        return False

    def call_Fn_blocks(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + 1, hidden_states + 10

    def _get_Fn_residual(self, original_hidden_states, hidden_states):
        return hidden_states - original_hidden_states

    def call_Mn_blocks(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.mn_call_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 2, hidden_states + 20, torch.full_like(hidden_states, 6)

    def call_Bn_blocks(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.bn_call_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 3, hidden_states + 30

    def _process_forward_outputs(self, hidden_states, encoder_hidden_states):
        return hidden_states, encoder_hidden_states


def _clone_batch_contexts(contexts):
    return [
        {name: copy.deepcopy(ctx) for name, ctx in ctx_map.items()}
        for ctx_map in contexts
    ]


def _fn_check_prefix(use_l1: bool) -> str:
    return "fake_Fn_hidden_states" if use_l1 else "fake_Fn_residual"


def _bn_prefix(residual: bool) -> str:
    return "fake_Bn_residual" if residual else "fake_Bn_hidden_states"


def _run_serial_base(pattern, batch_contexts, hidden_states, encoder_hidden_states):
    """Reference serial loop mirroring ``_forward_batched_base`` semantics."""
    cm = pattern.context_manager
    outputs_hs: list[torch.Tensor] = []
    outputs_enc: list[torch.Tensor] = []
    use_l1 = cm.is_l1_diff_enabled()
    for ctx_map, hs_slice in zip(
        batch_contexts, hidden_states.split(1, dim=0), strict=True
    ):
        enc_slice = None
        if encoder_hidden_states is not None:
            enc_slice = encoder_hidden_states[
                len(outputs_hs) : len(outputs_hs) + 1
            ].clone()
        hs_slice = hs_slice.clone()
        cm._current_context = ctx_map["ctx"]
        orig = hs_slice
        hs_slice, enc_slice = pattern.call_Fn_blocks(hs_slice, enc_slice)
        fn_residual = pattern._get_Fn_residual(orig, hs_slice)
        fn_hidden_states = hs_slice.clone()
        cm.mark_step_begin()
        can_cache = cm.can_cache(
            fn_hidden_states if use_l1 else fn_residual,
            prefix=_fn_check_prefix(use_l1),
        )
        if can_cache:
            cm.add_cached_step()
            hs_slice, enc_slice = cm.apply_cache(
                hs_slice,
                enc_slice,
                prefix=_bn_prefix(cm.is_cache_residual()),
                encoder_prefix=_bn_prefix(cm.is_encoder_cache_residual()),
            )
        else:
            cm.set_Fn_buffer(fn_residual, prefix="fake_Fn_residual")
            if use_l1:
                cm.set_Fn_buffer(fn_hidden_states, "fake_Fn_hidden_states")
            hs_slice, enc_slice, hs_residual, enc_residual = pattern.call_Mn_blocks(
                hs_slice, enc_slice
            )
            if cm.is_cache_residual():
                cm.set_Bn_buffer(hs_residual, prefix="fake_Bn_residual")
            else:
                cm.set_Bn_buffer(hs_slice, prefix="fake_Bn_hidden_states")
            if enc_slice is not None:
                if cm.is_encoder_cache_residual():
                    cm.set_Bn_encoder_buffer(enc_residual, prefix="fake_Bn_residual")
                else:
                    cm.set_Bn_encoder_buffer(enc_slice, prefix="fake_Bn_hidden_states")
        hs_slice, enc_slice = pattern.call_Bn_blocks(hs_slice, enc_slice)
        outputs_hs.append(hs_slice)
        if enc_slice is not None:
            outputs_enc.append(enc_slice)
    cm._current_context = None
    out_enc = None if encoder_hidden_states is None else torch.cat(outputs_enc, dim=0)
    return torch.cat(outputs_hs, dim=0), out_enc


def _run_serial_345(pattern, batch_contexts, hidden_states):
    """Reference serial loop mirroring ``_forward_batched_345`` semantics."""
    cm = pattern.context_manager
    outputs_hs: list[torch.Tensor] = []
    outputs_enc: list[torch.Tensor] = []
    use_l1 = cm.is_l1_diff_enabled()
    for ctx_map, hs_slice in zip(
        batch_contexts, hidden_states.split(1, dim=0), strict=True
    ):
        hs_slice = hs_slice.clone()
        cm._current_context = ctx_map["ctx"]
        orig = hs_slice
        hs_slice, enc_slice = pattern.call_Fn_blocks(hs_slice)
        fn_residual = pattern._get_Fn_residual(orig, hs_slice)
        fn_hidden_states = hs_slice.clone()
        cm.mark_step_begin()
        can_cache = cm.can_cache(
            fn_hidden_states if use_l1 else fn_residual,
            prefix=_fn_check_prefix(use_l1),
        )
        if can_cache:
            cm.add_cached_step()
            hs_slice, enc_slice = cm.apply_cache(
                hs_slice,
                enc_slice,
                prefix=_bn_prefix(cm.is_cache_residual()),
                encoder_prefix=_bn_prefix(cm.is_encoder_cache_residual()),
            )
        else:
            cm.set_Fn_buffer(fn_residual, prefix="fake_Fn_residual")
            if use_l1:
                cm.set_Fn_buffer(fn_hidden_states, "fake_Fn_hidden_states")
            hs_slice, enc_slice, hs_residual = pattern.call_Mn_blocks(hs_slice)
            if cm.is_cache_residual():
                cm.set_Bn_buffer(hs_residual, prefix="fake_Bn_residual")
            else:
                cm.set_Bn_buffer(hs_slice, prefix="fake_Bn_hidden_states")
            if enc_slice is not None:
                enc_residual = enc_slice - (orig + 10)
                if cm.is_encoder_cache_residual():
                    cm.set_Bn_encoder_buffer(enc_residual, prefix="fake_Bn_residual")
                else:
                    cm.set_Bn_encoder_buffer(enc_slice, prefix="fake_Bn_hidden_states")
        if cm.Bn_compute_blocks() > 0:
            hs_slice, enc_slice = pattern.call_Bn_blocks(hs_slice)
        outputs_hs.append(hs_slice)
        outputs_enc.append(enc_slice)
    cm._current_context = None
    return torch.cat(outputs_hs, dim=0), torch.cat(outputs_enc, dim=0)


def _seed_cached_buffers(
    ctx: _BatchCacheContext,
    *,
    hidden_shape: tuple[int, ...],
    encoder_shape: tuple[int, ...] | None = None,
    cache_residual: bool = False,
    encoder_cache_residual: bool = False,
) -> None:
    ctx.buffers[_bn_prefix(cache_residual)] = torch.full(hidden_shape, 50.0)
    if encoder_shape is not None:
        ctx.buffers[f"{_bn_prefix(encoder_cache_residual)}_encoder"] = torch.full(
            encoder_shape, 70.0
        )


def _assert_context_maps_match(actual_contexts, expected_contexts):
    assert len(actual_contexts) == len(expected_contexts)
    for actual_map, expected_map in zip(
        actual_contexts, expected_contexts, strict=True
    ):
        assert tuple(actual_map) == tuple(expected_map)
        for name in actual_map:
            actual = actual_map[name]
            expected = expected_map[name]
            assert actual.mark_step_calls == expected.mark_step_calls
            assert actual.cached_step_calls == expected.cached_step_calls
            assert actual.last_prefix == expected.last_prefix
            if expected.last_check_tensor is None:
                assert actual.last_check_tensor is None
            else:
                assert torch.allclose(
                    actual.last_check_tensor, expected.last_check_tensor
                )
            assert set(actual.buffers) == set(expected.buffers)
            for key in expected.buffers:
                assert torch.allclose(actual.buffers[key], expected.buffers[key])


# ---------------------------------------------------------------------------
# CacheManager batch-lifecycle fakes
# ---------------------------------------------------------------------------


class _FakeBatchCacheLifecycleDriver(CacheStateDriver):
    def __init__(self):
        self._next_slot_id = 0
        self.initialize_history: list[tuple[int, int]] = []
        self.install_batch_history: list[tuple[str, ...]] = []
        self.deactivate_batch_calls = 0
        self.clear_history: list[int] = []

    @property
    def backend_name(self) -> str:
        return "cache_dit"

    def create_empty_slot(self) -> CacheBackendSlot:
        self._next_slot_id += 1
        return CacheBackendSlot(
            backend_name=self.backend_name,
            payload={"slot_id": self._next_slot_id},
        )

    def install_slot(self, slot: CacheBackendSlot) -> None:
        slot.payload["installed_single"] = True

    def initialize_fresh_slot(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> None:
        slot.payload["initialized_for_steps"] = num_inference_steps
        self.initialize_history.append(
            (slot.payload["slot_id"], num_inference_steps)
        )

    def is_slot_compatible(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> bool:
        return slot.metadata.get("num_inference_steps") == num_inference_steps

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        del slot

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        self.clear_history.append(slot.payload["slot_id"])
        slot.payload["cleared"] = True
        slot.metadata.clear()
        slot.resident_bytes = 0

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        return 256 + slot.payload["slot_id"]

    def install_batch_slots(self, states):
        self.install_batch_history.append(tuple(state.req_id for state in states))

    def deactivate_batch_slots(self):
        self.deactivate_batch_calls += 1


# ---------------------------------------------------------------------------
# Cache-DiT batched runner fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BatchRunnerSnapshot:
    req_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    cache_decisions: tuple[bool, ...]
    batch_active: bool


class _FakeRunnerBatchCacheDiTDriver(CacheStateDriver):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._next_slot_id = 0
        self.install_batch_history: list[tuple[str, ...]] = []
        self.deactivate_batch_calls = 0

    @property
    def backend_name(self) -> str:
        return "cache_dit"

    def create_empty_slot(self) -> CacheBackendSlot:
        self._next_slot_id += 1
        return CacheBackendSlot(
            backend_name=self.backend_name,
            payload={"slot_id": self._next_slot_id, "history": []},
        )

    def install_slot(self, slot: CacheBackendSlot) -> None:
        self.pipeline.live_cache_slot = slot

    def initialize_fresh_slot(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> None:
        slot.payload["history"].clear()
        slot.payload["initialized_for_steps"] = num_inference_steps

    def is_slot_compatible(
        self, slot: CacheBackendSlot, num_inference_steps: int
    ) -> bool:
        return slot.metadata.get("num_inference_steps") == num_inference_steps

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        if self.pipeline.live_cache_slot is slot:
            self.pipeline.live_cache_slot = None

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        slot.payload["history"].clear()
        slot.metadata.clear()
        slot.resident_bytes = 0

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        return 1024 + 64 * len(slot.payload["history"])

    def install_batch_slots(self, states):
        self.install_batch_history.append(tuple(state.req_id for state in states))
        self.pipeline.live_batch_req_ids = [state.req_id for state in states]
        for state in states:
            state.cache_slot.payload["cache_plan"] = tuple(
                getattr(state.sampling, "cache_plan", ())
            )

    def deactivate_batch_slots(self):
        self.deactivate_batch_calls += 1
        self.pipeline.live_batch_req_ids = []


class _BatchCacheDiTPipeline(_BasePipeline):
    def __init__(self):
        super().__init__()
        self.live_cache_slot = None
        self.live_batch_req_ids: list[str] = []
        self.snapshots: list[_BatchRunnerSnapshot] = []

    def _make_latents(self) -> torch.Tensor:
        return torch.tensor([[0.0]], dtype=torch.float32)

    def _extend_prepared_state(self, state) -> None:
        state.extra["cache_plan"] = tuple(getattr(state.sampling, "cache_plan", ()))

    def denoise_step(self, input_batch, **kwargs):
        del kwargs
        assert self.runner is not None
        states = [self.runner.state_cache[req_id] for req_id in input_batch.req_ids]
        decisions = tuple(
            state.step_index in state.extra.get("cache_plan", ())
            for state in states
        )
        self.snapshots.append(
            _BatchRunnerSnapshot(
                req_ids=tuple(input_batch.req_ids),
                step_indices=tuple(state.step_index for state in states),
                cache_decisions=decisions,
                batch_active=self.runner.cache_manager._batch_active,
            )
        )
        for state in states:
            state.cache_slot.payload["history"].append(
                (state.req_id, state.step_index)
            )
        return input_batch.latents + 1


def _make_batch_cache_dit_runner():
    pipeline = _BatchCacheDiTPipeline()
    return _build_runner(pipeline, _FakeRunnerBatchCacheDiTDriver(pipeline))


# ---------------------------------------------------------------------------
# Cache-DiT state-driver (module-context) fakes
# ---------------------------------------------------------------------------


class _FakeDriverContext:
    def __init__(self, name: str, scale: float = 1.0):
        self.name = name
        self.scale = scale
        self.config_steps = None
        self.buffers: dict[str, torch.Tensor] = {}
        self._init_args = (name,)
        self._init_kwargs = {"scale": scale}

    def clear_buffers(self) -> None:
        self.buffers.clear()


class _FakeDriverContextManager:
    def __init__(self, *context_names: str):
        self._cached_context_manager = {
            name: _FakeDriverContext(name) for name in context_names
        }
        self._current_context = None
        self._batch_contexts = None
        self._batch_row_offsets = None
        self._batch_row_counts = None

    def get_context(self, name: str):
        return self._cached_context_manager[name]


class _FakeCacheDiTBackend:
    def __init__(self):
        self.force_refresh_history: list[int] = []

    def force_refresh(self, pipeline, num_inference_steps: int, verbose: bool = False):
        del verbose
        self.force_refresh_history.append(num_inference_steps)
        for module in CacheDiTStateDriver._candidate_modules(pipeline):
            context_manager = getattr(module, "_context_manager", None)
            if context_manager is None:
                continue
            for context in context_manager._cached_context_manager.values():
                context.config_steps = num_inference_steps
                context.buffers["config_steps"] = torch.tensor(
                    [num_inference_steps], dtype=torch.float32
                )


def _make_cache_dit_driver():
    manager_a = _FakeDriverContextManager("ctx_a", "ctx_b")
    manager_b = _FakeDriverContextManager("ctx_c")
    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(
            _context_manager=manager_a,
            _context_names=("ctx_a", "ctx_b"),
        ),
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                _context_manager=manager_b,
                _context_names=("ctx_c",),
            )
        ),
    )
    backend = _FakeCacheDiTBackend()
    driver = CacheDiTStateDriver(backend, pipeline)
    return driver, backend, pipeline, (manager_a, manager_b)


# ---------------------------------------------------------------------------
# TeaCache fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TeaCacheSnapshot:
    req_id: str
    step_index: int
    active_req_id: str | None
    waiting_req_ids: tuple[str, ...]
    resident_req_ids: tuple[str, ...]
    forward_cnt_before: int
    positive_cnt_before: int
    residual_sum_before: float


class _TeaHookRegistry:
    def __init__(self, hook: TeaCacheHook):
        self._hook = hook

    def get_hook(self, name: str):
        if name == TeaCacheHook._HOOK_NAME:
            return self._hook
        return None


class _TeaCachePipeline(_BasePipeline):
    def __init__(self):
        super().__init__()
        self.snapshots: list[_TeaCacheSnapshot] = []
        self.none_req_ids: set[str] = set()
        self.fail_req_ids: set[str] = set()
        self.hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="QwenImageTransformer2DModel",
                coefficients=[0.0, 0.0, 0.0, 0.0, 1.0],
            )
        )
        self.hook.state_manager.set_context("teacache")
        self.transformer = SimpleNamespace(_hook_registry=_TeaHookRegistry(self.hook))

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        assert self.runner is not None
        active_req_id = self.runner.cache_manager._active_req_id
        if active_req_id is None:
            raise AssertionError(
                "cache manager must activate one request before denoise_step"
            )
        state = self.runner.state_cache[active_req_id]
        resident_req_ids = tuple(
            sorted(
                req_id
                for req_id, cached_state in self.runner.state_cache.items()
                if cached_state.cache_slot is not None
            )
        )
        waiting_req_ids = tuple(
            rid for rid in resident_req_ids if rid != active_req_id
        )

        self.hook.state_manager.set_context("teacache_positive")
        positive_state = self.hook.state_manager.get_state()
        residual_sum_before = 0.0
        if positive_state.previous_residual is not None:
            residual_sum_before = float(positive_state.previous_residual.sum().item())

        self.snapshots.append(
            _TeaCacheSnapshot(
                req_id=state.req_id,
                step_index=state.step_index,
                active_req_id=active_req_id,
                waiting_req_ids=waiting_req_ids,
                resident_req_ids=resident_req_ids,
                forward_cnt_before=self.hook._forward_cnt,
                positive_cnt_before=positive_state.cnt,
                residual_sum_before=residual_sum_before,
            )
        )

        if state.req_id in self.fail_req_ids:
            raise RuntimeError(f"boom-{state.req_id}")
        if state.req_id in self.none_req_ids:
            return None

        positive_state.cnt += 1
        positive_state.previous_modulated_input = torch.full(
            (2,), float(state.step_index + 1)
        )
        positive_state.previous_residual = torch.full(
            (2,), float(10 * positive_state.cnt)
        )
        positive_state.previous_residual_encoder = torch.full(
            (1,), float(100 * positive_state.cnt)
        )

        self.hook.state_manager.set_context("teacache_negative")
        negative_state = self.hook.state_manager.get_state()
        negative_state.cnt = positive_state.cnt
        negative_state.previous_modulated_input = torch.full(
            (1,), float(1000 + state.step_index)
        )
        negative_state.previous_residual = torch.full(
            (1,), float(2000 + state.step_index)
        )

        self.hook.state_manager.set_context("teacache_positive")
        self.hook._forward_cnt += 1
        return torch.tensor([float(positive_state.cnt)])


def _make_teacache_runner():
    pipeline = _TeaCachePipeline()
    return _build_runner(
        pipeline, TeaCacheStateDriver(pipeline), backend_name="tea_cache"
    )


# ---------------------------------------------------------------------------
# Tests — Cache-DiT single-request stepwise
# ---------------------------------------------------------------------------


def _new_sched(req, *, step_id: int = 0, finished_req_ids=None):
    return _make_scheduler_output(
        new_reqs=[req], step_id=step_id, finished_req_ids=finished_req_ids
    )


def _cached_sched(req_id: str, *, step_id: int = 0, finished_req_ids=None):
    return _make_scheduler_output(
        cached_req_ids=[req_id], step_id=step_id, finished_req_ids=finished_req_ids
    )


class TestExecuteStepwiseCacheDiTCachePool:
    def test_cache_dit_switching_keeps_acceleration_state(self):
        runner = _make_cache_dit_runner()

        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        first = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot

        second = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_b, step_id=1))
        third = DiffusionModelRunner.execute_stepwise(runner, _cached_sched("req-a", step_id=2))

        assert first.req_id == "req-a"
        assert first.step_index == 1
        assert first.finished is False
        assert second.req_id == "req-b"
        assert second.step_index == 1
        assert second.finished is False
        assert third.req_id == "req-a"
        assert third.step_index == 2
        assert third.finished is False

        assert runner.state_cache["req-a"].cache_slot is slot_a
        assert runner.state_cache["req-a"].cache_slot.payload["trace"] == [
            "req-a:0",
            "req-a:1",
        ]
        assert runner.state_cache["req-b"].cache_slot.payload["trace"] == ["req-b:0"]
        assert [
            snapshot.trace_before
            for snapshot in runner.pipeline.snapshots
            if snapshot.req_id == "req-a"
        ] == [(), ("req-a:0",)]
        assert runner.cache_manager.driver.initialize_history == [(1, 3), (2, 3)]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_hbm_can_hold_multiple_waiting_slots_while_one_request_runs(self):
        runner = _make_cache_dit_runner()

        for step_id, req_id in enumerate(("req-a", "req-b", "req-c")):
            output = DiffusionModelRunner.execute_stepwise(
                runner,
                _new_sched(_make_request(req_id, num_inference_steps=4), step_id=step_id),
            )
            assert output.finished is False

        snapshot = runner.pipeline.snapshots[-1]
        resident_slots = {
            req_id: state.cache_slot for req_id, state in runner.state_cache.items()
        }

        assert snapshot.req_id == "req-c"
        assert snapshot.active_req_id == "req-c"
        assert snapshot.waiting_req_ids == ("req-a", "req-b")
        assert snapshot.resident_req_ids == ("req-a", "req-b", "req-c")
        assert set(resident_slots) == {"req-a", "req-b", "req-c"}
        assert (
            len({slot.payload["slot_id"] for slot in resident_slots.values()}) == 3
        )
        assert all(slot.resident_bytes > 0 for slot in resident_slots.values())
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_finished_req_ids_free_resident_slot_before_next_run(self):
        runner = _make_cache_dit_runner()

        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot

        output = DiffusionModelRunner.execute_stepwise(
            runner,
            _new_sched(req_b, step_id=1, finished_req_ids={"req-a"}),
        )

        assert output.req_id == "req-b"
        assert "req-a" not in runner.state_cache
        assert slot_a.metadata == {}
        assert slot_a.resident_bytes == 0
        assert runner.cache_manager.driver.clear_history == [slot_a.payload["slot_id"]]
        assert runner.state_cache["req-b"].cache_slot.payload["trace"] == ["req-b:0"]

    def test_denoise_none_path_cleans_up_cache_slot(self):
        runner = _make_cache_dit_runner()
        runner.pipeline.none_req_ids.add("req-a")

        output = DiffusionModelRunner.execute_stepwise(
            runner,
            _new_sched(_make_request("req-a", num_inference_steps=3), step_id=0),
        )

        assert output.req_id == "req-a"
        assert output.step_index == 0
        assert output.finished is True
        assert output.result is not None
        assert output.result.error == "stepwise denoise returned None"
        assert "req-a" not in runner.state_cache
        assert runner.cache_manager.driver.deactivate_history == [1]
        assert runner.cache_manager.driver.clear_history == [1]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_pipeline_error_still_deactivates_active_slot(self):
        runner = _make_cache_dit_runner()
        runner.pipeline.fail_req_ids.add("req-a")

        with pytest.raises(RuntimeError, match="boom-req-a"):
            DiffusionModelRunner.execute_stepwise(
                runner,
                _new_sched(_make_request("req-a", num_inference_steps=3), step_id=0),
            )

        assert runner.cache_manager.driver.deactivate_history == [1]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None
        assert runner.state_cache == {}
        assert runner.input_batch is None

    def test_stepwise_cache_backend_no_longer_requires_experiment_cachepool(
        self, monkeypatch
    ):
        runner = _make_cache_dit_runner()
        monkeypatch.setattr(diffusion_experimental, "EXPERIMENT_CACHEPOOL", False)

        output = DiffusionModelRunner.execute_stepwise(
            runner,
            _new_sched(_make_request("req-a", num_inference_steps=3), step_id=0),
        )

        assert output.req_id == "req-a"
        assert output.finished is False
        assert output.step_index == 1


# ---------------------------------------------------------------------------
# Tests — CacheManager batch lifecycle
# ---------------------------------------------------------------------------


class TestCacheManagerBatchLifecycle:
    def test_activate_batch_mixes_restored_and_fresh_slots(self):
        driver = _FakeBatchCacheLifecycleDriver()
        manager = CacheManager(driver)

        old_slot = driver.create_empty_slot()
        old_slot.metadata["num_inference_steps"] = 4
        req_a = _make_cache_state("req-a", num_inference_steps=4, slot=old_slot)
        req_b = _make_cache_state("req-b", num_inference_steps=4)

        restored = manager.activate([req_a, req_b])

        assert restored == [True, False]
        assert manager._batch_active is True
        assert driver.install_batch_history == [("req-a", "req-b")]
        assert driver.initialize_history == [(req_b.cache_slot.payload["slot_id"], 4)]
        assert req_a.cache_slot is old_slot
        assert req_b.cache_slot is not None
        assert req_b.cache_slot.metadata["num_inference_steps"] == 4

        manager.deactivate([req_a, req_b])

        assert manager._batch_active is False
        assert driver.deactivate_batch_calls == 1

    def test_free_during_batch_activation_releases_only_target_slot(self):
        driver = _FakeBatchCacheLifecycleDriver()
        manager = CacheManager(driver)

        req_a = _make_cache_state("req-a", num_inference_steps=3)
        req_b = _make_cache_state("req-b", num_inference_steps=3)
        manager.activate([req_a, req_b])
        slot_b = req_b.cache_slot
        slot_a_id = req_a.cache_slot.payload["slot_id"]

        manager.free(req_a)

        assert req_a.cache_slot is None
        assert req_b.cache_slot is slot_b
        assert driver.clear_history == [slot_a_id]
        assert manager._batch_active is True

        manager.deactivate([req_a, req_b])

        assert manager._batch_active is False
        assert req_b.cache_slot is slot_b

    def test_single_item_list_uses_single_request_lifecycle(self):
        driver = _FakeBatchCacheLifecycleDriver()
        manager = CacheManager(driver)
        req = _make_cache_state("req-a", num_inference_steps=2)

        restored = manager.activate([req])

        assert restored == [False]
        assert manager._active_req_id == "req-a"
        assert manager._batch_active is False
        assert driver.install_batch_history == []
        assert req.cache_slot.payload["installed_single"] is True

        manager.deactivate([req])

        assert manager._active_req_id is None


# ---------------------------------------------------------------------------
# Tests — CacheDiTStateDriver batch install
# ---------------------------------------------------------------------------


class TestCacheDiTStateDriverBatchInstall:
    def test_install_batch_slots_preserves_request_order_for_each_handle(self):
        driver, backend, _, managers = _make_cache_dit_driver()
        req_a = _make_cache_state("req-a", num_inference_steps=3, rows=2)
        req_b = _make_cache_state("req-b", num_inference_steps=5, rows=1)

        _attach_fresh_slots(driver, (req_a, req_b))
        driver.install_batch_slots([req_a, req_b])

        manager_a, manager_b = managers
        payload_a = CacheDiTStateDriver._get_payload(req_a.cache_slot)
        payload_b = CacheDiTStateDriver._get_payload(req_b.cache_slot)
        assert manager_a._batch_contexts == [payload_a[0], payload_b[0]]
        assert manager_b._batch_contexts == [payload_a[1], payload_b[1]]
        assert manager_a._batch_row_counts == [2, 1]
        assert manager_b._batch_row_counts == [2, 1]
        assert manager_a._batch_row_offsets == [0, 2, 3]
        assert manager_b._batch_row_offsets == [0, 2, 3]
        assert manager_a._batch_contexts[0]["ctx_a"] is payload_a[0]["ctx_a"]
        assert manager_a._batch_contexts[1]["ctx_b"] is payload_b[0]["ctx_b"]
        assert manager_b._batch_contexts[0]["ctx_c"] is payload_a[1]["ctx_c"]
        assert backend.force_refresh_history == [3, 5]

    def test_deactivate_batch_slots_clears_batch_mode_state(self):
        driver, _, _, managers = _make_cache_dit_driver()
        req_a = _make_cache_state("req-a", num_inference_steps=3, rows=1)
        req_b = _make_cache_state("req-b", num_inference_steps=3, rows=1)

        _attach_fresh_slots(driver, (req_a, req_b))
        driver.install_batch_slots([req_a, req_b])
        for manager in managers:
            manager._current_context = object()

        driver.deactivate_batch_slots()

        for manager in managers:
            assert manager._batch_contexts is None
            assert manager._batch_row_counts is None
            assert manager._batch_row_offsets is None
            assert manager._current_context is None

    def test_initialize_fresh_slot_keeps_per_request_configs_isolated(self):
        driver, _, _, _ = _make_cache_dit_driver()
        req_a = _make_cache_state("req-a", num_inference_steps=3, rows=1)
        req_b = _make_cache_state("req-b", num_inference_steps=5, rows=1)

        _attach_fresh_slots(driver, (req_a, req_b))
        payload_a = CacheDiTStateDriver._get_payload(req_a.cache_slot)
        payload_b = CacheDiTStateDriver._get_payload(req_b.cache_slot)

        assert payload_a[0]["ctx_a"].config_steps == 3
        assert payload_a[0]["ctx_b"].config_steps == 3
        assert payload_a[1]["ctx_c"].config_steps == 3
        assert payload_b[0]["ctx_a"].config_steps == 5
        assert payload_b[0]["ctx_b"].config_steps == 5
        assert payload_b[1]["ctx_c"].config_steps == 5

        driver.install_batch_slots([req_a, req_b])

        assert payload_a[0]["ctx_a"].config_steps == 3
        assert payload_b[0]["ctx_a"].config_steps == 5


# ---------------------------------------------------------------------------
# Tests — batched forward (numerical equivalence with a serial loop)
# ---------------------------------------------------------------------------


class TestCacheDiTBatchedForward:
    @pytest.mark.parametrize("use_l1", [False, True])
    @pytest.mark.parametrize("cache_residual", [False, True])
    @pytest.mark.parametrize("with_encoder", [False, True])
    def test_forward_batched_base_all_miss_matches_serial(
        self,
        use_l1: bool,
        cache_residual: bool,
        with_encoder: bool,
    ):
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        encoder_hidden_states = None
        if with_encoder:
            encoder_hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 100

        base_contexts = [
            {
                "ctx": _BatchCacheContext(
                    name=f"req-{idx}",
                    cache_decision=False,
                    is_l1_enabled=use_l1,
                    cache_residual=cache_residual,
                    encoder_cache_residual=cache_residual,
                )
            }
            for idx in range(3)
        ]

        batch_contexts = _clone_batch_contexts(base_contexts)
        serial_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager(
            is_l1_enabled=use_l1,
            cache_residual=cache_residual,
            encoder_cache_residual=cache_residual,
        )
        serial_manager = _BatchContextManager(
            is_l1_enabled=use_l1,
            cache_residual=cache_residual,
            encoder_cache_residual=cache_residual,
        )
        batch_pattern = _FakeBatchedPatternBase(batch_manager)
        serial_pattern = _FakeBatchedPatternBase(serial_manager)

        set_batch_contexts(batch_manager, batch_contexts, [1, 1, 1])
        try:
            batch_out = _forward_batched_base(
                batch_pattern,
                hidden_states.clone(),
                None if encoder_hidden_states is None else encoder_hidden_states.clone(),
            )
        finally:
            clear_batch_contexts(batch_manager)

        serial_out = _run_serial_base(
            serial_pattern,
            serial_contexts,
            hidden_states.clone(),
            None if encoder_hidden_states is None else encoder_hidden_states.clone(),
        )

        assert torch.allclose(batch_out[0], serial_out[0])
        if encoder_hidden_states is None:
            assert batch_out[1] is None
            assert serial_out[1] is None
        else:
            assert torch.allclose(batch_out[1], serial_out[1])
        assert batch_pattern.fn_call_shapes == [(3, 2)]
        assert batch_pattern.mn_call_shapes == [(3, 2)]
        assert batch_pattern.bn_call_shapes == [(3, 2)]
        _assert_context_maps_match(batch_contexts, serial_contexts)

    def test_forward_batched_base_all_hit_skips_mn_and_matches_serial(self):
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        encoder_hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 100
        base_contexts = []
        for idx in range(3):
            context = _BatchCacheContext(name=f"req-{idx}", cache_decision=True)
            _seed_cached_buffers(
                context, hidden_shape=(1, 2), encoder_shape=(1, 2)
            )
            base_contexts.append({"ctx": context})

        batch_contexts = _clone_batch_contexts(base_contexts)
        serial_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager()
        serial_manager = _BatchContextManager()
        batch_pattern = _FakeBatchedPatternBase(batch_manager)
        serial_pattern = _FakeBatchedPatternBase(serial_manager)

        set_batch_contexts(batch_manager, batch_contexts, [1, 1, 1])
        try:
            batch_out = _forward_batched_base(
                batch_pattern,
                hidden_states.clone(),
                encoder_hidden_states.clone(),
            )
        finally:
            clear_batch_contexts(batch_manager)

        serial_out = _run_serial_base(
            serial_pattern,
            serial_contexts,
            hidden_states.clone(),
            encoder_hidden_states.clone(),
        )

        assert torch.allclose(batch_out[0], serial_out[0])
        assert torch.allclose(batch_out[1], serial_out[1])
        assert batch_pattern.fn_call_shapes == [(3, 2)]
        assert batch_pattern.mn_call_shapes == []
        assert batch_pattern.bn_call_shapes == [(3, 2)]
        _assert_context_maps_match(batch_contexts, serial_contexts)

    def test_forward_batched_base_partial_hit_and_independent_blockers(self):
        hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        base_contexts = [
            {"ctx": _BatchCacheContext(name="req-0", cache_decision=True, cache_residual=True)},
            {
                "ctx": _BatchCacheContext(
                    name="req-1",
                    cache_decision=True,
                    warmup_blocks_cache=True,
                    cache_residual=True,
                )
            },
            {"ctx": _BatchCacheContext(name="req-2", cache_decision=True, cache_residual=True)},
            {
                "ctx": _BatchCacheContext(
                    name="req-3",
                    cache_decision=True,
                    block_due_to_continuous_limit=True,
                    cache_residual=True,
                )
            },
            {
                "ctx": _BatchCacheContext(
                    name="req-4",
                    cache_decision=True,
                    block_due_to_accumulated_diff=True,
                    cache_residual=True,
                )
            },
        ]
        _seed_cached_buffers(base_contexts[0]["ctx"], hidden_shape=(1, 2), cache_residual=True)
        _seed_cached_buffers(base_contexts[2]["ctx"], hidden_shape=(1, 2), cache_residual=True)

        batch_contexts = _clone_batch_contexts(base_contexts)
        serial_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager(cache_residual=True)
        serial_manager = _BatchContextManager(cache_residual=True)
        batch_pattern = _FakeBatchedPatternBase(batch_manager)
        serial_pattern = _FakeBatchedPatternBase(serial_manager)

        set_batch_contexts(batch_manager, batch_contexts, [1, 1, 1, 1, 1])
        try:
            batch_out = _forward_batched_base(
                batch_pattern, hidden_states.clone(), None
            )
        finally:
            clear_batch_contexts(batch_manager)

        serial_out = _run_serial_base(
            serial_pattern, serial_contexts, hidden_states.clone(), None
        )

        assert torch.allclose(batch_out[0], serial_out[0])
        assert batch_out[1] is None
        assert batch_pattern.mn_call_shapes == [(3, 2)]
        assert [ctx["ctx"].cached_step_calls for ctx in batch_contexts] == [1, 0, 1, 0, 0]
        _assert_context_maps_match(batch_contexts, serial_contexts)

    def test_forward_batched_base_slices_batch_aligned_kwargs_for_compute_subset(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        encoder_hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100
        base_contexts = [
            {"ctx": _BatchCacheContext(name="req-0", cache_decision=True)},
            {"ctx": _BatchCacheContext(name="req-1", cache_decision=False)},
            {"ctx": _BatchCacheContext(name="req-2", cache_decision=True)},
            {"ctx": _BatchCacheContext(name="req-3", cache_decision=False)},
        ]
        _seed_cached_buffers(base_contexts[0]["ctx"], hidden_shape=(1, 2), encoder_shape=(1, 2))
        _seed_cached_buffers(base_contexts[2]["ctx"], hidden_shape=(1, 2), encoder_shape=(1, 2))

        class _KwargInspectPattern(_FakeBatchedPatternBase):
            def __init__(self, context_manager):
                super().__init__(context_manager)
                self.mn_kwargs = None

            def call_Mn_blocks(
                self, hidden_states, encoder_hidden_states, *args, **kwargs
            ):
                del args
                self.mn_kwargs = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items()
                }
                return super().call_Mn_blocks(
                    hidden_states, encoder_hidden_states, **kwargs
                )

        batch_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager()
        batch_pattern = _KwargInspectPattern(batch_manager)

        temb = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        modulate_index = torch.arange(8, dtype=torch.int64).reshape(4, 2)
        encoder_hidden_states_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )
        hidden_states_mask = torch.tensor(
            [
                [True, False],
                [True, True],
                [False, False],
                [False, True],
            ],
            dtype=torch.bool,
        )

        set_batch_contexts(batch_manager, batch_contexts, [1, 1, 1, 1])
        try:
            _forward_batched_base(
                batch_pattern,
                hidden_states.clone(),
                encoder_hidden_states.clone(),
                temb=temb,
                modulate_index=modulate_index,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                hidden_states_mask=hidden_states_mask,
            )
        finally:
            clear_batch_contexts(batch_manager)

        compute_index = torch.tensor([1, 3], dtype=torch.long)
        doubled_compute_index = torch.tensor([1, 3, 5, 7], dtype=torch.long)

        assert batch_pattern.mn_call_shapes == [(2, 2)]
        assert batch_pattern.mn_kwargs is not None
        assert torch.equal(batch_pattern.mn_kwargs["temb"], temb[doubled_compute_index])
        assert torch.equal(
            batch_pattern.mn_kwargs["modulate_index"], modulate_index[compute_index]
        )
        assert torch.equal(
            batch_pattern.mn_kwargs["encoder_hidden_states_mask"],
            encoder_hidden_states_mask[compute_index],
        )
        assert torch.equal(
            batch_pattern.mn_kwargs["hidden_states_mask"],
            hidden_states_mask[compute_index],
        )

    def test_forward_batched_base_trims_encoder_cache_to_request_seq_len(self):
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        encoder_hidden_states = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2) + 100
        base_contexts = [
            {
                "ctx": _BatchCacheContext(
                    name="req-0",
                    cache_decision=True,
                    encoder_cache_residual=True,
                )
            },
            {"ctx": _BatchCacheContext(name="req-1", cache_decision=False)},
        ]
        _seed_cached_buffers(
            base_contexts[0]["ctx"],
            hidden_shape=(1, 2),
            encoder_shape=(1, 3, 2),
            encoder_cache_residual=True,
        )

        batch_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager(encoder_cache_residual=True)
        batch_pattern = _FakeBatchedPatternBase(batch_manager)

        encoder_hidden_states_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ],
            dtype=torch.bool,
        )

        set_batch_contexts(batch_manager, batch_contexts, [1, 1])
        try:
            out_hs, out_enc = _forward_batched_base(
                batch_pattern,
                hidden_states.clone(),
                encoder_hidden_states.clone(),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
            )
        finally:
            clear_batch_contexts(batch_manager)

        expected_cached_prefix = encoder_hidden_states[0:1, :3] + 10 + 70 + 30
        expected_cached_tail = encoder_hidden_states[0:1, 3:] + 10 + 30

        assert out_hs.shape == hidden_states.shape
        assert out_enc.shape == encoder_hidden_states.shape
        assert torch.equal(out_enc[0:1, :3], expected_cached_prefix)
        assert torch.equal(out_enc[0:1, 3:], expected_cached_tail)

    def test_forward_batched_345_partial_hit_matches_serial_without_bn(self):
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        base_contexts = []
        for idx, can_cache in enumerate((True, False, True)):
            context = _BatchCacheContext(
                name=f"req-{idx}",
                cache_decision=can_cache,
                is_l1_enabled=True,
            )
            if can_cache:
                _seed_cached_buffers(
                    context, hidden_shape=(1, 2), encoder_shape=(1, 2)
                )
            base_contexts.append({"ctx": context})

        batch_contexts = _clone_batch_contexts(base_contexts)
        serial_contexts = _clone_batch_contexts(base_contexts)
        batch_manager = _BatchContextManager(is_l1_enabled=True, bn_compute_blocks=0)
        serial_manager = _BatchContextManager(is_l1_enabled=True, bn_compute_blocks=0)
        batch_pattern = _FakeBatchedPattern345(batch_manager)
        serial_pattern = _FakeBatchedPattern345(serial_manager)

        set_batch_contexts(batch_manager, batch_contexts, [1, 1, 1])
        try:
            batch_out = _forward_batched_345(batch_pattern, hidden_states.clone())
        finally:
            clear_batch_contexts(batch_manager)

        serial_out = _run_serial_345(
            serial_pattern, serial_contexts, hidden_states.clone()
        )

        assert torch.allclose(batch_out[0], serial_out[0])
        assert torch.allclose(batch_out[1], serial_out[1])
        assert batch_pattern.mn_call_shapes == [(1, 2)]
        assert batch_pattern.bn_call_shapes == []
        _assert_context_maps_match(batch_contexts, serial_contexts)


# ---------------------------------------------------------------------------
# Tests — Cache-DiT batched runner end-to-end
# ---------------------------------------------------------------------------


def _batch_sched(*, new_reqs=(), cached_req_ids=(), step_id: int = 0):
    return _make_scheduler_output(
        new_reqs=new_reqs, cached_req_ids=cached_req_ids, step_id=step_id
    )


class TestExecuteStepwiseCacheDiTBatching:
    def test_runner_batches_two_requests_across_multiple_steps(self):
        runner = _make_batch_cache_dit_runner()
        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        first = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(new_reqs=[req_a, req_b], step_id=0)
        )
        slot_a = runner.state_cache["req-a"].cache_slot
        slot_b = runner.state_cache["req-b"].cache_slot
        second = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(cached_req_ids=["req-a", "req-b"], step_id=1)
        )

        assert first.req_id == ["req-a", "req-b"]
        assert first.step_index == [1, 1]
        assert first.finished == [False, False]
        assert second.step_index == [2, 2]
        assert second.finished == [False, False]
        assert slot_a.payload["history"] == [("req-a", 0), ("req-a", 1)]
        assert slot_b.payload["history"] == [("req-b", 0), ("req-b", 1)]

        third = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(cached_req_ids=["req-a", "req-b"], step_id=2)
        )

        assert third.step_index == [3, 3]
        assert third.finished == [True, True]
        assert runner.state_cache == {}
        assert runner.pipeline.snapshots == [
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(0, 0),
                cache_decisions=(False, False),
                batch_active=True,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(1, 1),
                cache_decisions=(False, False),
                batch_active=True,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(2, 2),
                cache_decisions=(False, False),
                batch_active=True,
            ),
        ]
        assert runner.cache_manager.driver.install_batch_history == [
            ("req-a", "req-b"),
            ("req-a", "req-b"),
            ("req-a", "req-b"),
        ]
        assert runner.cache_manager.driver.deactivate_batch_calls == 3

    def test_runner_allows_mixed_cache_decisions_in_same_step(self):
        runner = _make_batch_cache_dit_runner()
        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)
        req_b.sampling_params.cache_plan = (0, 1)

        output = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(new_reqs=[req_a, req_b], step_id=0)
        )

        assert output.req_id == ["req-a", "req-b"]
        assert output.step_index == [1, 1]
        assert output.finished == [False, False]
        assert runner.pipeline.snapshots == [
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(0, 0),
                cache_decisions=(False, True),
                batch_active=True,
            )
        ]

    def test_runner_handles_dynamic_row_layout_when_request_finishes(self):
        runner = _make_batch_cache_dit_runner()
        req_a = _make_request("req-a", num_inference_steps=2)
        req_b = _make_request("req-b", num_inference_steps=3)
        req_c = _make_request("req-c", num_inference_steps=2)

        first = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(new_reqs=[req_a, req_b], step_id=0)
        )
        second = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(cached_req_ids=["req-a", "req-b"], step_id=1)
        )
        third = DiffusionModelRunner.execute_stepwise(
            runner,
            _batch_sched(new_reqs=[req_c], cached_req_ids=["req-b"], step_id=2),
        )
        fourth = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(cached_req_ids=["req-c"], step_id=3)
        )

        assert first.finished == [False, False]
        assert second.finished == [True, False]
        assert third.req_id == ["req-c", "req-b"]
        assert third.step_index == [1, 3]
        assert third.finished == [False, True]
        assert fourth.req_id == "req-c"
        assert fourth.step_index == 2
        assert fourth.finished is True
        assert runner.pipeline.snapshots == [
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(0, 0),
                cache_decisions=(False, False),
                batch_active=True,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-a", "req-b"),
                step_indices=(1, 1),
                cache_decisions=(False, False),
                batch_active=True,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-c", "req-b"),
                step_indices=(0, 2),
                cache_decisions=(False, False),
                batch_active=True,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-c",),
                step_indices=(1,),
                cache_decisions=(False,),
                batch_active=False,
            ),
        ]
        assert runner.pipeline.decode_calls == ["req-a", "req-b", "req-c"]
        assert runner.state_cache == {}

    def test_single_request_path_remains_non_batch(self):
        runner = _make_batch_cache_dit_runner()
        req = _make_request("req-a", num_inference_steps=2)
        first = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(new_reqs=[req], step_id=0)
        )
        second = DiffusionModelRunner.execute_stepwise(
            runner, _batch_sched(cached_req_ids=["req-a"], step_id=1)
        )

        assert first.req_id == "req-a"
        assert first.step_index == 1
        assert first.finished is False
        assert second.req_id == "req-a"
        assert second.step_index == 2
        assert second.finished is True
        assert runner.cache_manager.driver.install_batch_history == []
        assert runner.pipeline.snapshots == [
            _BatchRunnerSnapshot(
                req_ids=("req-a",),
                step_indices=(0,),
                cache_decisions=(False,),
                batch_active=False,
            ),
            _BatchRunnerSnapshot(
                req_ids=("req-a",),
                step_indices=(1,),
                cache_decisions=(False,),
                batch_active=False,
            ),
        ]


# ---------------------------------------------------------------------------
# Tests — TeaCache stepwise
# ---------------------------------------------------------------------------


class TestExecuteStepwiseTeaCachePool:
    def test_teacache_switching_restores_hook_state(self):
        runner = _make_teacache_runner()
        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        first = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot
        second = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_b, step_id=1))
        third = DiffusionModelRunner.execute_stepwise(runner, _cached_sched("req-a", step_id=2))

        assert first.finished is False
        assert second.finished is False
        assert third.finished is False
        assert third.step_index == 2

        payload_a = slot_a.payload
        assert sorted(payload_a["states"]) == ["teacache_negative", "teacache_positive"]
        assert payload_a["forward_cnt"] == 2
        assert payload_a["states"]["teacache_positive"].cnt == 2
        assert payload_a["states"]["teacache_negative"].cnt == 2
        assert runner.state_cache["req-b"].cache_slot.payload["forward_cnt"] == 1
        assert (
            runner.state_cache["req-b"].cache_slot.payload["states"]["teacache_positive"].cnt
            == 1
        )

        req_a_snapshots = [
            snapshot
            for snapshot in runner.pipeline.snapshots
            if snapshot.req_id == "req-a"
        ]
        assert [snapshot.positive_cnt_before for snapshot in req_a_snapshots] == [0, 1]
        assert [snapshot.forward_cnt_before for snapshot in req_a_snapshots] == [0, 1]
        assert req_a_snapshots[1].residual_sum_before == pytest.approx(20.0)
        assert runner.pipeline.hook._forward_cnt == 0
        assert runner.pipeline.hook.state_manager._states == {}
        assert runner.cache_manager._active_req_id is None

    def test_runner_interleaves_two_teacache_requests_end_to_end(self):
        runner = _make_teacache_runner()
        req_a = _make_request("req-a", num_inference_steps=2)
        req_b = _make_request("req-b", num_inference_steps=2)

        first = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot
        second = DiffusionModelRunner.execute_stepwise(runner, _new_sched(req_b, step_id=1))
        slot_b = runner.state_cache["req-b"].cache_slot
        third = DiffusionModelRunner.execute_stepwise(runner, _cached_sched("req-a", step_id=2))
        fourth = DiffusionModelRunner.execute_stepwise(runner, _cached_sched("req-b", step_id=3))

        assert first.finished is False
        assert second.finished is False
        assert third.finished is True
        assert fourth.finished is True
        assert third.result is not None
        assert fourth.result is not None
        assert torch.equal(third.result.output, torch.tensor([2.0]))
        assert torch.equal(fourth.result.output, torch.tensor([2.0]))

        assert runner.state_cache == {}
        assert runner.pipeline.decode_calls == ["req-a", "req-b"]
        assert slot_a.metadata == {}
        assert slot_b.metadata == {}
        assert slot_a.payload["states"] == {}
        assert slot_b.payload["states"] == {}
        assert slot_a.payload["forward_cnt"] == 0
        assert slot_b.payload["forward_cnt"] == 0
        assert slot_a.resident_bytes == 0
        assert slot_b.resident_bytes == 0
        assert runner.pipeline.hook._forward_cnt == 0
        assert runner.pipeline.hook.state_manager._states == {}
