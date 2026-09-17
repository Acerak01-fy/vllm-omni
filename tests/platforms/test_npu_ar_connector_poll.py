# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm_ascend")

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_omni.core.prefix_cache.runner_mixin import PrefixCacheRunnerMixin
from vllm_omni.platforms.npu.worker import npu_ar_model_runner as module

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


@pytest.fixture
def poll_runner(monkeypatch):
    monkeypatch.setattr(module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: Mock())
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=0),
        speculative_config=None,
        omni_prefix_cache=None,
        _omni_prefix_cache_cfg=None,
        parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
        kv_extracted_req_ids=None,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False)),
        ascend_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False))
        ),
        execute_model_state=None,
        _warmup_state_cleared=True,
        model=SimpleNamespace(),
        kv_transfer_manager=SimpleNamespace(handle_finished_requests_kv_transfer=Mock(return_value=None)),
        kv_caches={},
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        _resolve_global_request_id=Mock(),
        kv_connector_no_forward=Mock(return_value=SimpleNamespace(finished_sending={"source"})),
        attach_omni_connector_output=lambda output: output,
        synchronize_input_prep=Mock(side_effect=RuntimeError("synchronized path")),
        _update_states=Mock(),
    )
    runner._prefix_cache_step_begin = MethodType(PrefixCacheRunnerMixin._prefix_cache_step_begin, runner)
    runner._is_connector_only_step = MethodType(module.NPUARModelRunner._is_connector_only_step, runner)
    return runner


@pytest.mark.parametrize("sender_placeholder", [False, True])
def test_idle_poll_still_returns_connector_completion_without_input_sync(poll_runner, sender_placeholder):
    if sender_placeholder:
        poll_runner._omni_connector = None
        poll_runner._async_chunk = False
        poll_runner._pending_full_payload_send = {}
        poll_runner.recv_full_payload_inputs = Mock()
    schedule = SchedulerOutput.make_empty()
    result = module.NPUARModelRunner.execute_model(poll_runner, schedule)
    assert result.finished_sending == {"source"}
    poll_runner.kv_connector_no_forward.assert_called_once_with(schedule, poll_runner.vllm_config)
    poll_runner.synchronize_input_prep.assert_not_called()
    poll_runner._update_states.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "total_num_scheduled_tokens",
        "scheduled_new_reqs",
        "num_scheduled_tokens",
        "finished_req_ids",
        "scheduled_spec_decode_tokens",
        "scheduled_encoder_inputs",
        "free_encoder_mm_hashes",
        "new_block_ids_to_zero",
        "kv_cache_block_copies",
        "ec_connector_metadata",
        "ec_manager_metadata",
        "finished_requests_needing_kv_transfer",
        "pending_input_registrations",
    ],
)
def test_state_changes_never_use_poll_fast_path(poll_runner, field):
    schedule = SchedulerOutput.make_empty()
    setattr(schedule, field, 1 if field == "total_num_scheduled_tokens" else ["update"])
    assert not poll_runner._is_connector_only_step(schedule)


@pytest.mark.parametrize("field", ["req_ids", "resumed_req_ids"])
def test_cached_request_updates_require_input_preparation(poll_runner, field):
    schedule = SchedulerOutput.make_empty()
    setattr(schedule.scheduled_cached_reqs, field, ["request"])
    assert not poll_runner._is_connector_only_step(schedule)


def test_first_empty_round_still_removes_previous_batch_under_sync(poll_runner):
    poll_runner.input_batch.num_reqs = 1
    with pytest.raises(RuntimeError, match="synchronized path"):
        module.NPUARModelRunner.execute_model(poll_runner, SchedulerOutput.make_empty())
    poll_runner.kv_connector_no_forward.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("parallel_config.data_parallel_size", 2),
        ("parallel_config.pipeline_parallel_size", 2),
        ("_omni_connector", object()),
        ("_async_chunk", True),
        ("_pending_full_payload_send", {"request": object()}),
    ],
)
def test_coordination_and_payload_work_keep_original_path(poll_runner, field, value):
    owner = poll_runner
    *parents, field = field.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    setattr(owner, field, value)
    assert not poll_runner._is_connector_only_step(SchedulerOutput.make_empty())


def test_no_connector_keeps_original_path(poll_runner, monkeypatch):
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    assert not poll_runner._is_connector_only_step(SchedulerOutput.make_empty())
