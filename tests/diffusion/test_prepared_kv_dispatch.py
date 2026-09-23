# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker import diffusion_worker as worker_module

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setattr(worker_module, "_all_gather_rank_values", lambda value: [value])
    config = SimpleNamespace(step_execution=False, parallel_config=SimpleNamespace(data_parallel_size=1), num_gpus=1)
    worker = object.__new__(worker_module.DiffusionWorker)
    worker.od_config = config
    worker.device = "test-device"
    monkeypatch.setattr(worker_module.current_omni_platform, "set_device", Mock())
    monkeypatch.setattr(
        worker_module.torch,
        "get_device_module",
        Mock(return_value=SimpleNamespace(current_stream=Mock(return_value=object()), stream=lambda _: nullcontext())),
    )
    worker._prepared_execution_lock = threading.Lock()
    worker._prepared_execution = None
    worker.model_runner = Mock()
    worker.model_runner.prepare_kv_for_forward.return_value = KVConnectorOutput(finished_recving={"A/kv"})
    worker.execute_model = Mock(return_value=DiffusionOutput(output=None))
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = config
    executor._closed = executor._is_failed = False
    executor._result_mq = object()
    executor._broadcast_mq = object()
    executor._rpc_id_lock = threading.Lock()
    executor._rpc_id_counter = 0
    executor.collective_rpc = Mock(
        side_effect=lambda method, args=(), kwargs=None, **_: getattr(worker, method)(*args, **(kwargs or {}))
    )
    request = NewRequestData("A", SimpleNamespace(request_id="A"), DiffusionKVMetadata("A", 7, ()))
    scheduled = DiffusionSchedulerOutput(
        step_id=1,
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
        kv_connector_metadata=object(),
        kv_transfer_request_ids={"A/kv"},
    )
    return executor, worker, scheduled


def test_preparation_needs_explicit_single_use_permit(dispatch):
    executor, worker, scheduled = dispatch
    output = executor.prepare_kv_for_forward(scheduled)
    assert output.finished_recving == {"A/kv"}
    worker.execute_model.assert_not_called()
    permit = executor._prepared_kv_request[0]
    result = executor.execute_request(scheduled)
    assert len(result.runner_outputs) == 1
    assert executor.collective_rpc.call_args.args == ("execute_prepared_model",)
    assert executor.collective_rpc.call_args.kwargs["args"] == (permit,)
    new_req = scheduled.scheduled_new_reqs[0]
    worker.execute_model.assert_called_once_with(
        new_req.req, worker.od_config, diffusion_kv_metadata=new_req.diffusion_kv_metadata
    )
    with pytest.raises(RuntimeError, match="stale"):
        worker.execute_prepared_model(permit)
    assert worker.execute_model.call_count == 1


@pytest.mark.parametrize("device", ["npu:1", "npu:3", "cuda:2"])
def test_execution_thread_restores_device_and_callers_stream(dispatch, monkeypatch, device):
    executor, worker, scheduled = dispatch
    worker.device = device
    local = threading.local()
    caller = threading.get_ident()
    original_stream = object()

    def capture_stream(actual_device):
        assert actual_device == device and threading.get_ident() == caller
        return original_stream

    def set_device(actual_device):
        assert threading.get_ident() != caller
        local.device = actual_device

    @contextmanager
    def stream_context(stream):
        assert local.device == device and stream is original_stream
        local.stream = stream
        try:
            yield
        finally:
            local.stream = None

    def compute(*args, **kwargs):
        assert local.device == device and local.stream is original_stream
        assert threading.get_ident() != caller
        return DiffusionOutput(output=None)

    module = SimpleNamespace(current_stream=capture_stream, stream=stream_context)
    monkeypatch.setattr(worker_module.torch, "get_device_module", lambda _: module)
    monkeypatch.setattr(worker_module.current_omni_platform, "set_device", set_device)
    worker.execute_model.side_effect = compute
    executor.prepare_kv_for_forward(scheduled)
    worker.execute_model.assert_not_called()
    result = executor.execute_request(scheduled)
    assert result.runner_outputs[0].result.error is None
    worker.execute_model.assert_called_once()


@pytest.mark.parametrize("action", ["cancel", "timeout"])
def test_engine_filters_terminal_request_before_permit(dispatch, action):
    executor, worker, scheduled = dispatch
    if action == "timeout":
        worker.model_runner.prepare_kv_for_forward.return_value = KVConnectorOutput()
    terminal = False

    def fail_incomplete(ids):
        nonlocal terminal
        if ids:
            terminal = True
        return {"A"} if terminal else set()

    engine = object.__new__(DiffusionEngine)
    engine.executor = executor
    engine.abort_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        update_kv_connector_output=Mock(),
        fail_incomplete_kv_loads=fail_incomplete,
        get_request_state=lambda _: SimpleNamespace(is_finished=lambda: terminal),
        completed_kv_drains=lambda: set(),
    )

    def cancel():
        nonlocal terminal
        engine.abort_queue.get_nowait()
        terminal = True
        executor.remove_diffusion_kv_requests([("A", 7)])

    engine._process_aborts_queue = cancel
    if action == "cancel":
        engine.abort_queue.put("A")
    engine._prepare_kv_for_forward(scheduled)
    assert scheduled.scheduled_request_ids == []
    assert scheduled.finished_req_ids == {"A"}
    executor.execute_request(scheduled)
    worker.execute_model.assert_not_called()
    assert all(call.args[0] != "execute_prepared_model" for call in executor.collective_rpc.call_args_list)


@pytest.mark.parametrize("target,cleared", [("A", True), (("A", 7), True), (("A", 6), False), ("B", False)])
def test_cleanup_respects_allocation_generation(dispatch, target, cleared):
    executor, worker, scheduled = dispatch
    executor.prepare_kv_for_forward(scheduled)
    executor.remove_diffusion_kv_requests([target])
    assert (worker._prepared_kv_request is None) == cleared
    assert (executor._prepared_kv_request is None) == cleared
    worker.execute_model.assert_not_called()


@pytest.mark.parametrize("remote_failure", [False, True])
def test_bad_permit_on_any_rank_prevents_computation(dispatch, monkeypatch, remote_failure):
    executor, worker, scheduled = dispatch
    executor.prepare_kv_for_forward(scheduled)
    permit = executor._prepared_kv_request[0]
    if remote_failure:
        monkeypatch.setattr(
            worker_module, "_all_gather_rank_values", lambda value: [value, (False, "stale remote permit")]
        )
    with pytest.raises(RuntimeError, match="stale"):
        worker.execute_prepared_model(permit if remote_failure else "wrong")
    worker.execute_model.assert_not_called()
    assert worker._prepared_kv_request is None


@pytest.mark.parametrize("failure", ["incomplete_rank", "invalid_block", "local_exception"])
def test_failed_receive_never_starts_compute(dispatch, monkeypatch, failure):
    executor, worker, scheduled = dispatch
    if failure == "incomplete_rank":
        executor.od_config.num_gpus = 2
    elif failure == "invalid_block":
        worker.model_runner.prepare_kv_for_forward.return_value = KVConnectorOutput(invalid_block_ids={1})
    else:
        worker.model_runner.prepare_kv_for_forward.side_effect = RuntimeError("receive failed")
    with pytest.raises(RuntimeError):
        executor.prepare_kv_for_forward(scheduled)
    worker.execute_model.assert_not_called()


@pytest.mark.parametrize("mode", ["step", "dp", "batch", "prefetch", "no_kv"])
def test_unsupported_modes_keep_existing_dispatch(dispatch, mode):
    executor, worker, scheduled = dispatch
    if mode == "step":
        executor.od_config.step_execution = True
    elif mode == "dp":
        executor.od_config.parallel_config.data_parallel_size = 2
    elif mode == "batch":
        scheduled.scheduled_new_reqs *= 2
    elif mode == "prefetch":
        scheduled.kv_prefetch_job = object()
    else:
        scheduled.kv_connector_metadata = None
    executor.prepare_kv_for_forward(scheduled)
    assert executor._prepared_kv_request is None
    worker.execute_model.assert_not_called()


def test_permit_uses_existing_async_compute_done_route(dispatch):
    from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind

    executor, _, _ = dispatch
    executor._futures_lock = threading.Lock()
    executor._rpc_futures = {}
    completed = AsyncDiffusionOutput(kind=AsyncOutputKind.COMPUTE_DONE, async_output_id="image-A")

    def reply(message):
        assert message["method"] == "execute_prepared_model"
        assert message["args"] == ("permit",)
        executor._rpc_futures.pop(message["rpc_id"]).set_result(completed)

    executor._broadcast_mq = SimpleNamespace(enqueue=reply)
    assert (
        MultiprocDiffusionExecutor.collective_rpc(
            executor, "execute_prepared_model", args=("permit",), unique_reply_rank=0, exec_all_ranks=True
        )
        is completed
    )
