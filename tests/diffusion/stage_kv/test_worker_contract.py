# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.diffusion.stage_kv.test_interface import make_init_config, make_metadata
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.stage_kv.interface import StageKVWorkerInitResult
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def make_runner() -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.stage_kv_init_config = None
    return runner


def test_model_runner_initialization_is_validation_only_and_idempotent() -> None:
    runner = make_runner()
    config = make_init_config()

    first = runner.initialize_stage_kv(config)
    second = runner.initialize_stage_kv(config)

    assert first == second
    assert runner.stage_kv_init_config is config
    assert not hasattr(runner, "kv_caches")
    assert not hasattr(runner, "block_tables")


def test_model_runner_requires_metadata_only_for_paged_new_requests() -> None:
    runner = make_runner()

    runner.install_stage_kv_metadata(new_request_ids=["dense"], metadata_by_request={})
    with pytest.raises(RuntimeError, match="before paged Worker initialization"):
        runner.install_stage_kv_metadata(
            new_request_ids=["dense"],
            metadata_by_request={"dense": make_metadata(request_id="dense")},
        )

    runner.initialize_stage_kv(make_init_config())
    with pytest.raises(RuntimeError, match="missing=\\['req-0'\\]"):
        runner.install_stage_kv_metadata(new_request_ids=["req-0"], metadata_by_request={})

    runner.install_stage_kv_metadata(
        new_request_ids=["req-0"],
        metadata_by_request={"req-0": make_metadata()},
    )
    # A cached step carries no new-request allocation metadata.
    runner.install_stage_kv_metadata(new_request_ids=[], metadata_by_request={})


def test_request_executor_keeps_prepared_layout_and_allocation_metadata_separate() -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace()
    executor._ensure_open = lambda: None
    calls = []

    def collective_rpc(method, *, args, unique_reply_rank, exec_all_ranks):
        calls.append((method, args, unique_reply_rank, exec_all_ranks))
        return DiffusionOutput(output=None)

    executor.collective_rpc = collective_rpc
    prepared_layout = object()
    allocation_metadata = make_metadata()
    req = SimpleNamespace(request_id="req-0", prepared_layout=prepared_layout)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id="req-0", req=req)],
        kv_prefetch_job=None,
        stage_kv_metadata={"req-0": allocation_metadata},
    )

    result = executor.execute_request(scheduler_output)

    assert result.request_ids == ["req-0"]
    rpc_req = calls[0][1][0]
    rpc_metadata = calls[0][1][3]
    assert rpc_req is req
    assert rpc_req.prepared_layout is prepared_layout
    assert rpc_metadata is allocation_metadata
    assert calls == [
        (
            "execute_model",
            (req, executor.od_config, None, allocation_metadata),
            0,
            True,
        )
    ]


def test_executor_initialization_uses_all_worker_control_rpc() -> None:
    config = make_init_config()
    expected = StageKVWorkerInitResult(
        cache_mode=config.cache_mode,
        cache_layout_fingerprint=config.cache_layout_fingerprint,
        num_blocks=config.kv_cache_config.num_blocks,
        physical_layout=config.physical_layout,
    )
    calls = []

    class Executor:
        def collective_rpc(
            self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None, exec_all_ranks=False
        ):
            calls.append((method, args, unique_reply_rank, exec_all_ranks))
            return [expected]

    result = DiffusionExecutor.initialize_stage_kv(Executor(), config)

    assert result is expected
    assert calls == [("initialize_stage_kv", (config,), None, False)]


def test_dense_request_executor_keeps_legacy_rpc_shape() -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace()
    executor._ensure_open = lambda: None
    calls = []

    def collective_rpc(method, *, args, unique_reply_rank, exec_all_ranks):
        calls.append((method, args, unique_reply_rank, exec_all_ranks))
        return DiffusionOutput(output=None)

    executor.collective_rpc = collective_rpc
    req = SimpleNamespace(request_id="dense")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id="dense", req=req)],
        kv_prefetch_job=None,
        stage_kv_metadata={},
    )

    executor.execute_request(scheduler_output)

    assert calls[0][1] == (req, executor.od_config, None)


def test_batch_and_step_paths_install_scheduler_metadata_before_forward() -> None:
    metadata = make_metadata()
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[
            SimpleNamespace(
                request_id="req-0",
                req=SimpleNamespace(request_id="req-0"),
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
        stage_kv_metadata={"req-0": metadata},
    )
    calls = []

    class Runner:
        pipeline = object()
        od_config = SimpleNamespace()

        def install_stage_kv_metadata(self, *, new_request_ids, metadata_by_request):
            calls.append((new_request_ids, metadata_by_request))

        def _execute_request_list(self, *args, **kwargs):
            return "batch-output"

        def _supports_step_mode(self):
            return False

    runner = Runner()
    assert DiffusionModelRunner.execute_model_batch(runner, scheduler_output, SimpleNamespace()) == "batch-output"
    with pytest.raises(ValueError, match="does not support step execution"):
        DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert calls == [
        (["req-0"], {"req-0": metadata}),
        (["req-0"], {"req-0": metadata}),
    ]


def test_engine_initializes_worker_contract_after_scheduler_provider() -> None:
    config = make_init_config()
    events = []

    class Scheduler:
        def initialize(self, od_config):
            events.append(("scheduler", od_config))

        def get_stage_kv_worker_init_config(self):
            events.append(("provider", None))
            return config

    class Executor:
        def initialize_stage_kv(self, received):
            events.append(("worker", received))
            return StageKVWorkerInitResult(
                cache_mode=received.cache_mode,
                cache_layout_fingerprint=received.cache_layout_fingerprint,
                num_blocks=received.kv_cache_config.num_blocks,
                physical_layout=received.physical_layout,
            )

    engine = object.__new__(DiffusionEngine)
    engine.executor = Executor()
    od_config = SimpleNamespace()

    engine._init_scheduler(od_config, Scheduler())

    assert events == [
        ("scheduler", od_config),
        ("provider", None),
        ("worker", config),
    ]


def test_engine_dense_scheduler_has_no_worker_init_side_effect() -> None:
    events = []

    class Scheduler:
        def initialize(self, od_config):
            events.append(("scheduler", od_config))

    class Executor:
        def initialize_stage_kv(self, config):
            events.append(("unexpected", config))

    engine = object.__new__(DiffusionEngine)
    engine.executor = Executor()
    od_config = SimpleNamespace()

    engine._init_scheduler(od_config, Scheduler())

    assert events == [("scheduler", od_config)]
