# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vllm.v1.worker.gpu.kv_connector as active_module

from vllm_omni.diffusion.diffusion_kv.kv_connector import commit_kv_load, wait_for_kv_load
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.platforms.npu.worker import npu_diffusion_kv_connector as npu_module

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


@pytest.mark.parametrize("reusable_tokens", [0, 5])
def test_native_mooncake_receives_pages_without_extending_reusable_prefix(reusable_tokens):
    native = pytest.importorskip("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector")
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler = object.__new__(native.MooncakeConnectorScheduler)
    scheduler._reqs_in_batch = set()
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    request = SimpleNamespace(
        request_id="cfg0",
        num_tokens=16,
        num_computed_tokens=0,
        kv_transfer_params={
            "num_transfer_tokens": 9,
            "do_remote_prefill": True,
            "remote_block_ids": ([10, 11, 12],),
            "remote_engine_id": "ar",
            "remote_request_id": "source",
            "remote_host": "localhost",
            "remote_port": 10000,
            "remote_block_size": 4,
        },
    )
    blocks = KVCacheBlocks(([SimpleNamespace(block_id=i, block_hash=None, is_null=False) for i in range(1, 5)],))
    manager = SimpleNamespace(
        get_blocks=lambda _: blocks,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))]),
    )
    commit_kv_load(scheduler, manager, (request,), [reusable_tokens])
    metadata = scheduler.build_connector_meta(None).requests["cfg0"]
    worker = object.__new__(native.MooncakeConnectorWorker)
    worker.block_size = 4
    worker.block_size_scale = [[1]]
    local, remote = worker._get_kernel_block_ids(
        [0], metadata, 0, {"kv_cache_spec_type": "FullAttentionSpec", "kv_cache_group_id": 0}
    )
    assert local == [1, 2, 3]
    assert remote == [10, 11, 12]
    assert request.num_computed_tokens == reusable_tokens


@pytest.fixture
def connector(monkeypatch):
    native = Mock()
    native.get_finished.return_value = (set(), set())
    native.get_block_ids_with_load_errors.return_value = set()
    monkeypatch.setattr(npu_module, "get_kv_transfer_group", lambda: native)
    monkeypatch.setattr(active_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(active_module, "get_forward_context", lambda: None)
    caches = {"layer": object()}
    active = npu_module.NPUActiveKVConnector(object(), caches)
    native.register_kv_caches.assert_called_once_with(caches)
    native.set_host_xfer_buffer_ops.assert_not_called()
    return active, native


@pytest.mark.parametrize("invalid_pages", [False, True])
def test_load_completion_and_errors_use_shared_lifecycle(connector, invalid_pages):
    active, native = connector
    native.get_finished.side_effect = [(set(), {"req/kv/0"}), (set(), set())]
    native.get_block_ids_with_load_errors.return_value = {3} if invalid_pages else set()
    scheduler = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=0,
        num_waiting_reqs=0,
        kv_connector_metadata=object(),
        kv_transfer_request_ids={"req/kv/0"},
        kv_finished_request_ids={"previous/kv/0"},
    )
    if invalid_pages:
        with pytest.raises(RuntimeError, match="invalid remote pages"):
            wait_for_kv_load(active, scheduler, timeout=1)
    else:
        assert wait_for_kv_load(active, scheduler, timeout=1).finished_recving == {"req/kv/0"}
        native.wait_for_save.assert_called_once()
        native.clear_connector_metadata.assert_called_once()
    native.bind_connector_metadata.assert_called_once_with(scheduler.kv_connector_metadata)
    native.start_load_kv.assert_called_once_with(None)


def _source_meta(request_id="source", *, virtual=False):
    return SimpleNamespace(
        do_virtual=virtual,
        remote_engine_id="ar-engine",
        remote_request_id=request_id,
        remote_pcp_size=1,
        remote_dcp_size=1,
        remote_ptp_size=4,
        remote_port=10000,
        remote_host="ar-host",
        remote_multi_nodes_meta_mapping={},
    )


def _fanout_worker(receiver, ranks):
    worker_type = type("MooncakeConnectorWorker", (), {})
    worker_type.__module__ = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"
    worker = worker_type()
    worker.kv_recv_thread = receiver
    worker._is_hma_required = False
    worker.pcp_size = worker.dcp_size = 1
    worker._prefill_tp_size = 4
    worker._get_remote_ranks_for_req = Mock(return_value=ranks)
    worker._get_remote_host_info_by_port = Mock(return_value=("ar-host", "ar-engine"))
    return worker


@pytest.mark.parametrize(
    "ranks,expected",
    [
        ([[0], [1], [2], [3]], {10000: 2, 10001: 2, 10002: 2, 10003: 2}),
        ([[0], [0], [1], [1]], {10000: 4, 10001: 4}),
        ([[0, 1], [2, 3]], {10000: 2, 10001: 2, 10002: 2, 10003: 2}),
        ([[0]], None),
    ],
)
def test_cfg_ack_counts_include_all_consumers_and_tp_ranks(ranks, expected):
    receiver = Mock()
    worker = _fanout_worker(receiver, ranks)
    metadata = SimpleNamespace(requests={"cfg0": _source_meta(), "cfg1": _source_meta()})
    if expected is None:
        metadata.requests = {"a": _source_meta("a"), "virtual": _source_meta("a", virtual=True), "b": _source_meta("b")}
    adapter = npu_module._MooncakeReceiveFanout(worker, metadata, receiver)
    for request_id in metadata.requests:
        adapter.add_request(request_id=request_id, remote_port_send_num=None, local_block_ids=([5],))
    for call in receiver.add_request.call_args_list:
        counts = call.kwargs["remote_port_send_num"]
        if expected is None:
            assert counts is None
        else:
            assert {port: value["num"] for port, value in counts.items()} == expected
            assert all(value["host"] == "ar-host" for value in counts.values())
        assert call.kwargs["local_block_ids"] == ([5],)
    assert adapter.task_tracker is receiver.task_tracker


@pytest.mark.parametrize("fail", [False, True])
def test_fanout_submission_restores_native_receiver(connector, fail):
    active, native = connector
    receiver = Mock()
    worker = _fanout_worker(receiver, [[0]])
    native.connector_worker = worker
    metadata = SimpleNamespace(requests={"cfg0": _source_meta(), "cfg1": _source_meta()})

    def submit(_):
        worker.kv_recv_thread.add_request(request_id="cfg0", remote_port_send_num=None)
        if fail:
            raise RuntimeError("submission failed")
        worker.kv_recv_thread.add_request(request_id="cfg1", remote_port_send_num=None)

    native.start_load_kv.side_effect = submit
    scheduler = SimpleNamespace(kv_connector_metadata=metadata, has_sync_kv_loads=True)
    if fail:
        with pytest.raises(RuntimeError, match="submission failed"):
            active.pre_forward(scheduler)
    else:
        active.pre_forward(scheduler)
    assert worker.kv_recv_thread is receiver
    assert receiver.add_request.call_args_list[0].kwargs["remote_port_send_num"][10000]["num"] == 2


@pytest.mark.parametrize("fail_second", [False, True])
def test_native_context_groups_without_ep(monkeypatch, fail_second):
    import vllm.distributed.parallel_state as native_state

    import vllm_omni.diffusion.distributed.parallel_state as diffusion_state

    monkeypatch.setattr(native_state, "_PCP", None)
    monkeypatch.setattr(native_state, "_DCP", None)
    monkeypatch.setattr(diffusion_state, "get_world_group", lambda: SimpleNamespace(world_size=4, local_rank=2))
    monkeypatch.setattr(native_state, "_WORLD", None)
    created = []

    def create(**kwargs):
        assert kwargs["group_ranks"] == [[0], [1], [2], [3]]
        assert kwargs["local_rank"] == 2
        assert kwargs["backend"] == "hccl"
        if fail_second and kwargs["group_name"] == "dcp":
            raise RuntimeError("group creation failed")
        group = SimpleNamespace(world_size=1, destroy=Mock())
        created.append(group)
        return group

    monkeypatch.setattr(diffusion_state, "init_vllm_model_parallel_group", create)
    config = SimpleNamespace(
        kv_transfer_config=object(),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="group creation failed"):
            npu_module.ensure_native_kv_context_groups(config)
        assert native_state._PCP is None and native_state._DCP is None
        created[0].destroy.assert_called_once()
    else:
        npu_module.ensure_native_kv_context_groups(config)
        npu_module.ensure_native_kv_context_groups(config)
        assert len(created) == 2
        assert native_state._PCP is created[0]
        assert native_state._DCP is created[1]


def test_native_context_groups_do_not_touch_legacy(monkeypatch):
    import vllm_omni.diffusion.distributed.parallel_state as diffusion_state

    unexpected = Mock(side_effect=AssertionError("legacy must not create groups"))
    monkeypatch.setattr(diffusion_state, "get_world_group", unexpected)
    npu_module.ensure_native_kv_context_groups(SimpleNamespace(kv_transfer_config=None))
    unexpected.assert_not_called()
