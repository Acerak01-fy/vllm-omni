# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend cache registration with the shared vLLM active KV lifecycle."""

from collections import Counter, defaultdict
from typing import Any

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


class NPUActiveKVConnector(ActiveKVConnector):
    """Reuse the initialized worker and upstream metadata/completion handling.

    Native Ascend Mooncake registers device pages directly. It does not need
    the CUDA host-buffer copy callback installed by the upstream constructor.
    The worker's existing initialization/shutdown owns the transfer singleton.
    """

    def __init__(self, vllm_config: Any, kv_caches: Any) -> None:
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        self.kv_connector.register_kv_caches(kv_caches)
        self._pending_load_start = False
        self._disabled = False

    def pre_forward(self, scheduler_output: Any) -> None:
        # Ascend's default TP-only protocol acknowledges each local sequence
        # as a complete source request. CFG has multiple local sequences that
        # consume the same source pages, so use its existing counted-ACK wire
        # protocol to retain those pages until every consumer has read them.
        worker = getattr(self.kv_connector, "connector_worker", None)
        if type(worker).__module__ != "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector":
            return super().pre_forward(scheduler_output)
        receiver = worker.kv_recv_thread
        metadata = scheduler_output.kv_connector_metadata
        if receiver is None or metadata is None:
            return super().pre_forward(scheduler_output)
        worker.kv_recv_thread = _MooncakeReceiveFanout(worker, metadata, receiver)
        try:
            return super().pre_forward(scheduler_output)
        finally:
            # Only synchronous task submission needs the adapter. Each queued
            # native task owns its count map; native threads and data movement
            # are unchanged, including failure-path completion notifications.
            worker.kv_recv_thread = receiver


class _MooncakeReceiveFanout:
    """Supply native ACK counts for one atomically scheduled CFG handoff."""

    def __init__(self, worker: Any, metadata: Any, receiver: Any) -> None:
        self._receiver = receiver
        self._counts: dict[str, dict[int, dict[str, Any]]] = {}
        sources = defaultdict(list)
        for request_id, meta in metadata.requests.items():
            if not meta.do_virtual:
                sources[(meta.remote_engine_id, meta.remote_request_id)].append((request_id, meta))
        for requests in sources.values():
            if len(requests) < 2:
                continue
            if worker._is_hma_required or worker.pcp_size != 1 or worker.dcp_size != 1:
                raise ValueError("NPU diffusion KV fanout requires TP-only non-hybrid attention")
            counts: Counter[int] = Counter()
            ports: dict[int, dict[str, Any]] = {}
            for _, meta in requests:
                if meta.remote_pcp_size != 1 or meta.remote_dcp_size != 1:
                    raise ValueError("NPU diffusion KV fanout requires source PCP=1 and DCP=1")
                tp_size = meta.remote_ptp_size or worker._prefill_tp_size
                for ranks in worker._get_remote_ranks_for_req(meta.remote_request_id, tp_size):
                    for rank in ranks:
                        port = meta.remote_port + rank
                        counts[port] += 1
                        host, _ = worker._get_remote_host_info_by_port(
                            meta.remote_port,
                            port,
                            meta.remote_host,
                            meta.remote_engine_id,
                            meta.remote_multi_nodes_meta_mapping,
                        )
                        ports[port] = {"host": host}
            for port, count in counts.items():
                ports[port]["num"] = count
            for request_id, _ in requests:
                self._counts[request_id] = ports

    def __getattr__(self, name: str) -> Any:
        return getattr(self._receiver, name)

    def add_request(self, **kwargs: Any) -> None:
        counts = self._counts.get(kwargs["request_id"])
        if counts is not None:
            kwargs["remote_port_send_num"] = counts
        self._receiver.add_request(**kwargs)


def ensure_native_kv_context_groups(vllm_config: Any) -> None:
    """Expose singleton PCP/DCP groups for native KV without enabling EP.

    Group ownership remains with diffusion parallel_state.destroy_model_parallel.
    This first native NPU topology supports TP-only context geometry.
    """
    if vllm_config.kv_transfer_config is None:
        return

    import vllm.distributed.parallel_state as native_state

    from vllm_omni.diffusion.distributed.parallel_state import get_world_group, init_vllm_model_parallel_group

    parallel_config = vllm_config.parallel_config
    for field in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel_config, field, 1) != 1:
            raise ValueError("Native diffusion KV on NPU currently requires PCP=1 and DCP=1")
    for attr in ("_PCP", "_DCP"):
        group = getattr(native_state, attr)
        if group is not None and group.world_size != 1:
            raise ValueError(f"Native diffusion KV on NPU requires singleton {attr} groups")

    # Diffusion owns its world coordinator; only the native model-parallel
    # groups are shared with vLLM. Native _WORLD need not be initialized.
    world = get_world_group()
    ranks = [[rank] for rank in range(world.world_size)]
    created = []
    try:
        for attr, name in (("_PCP", "pcp"), ("_DCP", "dcp")):
            if getattr(native_state, attr) is None:
                group = init_vllm_model_parallel_group(
                    group_ranks=ranks,
                    local_rank=world.local_rank,
                    backend="hccl",
                    group_name=name,
                )
                setattr(native_state, attr, group)
                created.append((attr, group))
    except BaseException:
        for attr, group in reversed(created):
            setattr(native_state, attr, None)
            group.destroy()
        raise
