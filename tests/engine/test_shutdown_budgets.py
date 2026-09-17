# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vllm.envs as vllm_envs
from vllm.v1.engine.core_client import MPClient

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.engine.shutdown import stage_shutdown_grace_s, worker_shutdown_grace_s
from vllm_omni.engine.stage_engine_core_client import DPLBStageEngineCoreClient, StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("client_cls", [StageEngineCoreClient, DPLBStageEngineCoreClient])
def test_ar_parent_allows_configured_worker_cleanup(monkeypatch, client_cls):
    monkeypatch.setattr(vllm_envs, "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 30)
    shutdown = Mock()
    monkeypatch.setattr(MPClient, "shutdown", shutdown)
    client = object.__new__(client_cls)

    client.shutdown()
    assert shutdown.call_args.kwargs["timeout"] > 30

    client.shutdown(timeout=2.5)
    shutdown.assert_called_with(timeout=2.5)


@pytest.mark.parametrize("native_worker_timeout", [5, 30])
def test_diffusion_parent_waits_for_worker_finalizers(monkeypatch, native_worker_timeout):
    monkeypatch.setattr(vllm_envs, "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", native_worker_timeout)
    events = []
    client = object.__new__(StageDiffusionClient)
    for attr in ("_request_socket", "_response_socket", "_zmq_ctx", "_encoder"):
        setattr(client, attr, Mock())

    def shutdown_process(*, timeout):
        # Device runtime finalizers run after the worker's shutdown method.
        assert timeout > worker_shutdown_grace_s() + 5
        events.append("worker-finalizers-complete")

    client._proc_manager = SimpleNamespace(proc=Mock(), shutdown=shutdown_process)
    client._proc_manager.proc.is_alive.return_value = True
    client._request_socket.close.side_effect = lambda **_: events.append("sockets-close")

    client.shutdown()

    assert events == ["worker-finalizers-complete", "sockets-close"]
    client._zmq_ctx.term.assert_called_once_with()
    assert stage_shutdown_grace_s() >= native_worker_timeout
