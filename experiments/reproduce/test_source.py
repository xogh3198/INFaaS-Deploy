"""The user's docker/v1 is the policy baseline; tests do not claim a GPU run."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.reproduce.infaas_entry import BASE,verify
from experiments.reproduce.infaas_host import settings,spec
from experiments.reproduce.infaas_trace import Client


def test_compose_uses_full_docker_v1_stack_and_does_not_install_the_cost_port():
    c=settings()
    services=spec(c)["services"]
    assert set(services)=={"redis","controller","worker","vm-autoscaler"}
    assert services["worker"]["command"][-1]=="infaas.worker.main"
    assert services["controller"]["command"][-1]=="infaas.controller.main"
    env=services["worker"]["environment"]
    assert "BATCH_SIZES" not in env
    assert env["PROFILE_MODE"]=="original"
    assert env["PREPROCESS_BACKEND"]=="auto"
    assert env["PREPROCESS_MODE"]=="cpu_decode_gpu"
    assert env["STATIC_WORKERS"]=="source-worker=v100@worker:9000"
    assert services["worker"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]==["2"]


def test_docker_build_pins_users_source_commit_and_retains_preprocessing():
    docker=Path(__file__).with_name("infaas.Dockerfile").read_text()
    assert BASE in docker
    assert "https://github.com/xogh3198/INFaaS-Deploy.git" in docker
    assert "infaas/single_gpu" not in docker
    assert "nvidia-dali-cuda120==" in docker


def test_metrics_do_not_change_native_request_or_subtract_cpu_from_rpc_latency(monkeypatch):
    from infaas.proto import queryfe_pb2,request_reply_pb2 as rr
    client=Client.__new__(Client)
    client.pb=queryfe_pb2
    client.success=rr.SUCCESS
    client.timeout=10
    sent=[]
    def query(request,timeout):
        sent.append(request)
        reply=queryfe_pb2.QueryOnlineResponse(variant="resnet-50__v100",latency_ms=118)
        reply.status.status=rr.SUCCESS
        reply.timings.update({"worker.decode":40,"worker.preprocess":60,"worker.lock_wait":8,
            "worker.forward":9,"worker.postprocess":1,"worker.load":200})
        return reply
    from types import SimpleNamespace
    client.stub=SimpleNamespace(QueryOnline=query)
    monkeypatch.setenv("SOURCE_REQUEST_MODEL","resnet-50")
    result=client.query(b"jpeg",200,"id")
    assert sent[0].parent_model=="resnet-50"
    assert sent[0].raw_input[0]==b"jpeg"
    assert sent[0].slo.LatencyInUSec==200_000
    assert result["server_ms"]==18
    assert result["timings"]["measurement.native_api_latency"]==118
    assert result["timings"]["worker.load"]==200
    assert result["e2e_ms"]>=0 and result["ok"]


def test_provided_source_can_be_verified_without_modification():
    # Optional local checkout; the container always clones and verifies BASE.
    path=Path(__file__).resolve().parents[3]/"infaas-docker-v1-base"
    if not path.exists():pytest.skip("Pinned source fixture not checked out on this machine")
    assert verify(path)["commit"]==BASE
    assert not subprocess.check_output(["git","-C",str(path),"diff","--name-only",BASE,"--","infaas"],text=True).strip()
