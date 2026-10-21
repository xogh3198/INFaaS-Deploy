"""CPU/fake-GPU mechanics only. Synthetic profiles are NOT experimental results."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import grpc
from infaas.common import config
from infaas.single_gpu.settings import Settings, registry
from infaas.single_gpu.policy import Candidate, choose, candidates
from infaas.single_gpu.service import Executor, Rejected
from infaas.single_gpu.server import add_services
from infaas.single_gpu.deploy import compose_document, read_env, DEFAULT, ROOT
from infaas.proto import queryfe_pb2 as pb, queryfe_pb2_grpc, modelreg_pb2 as reg, modelreg_pb2_grpc
from infaas.proto import request_reply_pb2 as rr


class FakeBackend:
    def __init__(self, settings):
        self.settings = settings
        self.fingerprint = {"schema": "FAKE-NOT-GPU-MEASUREMENTS", "registry": settings.registry_hash}
        self.resident = None
        self.gpu_threads = set()
        self.executions = []
        self.prepare_delay = 0
        self.load_delay = 0

    def prepare(self, image):
        start = time.monotonic()
        time.sleep(self.prepare_delay)
        return {"inputs": image, "size": (224, 224), "cpu_start": start,
                "cpu_end": time.monotonic(), "decode": 0.0, "preprocess": 0.0}

    def load(self, model):
        self.gpu_threads.add(threading.get_ident())
        loaded = self.resident != model
        if loaded:
            time.sleep(self.load_delay)
        self.resident = model
        return (1.0 if loaded else 0), loaded

    def unload(self):
        self.gpu_threads.add(threading.get_ident())
        self.resident = None

    def profile_batch(self, model, prepared, batch, runs):
        self.gpu_threads.add(threading.get_ident())
        timings = {"resnet-18": {1: 20, 2: 36, 4: 70, 8: 140},
                   "resnet-50": {1: 22, 2: 30, 4: 45, 8: 80}}
        latency = timings[model][batch]
        return {"p50_ms": latency, "p95_ms": latency + 1, "mean_ms": latency,
                "capacity_qps": batch * 1000 / latency,
                "memory_bytes": 100_000_000 if model == "resnet-18" else 200_000_000,
                "samples_ms": [latency] * runs, "runs": runs}

    def execute(self, model, prepared):
        assert self.resident == model
        self.gpu_threads.add(threading.get_ident())
        self.executions.append((model, len(prepared)))
        time.sleep(.001)
        return ["fake-label"] * len(prepared), 1.0


class CostPolicyTests(unittest.TestCase):
    def setUp(self):
        self.price = patch.dict(config.HW_COST, {"v100": 1})
        self.price.start()
        self.addCleanup(self.price.stop)

    def row(self, model, batch, capacity, memory, accuracy):
        return Candidate(model, "v100", batch, accuracy, 20, 21, capacity, 1, memory)

    def test_cost_not_highest_accuracy_is_primary_objective(self):
        low = self.row("resnet-18", 1, 50, 100_000_000, 70.93)
        high = self.row("resnet-50", 1, 60, 200_000_000, 77.36)
        self.assertEqual(choose([low, high], 1)[0], low)

    def test_higher_load_can_change_both_model_and_batch(self):
        low = self.row("resnet-18", 1, 50, 100_000_000, 70.93)
        high = self.row("resnet-50", 4, 90, 200_000_000, 77.36)
        self.assertEqual(choose([low, high], 60, low)[0], high)

    def test_no_switch_is_valid_if_one_candidate_dominates(self):
        low = self.row("resnet-18", 1, 200, 100_000_000, 70.93)
        high = self.row("resnet-50", 4, 90, 200_000_000, 77.36)
        self.assertEqual(choose([low, high], 60, low)[0], low)

    def test_overload_does_not_invent_extra_gpu_replicas(self):
        a = self.row("resnet-18", 1, 50, 100_000_000, 70.93)
        b = self.row("resnet-50", 4, 90, 200_000_000, 77.36)
        result, detail = choose([a, b], 200, a)
        self.assertEqual(result, b)
        self.assertFalse(detail["capacity_sufficient"])

    def test_minimum_accuracy_and_slo_half_are_constraints(self):
        profiles = {"resnet-18": {"load_ms": 1, "batches": {"1": {"p50_ms": 20, "p95_ms": 21,
            "capacity_qps": 50, "memory_bytes": 100}}},
            "resnet-50": {"load_ms": 1, "batches": {"8": {"p50_ms": 100, "p95_ms": 101,
            "capacity_qps": 80, "memory_bytes": 200}}}}
        entries = {"resnet-18": {"accuracy": 70.93}, "resnet-50": {"accuracy": 77.36}}
        self.assertEqual(candidates(profiles, entries, "v100", 200, 75), [])


class SingleGpuFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.price = patch.dict(config.HW_COST, {"v100": 1})
        self.price.start()
        self.addCleanup(self.price.stop)
        self.settings = Settings({"MODEL_REGISTRY": str(ROOT / "deploy/docker/infaas-single-gpu.models.json"),
            "PROFILE_DIR": str(self.root / "profiles"), "PROFILE_RUNS": "3", "ALLOCATION_INTERVAL": "10"})
        self.backend = FakeBackend(self.settings)
        self.executor = Executor(self.settings, self.backend)
        self.addCleanup(self.executor.close)

    def register(self):
        for model in self.settings.entries:
            self.executor.profile(model, b"fake-jpeg", 3)


class SingleGpuServiceTests(SingleGpuFixture):
    def test_requires_both_model_profiles(self):
        self.executor.profile("resnet-50", b"fake-jpeg", 3)
        with self.assertRaisesRegex(Rejected, "Register all"):
            self.executor.submit(b"jpeg", 200)
        self.executor.profile("resnet-18", b"fake-jpeg", 3)
        self.assertTrue(self.executor.ready())

    def test_profile_and_serving_share_one_gpu_thread(self):
        self.register()
        row = self.executor.submit(b"jpeg", 200)
        self.assertEqual(row["variant"], "resnet-18__v100")
        self.assertEqual(len(self.backend.gpu_threads), 1)
        self.assertEqual(row["timings"]["batch_size"], 1)

    def test_preprocessing_does_not_reset_deadline(self):
        self.register()
        self.backend.prepare_delay = .22
        with self.assertRaises(Rejected) as error:
            self.executor.submit(b"jpeg", 200)
        self.assertEqual(error.exception.kind, "deadline_drop")
        self.assertEqual(self.backend.executions, [])

    def test_cold_load_is_real_and_can_cause_slo_miss(self):
        self.register()
        self.backend.load_delay = .22
        result = self.executor.submit(b"jpeg", 200)
        self.assertEqual(result["path"], "inactive")
        self.assertEqual(result["timings"]["slo_missed"], 1)
        self.assertEqual(self.executor.status()["server_slo_misses"], 1)

    def test_profiling_cannot_race_outstanding_cpu_preparation(self):
        self.register()
        self.backend.prepare_delay = .03
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(self.executor.submit, b"jpeg", 200)
            until = time.monotonic() + 2
            while not self.executor.status()["outstanding_requests"] and time.monotonic() < until:
                time.sleep(.001)
            with self.assertRaisesRegex(Rejected, "no inference"):
                self.executor.profile("resnet-18", b"jpeg", 3)
            request.result(timeout=5)

    def test_model_switch_and_real_batch_are_executed(self):
        self.register()
        self.executor.submit(b"jpeg", 200)
        with self.executor.condition:
            self.executor.rate = 80
        barrier = threading.Barrier(8)
        def call(_):
            barrier.wait(timeout=5)
            return self.executor.submit(b"jpeg", 200)
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(call, range(8)))
        self.assertTrue(all(row["variant"] == "resnet-50__v100" for row in rows))
        self.assertTrue(any(size > 1 for _, size in self.backend.executions))
        state = self.executor.status()
        self.assertEqual(state["model_switches_after_initial_load"], 1)
        self.assertEqual(state["successful_requests_by_model"], {"resnet-18": 1, "resnet-50": 8})
        self.assertEqual(len(self.backend.gpu_threads), 1)

    def test_unchanged_valid_profiles_restore(self):
        self.register()
        other_backend = FakeBackend(self.settings)
        other = Executor(self.settings, other_backend)
        self.addCleanup(other.close)
        self.assertTrue(other.ready())

    def test_incompatible_profiles_are_not_silently_reused(self):
        self.register()
        other_backend = FakeBackend(self.settings)
        other_backend.fingerprint = {"different": True}
        other = Executor(self.settings, other_backend)
        self.addCleanup(other.close)
        self.assertFalse(other.ready())
        self.assertIn("fingerprint", other.status()["policy_error"])


class NativeRpcTests(SingleGpuFixture):
    def setUp(self):
        super().setUp()
        self.register()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=16))
        add_services(self.server, self.executor)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.addCleanup(lambda: self.server.stop(0).wait())
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port}")
        self.addCleanup(self.channel.close)
        self.stub = queryfe_pb2_grpc.QueryStub(self.channel)

    def test_pinned_model_request_is_explicitly_rejected(self):
        reply = self.stub.QueryOnline(pb.QueryOnlineRequest(parent_model="resnet-50", raw_input=[b"jpeg"],
            slo=pb.QuerySLO(LatencyInUSec=200_000)), timeout=5)
        self.assertEqual(reply.status.status, rr.UNAVAILABLE)
        self.assertEqual(reply.reject_kind, "bad_request")

    def test_application_request_reports_actual_selected_model_and_batch(self):
        info = self.stub.QueryModelInfo(pb.QueryModelInfoRequest(model="imagenet"), timeout=5)
        self.assertEqual(set(info.reply.all_models), {"resnet-18__v100", "resnet-50__v100"})
        reply = self.stub.QueryOnline(pb.QueryOnlineRequest(parent_model="imagenet", raw_input=[b"jpeg"],
            slo=pb.QuerySLO(LatencyInUSec=200_000)), timeout=5)
        self.assertEqual(reply.status.status, rr.SUCCESS)
        self.assertEqual(reply.variant, "resnet-18__v100")
        self.assertEqual(reply.timings["worker.batch_size"], 1)

    def test_minimum_accuracy_constraint_and_float32_rounding(self):
        def request(minimum):
            return self.stub.QueryOnline(pb.QueryOnlineRequest(parent_model="imagenet", raw_input=[b"jpeg"],
                slo=pb.QuerySLO(LatencyInUSec=200_000, MinAccuracy=minimum)), timeout=5)
        self.assertEqual(request(70.93).variant, "resnet-18__v100")
        self.assertEqual(request(77.36).variant, "resnet-50__v100")

    def test_trace_replay_task_preflight_and_reference_accuracy(self):
        from infaas.cli.trace_replay import main
        image = self.root / "image.jpg"
        image.write_bytes(b"fake-jpeg")
        trace = self.root / "trace.txt"
        trace.write_text("0,1,300\n5,1,300\n", encoding="utf-8", newline="\n")
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--controller", f"127.0.0.1:{self.port}", "--application", "imagenet",
                "--trace", str(trace), "--image", str(image), "--accuracy-registry", self.settings.registry_path,
                "--output", str(self.root / "results"), "--max-send-lag-ms", "2000"])
        self.assertEqual(result, 0)
        report = json.loads(next((self.root / "results").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["successful_requests_by_model"], {"resnet-18": 2})
        self.assertEqual(report["summary"]["mean_reference_accuracy_of_successful_replies"], 70.93)
        self.assertEqual(report["summary"]["actual_gpu_batch_execution_counts"], {"1": 2})


class ComposeTests(unittest.TestCase):
    def test_one_selected_gpu_separate_output_and_no_external_runtime(self):
        cfg = read_env(DEFAULT)
        for key in ("MODEL_DIR", "MODEL_REGISTRY", "DATA_DIR"):
            cfg[key] = str((ROOT / cfg[key]).resolve())
        cfg["MODEL_REVISIONS"] = "resnet-18:" + "a" * 40
        document = compose_document(cfg)
        worker = document["services"]["worker"]
        self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["2"])
        self.assertEqual(worker["environment"]["WORKER_THREADS"], "32")
        self.assertEqual(worker["command"], ["python", "-m", "infaas.single_gpu.server"])
        self.assertNotIn("proteus", worker["image"])
        self.assertIn("single-gpu-cost-v1", worker["volumes"][-1])


if __name__ == "__main__":
    unittest.main()
