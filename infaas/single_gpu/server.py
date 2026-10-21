"""Native INFaaS API for the opt-in single-GPU cost-policy experiment."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import signal
import time

import grpc

from infaas.proto import queryfe_pb2 as pb, queryfe_pb2_grpc, request_reply_pb2 as rr
from infaas.proto import modelreg_pb2 as reg, modelreg_pb2_grpc
from .service import Rejected


class Query(queryfe_pb2_grpc.QueryServicer):
    def __init__(self, executor):
        self.executor = executor

    def Heartbeat(self, request, context):
        return pb.HeartbeatResponse(status=rr.RequestReply(status=rr.SUCCESS))

    def QueryOnline(self, request, context):
        started = time.monotonic()
        response = pb.QueryOnlineResponse(worker=self.executor.settings.name,
                                         hardware=self.executor.settings.hardware)
        try:
            if request.parent_model != self.executor.settings.application or request.model_variant or request.grandparent_model:
                raise Rejected("bad_request", "Send APPLICATION, not a pinned model/variant, to enable cost selection")
            if len(request.raw_input) != 1:
                raise Rejected("bad_request", "One JPEG per request; server performs GPU batching")
            result = self.executor.submit(request.raw_input[0], request.slo.LatencyInUSec / 1000,
                                          request.slo.MinAccuracy)
            response.status.CopyFrom(rr.RequestReply(status=rr.SUCCESS))
            response.variant = result["variant"]
            response.label = result["label"]
            response.raw_output.append(result["label"].encode())
            response.img_width, response.img_height = result["size"]
            response.path = result["path"]
            response.timings.update({"worker." + key: value for key, value in result["timings"].items()})
        except Rejected as error:
            response.status.CopyFrom(rr.RequestReply(status=rr.UNAVAILABLE, msg=str(error)))
            response.reject_kind = error.kind
        except Exception as error:
            logging.exception("single-GPU query failed")
            response.status.CopyFrom(rr.RequestReply(status=rr.UNAVAILABLE, msg=str(error)))
            response.reject_kind = "worker_error"
        response.latency_ms = (time.monotonic() - started) * 1000
        response.timings["srv.total"] = response.latency_ms
        return response

    def QueryModelInfo(self, request, context):
        state = self.executor.status()
        names = list(state["profiles"]) if request.model == state["application"] else [request.model]
        variants = [f"{name}__{state['gpu_type']}" for name in names if name in state["profiles"]]
        ok = bool(variants) and state["policy_ready"]
        return pb.QueryModelInfoResponse(reply=rr.QueryModelReply(all_models=variants,
            status=rr.RequestReply(status=rr.SUCCESS if ok else rr.UNAVAILABLE)))

    def AllParentInfo(self, request, context):
        return pb.AllParResponse(reply=rr.AllParReply(all_models=[self.executor.settings.application],
            status=rr.RequestReply(status=rr.SUCCESS)))

    def QueryOffline(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Online single-GPU experiment only")


class Registrar(modelreg_pb2_grpc.ModelRegServicer):
    def __init__(self, executor):
        self.executor = executor

    def Heartbeat(self, request, context):
        return reg.HeartbeatResponse(status=rr.RequestReply(status=rr.SUCCESS))

    def RegisterModel(self, request, context):
        try:
            if request.submitter == "tester" or request.profile_json:
                raise ValueError("Use measured GPU registration, not an imported legacy profile")
            name = request.parent_model
            if name not in self.executor.settings.entries:
                raise ValueError("Register an enabled candidate model, not APPLICATION")
            if not request.profile_image:
                raise ValueError("Real experiment JPEG required for CPU preparation")
            if request.reprofile or name not in self.executor.profiles:
                self.executor.profile(name, request.profile_image)
            return reg.ModelRegResponse(status=rr.RequestReply(status=rr.SUCCESS),
                variants=[f"{name}__{self.executor.settings.hardware}"])
        except Exception as error:
            return reg.ModelRegResponse(status=rr.RequestReply(status=rr.UNAVAILABLE, msg=str(error)))


def add_services(server, executor):
    queryfe_pb2_grpc.add_QueryServicer_to_server(Query(executor), server)
    modelreg_pb2_grpc.add_ModelRegServicer_to_server(Registrar(executor), server)
    def status(request, context):
        return executor.status()
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("infaas.single_gpu.Admin", {
        "Status": grpc.unary_unary_rpc_method_handler(status,
            request_deserializer=lambda raw: json.loads(raw or b"{}"),
            response_serializer=lambda value: json.dumps(value).encode()),
    }),))


def main():
    from .settings import Settings
    from .backend import Backend
    from .service import Executor
    settings = Settings()
    # Registry checkpoint pins, not guessed defaults, define this mode's weights.
    os.environ["MODEL_REVISIONS"] = ",".join(f"{name}:{entry['revision']}" for name, entry in settings.entries.items())
    executor = Executor(settings, Backend(settings))
    server = grpc.server(ThreadPoolExecutor(max_workers=settings.threads), options=(
        ("grpc.max_receive_message_length", 20 << 20), ("grpc.max_send_message_length", 20 << 20)))
    add_services(server, executor)
    for port in (50052, 50053):
        if not server.add_insecure_port(f"0.0.0.0:{port}"):
            raise RuntimeError(f"Cannot bind internal port {port}")
    server.start()
    def stop(*_):
        server.stop(10)
        executor.close()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logging.info("single-GPU COST adapter ready; register both candidates before replay")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
