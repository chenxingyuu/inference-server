"""
gRPC subscriber demo for inference-server.

Setup:
    pip install grpcio grpcio-tools
    cd <repo-root>
    python -m grpc_tools.protoc -I proto \
        --python_out=examples/grpc_subscriber/python \
        --grpc_python_out=examples/grpc_subscriber/python \
        proto/inference.proto

Run:
    # Subscribe to all streams
    python subscriber.py

    # Subscribe to specific streams
    python subscriber.py --host 192.168.1.100 --streams cam_001 cam_002
"""

import argparse
import sys
import grpc
import inference_pb2
import inference_pb2_grpc


def run(host: str, port: int, stream_ids: list[str]) -> None:
    addr = f"{host}:{port}"
    print(f"Connecting to {addr} ...")

    with grpc.insecure_channel(addr) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        request = inference_pb2.SubscribeRequest(stream_ids=stream_ids)

        print(f"Subscribed to: {stream_ids or ['(all streams)']}\n")

        try:
            for frame in stub.Subscribe(request):
                _print_frame(frame)
        except grpc.RpcError as e:
            print(f"Stream ended: {e.code()} — {e.details()}", file=sys.stderr)


def _print_frame(frame) -> None:
    print(
        f"[{frame.stream_id}] seq={frame.frame_seq}"
        f" ts={frame.frame_ts:.3f}"
        f" latency={frame.latency_ms:.1f}ms"
        f" model={frame.model_id}"
        f" detections={len(frame.detections)}"
    )
    for det in frame.detections:
        track = f" track_id={det.track_id}" if det.track_id else ""
        attrs = f" attrs={dict(det.attributes)}" if det.attributes else ""
        print(
            f"  [{det.class_id}] {det.class_name}"
            f" conf={det.confidence:.2f}"
            f" bbox=({det.bbox.x1:.0f},{det.bbox.y1:.0f}"
            f",{det.bbox.x2:.0f},{det.bbox.y2:.0f})"
            f"{track}{attrs}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference-server gRPC subscriber")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--streams", nargs="*", default=[], metavar="STREAM_ID",
                        help="Stream IDs to subscribe to (default: all)")
    args = parser.parse_args()
    run(args.host, args.port, args.streams)
