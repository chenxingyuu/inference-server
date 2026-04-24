# gRPC Subscriber Examples

Demos for receiving detection results from inference-server via gRPC server-streaming.

Enable gRPC in your config first:

```yaml
publishers:
  grpc:
    enabled: true
    port: 50051
```

---

## Python

**Generate stubs** (run once from repo root):

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I proto \
    --python_out=examples/grpc_subscriber/python \
    --grpc_python_out=examples/grpc_subscriber/python \
    proto/inference.proto
```

**Run:**

```bash
cd examples/grpc_subscriber/python

# Subscribe to all streams
python subscriber.py

# Subscribe to specific streams
python subscriber.py --host 192.168.1.100 --streams cam_001 cam_002
```

---

## Go

**Generate stubs** (run once from repo root):

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

protoc -I proto \
    --go_out=examples/grpc_subscriber/go \
    --go-grpc_out=examples/grpc_subscriber/go \
    proto/inference.proto

cd examples/grpc_subscriber/go && go mod tidy
```

**Run:**

```bash
cd examples/grpc_subscriber/go

# Subscribe to all streams
go run main.go

# Subscribe to specific streams
go run main.go -host 192.168.1.100 -streams cam_001,cam_002
```

---

## Output format

```
[cam_001] seq=1234 ts=1714000000.123 latency=8.2ms model=yolov8n detections=3
  [2] car   conf=0.91 bbox=(120,80,320,240) track_id=7
  [0] person conf=0.87 bbox=(400,100,480,380) track_id=12
  [2] car   conf=0.76 bbox=(600,90,780,250) track_id=3
```

## Notes

| | |
|-|-|
| `--streams` empty | Subscribe to all streams |
| Multiple subscribers | Each connection receives independently |
| Slow consumer | Server drops frames after 64 pending; your consumer won't block inference |
| Reconnect | Implement retry loop on the client side; server is stateless |
| TLS | Replace `insecure` credentials with `credentials.NewClientTLSFromFile(...)` (Go) or `grpc.secure_channel(...)` (Python) |
