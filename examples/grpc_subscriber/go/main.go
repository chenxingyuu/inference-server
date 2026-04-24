// gRPC subscriber demo for inference-server.
//
// Setup:
//
//	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
//	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
//	cd <repo-root>
//	protoc -I proto \
//	    --go_out=examples/grpc_subscriber/go \
//	    --go-grpc_out=examples/grpc_subscriber/go \
//	    proto/inference.proto
//	cd examples/grpc_subscriber/go && go mod tidy
//
// Run:
//
//	# Subscribe to all streams
//	go run main.go
//
//	# Subscribe to specific streams
//	go run main.go -host 192.168.1.100 -streams cam_001,cam_002
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"strings"

	pb "github.com/chenxingyuu/inference-server/examples/grpc_subscriber/inference"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	host := flag.String("host", "localhost", "inference-server host")
	port := flag.Int("port", 50051, "gRPC port")
	streamsFlag := flag.String("streams", "", "comma-separated stream IDs (empty = all)")
	flag.Parse()

	var streamIDs []string
	if *streamsFlag != "" {
		streamIDs = strings.Split(*streamsFlag, ",")
	}

	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("Connecting to %s ...", addr)

	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	stream, err := client.Subscribe(context.Background(), &pb.SubscribeRequest{
		StreamIds: streamIDs,
	})
	if err != nil {
		log.Fatalf("subscribe: %v", err)
	}

	if len(streamIDs) == 0 {
		log.Println("Subscribed to: (all streams)")
	} else {
		log.Printf("Subscribed to: %v", streamIDs)
	}

	for {
		frame, err := stream.Recv()
		if err == io.EOF {
			log.Println("Stream closed by server.")
			return
		}
		if err != nil {
			log.Fatalf("recv: %v", err)
		}
		printFrame(frame)
	}
}

func printFrame(frame *pb.DetectionFrame) {
	fmt.Printf("[%s] seq=%d ts=%.3f latency=%.1fms model=%s detections=%d\n",
		frame.StreamId, frame.FrameSeq, frame.FrameTs, frame.LatencyMs,
		frame.ModelId, len(frame.Detections))

	for _, det := range frame.Detections {
		track := ""
		if det.TrackId != 0 {
			track = fmt.Sprintf(" track_id=%d", det.TrackId)
		}
		attrs := ""
		if len(det.Attributes) > 0 {
			attrs = fmt.Sprintf(" attrs=%v", det.Attributes)
		}
		fmt.Printf("  [%d] %s conf=%.2f bbox=(%.0f,%.0f,%.0f,%.0f)%s%s\n",
			det.ClassId, det.ClassName, det.Confidence,
			det.Bbox.X1, det.Bbox.Y1, det.Bbox.X2, det.Bbox.Y2,
			track, attrs)
	}
}
