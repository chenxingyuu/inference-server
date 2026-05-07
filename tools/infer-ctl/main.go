package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
)

const defaultSocket = "/var/run/infer.sock"

func socketPath() string {
	if s := os.Getenv("INFER_SOCKET"); s != "" {
		return s
	}
	return defaultSocket
}

func send(cmd map[string]string) (map[string]any, error) {
	conn, err := net.Dial("unix", socketPath())
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	defer conn.Close()

	data, _ := json.Marshal(cmd)
	if _, err := fmt.Fprintf(conn, "%s\n", data); err != nil {
		return nil, fmt.Errorf("send: %w", err)
	}

	var resp map[string]any
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&resp); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return resp, nil
}

func must(resp map[string]any, err error) map[string]any {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	return resp
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: infer-ctl <health|tasks|start <id>|stop <id>|metrics>")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "health":
		resp := must(send(map[string]string{"cmd": "health"}))
		fmt.Println(resp["status"])

	case "tasks":
		resp := must(send(map[string]string{"cmd": "list_tasks"}))
		data, _ := json.MarshalIndent(resp["data"], "", "  ")
		fmt.Println(string(data))

	case "start":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: infer-ctl start <id>")
			os.Exit(1)
		}
		resp := must(send(map[string]string{"cmd": "start_task", "id": os.Args[2]}))
		fmt.Println(resp["status"])

	case "stop":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: infer-ctl stop <id>")
			os.Exit(1)
		}
		resp := must(send(map[string]string{"cmd": "stop_task", "id": os.Args[2]}))
		fmt.Println(resp["status"])

	case "metrics":
		resp := must(send(map[string]string{"cmd": "metrics"}))
		if m, ok := resp["data"].(string); ok {
			fmt.Print(m)
		}

	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}
