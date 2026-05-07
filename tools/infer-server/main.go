package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"regexp"
	"strings"
)

var (
	socketPath = envOr("INFER_SOCKET", "/var/run/infer.sock")
	listenAddr = envOr("INFER_SERVER_ADDR", ":8080")
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func call(cmd map[string]string) (map[string]any, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	data, _ := json.Marshal(cmd)
	fmt.Fprintf(conn, "%s\n", data)

	var resp map[string]any
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&resp); err != nil {
		return nil, err
	}
	return resp, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func handleHealthz(w http.ResponseWriter, _ *http.Request) {
	resp, err := call(map[string]string{"cmd": "health"})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	if resp["status"] != "ok" {
		writeJSON(w, http.StatusServiceUnavailable, resp)
		return
	}
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "OK")
}

func handleMetrics(w http.ResponseWriter, _ *http.Request) {
	resp, err := call(map[string]string{"cmd": "metrics"})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	if m, ok := resp["data"].(string); ok {
		fmt.Fprint(w, m)
	}
}

func handleTasks(w http.ResponseWriter, r *http.Request) {
	resp, err := call(map[string]string{"cmd": "list_tasks"})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	writeJSON(w, http.StatusOK, resp["data"])
}

var taskActionRe = regexp.MustCompile(`^/tasks/([^/]+)/(start|stop)$`)

func handleTaskAction(w http.ResponseWriter, r *http.Request) {
	m := taskActionRe.FindStringSubmatch(r.URL.Path)
	if m == nil || r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}
	id, action := m[1], m[2]
	resp, err := call(map[string]string{"cmd": action + "_task", "id": id})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	code := http.StatusOK
	if resp["status"] != "ok" {
		code = http.StatusNotFound
	}
	writeJSON(w, code, resp)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", handleHealthz)
	mux.HandleFunc("/metrics", handleMetrics)
	mux.HandleFunc("/tasks", handleTasks)
	mux.HandleFunc("/tasks/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/start") || strings.HasSuffix(r.URL.Path, "/stop") {
			handleTaskAction(w, r)
		} else {
			http.NotFound(w, r)
		}
	})

	fmt.Printf("infer-server listening on %s (socket: %s)\n", listenAddr, socketPath)
	if err := http.ListenAndServe(listenAddr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
