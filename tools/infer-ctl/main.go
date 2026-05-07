package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"strings"
)

const defaultSocket = "/var/run/infer.sock"

var socket string

func init() {
	def := os.Getenv("INFER_SOCKET")
	if def == "" {
		def = defaultSocket
	}
	flag.StringVar(&socket, "s", def, "unix socket path (overrides INFER_SOCKET env var)")
	flag.StringVar(&socket, "socket", def, "unix socket path (alias for -s)")
}

func send(cmd map[string]any) (map[string]any, error) {
	conn, err := net.Dial("unix", socket)
	if err != nil {
		return nil, fmt.Errorf("connect %s: %w", socket, err)
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
	if resp["status"] == "error" {
		fmt.Fprintln(os.Stderr, "error:", resp["message"])
		os.Exit(1)
	}
	return resp
}

func dataRows(resp map[string]any) []map[string]any {
	items, _ := resp["data"].([]any)
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// ANSI color helpers
const (
	colorReset  = "\033[0m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorRed    = "\033[31m"
)

func colorState(state string) string {
	switch strings.ToUpper(state) {
	case "STREAMING":
		return colorGreen + state + colorReset
	case "DEGRADED", "RECONNECTING":
		return colorYellow + state + colorReset
	case "FAILED", "STOPPED":
		return colorRed + state + colorReset
	default:
		return state
	}
}

// stripAnsi removes ANSI escape sequences for width measurement.
func stripAnsi(s string) string {
	var b strings.Builder
	inEsc := false
	for _, c := range s {
		switch {
		case c == '\033':
			inEsc = true
		case inEsc && c == 'm':
			inEsc = false
		case !inEsc:
			b.WriteRune(c)
		}
	}
	return b.String()
}

func printTable(headers []string, tableRows [][]string) {
	if len(tableRows) == 0 {
		fmt.Println(strings.Join(headers, "   "))
		fmt.Println("(none)")
		return
	}
	widths := make([]int, len(headers))
	for i, h := range headers {
		widths[i] = len(h)
	}
	for _, row := range tableRows {
		for i, cell := range row {
			if i < len(widths) {
				if n := len(stripAnsi(cell)); n > widths[i] {
					widths[i] = n
				}
			}
		}
	}
	printRow := func(cells []string, pad func(i int, cell string) string) {
		for i, cell := range cells {
			if i < len(widths) {
				fmt.Print(pad(i, cell))
				if i < len(cells)-1 {
					fmt.Print("   ")
				}
			}
		}
		fmt.Println()
	}
	printRow(headers, func(i int, h string) string {
		return fmt.Sprintf("%-*s", widths[i], h)
	})
	for i, w := range widths {
		fmt.Print(strings.Repeat("-", w))
		if i < len(widths)-1 {
			fmt.Print("   ")
		}
	}
	fmt.Println()
	for _, row := range tableRows {
		printRow(row, func(i int, cell string) string {
			plain := stripAnsi(cell)
			return cell + strings.Repeat(" ", widths[i]-len(plain))
		})
	}
}

func str(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return fmt.Sprintf("%g", t)
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", t)
	}
}

func intStr(v any) string {
	if f, ok := v.(float64); ok {
		return fmt.Sprintf("%d", int(f))
	}
	return str(v)
}

// die prints a usage message and exits.
func die(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

// requireID returns args[0] or exits.
func requireID(args []string, usage string) string {
	if len(args) == 0 || args[0] == "" {
		die(usage)
	}
	return args[0]
}

// ── sources ───────────────────────────────────────────────────────────────────

func listSources() {
	resp := must(send(map[string]any{"cmd": "list_sources"}))
	var tableRows [][]string
	for _, item := range dataRows(resp) {
		tableRows = append(tableRows, []string{
			str(item["id"]),
			str(item["url"]),
			colorState(str(item["state"])),
			intStr(item["reconnect_count"]),
		})
	}
	printTable([]string{"ID", "URL", "STATE", "RECONNECTS"}, tableRows)
}

func sourcesAdd(args []string) {
	fs := flag.NewFlagSet("sources add", flag.ExitOnError)
	id                 := fs.String("id", "", "source id (required)")
	url                := fs.String("url", "", "stream URL (required)")
	reconnectDelay     := fs.Int("reconnect-delay-ms", 3000, "initial reconnect delay ms")
	maxReconnectDelay  := fs.Int("max-reconnect-delay-ms", 60000, "max reconnect delay ms")
	maxReconnectAttempts := fs.Int("max-reconnect-attempts", 5, "max reconnect attempts (0=unlimited)")
	fs.Parse(args)

	if *id == "" || *url == "" {
		die("usage: infer-ctl sources add --id <id> --url <url> [--reconnect-delay-ms N] [--max-reconnect-delay-ms N] [--max-reconnect-attempts N]")
	}
	resp := must(send(map[string]any{
		"cmd":                    "add_source",
		"id":                     *id,
		"url":                    *url,
		"reconnect_delay_ms":     *reconnectDelay,
		"max_reconnect_delay_ms": *maxReconnectDelay,
		"max_reconnect_attempts": *maxReconnectAttempts,
	}))
	fmt.Printf("added source %s\n", resp["id"])
}

func sourcesRemove(args []string) {
	id := requireID(args, "usage: infer-ctl sources remove <id>")
	resp := must(send(map[string]any{"cmd": "remove_source", "id": id}))
	fmt.Printf("removed source %s\n", resp["id"])
}

func cmdSources(args []string) {
	if len(args) == 0 {
		listSources()
		return
	}
	switch args[0] {
	case "add":
		sourcesAdd(args[1:])
	case "remove", "rm", "delete", "del":
		sourcesRemove(args[1:])
	default:
		die(fmt.Sprintf("unknown subcommand %q\n\nusage: infer-ctl sources [add|remove] ...", args[0]))
	}
}

// ── pipelines ─────────────────────────────────────────────────────────────────

func listPipelines() {
	resp := must(send(map[string]any{"cmd": "list_pipelines"}))
	var tableRows [][]string
	for _, item := range dataRows(resp) {
		var parts []string
		if ns, ok := item["nodes"].([]any); ok {
			for _, n := range ns {
				parts = append(parts, str(n))
			}
		}
		tableRows = append(tableRows, []string{
			str(item["id"]),
			strings.Join(parts, ", "),
			intStr(item["edge_count"]),
		})
	}
	printTable([]string{"ID", "NODES", "EDGES"}, tableRows)
}

func pipelinesAdd(args []string) {
	fs := flag.NewFlagSet("pipelines add", flag.ExitOnError)
	id   := fs.String("id", "", "pipeline id (required)")
	file := fs.String("file", "", "JSON file with {nodes:[...], edges:[...]} (required)")
	fs.Parse(args)

	if *id == "" || *file == "" {
		die("usage: infer-ctl pipelines add --id <id> --file pipeline.json")
	}
	raw, err := os.ReadFile(*file)
	if err != nil {
		die(fmt.Sprintf("error reading %s: %v", *file, err))
	}
	var def map[string]any
	if err := json.Unmarshal(raw, &def); err != nil {
		die(fmt.Sprintf("invalid JSON in %s: %v", *file, err))
	}
	req := map[string]any{
		"cmd": "add_pipeline",
		"id":  *id,
	}
	if nodes, ok := def["nodes"]; ok {
		req["nodes"] = nodes
	}
	if edges, ok := def["edges"]; ok {
		req["edges"] = edges
	}
	resp := must(send(req))
	fmt.Printf("added pipeline %s\n", resp["id"])
}

func pipelinesRemove(args []string) {
	id := requireID(args, "usage: infer-ctl pipelines remove <id>")
	resp := must(send(map[string]any{"cmd": "remove_pipeline", "id": id}))
	fmt.Printf("removed pipeline %s\n", resp["id"])
}

func cmdPipelines(args []string) {
	if len(args) == 0 {
		listPipelines()
		return
	}
	switch args[0] {
	case "add":
		pipelinesAdd(args[1:])
	case "remove", "rm", "delete", "del":
		pipelinesRemove(args[1:])
	default:
		die(fmt.Sprintf("unknown subcommand %q\n\nusage: infer-ctl pipelines [add|remove] ...", args[0]))
	}
}

// ── tasks ─────────────────────────────────────────────────────────────────────

func listTasks() {
	resp := must(send(map[string]any{"cmd": "list_tasks"}))
	var tableRows [][]string
	for _, item := range dataRows(resp) {
		tableRows = append(tableRows, []string{str(item["id"]), str(item["state"])})
	}
	printTable([]string{"ID", "STATE"}, tableRows)
}

func tasksAdd(args []string) {
	fs := flag.NewFlagSet("tasks add", flag.ExitOnError)
	id         := fs.String("id", "", "task id (required)")
	source     := fs.String("source", "", "source id (required)")
	pipeline   := fs.String("pipeline", "", "pipeline id (required)")
	fps        := fs.Float64("fps", 0, "sample fps (0 = use all frames)")
	useHwdec   := fs.Bool("hwdec", false, "enable hardware decoding")
	fs.Parse(args)

	if *id == "" || *source == "" || *pipeline == "" {
		die("usage: infer-ctl tasks add --id <id> --source <source-id> --pipeline <pipeline-id> [--fps N] [--hwdec]")
	}
	resp := must(send(map[string]any{
		"cmd":         "add_task",
		"id":          *id,
		"source_id":   *source,
		"pipeline_id": *pipeline,
		"sample_fps":  *fps,
		"use_hwdec":   *useHwdec,
	}))
	fmt.Printf("added task %s\n", resp["id"])
}

func tasksRemove(args []string) {
	id := requireID(args, "usage: infer-ctl tasks remove <id>")
	resp := must(send(map[string]any{"cmd": "remove_task", "id": id}))
	fmt.Printf("removed task %s\n", resp["id"])
}

func cmdTasks(args []string) {
	if len(args) == 0 {
		listTasks()
		return
	}
	switch args[0] {
	case "add":
		tasksAdd(args[1:])
	case "remove", "rm", "delete", "del":
		tasksRemove(args[1:])
	default:
		die(fmt.Sprintf("unknown subcommand %q\n\nusage: infer-ctl tasks [add|remove] ...", args[0]))
	}
}

// ── models ────────────────────────────────────────────────────────────────────

func listModels() {
	resp := must(send(map[string]any{"cmd": "list_models"}))
	var tableRows [][]string
	for _, item := range dataRows(resp) {
		tableRows = append(tableRows, []string{
			str(item["id"]),
			str(item["backend"]),
			str(item["version"]),
			str(item["input_shape"]),
			intStr(item["batch_size"]),
			intStr(item["instance_count"]),
		})
	}
	printTable([]string{"ID", "BACKEND", "VERSION", "INPUT", "BATCH", "INSTANCES"}, tableRows)
}

func modelsLoad(args []string) {
	fs := flag.NewFlagSet("models load", flag.ExitOnError)
	id         := fs.String("id", "", "model id (required)")
	backend    := fs.String("backend", "cpu", "backend: cpu|tensorrt|ascend|mps")
	onnxPath   := fs.String("onnx", "", "path to ONNX model file")
	enginePath := fs.String("engine", "", "path to pre-built engine/OM file")
	batch      := fs.Int("batch", 1, "batch size")
	channels   := fs.Int("channels", 3, "input channels")
	height     := fs.Int("height", 640, "input height")
	width      := fs.Int("width", 640, "input width")
	fs.Parse(args)

	if *id == "" {
		die("usage: infer-ctl models load --id <id> --backend <backend> [--onnx <path>] [--engine <path>] [--batch N] [--height H] [--width W]")
	}
	resp := must(send(map[string]any{
		"cmd":         "load_model",
		"id":          *id,
		"backend":     *backend,
		"onnx_path":   *onnxPath,
		"engine_path": *enginePath,
		"batch_size":  *batch,
		"input_channels": *channels,
		"input_height": *height,
		"input_width":  *width,
	}))
	fmt.Printf("loaded model %s\n", resp["id"])
}

func modelsUnload(args []string) {
	id := requireID(args, "usage: infer-ctl models unload <id>")
	resp := must(send(map[string]any{"cmd": "unload_model", "id": id}))
	fmt.Printf("unloaded model %s\n", resp["id"])
}

func cmdModels(args []string) {
	if len(args) == 0 {
		listModels()
		return
	}
	switch args[0] {
	case "load":
		modelsLoad(args[1:])
	case "unload", "remove", "rm":
		modelsUnload(args[1:])
	default:
		die(fmt.Sprintf("unknown subcommand %q\n\nusage: infer-ctl models [load|unload] ...", args[0]))
	}
}

// ── main ──────────────────────────────────────────────────────────────────────

var usage = `usage: infer-ctl [-s <socket>] <command> [subcommand] [args]

read commands:
  health                              check daemon health
  tasks                               list all tasks
  sources  (alias: streams)           list sources and health state
  pipelines                           list pipeline templates
  models                              list loaded models
  metrics                             dump Prometheus metrics text

write commands:
  sources add  --id <id> --url <url>  add a new source
  sources remove <id>                 remove a source
  pipelines add --id <id> --file F    add a pipeline from JSON file
  pipelines remove <id>               remove a pipeline
  tasks add --id <id> --source <s> --pipeline <p>   add a task
  tasks remove <id>                   remove a task
  models load --id <id> [--backend B] [--onnx P]    load a model
  models unload <id>                  unload a model

task shortcuts:
  start <id>                          start a stopped task
  stop  <id>                          stop a running task

flags:
`

func main() {
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	switch args[0] {
	case "health":
		resp := must(send(map[string]any{"cmd": "health"}))
		fmt.Println(resp["status"])

	case "tasks":
		cmdTasks(args[1:])

	case "start":
		if len(args) < 2 {
			die("usage: infer-ctl start <id>")
		}
		resp := must(send(map[string]any{"cmd": "start_task", "id": args[1]}))
		fmt.Printf("started %s\n", resp["id"])

	case "stop":
		if len(args) < 2 {
			die("usage: infer-ctl stop <id>")
		}
		resp := must(send(map[string]any{"cmd": "stop_task", "id": args[1]}))
		fmt.Printf("stopped %s\n", resp["id"])

	case "sources", "streams":
		cmdSources(args[1:])

	case "pipelines":
		cmdPipelines(args[1:])

	case "models":
		cmdModels(args[1:])

	case "metrics":
		resp := must(send(map[string]any{"cmd": "metrics"}))
		if m, ok := resp["data"].(string); ok {
			fmt.Print(m)
		}

	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", args[0])
		flag.Usage()
		os.Exit(1)
	}
}
