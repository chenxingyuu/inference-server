package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
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

func call(cmd map[string]any) (map[string]any, error) {
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

func handleHealthz(c *gin.Context) {
	resp, err := call(map[string]any{"cmd": "health"})
	if err != nil {
		c.String(http.StatusServiceUnavailable, err.Error())
		return
	}
	if resp["status"] != "ok" {
		c.JSON(http.StatusServiceUnavailable, resp)
		return
	}
	c.String(http.StatusOK, "OK")
}

func handleMetrics(c *gin.Context) {
	resp, err := call(map[string]any{"cmd": "metrics"})
	if err != nil {
		c.String(http.StatusServiceUnavailable, err.Error())
		return
	}
	c.Header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	if m, ok := resp["data"].(string); ok {
		c.String(http.StatusOK, m)
	}
}

func listHandler(listCmd string) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := call(map[string]any{"cmd": listCmd})
		if err != nil {
			c.String(http.StatusServiceUnavailable, err.Error())
			return
		}
		c.JSON(http.StatusOK, resp["data"])
	}
}

func createHandler(addCmd string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var body map[string]any
		if err := c.ShouldBindJSON(&body); err != nil {
			c.String(http.StatusBadRequest, "invalid JSON body")
			return
		}
		body["cmd"] = addCmd
		resp, err := call(body)
		if err != nil {
			c.String(http.StatusServiceUnavailable, err.Error())
			return
		}
		code := http.StatusCreated
		if resp["status"] != "ok" {
			code = http.StatusBadRequest
		}
		c.JSON(code, resp)
	}
}

func deleteHandler(removeCmd string) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("id")
		resp, err := call(map[string]any{"cmd": removeCmd, "id": id})
		if err != nil {
			c.String(http.StatusServiceUnavailable, err.Error())
			return
		}
		code := http.StatusOK
		if resp["status"] != "ok" {
			code = http.StatusNotFound
		}
		c.JSON(code, resp)
	}
}

func taskActionHandler(c *gin.Context) {
	id := c.Param("id")
	action := c.Param("action")
	if action != "start" && action != "stop" {
		c.Status(http.StatusNotFound)
		return
	}
	resp, err := call(map[string]any{"cmd": action + "_task", "id": id})
	if err != nil {
		c.String(http.StatusServiceUnavailable, err.Error())
		return
	}
	code := http.StatusOK
	if resp["status"] != "ok" {
		code = http.StatusNotFound
	}
	c.JSON(code, resp)
}

func main() {
	r := gin.Default()

	r.GET("/healthz", handleHealthz)
	r.GET("/metrics", handleMetrics)

	r.GET("/sources", listHandler("list_sources"))
	r.POST("/sources", createHandler("add_source"))
	r.DELETE("/sources/:id", deleteHandler("remove_source"))

	r.GET("/pipelines", listHandler("list_pipelines"))
	r.POST("/pipelines", createHandler("add_pipeline"))
	r.DELETE("/pipelines/:id", deleteHandler("remove_pipeline"))

	r.GET("/models", listHandler("list_models"))
	r.POST("/models", createHandler("load_model"))
	r.DELETE("/models/:id", deleteHandler("unload_model"))

	r.GET("/tasks", listHandler("list_tasks"))
	r.POST("/tasks", createHandler("add_task"))
	r.DELETE("/tasks/:id", deleteHandler("remove_task"))
	r.POST("/tasks/:id/:action", taskActionHandler)

	fmt.Printf("infer-server listening on %s (socket: %s)\n", listenAddr, socketPath)
	if err := r.Run(listenAddr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
