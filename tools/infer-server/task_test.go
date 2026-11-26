package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestUpdateTaskAllowsCrossOriginPreflight(t *testing.T) {
	router := gin.New()
	router.Use(corsMiddleware())
	req := httptest.NewRequest(http.MethodOptions, "/tasks/task", nil)
	req.Header.Set("Origin", "https://infer-web.example")
	req.Header.Set("Access-Control-Request-Method", http.MethodPut)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	methods := strings.Split(rec.Header().Get("Access-Control-Allow-Methods"), ", ")
	if rec.Code != http.StatusNoContent || !slices.Contains(methods, http.MethodPut) {
		t.Fatalf("preflight returned HTTP %d and methods %v", rec.Code, methods)
	}
}

func TestTaskWritesForwardAndValidateParameters(t *testing.T) {
	const taskBody = `"id":"body-id","source_id":"camera","pipeline_id":"graph"`
	for _, route := range []struct {
		method  string
		path    string
		command string
		id      string
		status  int
	}{
		{http.MethodPost, "/tasks", "add_task", "body-id", http.StatusCreated},
		{http.MethodPut, "/tasks/path-id", "update_task", "path-id", http.StatusOK},
	} {
		t.Run(route.command, func(t *testing.T) {
			for _, scenario := range []struct {
				name      string
				fields    string
				options   map[string]any
				errorText string
			}{
				{
					name:   "all parameters",
					fields: `,"sample_fps":12,"sampling_mode":"time_based","use_hwdec":true,"use_ascend_dvpp":true,"ascend_device_id":2`,
					options: map[string]any{
						"sample_fps": float64(12), "sampling_mode": "time_based",
						"use_hwdec": true, "use_ascend_dvpp": true, "ascend_device_id": float64(2),
					},
				},
				{name: "omitted fps"},
				{
					name:      "explicit zero reaches engine",
					fields:    `,"sample_fps":0`,
					options:   map[string]any{"sample_fps": float64(0)},
					errorText: "sample_fps must be an integer between 1 and 2147483647",
				},
				{
					name:      "invalid JSON field type",
					fields:    `,"use_hwdec":"true"`,
					errorText: "use_hwdec must be of type bool",
				},
			} {
				t.Run(scenario.name, func(t *testing.T) {
					state, err := NewStore(filepath.Join(t.TempDir(), "state.json"))
					if err != nil {
						t.Fatal(err)
					}
					oldStore, oldCall := store, callFn
					t.Cleanup(func() { store, callFn = oldStore, oldCall })
					store = state
					wantCommand := map[string]any{
						"cmd": route.command, "id": route.id, "source_id": "camera", "pipeline_id": "graph",
						"use_hwdec": false, "use_ascend_dvpp": false, "ascend_device_id": float64(0),
					}
					for key, value := range scenario.options {
						wantCommand[key] = value
					}
					calls := 0
					callFn = func(command map[string]any) (map[string]any, error) {
						calls++
						if !reflect.DeepEqual(command, wantCommand) {
							t.Fatalf("forwarded %#v, want %#v", command, wantCommand)
						}
						if scenario.errorText != "" {
							return map[string]any{
								"status": "error", "code": "invalid_argument", "message": scenario.errorText,
							}, nil
						}
						return map[string]any{"status": "ok", "id": route.id}, nil
					}
					router := gin.New()
					router.POST("/tasks", handleAddTask)
					router.PUT("/tasks/:id", handleUpdateTask)
					req := httptest.NewRequest(route.method, route.path,
						strings.NewReader("{"+taskBody+scenario.fields+"}"))
					req.Header.Set("Content-Type", "application/json")
					rec := httptest.NewRecorder()
					router.ServeHTTP(rec, req)

					wantStatus, wantCalls := route.status, 1
					if scenario.errorText != "" {
						wantStatus = http.StatusBadRequest
						if scenario.options == nil {
							wantCalls = 0
						}
					}
					if rec.Code != wantStatus || calls != wantCalls {
						t.Fatalf("HTTP %d and %d socket calls, want %d and %d: %s",
							rec.Code, calls, wantStatus, wantCalls, rec.Body.String())
					}
					_, saved := store.Snapshot().Tasks[route.id]
					if scenario.errorText == "" {
						if !saved {
							t.Fatal("accepted task was not saved")
						}
						return
					}
					if saved {
						t.Fatal("rejected task was saved")
					}
					var body ErrorResponse
					if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
						t.Fatal(err)
					}
					if body.Code != "invalid_argument" || body.Message != scenario.errorText {
						t.Fatalf("unexpected error response: %+v", body)
					}
				})
			}
		})
	}
}

func TestTaskLifecycleListAndConflicts(t *testing.T) {
	oldCall := callFn
	t.Cleanup(func() { callFn = oldCall })
	callFn = func(command map[string]any) (map[string]any, error) {
		if command["cmd"] == "list_tasks" {
			return map[string]any{"status": "ok", "data": []TaskInfo{
				{ID: "waiting", State: "starting"},
				{ID: "failed", State: "failed", CleanupPending: true,
					Failure: &TaskFailure{Code: "startup_timeout", Message: "Startup timed out"}},
			}}, nil
		}
		return map[string]any{"status": "error", "code": "task_busy", "message": "cleanup pending"}, nil
	}
	router := gin.New()
	router.GET("/tasks", handleListTasks)
	router.POST("/tasks/:id/:action", handleTaskAction)
	router.PUT("/tasks/:id", handleUpdateTask)

	result := httptest.NewRecorder()
	router.ServeHTTP(result, httptest.NewRequest(http.MethodGet, "/tasks", nil))
	var tasks []TaskInfo
	if result.Code != http.StatusOK || json.Unmarshal(result.Body.Bytes(), &tasks) != nil || len(tasks) != 2 {
		t.Fatalf("list tasks returned %d: %s", result.Code, result.Body)
	}
	if tasks[0].State != "starting" || tasks[1].State != "failed" ||
		!tasks[1].CleanupPending || tasks[1].Failure == nil || tasks[1].Failure.Code != "startup_timeout" {
		t.Fatalf("lifecycle fields were lost: %+v", tasks)
	}
	for _, req := range []*http.Request{
		httptest.NewRequest(http.MethodPost, "/tasks/failed/start", nil),
		httptest.NewRequest(http.MethodPut, "/tasks/failed",
			strings.NewReader(`{"id":"failed","source_id":"cam","pipeline_id":"graph"}`)),
	} {
		req.Header.Set("Content-Type", "application/json")
		result = httptest.NewRecorder()
		router.ServeHTTP(result, req)
		if result.Code != http.StatusConflict {
			t.Fatalf("%s returned %d: %s", req.Method, result.Code, result.Body)
		}
	}
}
