package main

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestUpdateSource(t *testing.T) {
	oldKey, oldCall, oldStore := apiKey, callFn, store
	t.Cleanup(func() { apiKey, callFn, store = oldKey, oldCall, oldStore })
	apiKey = "local-test-key"
	path := filepath.Join(t.TempDir(), "state.json")
	var err error
	store, err = NewStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SaveSource(SourceCreate{ID: "camera", URL: "rtsp://old/stream"}); err != nil {
		t.Fatal(err)
	}

	router := gin.New()
	router.PUT("/sources/:id", authMiddleware(), handleUpdateSource)
	for _, test := range []struct {
		name, id, bearer, body, engineCode string
		status                             int
		engineCalled                       bool
	}{
		{"unauthorized", "camera", "", `{"url":"rtsp://new/stream"}`, "", 401, false},
		{"missing address", "camera", apiKey, `{"open_timeout_ms":1000}`, "", 400, false},
		{"invalid setting", "camera", apiKey, `{"url":"rtsp://new/stream","open_timeout_ms":"slow"}`, "", 400, false},
		{"changed type", "camera", apiKey, `{"url":"/video.mp4"}`, "invalid_argument", 400, true},
		{"busy", "camera", apiKey, `{"url":"rtsp://new/stream"}`, "in_use", 409, true},
		{"missing source", "missing", apiKey, `{"url":"rtsp://new/stream"}`, "not_found", 404, true},
		{"persistence failure", "camera", apiKey, `{"url":"rtsp://new/stream"}`, "persistence_failed", 500, true},
		{"edit configured source", "yaml-camera", apiKey, `{"url":"rtsp://new/stream"}`, "", 200, true},
		{"edit API source", "camera", apiKey, `{"id":"other","url":"rtsp://new/stream","max_reconnect_attempts":0}`, "", 200, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			called := false
			callFn = func(command map[string]any) (map[string]any, error) {
				called = true
				if command["cmd"] != "update_source" || command["id"] != test.id {
					t.Fatalf("unexpected command: %#v", command)
				}
				if test.name == "edit API source" && command["max_reconnect_attempts"] != float64(0) {
					t.Fatalf("explicit zero disappeared: %#v", command)
				}
				if _, present := command["read_timeout_ms"]; present {
					t.Fatalf("omitted setting was sent: %#v", command)
				}
				if test.engineCode != "" {
					return map[string]any{"status": "error", "code": test.engineCode}, nil
				}
				return map[string]any{"status": "ok", "id": test.id, "data": map[string]any{
					"id": test.id, "url": "rtsp://new/stream",
					"config": map[string]any{"max_reconnect_attempts": 0, "read_timeout_ms": 5000},
				}}, nil
			}
			request := httptest.NewRequest(http.MethodPut, "/sources/"+test.id, strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer "+test.bearer)
			result := httptest.NewRecorder()
			router.ServeHTTP(result, request)
			if result.Code != test.status || called != test.engineCalled {
				t.Fatalf("status %d, called %v: %s", result.Code, called, result.Body)
			}
		})
	}

	reopened, err := NewStore(path)
	if err != nil {
		t.Fatal(err)
	}
	sources := reopened.Snapshot().Sources
	saved := sources["camera"]
	if len(sources) != 1 || saved.URL != "rtsp://new/stream" ||
		saved.MaxReconnectAttempts == nil || *saved.MaxReconnectAttempts != 0 ||
		saved.ReadTimeoutMs == nil || *saved.ReadTimeoutMs != 5000 {
		t.Fatalf("unexpected source replay state: %#v", sources)
	}
}
