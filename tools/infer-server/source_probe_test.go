package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestSourceProbeRequiresAuthentication(t *testing.T) {
	oldKey, oldCall := apiKey, callFn
	t.Cleanup(func() { apiKey, callFn = oldKey, oldCall })
	callFn = func(map[string]any) (map[string]any, error) {
		t.Fatal("unauthorized request reached the engine")
		return nil, nil
	}

	router := gin.New()
	registerSourceProbeRoute(router)
	for _, scenario := range []struct {
		key, authorization string
		status             int
	}{
		{"", "", http.StatusServiceUnavailable},
		{"local-test-key", "", http.StatusUnauthorized},
		{"local-test-key", "Bearer wrong", http.StatusUnauthorized},
	} {
		apiKey = scenario.key
		request := httptest.NewRequest(http.MethodPost, "/sources/cam/probe", nil)
		request.Header.Set("Authorization", scenario.authorization)
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != scenario.status {
			t.Fatalf("got status %d, want %d", response.Code, scenario.status)
		}
	}
}

func TestSourceProbeForwardsRegisteredIDAndMapsResponses(t *testing.T) {
	oldKey, oldCall := apiKey, callFn
	t.Cleanup(func() { apiKey, callFn = oldKey, oldCall })
	apiKey = "local-test-key"
	router := gin.New()
	registerSourceProbeRoute(router)
	for _, scenario := range []struct {
		code   string
		status int
	}{
		{"connection_failed", http.StatusOK},
		{"source_not_found", http.StatusNotFound},
		{"source_probe_in_progress", http.StatusConflict},
		{"source_probe_unsupported", http.StatusUnprocessableEntity},
		{"source_probe_busy", http.StatusTooManyRequests},
		{"source_probe_unavailable", http.StatusServiceUnavailable},
	} {
		t.Run(scenario.code, func(t *testing.T) {
			callFn = func(command map[string]any) (map[string]any, error) {
				if len(command) != 2 || command["cmd"] != "probe_source" || command["id"] != "cam" {
					t.Fatalf("unexpected command: %v", command)
				}
				switch scenario.status {
				case http.StatusOK:
					return map[string]any{"status": "ok", "data": map[string]any{
						"id": "cam", "last_probe": map[string]any{"passed": false, "reason": scenario.code},
					}}, nil
				case http.StatusServiceUnavailable:
					return nil, errors.New("private connection details")
				default:
					return map[string]any{"status": "error", "code": scenario.code}, nil
				}
			}
			request := httptest.NewRequest(http.MethodPost, "/sources/cam/probe", strings.NewReader(`{"url":"rtsp://other"}`))
			request.Header.Set("Authorization", "Bearer local-test-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != scenario.status || !strings.Contains(response.Body.String(), scenario.code) {
				t.Fatalf("unexpected response %d: %s", response.Code, response.Body)
			}
			if strings.Contains(response.Body.String(), "private connection details") {
				t.Fatalf("engine error leaked: %s", response.Body)
			}
		})
	}
}
