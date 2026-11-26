package main

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func modelTestRouter(t *testing.T, key string, fake func(map[string]any) (map[string]any, error)) http.Handler {
	t.Helper()
	oldKey, oldCall := apiKey, callFn
	apiKey, callFn = key, fake
	t.Cleanup(func() { apiKey, callFn = oldKey, oldCall })

	router := gin.New()
	registerModelUploadRoutes(router)
	registerModelExportRoute(router)
	registerModelTestRoute(router)
	protected := router.Group("/", authMiddleware())
	protected.GET("/models/detail/:id", handleModelDetail)
	protected.GET("/models/repository", handleListRepository)
	protected.DELETE("/models/:id", handleUnloadModel)
	return router
}

func testImage(t *testing.T, width, height int) []byte {
	t.Helper()
	var data bytes.Buffer
	if err := png.Encode(&data, image.NewGray(image.Rect(0, 0, width, height))); err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func imageRequest(t *testing.T, data []byte, count int, authorization string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for i := 0; i < count; i++ {
		part, err := writer.CreateFormFile("image", "sample.png")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := part.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, "/models/test/vehicle", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.Header.Set("Authorization", authorization)
	return request
}

func TestModelImageForwardsOnlyOneValidatedImage(t *testing.T) {
	png := testImage(t, 8, 12)
	calls := 0
	router := modelTestRouter(t, "key", func(command map[string]any) (map[string]any, error) {
		calls++
		if command["cmd"] != "test_model" || command["id"] != "vehicle" || !bytes.Equal(command["image"].([]byte), png) {
			t.Fatalf("wrong command: %v", command)
		}
		return map[string]any{"status": "ok", "data": map[string]any{"width": 8, "height": 12}}, nil
	})

	for _, scenario := range []struct {
		name          string
		image         []byte
		count         int
		authorization string
		status        int
	}{
		{"correct image", png, 1, "Bearer key", http.StatusOK},
		{"bad key", nil, 0, "Bearer other", http.StatusUnauthorized},
		{"missing image", png, 0, "Bearer key", http.StatusBadRequest},
		{"extra image", png, 2, "Bearer key", http.StatusBadRequest},
		{"garbage image", []byte("garbage"), 1, "Bearer key", http.StatusBadRequest},
		{"oversized pixels", testImage(t, 4097, 1), 1, "Bearer key", http.StatusBadRequest},
		{"oversized upload", make([]byte, maxModelTestImageBytes+1), 1, "Bearer key", http.StatusRequestEntityTooLarge},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			router.ServeHTTP(response, imageRequest(t, scenario.image, scenario.count, scenario.authorization))
			if response.Code != scenario.status {
				t.Fatalf("status %d, want %d: %s", response.Code, scenario.status, response.Body.String())
			}
		})
	}
	if calls != 1 {
		t.Fatalf("forwarded %d images, want 1", calls)
	}
}

func TestModelImageRequiresConfiguredAPIKey(t *testing.T) {
	router := modelTestRouter(t, "", func(map[string]any) (map[string]any, error) {
		t.Fatal("unconfigured route reached engine")
		return nil, nil
	})
	response := httptest.NewRecorder()
	router.ServeHTTP(response, imageRequest(t, nil, 0, ""))
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), "model_test_auth_required") {
		t.Fatalf("unexpected response: %d %s", response.Code, response.Body.String())
	}
}

func TestModelTestAndDetailMapEngineResults(t *testing.T) {
	png := testImage(t, 1, 1)
	for _, scenario := range []struct {
		code   string
		status int
		err    error
	}{
		{"model_not_registered", http.StatusConflict, nil},
		{"model_test_busy", http.StatusConflict, nil},
		{"model_test_timeout", http.StatusGatewayTimeout, nil},
		{"model_test_failed", http.StatusUnprocessableEntity, nil},
		{"engine_unavailable", http.StatusServiceUnavailable, errors.New("socket unavailable")},
	} {
		t.Run(scenario.code, func(t *testing.T) {
			router := modelTestRouter(t, "key", func(map[string]any) (map[string]any, error) {
				return map[string]any{"status": "error", "code": scenario.code, "message": "test error"}, scenario.err
			})
			response := httptest.NewRecorder()
			router.ServeHTTP(response, imageRequest(t, png, 1, "Bearer key"))
			if response.Code != scenario.status || !strings.Contains(response.Body.String(), scenario.code) {
				t.Fatalf("unexpected response: %d %s", response.Code, response.Body.String())
			}
		})
	}

	router := modelTestRouter(t, "key", func(command map[string]any) (map[string]any, error) {
		if command["cmd"] == "unload_model" {
			return map[string]any{"status": "error", "code": "in_use"}, nil
		}
		if command["cmd"] != "describe_model" {
			t.Fatalf("wrong command: %v", command)
		}
		if command["id"] == "missing" {
			return map[string]any{"status": "error", "code": "model_not_found", "message": "model not found"}, nil
		}
		return map[string]any{"status": "ok", "data": map[string]any{"registered": true}}, nil
	})

	for _, scenario := range []struct {
		method string
		url    string
		code   int
	}{
		{http.MethodGet, "/models/detail/repository", http.StatusOK},
		{http.MethodGet, "/models/detail/missing", http.StatusNotFound},
		{http.MethodDelete, "/models/vehicle", http.StatusConflict},
	} {
		request := httptest.NewRequest(scenario.method, scenario.url, nil)
		request.Header.Set("Authorization", "Bearer key")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != scenario.code {
			t.Fatalf("unexpected response for %s: %d %s", scenario.url, response.Code, response.Body.String())
		}
	}
}
