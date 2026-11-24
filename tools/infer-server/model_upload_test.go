package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/goccy/go-yaml"
)

func modelUploadRouter(t *testing.T) (*gin.Engine, string) {
	t.Helper()
	repo := t.TempDir()
	oldPath, oldCall := modelRepositoryPath, callFn
	modelRepositoryPath = repo
	t.Cleanup(func() { modelRepositoryPath, callFn = oldPath, oldCall })
	router := gin.New()
	router.POST("/models/upload", handleUploadModel)
	router.POST("/models/upload/analyze", handleAnalyzeArchive)
	return router, repo
}

func uploadMultipart(t *testing.T, router http.Handler, path string, fields map[string]string,
	field, filename string, content []byte) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			t.Fatal(err)
		}
	}
	part, err := writer.CreateFormFile(field, filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, path, &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

func uploadArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for path, data := range files {
		f, err := w.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func postPublish(t *testing.T, router http.Handler, id, token string, fields url.Values) *httptest.ResponseRecorder {
	t.Helper()
	if fields == nil {
		fields = url.Values{}
	}
	fields.Set("id", id)
	fields.Set("token", token)
	request := httptest.NewRequest(http.MethodPost, "/models/upload", strings.NewReader(fields.Encode()))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

func getUploadPreview(t *testing.T, response *httptest.ResponseRecorder) ArchiveAnalysisResponse {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("analyze returned %d: %s", response.Code, response.Body)
	}
	var preview ArchiveAnalysisResponse
	if err := json.Unmarshal(response.Body.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if !uploadTokenPattern.MatchString(preview.Token) {
		t.Fatalf("invalid draft token: %v", preview.Token)
	}
	return preview
}

func requireUploadStatus(t *testing.T, response *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if response.Code != status {
		t.Fatalf("expected %d, got %d: %s", status, response.Code, response.Body)
	}
	if code != "" {
		var result ErrorResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || result.Code != code {
			t.Fatalf("expected error code %q: %s", code, response.Body)
		}
	}
}

func TestUploadRetainsArchiveConfigAndRetriesWithSameFiles(t *testing.T) {
	router, repo := modelUploadRouter(t)
	config := "id: vehicle\nbackend: onnx_cpu\nversion: yolov8\ninput_size: [384, 512]\n" +
		"batch_size: 1\nconf_thresh: 0\nclass_names: [car, bus]\nnum_classes: 2\n" +
		"instance_count: 2\ndevice_ids: [0, 2]\ncascade:\n  - model_id: color\n    attribute_key: paint\n"
	body := uploadArchive(t, map[string]string{
		"detector/config.yaml":     config,
		"detector/3/net.onnx":      "weights",
		"detector/3/net.onnx.data": "auxiliary data",
	})
	preview := getUploadPreview(t, uploadMultipart(t, router, "/models/upload/analyze", nil, "archive", "vehicle.zip", body))
	if preview.SuggestedID != "vehicle" || preview.Backend != "cpu" ||
		len(preview.InputSize) != 2 || preview.InputSize[0] != 384 ||
		preview.ConfThresh != 0 || len(preview.DeviceIDs) != 2 {
		t.Fatalf("unexpected preview: %+v", preview)
	}

	calls := 0
	callFn = func(command map[string]any) (map[string]any, error) {
		calls++
		candidate := filepath.Join(repo, ".uploads", preview.Token, command["attempt"].(string), "vehicle")
		data, err := os.ReadFile(filepath.Join(candidate, "config.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		var model struct {
			InputSize  []int    `yaml:"input_size"`
			DeviceIDs  []int    `yaml:"device_ids"`
			ClassNames []string `yaml:"class_names"`
			Cascade    []struct {
				ModelID string `yaml:"model_id"`
			} `yaml:"cascade"`
			NumClasses int     `yaml:"num_classes"`
			ConfThresh float64 `yaml:"conf_thresh"`
			OnnxPath   string  `yaml:"onnx_path"`
		}
		if err := yaml.Unmarshal(data, &model); err != nil {
			t.Fatal(err)
		}
		if len(model.InputSize) != 2 || model.InputSize[1] != 512 ||
			len(model.DeviceIDs) != 2 || model.DeviceIDs[1] != 2 || len(model.ClassNames) != 2 ||
			len(model.Cascade) != 1 || model.Cascade[0].ModelID != "color" ||
			model.NumClasses != 3 || model.ConfThresh != 0 || model.OnnxPath != "net.onnx" {
			t.Fatalf("config changed unexpectedly: %s", data)
		}
		weights, err := os.ReadFile(filepath.Join(candidate, "1", "net.onnx"))
		if string(weights) != "weights" || err != nil {
			t.Fatalf("weight missing: %v", err)
		}
		if _, err := os.Stat(filepath.Join(candidate, "1", "net.onnx.data")); err != nil {
			t.Fatal(err)
		}
		if calls == 1 {
			return nil, errors.New("socket offline")
		}
		if calls == 2 {
			return map[string]any{"status": "error", "code": "missing_dependency", "message": "register color"}, nil
		}
		return map[string]any{"status": "ok", "id": "vehicle"}, nil
	}

	for _, check := range []struct {
		status int
		code   string
	}{
		{http.StatusServiceUnavailable, "engine_unavailable"},
		{http.StatusBadRequest, "missing_dependency"},
		{http.StatusCreated, ""},
	} {
		response := postPublish(t, router, "vehicle", preview.Token, url.Values{"num_classes": {"3"}})
		requireUploadStatus(t, response, check.status, check.code)
		if check.code != "" {
			if _, err := os.Stat(filepath.Join(repo, "vehicle")); !os.IsNotExist(err) {
				t.Fatalf("failed attempt left a final directory: %v", err)
			}
		}
	}
	if calls != 3 {
		t.Fatalf("expected 3 calls, got %d", calls)
	}
}

func TestUploadRetriesACommittedModelAfterLostResponse(t *testing.T) {
	router, repo := modelUploadRouter(t)
	preview := getUploadPreview(t, uploadMultipart(t, router, "/models/upload/analyze", nil, "file", "detector.onnx", []byte("weights")))
	if preview.InputSize != nil || preview.SuggestedID != "detector" || preview.Backend != "cpu" {
		t.Fatalf("unexpected bare weight metadata: %+v", preview)
	}
	calls := 0
	callFn = func(command map[string]any) (map[string]any, error) {
		calls++
		attempt := command["attempt"].(string)
		target := filepath.Join(repo, "detector")
		if calls == 1 {
			candidate := filepath.Join(repo, ".uploads", preview.Token, attempt, "detector")
			if err := os.WriteFile(filepath.Join(candidate, ".publication-token"), []byte(preview.Token), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(candidate, target); err != nil {
				t.Fatal(err)
			}
			return nil, errors.New("reply lost")
		}
		if attempt != "" {
			t.Fatalf("retry should reconcile the committed model, got attempt %q", attempt)
		}
		return map[string]any{"status": "ok", "id": "detector"}, nil
	}
	first := postPublish(t, router, "detector", preview.Token, url.Values{"input_height": {"384"}, "input_width": {"512"}})
	requireUploadStatus(t, first, http.StatusServiceUnavailable, "engine_unavailable")
	retry := postPublish(t, router, "detector", preview.Token, nil)
	requireUploadStatus(t, retry, http.StatusCreated, "")
}

func TestLegacyUploadUsesPublicationAndReturnsConflict(t *testing.T) {
	router, repo := modelUploadRouter(t)
	callFn = func(command map[string]any) (map[string]any, error) {
		if command["cmd"] != "publish_model" || command["id"] != "legacy" {
			t.Fatalf("unexpected engine request: %v", command)
		}
		cfg := filepath.Join(repo, ".uploads", command["token"].(string), command["attempt"].(string), "legacy", "config.yaml")
		if _, err := os.Stat(cfg); err != nil {
			t.Fatal(err)
		}
		return map[string]any{"status": "error", "code": "already_exists", "message": "model exists"}, nil
	}
	response := uploadMultipart(t, router, "/models/upload", map[string]string{"id": "legacy", "backend": "trt"}, "engine", "model.engine", []byte("weights"))
	requireUploadStatus(t, response, http.StatusConflict, "already_exists")
}

func TestUploadRejectsMultipleModelsAndConflictingBackend(t *testing.T) {
	router, _ := modelUploadRouter(t)
	for _, files := range []map[string]string{
		{"a.onnx": "a", "b.onnx": "b"},
		{"a.onnx": "a", "b.engine": "b"},
		{"a.onnx": "a", "config.yaml": "backend: trt\n"},
		{"a.onnx": "a", "../outside": "unexpected"},
		{"a.onnx": "a", "a/config.yaml": "{}\n", "b/config.yaml": "{}\n"},
	} {
		body := uploadArchive(t, files)
		response := uploadMultipart(t, router, "/models/upload/analyze", nil, "archive", "bad.zip", body)
		requireUploadStatus(t, response, http.StatusBadRequest, "invalid_model_files")
	}
}

func TestAscendArchiveKeepsExplicitBatchMapping(t *testing.T) {
	router, repo := modelUploadRouter(t)
	body := uploadArchive(t, map[string]string{
		"config.yaml": "backend: ascend\ninput_size: [640, 640]\nom_paths: {1: base.om, 4: four.om}\n",
		"base.om":     "one", "four.om": "four",
	})
	preview := getUploadPreview(t, uploadMultipart(t, router, "/models/upload/analyze", nil, "archive", "weights.zip", body))
	if preview.Backend != "ascend" || preview.BatchSize != 4 {
		t.Fatalf("unexpected Ascend preview: %+v", preview)
	}
	callFn = func(command map[string]any) (map[string]any, error) {
		path := filepath.Join(repo, ".uploads", preview.Token, command["attempt"].(string), "ascend", "config.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var config modelUploadConfig
		if err := yaml.Unmarshal(data, &config); err != nil {
			t.Fatal(err)
		}
		if config.OMPaths[1] != "base.om" || config.OMPaths[4] != "four.om" {
			t.Fatalf("wrong Ascend batches: %s", data)
		}
		return map[string]any{"status": "ok", "id": "ascend"}, nil
	}
	requireUploadStatus(t, postPublish(t, router, "ascend", preview.Token, nil), http.StatusCreated, "")
}
