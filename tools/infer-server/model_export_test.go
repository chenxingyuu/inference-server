package main

import (
	"archive/zip"
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
)

const exportSourceID = "vehicle"

func exportRequest(router http.Handler, id, authorization string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodGet, "/models/repository/"+id+"/export", nil)
	request.Header.Set("Authorization", authorization)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

func writeExportFixture(t *testing.T, base string, files map[string]string) {
	t.Helper()
	for name, contents := range files {
		target := filepath.Join(base, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, []byte(contents), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func readExportZIP(t *testing.T, data []byte) map[string]string {
	t.Helper()
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	files := make(map[string]string)
	for _, f := range reader.File {
		entry, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		content, err := io.ReadAll(entry)
		entry.Close()
		if err != nil {
			t.Fatal(err)
		}
		files[f.Name] = string(content)
	}
	return files
}

func TestExportRoundTripsThroughModelUpload(t *testing.T) {
	for _, scenario := range []struct {
		backend, archiveBackend, config string
		weights                         []string
		aux                             []string
	}{
		{"cpu", "cpu", "onnx_path: model/net.onnx\n", []string{"model/net.onnx"},
			[]string{"model/data/weights.bin", "model/data/weights.engine", "model/data/weights.om", "model/data/weights.onnx"}},
		{"tensorrt", "trt", "engine_path: net.engine\n", []string{"net.engine"}, nil},
		{"ascend", "ascend", "om_paths: {1: small.om, 4: large.om}\n", []string{"small.om", "large.om"}, nil},
	} {
		t.Run(scenario.backend, func(t *testing.T) {
			router, repo := modelUploadRouter(t)
			registerModelExportRoute(router)
			base := filepath.Join(repo, exportSourceID)
			original := "version: yolov8\ninput_size: [384, 512]\nconf_thresh: 0\n" +
				"class_names: [\"01\", \"true\"]\nnum_classes: 2\ninstance_count: 2\n" +
				"device_ids: [0, 2]\ncascade: [{model_id: color, attribute_key: paint}]\n" +
				"active_version: 3\nweight_file: /old/path\n"
			files := map[string]string{
				"config.yaml": original, ".publication-token": "private",
				"3/.hidden": "private", "3/unused.onnx": "unselected", "4/different.onnx": "other version",
				"3/unrelated.txt": "not an ONNX dependency",
			}
			expected := make(map[string]string)
			for _, name := range append(append([]string{}, scenario.weights...), scenario.aux...) {
				files["3/"+name] = "data for " + name
				expected["1/"+name] = files["3/"+name]
			}
			if scenario.backend == "cpu" {
				var locations []string
				for _, file := range scenario.aux {
					locations = append(locations, strings.TrimPrefix(file, "model/"))
				}
				model := fixtureONNXExternalData(locations...)
				files["3/model/net.onnx"] = model
				expected["1/model/net.onnx"] = model
			}
			writeExportFixture(t, base, files)

			desc := modelExportDescription{
				Version: "3", ConfigYAML: original, WeightPaths: scenario.weights,
				OverridesYAML: "active_version: 1\nbackend: " + scenario.backend +
					"\nbatch_size: 4\nnms_thresh: 0.5\n" + scenario.config,
			}
			callFn = func(command map[string]any) (map[string]any, error) {
				switch command["cmd"] {
				case "describe_model_export":
					if command["id"] != exportSourceID {
						t.Fatalf("unexpected description request: %v", command)
					}
					return map[string]any{"status": "ok", "data": desc}, nil
				case "publish_model":
					id := command["id"].(string)
					if id == exportSourceID {
						return map[string]any{"status": "error", "code": "already_exists"}, nil
					}
					candidate := filepath.Join(repo, ".uploads", command["token"].(string), command["attempt"].(string), id)
					if err := os.Rename(candidate, filepath.Join(repo, id)); err != nil {
						t.Fatal(err)
					}
					return map[string]any{"status": "ok"}, nil
				default:
					t.Fatalf("unexpected engine command: %v", command)
					return nil, nil
				}
			}

			response := exportRequest(router, exportSourceID, "Bearer "+testModelUploadKey)
			requireUploadStatus(t, response, http.StatusOK, "")
			if headers := response.Header(); headers.Get("Content-Type") != "application/zip" ||
				headers.Get("Content-Disposition") != "attachment; filename=vehicle-v3.zip" ||
				headers.Get("Cache-Control") != "no-store" || headers.Get("Content-Length") == "" {
				t.Fatalf("unexpected download headers: %v", headers)
			}
			archive := readExportZIP(t, response.Body.Bytes())
			zipConfig := archive["config.yaml"]
			if zipConfig == "" || strings.Contains(zipConfig, "/old/path") {
				t.Fatalf("invalid exported config: %q", zipConfig)
			}
			delete(archive, "config.yaml")
			if !reflect.DeepEqual(archive, expected) {
				t.Fatalf("exported files = %v, want %v", archive, expected)
			}

			preview := getUploadPreview(t, uploadMultipart(t, router, "/models/upload/analyze", nil,
				"archive", "vehicle-v3.zip", response.Body.Bytes()))
			if preview.SuggestedID != exportSourceID || preview.Backend != scenario.archiveBackend ||
				preview.BatchSize != 4 || preview.ConfThresh != 0 || preview.NmsThresh != 0.5 ||
				!reflect.DeepEqual(preview.InputSize, []int{384, 512}) {
				t.Fatalf("unexpected reimport preview: %+v", preview)
			}
			requireUploadStatus(t, postPublish(t, router, exportSourceID, preview.Token, nil),
				http.StatusConflict, "already_exists")
			const copyID = "vehicle-smoke"
			requireUploadStatus(t, postPublish(t, router, copyID, preview.Token, nil), http.StatusCreated, "")

			data, err := os.ReadFile(filepath.Join(repo, copyID, "config.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			var published struct {
				ID            string           `yaml:"id"`
				ActiveVersion int              `yaml:"active_version"`
				ClassNames    []string         `yaml:"class_names"`
				DeviceIDs     []int            `yaml:"device_ids"`
				Cascade       []map[string]any `yaml:"cascade"`
				OMPaths       map[int]string   `yaml:"om_paths"`
			}
			if err := yaml.Unmarshal(data, &published); err != nil {
				t.Fatal(err)
			}
			if published.ID != copyID || published.ActiveVersion != 1 ||
				!reflect.DeepEqual(published.ClassNames, []string{"01", "true"}) ||
				!reflect.DeepEqual(published.DeviceIDs, []int{0, 2}) || len(published.Cascade) != 1 ||
				published.Cascade[0]["model_id"] != "color" ||
				(scenario.backend == "ascend" && !reflect.DeepEqual(published.OMPaths, map[int]string{1: "small.om", 4: "large.om"})) {
				t.Fatalf("reimport changed config: %s", data)
			}
			for name, want := range expected {
				got, err := os.ReadFile(filepath.Join(repo, copyID, filepath.FromSlash(name)))
				if err != nil || string(got) != want {
					t.Fatalf("reimported %q = %q, want %q (%v)", name, got, want, err)
				}
			}
			if got, err := os.ReadFile(filepath.Join(base, "config.yaml")); err != nil || string(got) != original {
				t.Fatalf("original config was changed: %s (%v)", got, err)
			}
		})
	}
}

func TestExportPreservesNumericModelID(t *testing.T) {
	desc := modelExportDescription{ConfigYAML: "backend: cpu", OverridesYAML: "onnx_path: net.onnx"}
	data, err := desc.portableConfig("001")
	if err != nil {
		t.Fatal(err)
	}
	var config modelUploadConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	if config.ID != "001" {
		t.Fatalf("export changed numeric model ID: %s", data)
	}
}

func TestExportRejectsUnavailableOrUnsafeSources(t *testing.T) {
	router, repo := modelUploadRouter(t)
	registerModelExportRoute(router)
	const bearer = "Bearer " + testModelUploadKey
	callFn = func(command map[string]any) (map[string]any, error) {
		t.Fatalf("unexpected engine request: %v", command)
		return nil, nil
	}
	apiKey = ""
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusServiceUnavailable, "export_auth_required")
	apiKey = testModelUploadKey
	requireUploadStatus(t, exportRequest(router, exportSourceID, ""), http.StatusUnauthorized, "unauthorized")
	requireUploadStatus(t, exportRequest(router, exportSourceID, "Bearer invalid"), http.StatusUnauthorized, "unauthorized")
	requireUploadStatus(t, exportRequest(router, "vehicle.bad", bearer), http.StatusBadRequest, "invalid_model_id")

	callFn = func(map[string]any) (map[string]any, error) {
		return map[string]any{"status": "error", "code": "model_not_found"}, nil
	}
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusNotFound, "model_not_found")

	desc := modelExportDescription{
		Version: "3", ConfigYAML: "backend: cpu\n", OverridesYAML: "onnx_path: net.onnx\n",
		WeightPaths: []string{"../net.onnx"},
	}
	callFn = func(map[string]any) (map[string]any, error) {
		return map[string]any{"status": "ok", "data": desc}, nil
	}
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusServiceUnavailable, "export_unavailable")
	desc.WeightPaths = []string{"net.onnx"}
	version := filepath.Join(repo, exportSourceID, "3")
	writeExportFixture(t, version, map[string]string{"data": "sidecar"})
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusUnprocessableEntity, "model_not_exportable")

	outside := t.TempDir()
	writeExportFixture(t, outside, map[string]string{"net.onnx": "outside"})
	weight := filepath.Join(version, "net.onnx")
	if err := os.Symlink(filepath.Join(outside, "net.onnx"), weight); err != nil {
		t.Fatal(err)
	}
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusUnprocessableEntity, "model_not_exportable")
	if err := os.Remove(weight); err != nil {
		t.Fatal(err)
	}
	writeExportFixture(t, version, map[string]string{"net.onnx": ""})
	if err := os.Truncate(weight, maxExportArchiveBytes); err != nil {
		t.Fatal(err)
	}
	requireUploadStatus(t, exportRequest(router, exportSourceID, bearer), http.StatusRequestEntityTooLarge, "export_too_large")
}

func TestONNXExportRejectsUnimportableExternalData(t *testing.T) {
	for _, scenario := range []struct{ label, location, message string }{
		{"hidden", ".hidden/weights.bin", "cannot be imported"},
		{"missing", "missing.bin", "must be a regular file"},
		{"outside version", "../weights.bin", "must stay within the model directory"},
		{"symlink", "linked.bin", "must be a regular file"},
	} {
		t.Run(scenario.label, func(t *testing.T) {
			router, repo := modelUploadRouter(t)
			registerModelExportRoute(router)
			version := filepath.Join(repo, exportSourceID, "1")
			writeExportFixture(t, version, map[string]string{
				"net.onnx": fixtureONNXExternalData(scenario.location), ".hidden/weights.bin": "external data",
				"weights.bin": "external data",
			})
			if err := os.Symlink("weights.bin", filepath.Join(version, "linked.bin")); err != nil {
				t.Fatal(err)
			}
			desc := modelExportDescription{
				Version: "1", ConfigYAML: "backend: cpu\n", OverridesYAML: "onnx_path: net.onnx\n",
				WeightPaths: []string{"net.onnx"},
			}
			callFn = func(map[string]any) (map[string]any, error) {
				return map[string]any{"status": "ok", "data": desc}, nil
			}

			response := exportRequest(router, exportSourceID, "Bearer "+testModelUploadKey)
			requireUploadStatus(t, response, http.StatusUnprocessableEntity, "model_not_exportable")
			if !strings.Contains(response.Body.String(), scenario.message) || response.Header().Get("Content-Disposition") != "" {
				t.Fatalf("expected a dependency error, got %s", response.Body)
			}
		})
	}
}
