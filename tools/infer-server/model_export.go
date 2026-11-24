package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/goccy/go-yaml"
)

// Leave room for the file part and envelope when the ZIP is uploaded again.
const maxExportArchiveBytes = maxUploadBytes - (64 << 10)

var (
	errExportTooLarge = errors.New("model archive would exceed the upload limit")
	errExportFiles    = errors.New("active version contains missing or unsupported model files")
)

type modelExportDescription struct {
	Version       string   `json:"version"`
	ConfigYAML    string   `json:"config_yaml"`
	OverridesYAML string   `json:"overrides_yaml"`
	WeightPaths   []string `json:"weight_paths"`
}

func registerModelExportRoute(router gin.IRoutes) {
	authenticate := authMiddleware()
	router.GET("/models/repository/:id/export", func(c *gin.Context) {
		if apiKey == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, ErrorResponse{
				Status: "error", Code: "export_auth_required",
				Message: "Configure INFER_API_KEY on the management server to enable model export",
			})
			return
		}
		authenticate(c)
	}, handleExportModel)
}

// handleExportModel godoc
//
//	@Summary	Download the active version of a repository model
//	@Tags		models
//	@Produce	application/zip
//	@Security	ApiKeyAuth
//	@Param		id	path		string	true	"Model ID"
//	@Success	200	{file}		file
//	@Failure	400	{object}	ErrorResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	404	{object}	ErrorResponse
//	@Failure	413	{object}	ErrorResponse
//	@Failure	422	{object}	ErrorResponse
//	@Failure	500	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse
//	@Router		/models/repository/{id}/export [get]
func handleExportModel(c *gin.Context) {
	fail := func(status int, code, message string) {
		c.JSON(status, ErrorResponse{Status: "error", Code: code, Message: message})
	}
	id := c.Param("id")
	if !modelIDPattern.MatchString(id) {
		fail(http.StatusBadRequest, "invalid_model_id", "invalid model ID")
		return
	}
	if modelRepositoryPath == "" {
		fail(http.StatusServiceUnavailable, "no_repository", "MODEL_REPOSITORY is not configured on the management server")
		return
	}
	resp, err := call(map[string]any{"cmd": "describe_model_export", "id": id})
	if err != nil {
		fail(http.StatusServiceUnavailable, "export_unavailable", "engine could not describe the model for export")
		return
	}
	if resp["status"] != "ok" {
		code, _ := resp["code"].(string)
		message, _ := resp["message"].(string)
		switch code {
		case "model_not_found":
			fail(http.StatusNotFound, code, message)
		case "model_not_exportable":
			fail(http.StatusUnprocessableEntity, code, message)
		case "no_repository":
			fail(http.StatusServiceUnavailable, code, message)
		default:
			fail(http.StatusServiceUnavailable, "export_unavailable", "model export is unavailable on the engine")
		}
		return
	}

	var desc modelExportDescription
	data, err := json.Marshal(resp["data"])
	if err != nil || json.Unmarshal(data, &desc) != nil || !desc.valid() {
		fail(http.StatusServiceUnavailable, "export_unavailable", "engine returned an invalid model export description")
		return
	}

	ctx := c.Request.Context()
	config, err := desc.portableConfig(id)
	if err != nil {
		fail(http.StatusUnprocessableEntity, "model_not_exportable", "could not convert the model config to upload format")
		return
	}
	archive, err := os.CreateTemp("", "infer-model-export-*.zip")
	if err == nil {
		defer os.Remove(archive.Name())
		defer archive.Close()
		err = writeModelExport(ctx, archive, id, desc, config)
	}
	if ctx.Err() != nil {
		return
	}
	if err != nil {
		switch {
		case errors.Is(err, errExportTooLarge):
			fail(http.StatusRequestEntityTooLarge, "export_too_large", errExportTooLarge.Error())
		case errors.Is(err, errExportFiles):
			fail(http.StatusUnprocessableEntity, "model_not_exportable", errExportFiles.Error())
		default:
			log.Printf("export model %s: %v", id, err)
			fail(http.StatusInternalServerError, "export_io_error", "could not build the model archive")
		}
		return
	}
	size, err := archive.Seek(0, io.SeekCurrent)
	if err == nil {
		_, err = archive.Seek(0, io.SeekStart)
	}
	if err != nil {
		fail(http.StatusInternalServerError, "export_io_error", "could not read the model archive")
		return
	}
	filename := id + "-v" + desc.Version + ".zip"
	c.Header("Cache-Control", "no-store")
	c.Header("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": filename}))
	c.DataFromReader(http.StatusOK, size, "application/zip", archive, nil)
}

func (d modelExportDescription) valid() bool {
	version, err := strconv.ParseUint(d.Version, 10, 64)
	if err != nil || version == 0 || strconv.FormatUint(version, 10) != d.Version ||
		d.ConfigYAML == "" || len(d.ConfigYAML) > maxModelConfigBytes ||
		d.OverridesYAML == "" || len(d.OverridesYAML) > maxModelConfigBytes || len(d.WeightPaths) == 0 {
		return false
	}
	backend := weightBackend(d.WeightPaths[0])
	if backend == "" || (backend != "ascend" && len(d.WeightPaths) != 1) {
		return false
	}
	seen := make(map[string]bool)
	for _, name := range d.WeightPaths {
		if !validExportPath(name) || seen[name] || weightBackend(name) != backend {
			return false
		}
		seen[name] = true
	}
	return true
}

func (d modelExportDescription) portableConfig(id string) ([]byte, error) {
	var config, overrides map[string]any
	if err := yaml.Unmarshal([]byte(d.ConfigYAML), &config); err != nil {
		return nil, err
	}
	if err := yaml.Unmarshal([]byte(d.OverridesYAML), &overrides); err != nil {
		return nil, err
	}
	if config == nil || overrides == nil {
		return nil, errExportFiles
	}
	for _, field := range []string{"weight_file", "onnx_path", "engine_path", "om_paths"} {
		delete(config, field)
	}
	for field, value := range overrides {
		config[field] = value
	}
	config["id"] = id
	data, err := yaml.Marshal(config)
	if err == nil && len(data) > maxModelConfigBytes {
		return nil, errExportFiles
	}
	return data, err
}

func validExportPath(name string) bool {
	if !fs.ValidPath(name) || name == "." || strings.ContainsAny(name, "\\\x00") ||
		strings.EqualFold(path.Base(name), "config.yaml") {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if strings.HasPrefix(part, ".") || part == "__MACOSX" {
			return false
		}
	}
	return true
}

type exportFile struct {
	name string
	size int64
}

func collectExportFiles(ctx context.Context, version *os.Root, weights []string, configSize int) ([]exportFile, error) {
	var files []exportFile
	remaining := int64(maxExportArchiveBytes) - int64(configSize)
	add := func(name string, size int64) error {
		if len(files) >= maxArchiveEntries-1 || size > remaining {
			return errExportTooLarge
		}
		files = append(files, exportFile{name, size})
		remaining -= size
		return nil
	}
	for _, name := range weights {
		prefix := ""
		for _, part := range strings.Split(path.Dir(name), "/") {
			if part == "." {
				break
			}
			prefix = path.Join(prefix, part)
			info, err := version.Lstat(filepath.FromSlash(prefix))
			if err != nil || !info.IsDir() {
				return nil, errExportFiles
			}
		}
		info, err := version.Lstat(filepath.FromSlash(name))
		if err != nil || !info.Mode().IsRegular() {
			return nil, errExportFiles
		}
		if err := add(name, info.Size()); err != nil {
			return nil, err
		}
	}
	if weightBackend(weights[0]) != "cpu" {
		return files, nil
	}
	// ONNX external data uses paths relative to the model and can have any extension.
	err := fs.WalkDir(version.FS(), ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if name == "." {
			return nil
		}
		if !validExportPath(name) {
			if entry.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if !entry.Type().IsRegular() || weightBackend(name) != "" {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		return add(name, info.Size())
	})
	return files, err
}

type exportArchiveWriter struct {
	io.Writer
	ctx       context.Context
	remaining int64
}

func (w *exportArchiveWriter) Write(p []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	if int64(len(p)) > w.remaining {
		return 0, errExportTooLarge
	}
	n, err := w.Writer.Write(p)
	w.remaining -= int64(n)
	return n, err
}

func writeModelExport(ctx context.Context, dst io.Writer, id string, desc modelExportDescription, config []byte) error {
	repo, err := os.OpenRoot(modelRepositoryPath)
	if err != nil {
		return err
	}
	defer repo.Close()
	versionPath := filepath.Join(id, desc.Version)
	for _, name := range []string{id, versionPath} {
		info, err := repo.Lstat(name)
		if err != nil || !info.IsDir() {
			return errExportFiles
		}
	}
	version, err := repo.OpenRoot(versionPath)
	if err != nil {
		return errExportFiles
	}
	defer version.Close()

	files, err := collectExportFiles(ctx, version, desc.WeightPaths, len(config))
	if err != nil {
		return err
	}
	archive := zip.NewWriter(&exportArchiveWriter{Writer: dst, ctx: ctx, remaining: maxExportArchiveBytes})
	configEntry, err := archive.CreateHeader(&zip.FileHeader{Name: "config.yaml", Method: zip.Store})
	if err != nil {
		return err
	}
	if _, err := configEntry.Write(config); err != nil {
		return err
	}
	for _, entry := range files {
		file, err := version.Open(filepath.FromSlash(entry.name))
		if err != nil {
			return errExportFiles
		}
		info, err := file.Stat()
		if err != nil || !info.Mode().IsRegular() || info.Size() != entry.size {
			file.Close()
			return errExportFiles
		}
		target, err := archive.CreateHeader(&zip.FileHeader{Name: path.Join("1", entry.name), Method: zip.Store})
		if err == nil {
			_, err = io.CopyN(target, file, entry.size)
		}
		file.Close()
		if err != nil {
			return err
		}
	}
	return archive.Close()
}
