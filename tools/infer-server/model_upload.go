package main

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/goccy/go-yaml"
)

const (
	uploadTTL           = 30 * time.Minute
	maxUploadBytes      = 2 << 30
	maxModelConfigBytes = 1 << 20
)

var (
	modelIDPattern     = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	uploadTokenPattern = regexp.MustCompile(`^[0-9a-f]{32}$`)
	omBatchPattern     = regexp.MustCompile(`(?:^|[_-])(?:b|batch[_-]?)([0-9]+)\.om$`)
	// Holds cleanup off while a candidate is being published.
	publishDraftMu sync.Mutex
)

type uploadDraft struct {
	Config    []byte
	Files     map[string]string // path relative to version 1 → file under the draft's files directory
	Backend   string
	Weight    string
	OMPaths   map[int]string
	ExpiresAt time.Time
}

type modelUploadConfig struct {
	ID            string         `yaml:"id"`
	Backend       string         `yaml:"backend"`
	Version       string         `yaml:"version"`
	ModelType     string         `yaml:"model_type"`
	InputSize     []int          `yaml:"input_size"`
	BatchSize     int            `yaml:"batch_size"`
	ConfThresh    float64        `yaml:"conf_thresh"`
	NmsThresh     float64        `yaml:"nms_thresh"`
	NumClasses    int            `yaml:"num_classes"`
	InstanceCount int            `yaml:"instance_count"`
	DeviceID      int            `yaml:"device_id"`
	DeviceIDs     []int          `yaml:"device_ids"`
	OMPaths       map[int]string `yaml:"om_paths"`
	WeightFile    string         `yaml:"weight_file"`
	EnginePath    string         `yaml:"engine_path"`
	OnnxPath      string         `yaml:"onnx_path"`
}

// ArchiveAnalysisResponse previews a staged model before publication.
type ArchiveAnalysisResponse struct {
	Token         string    `json:"token"`
	SuggestedID   string    `json:"suggested_id,omitempty"`
	Backend       string    `json:"backend"`
	YoloVersion   string    `json:"yolo_version"`
	ModelType     string    `json:"model_type"`
	BatchSize     int       `json:"batch_size"`
	ConfThresh    float64   `json:"conf_thresh"`
	NmsThresh     float64   `json:"nms_thresh"`
	NumClasses    int       `json:"num_classes"`
	InstanceCount int       `json:"instance_count"`
	InputSize     []int     `json:"input_size,omitempty"`
	DeviceID      int       `json:"device_id"`
	DeviceIDs     []int     `json:"device_ids,omitempty"`
	HasConfig     bool      `json:"has_config"`
	FilesFound    []string  `json:"files_found"`
	ExpiresAt     time.Time `json:"expires_at"`
}

func newToken() string {
	var token [16]byte
	_, _ = rand.Read(token[:])
	return fmt.Sprintf("%x", token)
}

func readDraft(dir string) (uploadDraft, error) {
	var draft uploadDraft
	data, err := os.ReadFile(filepath.Join(dir, ".draft.json"))
	if err != nil {
		return draft, err
	}
	err = json.Unmarshal(data, &draft)
	return draft, err
}

func startTokenCleanup() {
	if modelRepositoryPath == "" {
		return
	}
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for now := range ticker.C {
			if !publishDraftMu.TryLock() {
				continue
			}
			root := filepath.Join(modelRepositoryPath, ".uploads")
			entries, _ := os.ReadDir(root)
			for _, entry := range entries {
				if !entry.IsDir() || !uploadTokenPattern.MatchString(entry.Name()) {
					continue
				}
				dir := filepath.Join(root, entry.Name())
				draft, err := readDraft(dir)
				if err == nil && now.After(draft.ExpiresAt) {
					_ = os.RemoveAll(dir)
				} else if err != nil {
					// Covers interrupted analysis and protects newly created drafts.
					if info, statErr := entry.Info(); statErr == nil && now.Sub(info.ModTime()) > uploadTTL {
						_ = os.RemoveAll(dir)
					}
				}
			}
			publishDraftMu.Unlock()
		}
	}()
}

func backendName(value string) string {
	switch strings.ToLower(value) {
	case "trt", "tensorrt", "cuda":
		return "trt"
	case "cpu", "onnx", "onnx_cpu":
		return "cpu"
	case "mps", "metal", "onnx_mps":
		return "mps"
	case "ascend", "acl", "npu":
		return "ascend"
	default:
		return ""
	}
}

func weightBackend(name string) string {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".om":
		return "ascend"
	case ".engine":
		return "trt"
	case ".onnx":
		return "cpu"
	default:
		return ""
	}
}

func compatibleBackend(configured, detected string) bool {
	return configured == detected || (configured == "mps" && detected == "cpu")
}

func archivePath(base, file string) (string, bool) {
	rel, err := filepath.Rel(base, file)
	return rel, err == nil && filepath.IsLocal(rel) && rel != "."
}

func inspectUpload(dir string) (uploadDraft, ArchiveAnalysisResponse, error) {
	draft := uploadDraft{Files: make(map[string]string), OMPaths: make(map[int]string), ExpiresAt: time.Now().Add(uploadTTL)}
	var preview ArchiveAnalysisResponse
	var configName string
	var weights []string
	err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		preview.FilesFound = append(preview.FilesFound, rel)
		if strings.EqualFold(entry.Name(), "config.yaml") {
			if configName != "" {
				return fmt.Errorf("upload one model configuration at a time")
			}
			configName = rel
		}
		if backend := weightBackend(entry.Name()); backend != "" {
			if draft.Backend != "" && draft.Backend != backend {
				return fmt.Errorf("files for different backends cannot be published together")
			}
			draft.Backend = backend
			weights = append(weights, rel)
		}
		return nil
	})
	if err != nil {
		return draft, preview, err
	}
	if len(weights) == 0 {
		return draft, preview, fmt.Errorf("no .onnx, .engine or .om model weights found")
	}

	var config modelUploadConfig
	base := filepath.Dir(weights[0])
	if configName != "" {
		configPath := filepath.Join(dir, configName)
		info, err := os.Stat(configPath)
		if err != nil || info.Size() > maxModelConfigBytes {
			return draft, preview, fmt.Errorf("cannot read model config.yaml or it exceeds 1 MiB")
		}
		draft.Config, err = os.ReadFile(configPath)
		if err != nil {
			return draft, preview, err
		}
		if err = yaml.Unmarshal(draft.Config, &config); err != nil {
			return draft, preview, fmt.Errorf("invalid config.yaml: %w", err)
		}
		if config.Backend != "" {
			selected := backendName(config.Backend)
			if !compatibleBackend(selected, draft.Backend) {
				return draft, preview, fmt.Errorf("configured backend does not match the uploaded weights")
			}
			draft.Backend = selected
		}
		base = filepath.Dir(configName)
		version := ""
		unversioned := false
		for _, weight := range weights {
			rel, ok := archivePath(base, weight)
			if !ok {
				return draft, preview, fmt.Errorf("weights must be in the config.yaml directory or a subdirectory")
			}
			component, _, hasDir := strings.Cut(rel, string(os.PathSeparator))
			n, _ := strconv.Atoi(component)
			if !hasDir || n < 1 {
				unversioned = true
				continue
			}
			if version != "" && version != component {
				return draft, preview, fmt.Errorf("upload one model version at a time")
			}
			version = component
		}
		if version != "" {
			if unversioned {
				return draft, preview, fmt.Errorf("upload one model version at a time")
			}
			base = filepath.Join(base, version)
		}
	} else {
		for _, weight := range weights[1:] {
			for _, ok := archivePath(base, weight); !ok; _, ok = archivePath(base, weight) {
				base = filepath.Dir(base)
			}
		}
	}

	weightsByName := make(map[string]bool, len(weights))
	for _, weight := range weights {
		name, _ := archivePath(base, weight)
		weightsByName[name] = true
	}
	if draft.Backend == "ascend" {
		if config.WeightFile != "" && len(config.OMPaths) == 0 {
			config.OMPaths = map[int]string{max(1, config.BatchSize): config.WeightFile}
		}
		if len(config.OMPaths) > 0 {
			seen := make(map[string]bool)
			for batch, path := range config.OMPaths {
				name := filepath.Clean(path)
				if batch < 1 || !weightsByName[name] || seen[name] {
					return draft, preview, fmt.Errorf("invalid or duplicate weight in om_paths for batch %d", batch)
				}
				draft.OMPaths[batch] = name
				seen[name] = true
			}
			if len(seen) != len(weights) {
				return draft, preview, fmt.Errorf("om_paths must identify every uploaded Ascend weight")
			}
		} else {
			for name := range weightsByName {
				batch := 1
				if len(weightsByName) == 1 && config.BatchSize > 0 {
					batch = config.BatchSize
				} else if match := omBatchPattern.FindStringSubmatch(strings.ToLower(filepath.Base(name))); len(match) == 2 {
					batch, _ = strconv.Atoi(match[1])
				}
				if batch < 1 || draft.OMPaths[batch] != "" {
					return draft, preview, fmt.Errorf("ambiguous Ascend weights, specify om_paths or use names such as model_b1.om")
				}
				draft.OMPaths[batch] = name
			}
		}
	} else {
		if len(weights) != 1 {
			return draft, preview, fmt.Errorf("upload one weight file for this backend")
		}
		draft.Weight, _ = archivePath(base, weights[0])
		for _, path := range []string{config.WeightFile, config.EnginePath, config.OnnxPath} {
			if path != "" && filepath.Clean(path) != draft.Weight {
				return draft, preview, fmt.Errorf("configured weight path does not match the uploaded file")
			}
		}
	}

	for _, path := range preview.FilesFound {
		if name, ok := archivePath(base, path); ok && path != configName {
			draft.Files[name] = path
		}
	}
	preview.SuggestedID = config.ID
	if !modelIDPattern.MatchString(preview.SuggestedID) {
		name := filepath.Base(weights[0])
		if configName != "" && filepath.Dir(configName) != "." {
			name = filepath.Dir(configName)
		}
		preview.SuggestedID = strings.TrimSuffix(filepath.Base(name), filepath.Ext(name))
		if !modelIDPattern.MatchString(preview.SuggestedID) {
			preview.SuggestedID = ""
		}
	}
	return draft, preview, nil
}

func defaultUploadConfig(draft uploadDraft) modelUploadConfig {
	config := modelUploadConfig{
		Backend: draft.Backend, Version: "v8", ModelType: "detector",
		BatchSize: 1, ConfThresh: 0.25, NmsThresh: 0.45, NumClasses: 80, InstanceCount: 1,
	}
	for batch := range draft.OMPaths {
		config.BatchSize = max(config.BatchSize, batch)
	}
	return config
}

func fillPreview(draft uploadDraft, preview *ArchiveAnalysisResponse) error {
	config := defaultUploadConfig(draft)
	if len(draft.Config) > 0 {
		if err := yaml.Unmarshal(draft.Config, &config); err != nil {
			return err
		}
	}
	if len(config.InputSize) != 0 && (len(config.InputSize) != 2 ||
		config.InputSize[0] < 1 || config.InputSize[1] < 1) {
		return fmt.Errorf("input_size must contain positive height and width")
	}
	preview.Backend = draft.Backend
	preview.YoloVersion = strings.TrimPrefix(strings.ToLower(config.Version), "yolo")
	if !strings.HasPrefix(preview.YoloVersion, "v") {
		preview.YoloVersion = "v" + preview.YoloVersion
	}
	preview.ModelType = config.ModelType
	preview.BatchSize = config.BatchSize
	preview.ConfThresh = config.ConfThresh
	preview.NmsThresh = config.NmsThresh
	preview.NumClasses = config.NumClasses
	preview.InstanceCount = config.InstanceCount
	preview.InputSize = config.InputSize
	preview.DeviceID = config.DeviceID
	preview.DeviceIDs = config.DeviceIDs
	preview.HasConfig = len(draft.Config) > 0
	preview.ExpiresAt = draft.ExpiresAt
	return nil
}

func writeUploadError(c *gin.Context, status int, code, message, token string) {
	body := gin.H{"status": "error", "code": code, "message": message}
	if token != "" {
		body["token"] = token
	}
	c.JSON(status, body)
}

func registerModelUploadRoutes(router gin.IRoutes) {
	authenticate := authMiddleware()
	requireKey := func(c *gin.Context) {
		if apiKey == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, ErrorResponse{
				Status: "error", Code: "publication_auth_required",
				Message: "Configure INFER_API_KEY on the management server to enable model publication",
			})
			return
		}
		authenticate(c)
	}
	router.POST("/models/upload/analyze", requireKey, handleAnalyzeArchive)
	router.POST("/models/upload", requireKey, handleUploadModel)
}

func parseUploadRequest(c *gin.Context) bool {
	if modelRepositoryPath == "" {
		writeUploadError(c, http.StatusBadRequest, "no_repository",
			"MODEL_REPOSITORY is not configured on the management server", "")
		return false
	}
	contentType := c.ContentType()
	if contentType == "multipart/form-data" {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxUploadBytes)
		if err := c.Request.ParseMultipartForm(8 << 20); err != nil {
			var sizeError *http.MaxBytesError
			if errors.As(err, &sizeError) {
				writeUploadError(c, http.StatusRequestEntityTooLarge, "upload_too_large",
					"upload exceeds the 2 GiB limit", "")
			} else {
				writeUploadError(c, http.StatusBadRequest, "invalid_upload", "invalid multipart upload", "")
			}
			return false
		}
	} else if err := c.Request.ParseForm(); err != nil {
		writeUploadError(c, http.StatusBadRequest, "invalid_upload", "invalid upload form", "")
		return false
	}
	return true
}

func saveUploadFiles(c *gin.Context, token, dir string, source *multipart.FileHeader,
	archive bool) (ArchiveAnalysisResponse, error) {
	var preview ArchiveAnalysisResponse
	files := filepath.Join(dir, "files")
	if err := os.MkdirAll(files, 0o755); err != nil {
		return preview, err
	}
	if archive {
		compressed := filepath.Join(dir, ".archive")
		if err := c.SaveUploadedFile(source, compressed); err != nil {
			return preview, err
		}
		err := extractModelArchive(compressed, strings.ToLower(source.Filename), files)
		_ = os.Remove(compressed)
		if err != nil {
			return preview, err
		}
	} else {
		name := filepath.Base(source.Filename)
		backend := weightBackend(name)
		if backend == "" {
			return preview, fmt.Errorf("select a .onnx, .engine, .om or model archive")
		}
		if backend == "ascend" {
			// A standalone .om is Batch 1. Explicit multipart slots retain their batch.
			name = "model_b1.om"
			for _, batch := range []int{4, 8, 16} {
				file, err := c.FormFile(fmt.Sprintf("batch_%d", batch))
				if errors.Is(err, http.ErrMissingFile) {
					continue
				}
				if err != nil || weightBackend(file.Filename) != "ascend" {
					return preview, fmt.Errorf("invalid .om weight for batch %d", batch)
				}
				if err := c.SaveUploadedFile(file, filepath.Join(files, fmt.Sprintf("model_b%d.om", batch))); err != nil {
					return preview, err
				}
			}
		}
		if err := c.SaveUploadedFile(source, filepath.Join(files, name)); err != nil {
			return preview, err
		}
	}
	draft, preview, err := inspectUpload(files)
	if err != nil {
		return preview, err
	}
	if !archive {
		name := filepath.Base(source.Filename)
		suggested := strings.TrimSuffix(name, filepath.Ext(name))
		if modelIDPattern.MatchString(suggested) {
			preview.SuggestedID = suggested
		} else {
			preview.SuggestedID = ""
		}
		// The analyst sees the original filename even when we normalize the stored batch name.
		if len(preview.FilesFound) == 1 {
			preview.FilesFound[0] = name
		}
	}
	if err := fillPreview(draft, &preview); err != nil {
		return preview, err
	}
	data, err := json.Marshal(draft)
	if err != nil {
		return preview, err
	}
	if err := os.WriteFile(filepath.Join(dir, ".draft.json"), data, 0o600); err != nil {
		return preview, err
	}
	preview.Token = token
	return preview, nil
}

func stageUpload(c *gin.Context, file *multipart.FileHeader, archive bool) (ArchiveAnalysisResponse, bool) {
	token := newToken()
	dir := filepath.Join(modelRepositoryPath, ".uploads", token)
	preview, err := saveUploadFiles(c, token, dir, file, archive)
	if err != nil {
		_ = os.RemoveAll(dir)
		var pathError *os.PathError
		if errors.As(err, &pathError) {
			writeUploadError(c, http.StatusInternalServerError, "upload_io_error", err.Error(), "")
		} else {
			writeUploadError(c, http.StatusBadRequest, "invalid_model_files", err.Error(), "")
		}
		return preview, false
	}
	return preview, true
}

// handleAnalyzeArchive godoc
//
//	@Summary	Analyze uploaded weights or a single-model archive
//	@Tags		models
//	@Accept		mpfd
//	@Produce	json
//	@Security	ApiKeyAuth
//	@Param		archive	formData	file	false	"Model archive (.zip, .tar.gz or .tgz)"
//	@Param		file	formData	file	false	"Model weight (.onnx, .engine or .om)"
//	@Param		batch_4	formData	file	false	"Optional Ascend Batch 4 weight"
//	@Param		batch_8	formData	file	false	"Optional Ascend Batch 8 weight"
//	@Param		batch_16	formData	file	false	"Optional Ascend Batch 16 weight"
//	@Success	200	{object}	ArchiveAnalysisResponse
//	@Failure	400	{object}	ErrorResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	413	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse "Model publication requires INFER_API_KEY"
//	@Router		/models/upload/analyze [post]
func handleAnalyzeArchive(c *gin.Context) {
	if !parseUploadRequest(c) {
		return
	}
	archive, archiveErr := c.FormFile("archive")
	file, fileErr := c.FormFile("file")
	if (archiveErr == nil) == (fileErr == nil) {
		writeUploadError(c, http.StatusBadRequest, "missing_file",
			"provide one weight file or one model archive", "")
		return
	}
	if archiveErr == nil {
		file = archive
	}
	if preview, ok := stageUpload(c, file, archiveErr == nil); ok {
		c.JSON(http.StatusOK, preview)
	}
}

func candidateConfig(c *gin.Context, id string, draft uploadDraft) ([]byte, error) {
	defaults := defaultUploadConfig(draft)
	config := map[string]any{
		"version": defaults.Version, "model_type": defaults.ModelType, "batch_size": defaults.BatchSize,
		"conf_thresh": defaults.ConfThresh, "nms_thresh": defaults.NmsThresh,
		"num_classes": defaults.NumClasses, "instance_count": defaults.InstanceCount,
		"input_size": []int{640, 640}, // legacy clients without dimension fields
	}
	if len(draft.Config) > 0 {
		var original map[string]any
		if err := yaml.Unmarshal(draft.Config, &original); err != nil {
			return nil, err
		}
		for key, value := range original {
			config[key] = value
		}
	}
	backend := draft.Backend
	if field, present := c.GetPostForm("backend"); present {
		backend = backendName(field)
		if !compatibleBackend(backend, draft.Backend) {
			return nil, fmt.Errorf("selected backend does not match the uploaded weights")
		}
	}
	config["backend"] = backend
	config["id"] = id
	config["active_version"] = 1
	for _, pathField := range []string{"engine_path", "onnx_path", "om_paths", "weight_file"} {
		delete(config, pathField)
	}
	switch backend {
	case "ascend":
		config["om_paths"] = draft.OMPaths
	case "trt":
		config["engine_path"] = draft.Weight
	default:
		config["onnx_path"] = draft.Weight
	}

	for _, pair := range [][2]string{{"yolo_version", "version"}, {"model_type", "model_type"}} {
		if val, ok := c.GetPostForm(pair[0]); ok {
			config[pair[1]] = val
		}
	}
	for _, key := range []string{"batch_size", "num_classes", "instance_count", "device_id"} {
		if raw, ok := c.GetPostForm(key); ok {
			value, err := strconv.Atoi(raw)
			if err != nil || value < 0 || (key != "device_id" && value == 0) {
				return nil, fmt.Errorf("invalid %s", key)
			}
			if key == "batch_size" && backend == "ascend" && c.PostForm("token") == "" {
				// Keep the direct upload contract for clients providing larger batch weights.
				value = max(value, defaults.BatchSize)
			}
			config[key] = value
		}
	}
	if raw, ok := c.GetPostForm("device_ids"); ok {
		if strings.TrimSpace(raw) == "" {
			delete(config, "device_ids")
		} else {
			var devices []int
			for _, part := range strings.Split(raw, ",") {
				device, err := strconv.Atoi(strings.TrimSpace(part))
				if err != nil || device < 0 {
					return nil, fmt.Errorf("device IDs must be nonnegative integers")
				}
				devices = append(devices, device)
			}
			config["device_ids"] = devices
		}
	}
	for _, key := range []string{"conf_thresh", "nms_thresh"} {
		if raw, ok := c.GetPostForm(key); ok {
			value, err := strconv.ParseFloat(raw, 64)
			if err != nil || value < 0 || value > 1 || math.IsNaN(value) {
				return nil, fmt.Errorf("%s must be between 0 and 1", key)
			}
			config[key] = value
		}
	}
	height, hasHeight := c.GetPostForm("input_height")
	width, hasWidth := c.GetPostForm("input_width")
	if hasHeight || hasWidth {
		h, hErr := strconv.Atoi(height)
		w, wErr := strconv.Atoi(width)
		if hErr != nil || wErr != nil || h < 1 || w < 1 {
			return nil, fmt.Errorf("enter a positive input height and width")
		}
		config["input_size"] = []int{h, w}
	}
	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, err
	}
	var validated modelUploadConfig
	if err := yaml.Unmarshal(data, &validated); err != nil {
		return nil, err
	}
	if validated.ModelType != "detector" && validated.ModelType != "classifier" {
		return nil, fmt.Errorf("model_type must be detector or classifier")
	}
	for _, threshold := range []float64{validated.ConfThresh, validated.NmsThresh} {
		if math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
			return nil, fmt.Errorf("confidence and NMS thresholds must be between 0 and 1")
		}
	}
	return data, nil
}

func prepareCandidate(dir, attempt, id string, draft uploadDraft, config []byte) error {
	candidate := filepath.Join(dir, attempt, id)
	version := filepath.Join(candidate, "1")
	if err := os.MkdirAll(version, 0o755); err != nil {
		return err
	}
	for name, file := range draft.Files {
		if !filepath.IsLocal(name) || !filepath.IsLocal(file) {
			return fmt.Errorf("invalid staged model file")
		}
		destination := filepath.Join(version, name)
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			return err
		}
		source := filepath.Join(dir, "files", file)
		if err := linkOrCopyWeight(source, destination); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(candidate, "config.yaml"), config, 0o644)
}

func linkOrCopyWeight(source, destination string) error {
	if err := os.Link(source, destination); err == nil {
		return nil
	}
	// Some shared volumes do not support hard links.
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, input)
	closeErr := output.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func publishUpload(c *gin.Context, id, token string) {
	if !uploadTokenPattern.MatchString(token) {
		writeUploadError(c, http.StatusBadRequest, "invalid_upload", "invalid upload token", "")
		return
	}
	if !publishDraftMu.TryLock() {
		writeUploadError(c, http.StatusConflict, "publish_busy", "another model is being checked, retry shortly", token)
		return
	}
	defer publishDraftMu.Unlock()

	dir := filepath.Join(modelRepositoryPath, ".uploads", token)
	attempt := ""
	marker, _ := os.ReadFile(filepath.Join(modelRepositoryPath, id, ".publication-token"))
	if strings.TrimSpace(string(marker)) != token {
		draft, err := readDraft(dir)
		if err != nil || !time.Now().Before(draft.ExpiresAt) {
			writeUploadError(c, http.StatusGone, "upload_expired", "upload expired, select the model files again", "")
			return
		}
		if previous, _ := os.ReadFile(filepath.Join(dir, ".published-id")); strings.TrimSpace(string(previous)) != "" {
			previousID := strings.TrimSpace(string(previous))
			if previousID != id && modelIDPattern.MatchString(previousID) {
				previousToken, _ := os.ReadFile(filepath.Join(modelRepositoryPath, previousID, ".publication-token"))
				if strings.TrimSpace(string(previousToken)) == token {
					writeUploadError(c, http.StatusConflict, "already_published", "this upload was already published as "+previousID, token)
					return
				}
			}
		}
		config, err := candidateConfig(c, id, draft)
		if err != nil {
			writeUploadError(c, http.StatusBadRequest, "invalid_model", err.Error(), token)
			return
		}
		attempt = newToken()
		if err := prepareCandidate(dir, attempt, id, draft, config); err != nil {
			_ = os.RemoveAll(filepath.Join(dir, attempt))
			writeUploadError(c, http.StatusInternalServerError, "publish_io_error", err.Error(), token)
			return
		}
	}

	resp, err := call(map[string]any{"cmd": "publish_model", "id": id, "token": token, "attempt": attempt})
	if err != nil {
		// The engine may have committed before the connection was lost.
		// Retain this attempt until a retry or expiry.
		writeUploadError(c, http.StatusServiceUnavailable, "engine_unavailable",
			"engine did not confirm publication, retry with the same upload", token)
		return
	}
	if attempt != "" {
		_ = os.RemoveAll(filepath.Join(dir, attempt))
	}
	if resp["status"] != "ok" {
		code, _ := resp["code"].(string)
		message, _ := resp["message"].(string)
		if code == "" {
			code = "publish_unavailable"
		}
		if message == "" {
			message = "engine could not publish this model"
		}
		status := http.StatusBadRequest
		switch code {
		case "already_exists", "already_published", "publish_busy", "startup_config_invalid":
			status = http.StatusConflict
		case "upload_expired":
			status = http.StatusGone
		case "publish_io_error":
			status = http.StatusInternalServerError
		case "upload_missing", "publish_unavailable":
			status = http.StatusServiceUnavailable
		}
		writeUploadError(c, status, code, message, token)
		return
	}
	_ = os.RemoveAll(filepath.Join(dir, "files"))
	c.JSON(http.StatusCreated, OkResponse{Status: "ok", ID: id})
}

// handleUploadModel godoc
//
//	@Summary	Validate model weights and publish to the repository
//	@Tags		models
//	@Accept		mpfd,x-www-form-urlencoded
//	@Produce	json
//	@Security	ApiKeyAuth
//	@Param		id	formData	string	true	"Model ID (letters, numbers, _ and -)"
//	@Param		token	formData	string	false	"Token returned by upload/analyze"
//	@Param		backend	formData	string	false	"Backend for a direct upload"
//	@Param		onnx	formData	file	false	"ONNX weight for a direct upload"
//	@Param		engine	formData	file	false	"TensorRT weight for a direct upload"
//	@Param		batch_1	formData	file	false	"Ascend Batch 1 weight for a direct upload"
//	@Param		batch_4	formData	file	false	"Optional Ascend Batch 4 weight for a direct upload"
//	@Param		batch_8	formData	file	false	"Optional Ascend Batch 8 weight for a direct upload"
//	@Param		batch_16	formData	file	false	"Optional Ascend Batch 16 weight for a direct upload"
//	@Param		yolo_version	formData	string	false	"YOLO version"
//	@Param		model_type	formData	string	false	"Model type"
//	@Param		input_height	formData	int	false	"Model input height"
//	@Param		input_width	formData	int	false	"Model input width"
//	@Param		batch_size	formData	int	false	"Batch size"
//	@Param		conf_thresh	formData	number	false	"Confidence threshold"
//	@Param		nms_thresh	formData	number	false	"NMS threshold"
//	@Param		num_classes	formData	int	false	"Number of classes"
//	@Param		instance_count	formData	int	false	"Worker instances"
//	@Param		device_id	formData	int	false	"Default device ID"
//	@Param		device_ids	formData	string	false	"Comma-separated device IDs"
//	@Success	201	{object}	OkResponse
//	@Failure	400	{object}	ErrorResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	409	{object}	ErrorResponse
//	@Failure	410	{object}	ErrorResponse
//	@Failure	413	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse "Engine unavailable or publication requires INFER_API_KEY"
//	@Router		/models/upload [post]
func handleUploadModel(c *gin.Context) {
	if !parseUploadRequest(c) {
		return
	}
	id := c.PostForm("id")
	if !modelIDPattern.MatchString(id) {
		writeUploadError(c, http.StatusBadRequest, "invalid_id", "model ID must contain only letters, digits, _ and -", "")
		return
	}
	token := c.PostForm("token")
	if token == "" {
		backend := backendName(c.DefaultPostForm("backend", "cpu"))
		field := map[string]string{"cpu": "onnx", "mps": "onnx", "trt": "engine", "ascend": "batch_1"}[backend]
		if field == "" {
			writeUploadError(c, http.StatusBadRequest, "invalid_backend", "unsupported model backend", "")
			return
		}
		file, err := c.FormFile(field)
		if err != nil {
			writeUploadError(c, http.StatusBadRequest, "missing_file", "provide weight file in "+field, "")
			return
		}
		if !compatibleBackend(backend, weightBackend(file.Filename)) {
			writeUploadError(c, http.StatusBadRequest, "invalid_model_files", "weight file has the wrong extension", "")
			return
		}
		preview, ok := stageUpload(c, file, false)
		if !ok {
			return
		}
		token = preview.Token
	}
	publishUpload(c, id, token)
}
