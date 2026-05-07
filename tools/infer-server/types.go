package main

// SourceInfo describes a registered video source.
type SourceInfo struct {
	ID             string `json:"id"`
	URL            string `json:"url"`
	State          string `json:"state" enums:"CONNECTING,STREAMING,RECONNECTING,DEGRADED,STOPPED"`
	ReconnectCount int    `json:"reconnect_count"`
}

// SourceCreate is the request body for POST /sources.
type SourceCreate struct {
	ID                   string `json:"id"                              binding:"required"`
	URL                  string `json:"url"                             binding:"required"`
	ReconnectDelayMs     int    `json:"reconnect_delay_ms,omitempty"`
	MaxReconnectDelayMs  int    `json:"max_reconnect_delay_ms,omitempty"`
	DegradedThreshold    int    `json:"degraded_threshold,omitempty"`
	MaxReconnectAttempts int    `json:"max_reconnect_attempts,omitempty"`
}

// PipelineInfo describes a registered pipeline.
type PipelineInfo struct {
	ID        string   `json:"id"`
	Nodes     []string `json:"nodes"`
	EdgeCount int      `json:"edge_count"`
}

// StageConfig is a single node in a pipeline.
type StageConfig struct {
	ID   string            `json:"id"   binding:"required"`
	Type string            `json:"type" binding:"required"`
	With map[string]string `json:"with,omitempty"`
}

// EdgeConfig connects two nodes in a pipeline.
type EdgeConfig struct {
	From string `json:"from" binding:"required"`
	To   string `json:"to"   binding:"required"`
}

// PipelineCreate is the request body for POST /pipelines.
type PipelineCreate struct {
	ID    string        `json:"id"    binding:"required"`
	Nodes []StageConfig `json:"nodes" binding:"required"`
	Edges []EdgeConfig  `json:"edges,omitempty"`
}

// ModelInfo describes a loaded model.
type ModelInfo struct {
	ID            string `json:"id"`
	Backend       string `json:"backend"`
	Version       string `json:"version"`
	InputShape    string `json:"input_shape" example:"3x640x640"`
	BatchSize     int    `json:"batch_size"`
	InstanceCount int    `json:"instance_count"`
}

// InputShape specifies model input dimensions.
type InputShape struct {
	C int `json:"c" example:"3"`
	H int `json:"h" example:"640"`
	W int `json:"w" example:"640"`
}

// ModelLoad is the request body for POST /models.
type ModelLoad struct {
	ID         string     `json:"id"                       binding:"required"`
	Backend    string     `json:"backend,omitempty"        enums:"cpu,trt,ascend"`
	OnnxPath   string     `json:"onnx_path,omitempty"`
	EnginePath string     `json:"engine_path,omitempty"`
	BatchSize  int        `json:"batch_size,omitempty"`
	InputShape InputShape `json:"input_shape,omitempty"`
}

// TaskInfo describes a task's current state.
type TaskInfo struct {
	ID    string `json:"id"`
	State string `json:"state" enums:"running,stopped"`
}

// TaskCreate is the request body for POST /tasks.
type TaskCreate struct {
	ID         string  `json:"id"          binding:"required"`
	SourceID   string  `json:"source_id"   binding:"required"`
	PipelineID string  `json:"pipeline_id" binding:"required"`
	SampleFPS  float64 `json:"sample_fps,omitempty"`
}

// OkResponse is returned on successful write operations.
type OkResponse struct {
	Status string `json:"status" example:"ok"`
	ID     string `json:"id"     example:"cam0"`
}

// ErrorResponse is returned on errors.
type ErrorResponse struct {
	Status  string `json:"status"           example:"error"`
	Code    string `json:"code,omitempty"   example:"in_use"`
	Message string `json:"message"          example:"source in use by a task: cam0"`
}
