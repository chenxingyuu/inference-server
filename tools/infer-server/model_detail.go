package main

import (
	"bytes"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

const (
	maxModelTestImageBytes    = 8 << 20
	maxModelTestImageSide     = 4096
	maxModelTestResponseBytes = 4 << 20
)

type ModelDetail struct {
	Model           ModelInfo              `json:"model"`
	Registered      bool                   `json:"registered"`
	Repository      *ModelDetailRepository `json:"repository"`
	RepositoryError bool                   `json:"repository_error"`
	Pipelines       []ModelPipelineRef     `json:"pipelines"`
	CascadeParents  []string               `json:"cascade_parents"`
	CascadeChildren []ModelCascadeRef      `json:"cascade_children"`
}

type ModelDetailRepository struct {
	Version     string   `json:"version"`
	WeightPaths []string `json:"weight_paths"`
}

type ModelPipelineRef struct {
	ID           string                 `json:"id"`
	Nodes        []string               `json:"nodes"`
	CascadeNodes []string               `json:"cascade_nodes"`
	Tasks        []ModelPipelineTaskRef `json:"tasks"`
}

type ModelPipelineTaskRef struct {
	ID    string `json:"id"`
	State string `json:"state" enums:"running,stopped"`
}

type ModelCascadeRef struct {
	ModelID        string  `json:"model_id"`
	AttributeKey   string  `json:"attribute_key"`
	TriggerClasses []int   `json:"trigger_classes"`
	CropExpand     float64 `json:"crop_expand"`
}

type ModelTestResult struct {
	ModelType  string            `json:"model_type"`
	Backend    string            `json:"backend"`
	ConfThresh float64           `json:"conf_thresh"`
	NMSThresh  *float64          `json:"nms_thresh"`
	Width      int               `json:"width"`
	Height     int               `json:"height"`
	Preview    string            `json:"preview_jpeg"`
	Results    []ModelPrediction `json:"results"`
}

type ModelPrediction struct {
	ClassID    int       `json:"class_id"`
	ClassName  string    `json:"class_name"`
	Confidence float64   `json:"confidence"`
	BBox       *ModelBox `json:"bbox,omitempty"`
}

type ModelBox struct {
	X1 float64 `json:"x1"`
	Y1 float64 `json:"y1"`
	X2 float64 `json:"x2"`
	Y2 float64 `json:"y2"`
}

// handleModelDetail godoc
//
//	@Summary	Describe a model, its repository version, and references
//	@Tags		models
//	@Produce	json
//	@Security	ApiKeyAuth
//	@Param		id	path	string	true	"Model ID"
//	@Success	200	{object}	ModelDetail
//	@Failure	400	{object}	ErrorResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	404	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse
//	@Router		/models/detail/{id} [get]
func handleModelDetail(c *gin.Context) {
	forwardModelRequest(c, map[string]any{"cmd": "describe_model", "id": c.Param("id")})
}

func registerModelTestRoute(router gin.IRoutes) {
	authenticate := authMiddleware()
	router.POST("/models/test/:id", func(c *gin.Context) {
		if apiKey == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, ErrorResponse{
				Status: "error", Code: "model_test_auth_required",
				Message: "Configure INFER_API_KEY to enable model testing",
			})
			return
		}
		authenticate(c)
	}, handleModelTest)
}

// handleModelTest godoc
//
//	@Summary	Run single-image inference with a registered model
//	@Tags		models
//	@Accept		mpfd
//	@Produce	json
//	@Security	ApiKeyAuth
//	@Param		id		path		string	true	"Model ID"
//	@Param		image	formData	file	true	"One JPEG or PNG, at most 8 MiB and 4096 pixels per side"
//	@Success	200	{object}	ModelTestResult
//	@Failure	400	{object}	ErrorResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	409	{object}	ErrorResponse
//	@Failure	413	{object}	ErrorResponse
//	@Failure	422	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse
//	@Failure	504	{object}	ErrorResponse
//	@Router		/models/test/{id} [post]
func handleModelTest(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxModelTestImageBytes+(64<<10))
	data, err := readModelTestImage(c.Request)
	if err != nil {
		var maxBytes *http.MaxBytesError
		if errors.As(err, &maxBytes) {
			modelRequestError(c, http.StatusRequestEntityTooLarge, "image_too_large", "image exceeds 8 MiB")
		} else {
			modelRequestError(c, http.StatusBadRequest, "invalid_image", err.Error())
		}
		return
	}

	forwardModelRequest(c, map[string]any{"cmd": "test_model", "id": c.Param("id"), "image": data})
}

func readModelTestImage(req *http.Request) ([]byte, error) {
	form, err := req.MultipartReader()
	if err != nil {
		return nil, errors.New("upload one JPEG or PNG in the image field")
	}
	part, err := form.NextPart()
	if err != nil {
		var maxBytes *http.MaxBytesError
		if errors.As(err, &maxBytes) {
			return nil, err
		}
		return nil, errors.New("upload one JPEG or PNG in the image field")
	}
	if part.FormName() != "image" || part.FileName() == "" {
		return nil, errors.New("upload one JPEG or PNG in the image field")
	}
	data, err := io.ReadAll(io.LimitReader(part, maxModelTestImageBytes+1))
	if err != nil {
		var maxBytes *http.MaxBytesError
		if errors.As(err, &maxBytes) {
			return nil, err
		}
		return nil, errors.New("could not read the uploaded image")
	}
	if len(data) > maxModelTestImageBytes {
		return nil, &http.MaxBytesError{Limit: maxModelTestImageBytes}
	}
	if _, err := form.NextPart(); err != io.EOF {
		var maxBytes *http.MaxBytesError
		if errors.As(err, &maxBytes) {
			return nil, err
		}
		return nil, errors.New("upload exactly one image")
	}

	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || (format != "jpeg" && format != "png") {
		return nil, errors.New("upload a valid JPEG or PNG")
	}
	if config.Width < 1 || config.Height < 1 ||
		config.Width > maxModelTestImageSide || config.Height > maxModelTestImageSide {
		return nil, errors.New("image dimensions must be between 1 and 4096 pixels per side")
	}
	return data, nil
}

func modelRequestError(c *gin.Context, status int, code, message string) {
	c.JSON(status, ErrorResponse{Status: "error", Code: code, Message: message})
}

func forwardModelRequest(c *gin.Context, command map[string]any) {
	result, err := call(command)
	if err != nil {
		var timeout interface{ Timeout() bool }
		if errors.As(err, &timeout) && timeout.Timeout() && command["cmd"] == "test_model" {
			modelRequestError(c, http.StatusGatewayTimeout, "model_test_timeout", "model test timed out")
			return
		}
		modelRequestError(c, http.StatusServiceUnavailable, "engine_unavailable", "the engine could not complete the request")
		return
	}
	if result["status"] == "ok" && result["data"] != nil {
		c.JSON(http.StatusOK, result["data"])
		return
	}

	code, _ := result["code"].(string)
	message, _ := result["message"].(string)
	status := http.StatusServiceUnavailable
	switch code {
	case "invalid_model_id", "invalid_image":
		status = http.StatusBadRequest
	case "model_not_found":
		status = http.StatusNotFound
	case "model_not_registered", "model_test_busy":
		status = http.StatusConflict
	case "image_too_large", "request_too_large":
		status = http.StatusRequestEntityTooLarge
	case "model_test_failed", "model_test_unsupported_model", "model_test_result_too_large":
		status = http.StatusUnprocessableEntity
	case "model_test_timeout":
		status = http.StatusGatewayTimeout
	case "model_repository_unavailable", "model_test_unavailable":
	default:
		code, message = "engine_unavailable", "the engine returned an invalid response"
	}
	modelRequestError(c, status, code, message)
}
