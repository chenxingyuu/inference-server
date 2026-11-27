package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func registerSourceProbeRoute(router gin.IRoutes) {
	authenticate := authMiddleware()
	router.POST("/sources/:id/probe", func(c *gin.Context) {
		if apiKey == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, ErrorResponse{
				Status: "error", Code: "source_probe_auth_required",
				Message: "Configure INFER_API_KEY to enable connection tests",
			})
			return
		}
		authenticate(c)
	}, handleSourceProbe)
}

// handleSourceProbe godoc
//
//	@Summary	Check whether the engine can decode a frame from a registered RTSP source
//	@Description	Opens the configured RTSP stream independently of tasks and updates the engine cache. A completed check returns 200 for both online and offline results.
//	@Tags		sources
//	@Produce	json
//	@Security	ApiKeyAuth
//	@Param		id	path		string	true	"Source ID"
//	@Success	200	{object}	SourceProbeResponse
//	@Failure	401	{object}	ErrorResponse
//	@Failure	404	{object}	ErrorResponse
//	@Failure	409	{object}	ErrorResponse
//	@Failure	422	{object}	ErrorResponse
//	@Failure	429	{object}	ErrorResponse
//	@Failure	503	{object}	ErrorResponse
//	@Router		/sources/{id}/probe [post]
func handleSourceProbe(c *gin.Context) {
	response, err := call(map[string]any{"cmd": "probe_source", "id": c.Param("id")})
	if err == nil && response["status"] == "ok" {
		if data, ok := response["data"].(map[string]any); ok {
			c.JSON(http.StatusOK, data)
			return
		}
	}

	code, _ := response["code"].(string)
	status, message := http.StatusServiceUnavailable, "The engine or connection test is unavailable"
	switch code {
	case "source_not_found":
		status, message = http.StatusNotFound, "Source not found"
	case "source_probe_in_progress":
		status, message = http.StatusConflict, "This source already has a connection test in progress"
	case "source_probe_unsupported":
		status, message = http.StatusUnprocessableEntity, "Connection tests require an RTSP source"
	case "source_probe_busy":
		status, message = http.StatusTooManyRequests, "The engine is running other connection tests"
	default:
		code = "source_probe_unavailable"
	}
	c.JSON(status, ErrorResponse{Status: "error", Code: code, Message: message})
}
