package main

import (
	"sync"
	"testing"
)

func TestReplayState_PipelineAlreadyExistsCallsUpdate(t *testing.T) {
	state := AppState{
		Sources: map[string]SourceCreate{},
		Pipelines: map[string]PipelineCreate{
			"p1": {ID: "p1", Nodes: []StageConfig{{ID: "n1", Type: "detector"}}, Edges: []EdgeConfig{
				{From: "a", To: "b", Capacity: 32, DropPolicy: "drop_oldest"},
			}},
		},
		Tasks: map[string]TaskEntry{},
	}

	var mu sync.Mutex
	var cmds []string
	fakeFn := func(cmd map[string]any) (map[string]any, error) {
		mu.Lock()
		cmds = append(cmds, cmd["cmd"].(string))
		mu.Unlock()
		if cmd["cmd"] == "add_pipeline" {
			return map[string]any{"status": "error", "message": "already exists: p1"}, nil
		}
		return map[string]any{"status": "ok"}, nil
	}

	if err := replayState(state, fakeFn, 1, 0); err != nil {
		t.Fatalf("replayState: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(cmds) != 2 || cmds[0] != "add_pipeline" || cmds[1] != "update_pipeline" {
		t.Fatalf("want add_pipeline then update_pipeline, got %v", cmds)
	}
}
