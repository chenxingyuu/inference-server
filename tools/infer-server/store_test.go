package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// ---- NewStore ----

func TestNewStore_CreatesEmptyStateWhenFileAbsent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	snap := s.Snapshot()
	if len(snap.Models) != 0 || len(snap.Sources) != 0 || len(snap.Pipelines) != 0 || len(snap.Tasks) != 0 {
		t.Fatal("expected empty state for new store")
	}
}

func TestNewStore_MapsAreNotNil(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	snap := s.Snapshot()
	if snap.Models == nil || snap.Sources == nil || snap.Pipelines == nil || snap.Tasks == nil {
		t.Fatal("maps must be initialised, not nil")
	}
}

func TestNewStore_LoadsExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")

	initial := AppState{
		Models:    map[string]ModelLoad{"m1": {ID: "m1", Backend: "trt"}},
		Sources:   map[string]SourceCreate{"s1": {ID: "s1", URL: "rtsp://cam"}},
		Pipelines: map[string]PipelineCreate{},
		Tasks:     map[string]TaskEntry{},
	}
	data, _ := json.Marshal(initial)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	snap := s.Snapshot()
	if snap.Models["m1"].ID != "m1" {
		t.Error("model m1 not loaded")
	}
	if snap.Sources["s1"].URL != "rtsp://cam" {
		t.Error("source s1 not loaded")
	}
}

func TestNewStore_ErrorOnCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte("not json {{{"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := NewStore(path)
	if err == nil {
		t.Fatal("expected error for corrupt JSON file")
	}
}

func TestNewStore_CreatesParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a", "b", "c", "state.json")
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = s.SaveSource(SourceCreate{ID: "x", URL: "rtsp://x"})
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("parent dirs were not created")
	}
}

// ---- Model ----

func TestSaveModel_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	if err := s.SaveModel(ModelLoad{ID: "det0", Backend: "trt"}); err != nil {
		t.Fatal(err)
	}

	s2, _ := NewStore(path)
	snap := s2.Snapshot()
	if snap.Models["det0"].Backend != "trt" {
		t.Error("model not persisted to disk")
	}
}

func TestSaveModel_Deduplication(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveModel(ModelLoad{ID: "det0", Backend: "trt"})
	s.SaveModel(ModelLoad{ID: "det0", Backend: "ascend"})

	snap := s.Snapshot()
	if len(snap.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(snap.Models))
	}
	if snap.Models["det0"].Backend != "ascend" {
		t.Error("second save should replace first")
	}
}

func TestDeleteModel_RemovesFromState(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveModel(ModelLoad{ID: "det0"})
	if err := s.DeleteModel("det0"); err != nil {
		t.Fatal(err)
	}
	snap := s.Snapshot()
	if _, ok := snap.Models["det0"]; ok {
		t.Error("model should have been deleted")
	}
}

func TestDeleteModel_IdempotentOnMissingID(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	if err := s.DeleteModel("nonexistent"); err != nil {
		t.Errorf("delete of missing ID should be no-op, got: %v", err)
	}
}

// ---- Source ----

func TestSaveSource_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SaveSource(SourceCreate{ID: "cam0", URL: "rtsp://cam0"})

	s2, _ := NewStore(path)
	if s2.Snapshot().Sources["cam0"].URL != "rtsp://cam0" {
		t.Error("source not persisted")
	}
}

func TestSaveSource_Deduplication(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveSource(SourceCreate{ID: "cam0", URL: "rtsp://old"})
	s.SaveSource(SourceCreate{ID: "cam0", URL: "rtsp://new"})
	if s.Snapshot().Sources["cam0"].URL != "rtsp://new" {
		t.Error("second save should replace first")
	}
}

func TestDeleteSource_RemovesFromState(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveSource(SourceCreate{ID: "cam0", URL: "rtsp://cam0"})
	s.DeleteSource("cam0")
	if _, ok := s.Snapshot().Sources["cam0"]; ok {
		t.Error("source should have been deleted")
	}
}

func TestDeleteSource_IdempotentOnMissingID(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	if err := s.DeleteSource("none"); err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

// ---- Pipeline ----

func TestSavePipeline_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SavePipeline(PipelineCreate{ID: "pipe0", Nodes: []StageConfig{{ID: "n1", Type: "detector"}}})

	s2, _ := NewStore(path)
	snap := s2.Snapshot()
	if len(snap.Pipelines["pipe0"].Nodes) != 1 {
		t.Error("pipeline not persisted")
	}
}

func TestSavePipeline_Deduplication(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SavePipeline(PipelineCreate{ID: "pipe0", Nodes: []StageConfig{{ID: "n1", Type: "a"}}})
	s.SavePipeline(PipelineCreate{ID: "pipe0", Nodes: []StageConfig{{ID: "n1", Type: "b"}}})
	snap := s.Snapshot()
	if len(snap.Pipelines) != 1 || snap.Pipelines["pipe0"].Nodes[0].Type != "b" {
		t.Error("second save should replace first")
	}
}

func TestDeletePipeline_RemovesFromState(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SavePipeline(PipelineCreate{ID: "pipe0"})
	s.DeletePipeline("pipe0")
	if _, ok := s.Snapshot().Pipelines["pipe0"]; ok {
		t.Error("pipeline should have been deleted")
	}
}

func TestDeletePipeline_IdempotentOnMissingID(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	if err := s.DeletePipeline("none"); err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

// ---- Task ----

func TestSaveTask_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SaveTask(TaskCreate{ID: "t1", SourceID: "cam0", PipelineID: "pipe0"})

	s2, _ := NewStore(path)
	snap := s2.Snapshot()
	entry, ok := snap.Tasks["t1"]
	if !ok {
		t.Fatal("task not persisted")
	}
	if entry.Running {
		t.Error("newly saved task should have running=false")
	}
	if entry.Create.SourceID != "cam0" {
		t.Error("task source not persisted")
	}
}

func TestSaveTask_Deduplication(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveTask(TaskCreate{ID: "t1", SourceID: "old"})
	s.SaveTask(TaskCreate{ID: "t1", SourceID: "new"})
	snap := s.Snapshot()
	if len(snap.Tasks) != 1 || snap.Tasks["t1"].Create.SourceID != "new" {
		t.Error("second save should replace first")
	}
}

func TestDeleteTask_RemovesFromState(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveTask(TaskCreate{ID: "t1"})
	s.DeleteTask("t1")
	if _, ok := s.Snapshot().Tasks["t1"]; ok {
		t.Error("task should have been deleted")
	}
}

func TestDeleteTask_IdempotentOnMissingID(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	if err := s.DeleteTask("none"); err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

// ---- SetTaskRunning ----

func TestSetTaskRunning_StartsAndStops(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SaveTask(TaskCreate{ID: "t1"})

	if err := s.SetTaskRunning("t1", true); err != nil {
		t.Fatal(err)
	}
	if !s.Snapshot().Tasks["t1"].Running {
		t.Error("task should be running")
	}

	s.SetTaskRunning("t1", false)
	if s.Snapshot().Tasks["t1"].Running {
		t.Error("task should be stopped")
	}
}

func TestSetTaskRunning_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SaveTask(TaskCreate{ID: "t1"})
	s.SetTaskRunning("t1", true)

	s2, _ := NewStore(path)
	if !s2.Snapshot().Tasks["t1"].Running {
		t.Error("running state not persisted to disk")
	}
}

func TestSetTaskRunning_NoopOnMissingTask(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	if err := s.SetTaskRunning("nonexistent", true); err != nil {
		t.Errorf("expected no-op, got: %v", err)
	}
}

// ---- Snapshot ----

func TestSnapshot_ReturnsDeepCopy(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))
	s.SaveModel(ModelLoad{ID: "m1", Backend: "trt"})

	snap1 := s.Snapshot()
	snap1.Models["m1"] = ModelLoad{ID: "m1", Backend: "mutated"}
	snap1.Models["injected"] = ModelLoad{ID: "injected"}

	snap2 := s.Snapshot()
	if snap2.Models["m1"].Backend == "mutated" {
		t.Error("mutation of snapshot bled through to store")
	}
	if _, ok := snap2.Models["injected"]; ok {
		t.Error("injected key bled through to store")
	}
}

// ---- replayState ----

type callRecord struct {
	mu   sync.Mutex
	cmds []string
}

func (r *callRecord) fn(cmd map[string]any) (map[string]any, error) {
	r.mu.Lock()
	r.cmds = append(r.cmds, cmd["cmd"].(string))
	r.mu.Unlock()
	return map[string]any{"status": "ok"}, nil
}

func (r *callRecord) ordered() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.cmds))
	copy(out, r.cmds)
	return out
}

func TestReplayState_DependencyOrder(t *testing.T) {
	// Models are NOT replayed (model_repository is the single source of truth).
	state := AppState{
		Models:    map[string]ModelLoad{"m1": {ID: "m1"}}, // present but not replayed
		Sources:   map[string]SourceCreate{"s1": {ID: "s1", URL: "rtsp://s1"}},
		Pipelines: map[string]PipelineCreate{"p1": {ID: "p1"}},
		Tasks: map[string]TaskEntry{
			"t1": {Create: TaskCreate{ID: "t1", SourceID: "s1", PipelineID: "p1"}, Running: true},
		},
	}

	rec := &callRecord{}
	err := replayState(state, rec.fn, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cmds := rec.ordered()
	wantOrder := []string{"add_source", "add_pipeline", "add_task", "start_task"}
	if len(cmds) != len(wantOrder) {
		t.Fatalf("expected %d calls, got %d: %v", len(wantOrder), len(cmds), cmds)
	}
	for i, want := range wantOrder {
		if cmds[i] != want {
			t.Errorf("cmd[%d]: want %q, got %q", i, want, cmds[i])
		}
	}
	for _, cmd := range cmds {
		if cmd == "load_model" {
			t.Error("load_model must not be replayed; model_repository is the source of truth")
		}
	}
}

func TestReplayState_DoesNotStartStoppedTasks(t *testing.T) {
	state := AppState{
		Models:    map[string]ModelLoad{},
		Sources:   map[string]SourceCreate{},
		Pipelines: map[string]PipelineCreate{},
		Tasks: map[string]TaskEntry{
			"t1": {Create: TaskCreate{ID: "t1"}, Running: false},
		},
	}

	rec := &callRecord{}
	replayState(state, rec.fn, 1, 0)

	for _, cmd := range rec.ordered() {
		if cmd == "start_task" {
			t.Error("start_task should not be called for stopped tasks")
		}
	}
}

func TestReplayState_RetriesOnSocketError(t *testing.T) {
	state := AppState{
		Models:    map[string]ModelLoad{},
		Sources:   map[string]SourceCreate{"s1": {ID: "s1", URL: "rtsp://s1"}},
		Pipelines: map[string]PipelineCreate{},
		Tasks:     map[string]TaskEntry{},
	}

	attempts := 0
	fakeFn := func(cmd map[string]any) (map[string]any, error) {
		attempts++
		if attempts < 3 {
			return nil, errors.New("connection refused")
		}
		return map[string]any{"status": "ok"}, nil
	}

	err := replayState(state, fakeFn, 5, 0)
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if attempts < 3 {
		t.Errorf("expected at least 3 attempts, got %d", attempts)
	}
}

func TestReplayState_ExceedsMaxAttempts(t *testing.T) {
	state := AppState{
		Models:    map[string]ModelLoad{},
		Sources:   map[string]SourceCreate{"s1": {ID: "s1", URL: "rtsp://s1"}},
		Pipelines: map[string]PipelineCreate{},
		Tasks:     map[string]TaskEntry{},
	}

	fakeFn := func(cmd map[string]any) (map[string]any, error) {
		return nil, errors.New("connection refused")
	}

	err := replayState(state, fakeFn, 2, 0)
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
}

func TestReplayState_SkipsEngineErrorsAndContinues(t *testing.T) {
	// Engine-level errors (status != "ok") on add_source should not abort the replay.
	state := AppState{
		Models:    map[string]ModelLoad{},
		Sources:   map[string]SourceCreate{"s1": {ID: "s1", URL: "rtsp://s1"}, "s2": {ID: "s2", URL: "rtsp://s2"}},
		Pipelines: map[string]PipelineCreate{},
		Tasks:     map[string]TaskEntry{},
	}

	rec := &callRecord{}
	fakeFn := func(cmd map[string]any) (map[string]any, error) {
		_, _ = rec.fn(cmd)
		if cmd["id"] == "s1" {
			return map[string]any{"status": "error", "message": "already exists"}, nil
		}
		return map[string]any{"status": "ok"}, nil
	}

	err := replayState(state, fakeFn, 1, 0)
	if err != nil {
		t.Fatalf("engine-level errors should not cause replayState to return error: %v", err)
	}

	cmds := rec.ordered()
	count := 0
	for _, c := range cmds {
		if c == "add_source" {
			count++
		}
	}
	if count < 2 {
		t.Errorf("both add_source calls should be attempted; got %d", count)
	}
}

// ---- Concurrency ----

func TestStore_ConcurrentWrites(t *testing.T) {
	s, _ := NewStore(filepath.Join(t.TempDir(), "state.json"))

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		id := "m" + string(rune('0'+i))
		go func(id string) {
			defer wg.Done()
			s.SaveModel(ModelLoad{ID: id, Backend: "trt"})
		}(id)
	}
	wg.Wait()

	snap := s.Snapshot()
	if len(snap.Models) != 10 {
		t.Errorf("expected 10 models, got %d", len(snap.Models))
	}
}

// ---- Atomic write ----

func TestSave_NoTmpFileAfterWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	s, _ := NewStore(path)
	s.SaveModel(ModelLoad{ID: "m1"})

	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error(".tmp file should not exist after save completes")
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("state.json should exist after save")
	}
}

// ensure replayState uses the time import properly (compile check)
var _ = time.Second
