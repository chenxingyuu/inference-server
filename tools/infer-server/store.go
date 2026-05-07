package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// TaskEntry wraps a task's creation payload and its last-known running state.
type TaskEntry struct {
	Create  TaskCreate `json:"create"`
	Running bool       `json:"running"`
}

// AppState is the full persisted state written to disk.
type AppState struct {
	Models    map[string]ModelLoad     `json:"models"`
	Sources   map[string]SourceCreate  `json:"sources"`
	Pipelines map[string]PipelineCreate `json:"pipelines"`
	Tasks     map[string]TaskEntry     `json:"tasks"`
}

// Store serialises AppState to a JSON file on every mutation.
type Store struct {
	mu    sync.Mutex
	path  string
	state AppState
}

func initAppState() AppState {
	return AppState{
		Models:    make(map[string]ModelLoad),
		Sources:   make(map[string]SourceCreate),
		Pipelines: make(map[string]PipelineCreate),
		Tasks:     make(map[string]TaskEntry),
	}
}

// NewStore loads state from path if the file exists, otherwise starts empty.
// Parent directories are created as needed.
func NewStore(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create state dir: %w", err)
	}

	s := &Store{path: path, state: initAppState()}

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read state file: %w", err)
	}

	var loaded AppState
	if err := json.Unmarshal(data, &loaded); err != nil {
		return nil, fmt.Errorf("parse state file: %w", err)
	}
	// Ensure maps are never nil even for fields absent in the file.
	if loaded.Models == nil {
		loaded.Models = make(map[string]ModelLoad)
	}
	if loaded.Sources == nil {
		loaded.Sources = make(map[string]SourceCreate)
	}
	if loaded.Pipelines == nil {
		loaded.Pipelines = make(map[string]PipelineCreate)
	}
	if loaded.Tasks == nil {
		loaded.Tasks = make(map[string]TaskEntry)
	}
	s.state = loaded
	return s, nil
}

// save atomically writes state to disk. Must be called with mu held.
func (s *Store) save() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("write tmp state: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("rename state file: %w", err)
	}
	return nil
}

// Snapshot returns a deep copy of the current state.
func (s *Store) Snapshot() AppState {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, _ := json.Marshal(s.state)
	var copy AppState
	json.Unmarshal(data, &copy)
	if copy.Models == nil {
		copy.Models = make(map[string]ModelLoad)
	}
	if copy.Sources == nil {
		copy.Sources = make(map[string]SourceCreate)
	}
	if copy.Pipelines == nil {
		copy.Pipelines = make(map[string]PipelineCreate)
	}
	if copy.Tasks == nil {
		copy.Tasks = make(map[string]TaskEntry)
	}
	return copy
}

// SaveModel upserts a model by ID and persists.
func (s *Store) SaveModel(m ModelLoad) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Models[m.ID] = m
	return s.save()
}

// DeleteModel removes a model by ID. No-op if absent.
func (s *Store) DeleteModel(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state.Models, id)
	return s.save()
}

// SaveSource upserts a source by ID and persists.
func (s *Store) SaveSource(src SourceCreate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Sources[src.ID] = src
	return s.save()
}

// DeleteSource removes a source by ID. No-op if absent.
func (s *Store) DeleteSource(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state.Sources, id)
	return s.save()
}

// SavePipeline upserts a pipeline by ID and persists.
func (s *Store) SavePipeline(p PipelineCreate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Pipelines[p.ID] = p
	return s.save()
}

// DeletePipeline removes a pipeline by ID. No-op if absent.
func (s *Store) DeletePipeline(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state.Pipelines, id)
	return s.save()
}

// SaveTask upserts a task by ID with running=false and persists.
func (s *Store) SaveTask(t TaskCreate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	existing := s.state.Tasks[t.ID]
	s.state.Tasks[t.ID] = TaskEntry{Create: t, Running: existing.Running}
	return s.save()
}

// DeleteTask removes a task by ID. No-op if absent.
func (s *Store) DeleteTask(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state.Tasks, id)
	return s.save()
}

// SetTaskRunning updates the running state of a task. No-op if task not found.
func (s *Store) SetTaskRunning(id string, running bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.state.Tasks[id]
	if !ok {
		return nil
	}
	entry.Running = running
	s.state.Tasks[id] = entry
	return s.save()
}

// mustToMap converts a typed struct to map[string]any via JSON round-trip.
func mustToMap(v any) map[string]any {
	data, _ := json.Marshal(v)
	var m map[string]any
	json.Unmarshal(data, &m)
	return m
}

// replayState sends the full state to the C++ engine in dependency order.
// It retries on socket errors up to maxAttempts times with retryDelay between attempts.
// Engine-level errors (status != "ok") are logged but not fatal.
func replayState(state AppState, callFn func(map[string]any) (map[string]any, error), maxAttempts int, retryDelay time.Duration) error {
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		lastErr = replayOnce(state, callFn)
		if lastErr == nil {
			return nil
		}
		if attempt < maxAttempts && retryDelay > 0 {
			time.Sleep(retryDelay)
		}
	}
	return fmt.Errorf("replay failed after %d attempts: %w", maxAttempts, lastErr)
}

func replayOnce(state AppState, callFn func(map[string]any) (map[string]any, error)) error {
	for _, m := range state.Models {
		cmd := mustToMap(m)
		cmd["cmd"] = "load_model"
		if _, err := callFn(cmd); err != nil {
			return err
		}
	}
	for _, src := range state.Sources {
		cmd := mustToMap(src)
		cmd["cmd"] = "add_source"
		if _, err := callFn(cmd); err != nil {
			return err
		}
	}
	for _, p := range state.Pipelines {
		cmd := mustToMap(p)
		cmd["cmd"] = "add_pipeline"
		if _, err := callFn(cmd); err != nil {
			return err
		}
	}
	for _, entry := range state.Tasks {
		cmd := mustToMap(entry.Create)
		cmd["cmd"] = "add_task"
		if _, err := callFn(cmd); err != nil {
			return err
		}
	}
	for id, entry := range state.Tasks {
		if entry.Running {
			if _, err := callFn(map[string]any{"cmd": "start_task", "id": id}); err != nil {
				return err
			}
		}
	}
	return nil
}
