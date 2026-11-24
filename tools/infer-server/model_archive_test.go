package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTarExtractionBudgetCountsSkippedEntries(t *testing.T) {
	var expanded bytes.Buffer
	writer := tar.NewWriter(&expanded)
	for _, entry := range []struct{ name, data string }{
		{".ignored/first", strings.Repeat("a", 768)},
		{"__MACOSX/second", strings.Repeat("b", 768)},
		{"model.onnx", "weight"},
	} {
		if err := writer.WriteHeader(&tar.Header{Name: entry.name, Mode: 0o644, Size: int64(len(entry.data))}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Write([]byte(entry.data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(t.TempDir(), "test.tar.gz")
	file, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	compressed := gzip.NewWriter(file)
	if _, err := compressed.Write(expanded.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := compressed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	target := t.TempDir()
	if err := extractModelArchiveWithLimit(archivePath, "test.tar.gz", target, 2560); !errors.Is(err, errArchiveExtractionLimit) {
		t.Fatalf("expected the skipped entries to exceed the budget, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(target, "model.onnx")); !os.IsNotExist(err) {
		t.Fatalf("extraction continued after exhausting its budget: %v", err)
	}
	if err := extractModelArchiveWithLimit(archivePath, "test.tar.gz", target, int64(expanded.Len())); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(target)
	if err != nil || len(entries) != 1 || entries[0].Name() != "model.onnx" {
		t.Fatalf("expected only the model weight, got %v (%v)", entries, err)
	}
}
