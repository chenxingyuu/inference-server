package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"reflect"
	"strings"
	"testing"
)

func onnxBytesField(number uint64, value []byte) []byte {
	field := binary.AppendUvarint(nil, number<<3|2)
	field = binary.AppendUvarint(field, uint64(len(value)))
	return append(field, value...)
}

func onnxExternalTensor(location string) []byte {
	entry := append(onnxBytesField(1, []byte("location")), onnxBytesField(2, []byte(location))...)
	tensor := onnxBytesField(13, entry)
	tensor = binary.AppendUvarint(tensor, 14<<3)
	return binary.AppendUvarint(tensor, 1)
}

func fixtureONNXExternalData(locations ...string) string {
	var graph []byte
	for _, location := range locations {
		graph = append(graph, onnxBytesField(5, onnxExternalTensor(location))...)
	}
	return string(onnxBytesField(7, graph))
}

type countedONNXReader struct {
	*bytes.Reader
	read int
}

func (r *countedONNXReader) ReadAt(p []byte, offset int64) (int, error) {
	n, err := r.Reader.ReadAt(p, offset)
	r.read += n
	return n, err
}

func TestONNXExternalLocationsFollowNestedMessagesAndSkipInlineData(t *testing.T) {
	inlineTensor := onnxBytesField(9, bytes.Repeat([]byte{0xff}, 1<<20))
	graph := onnxBytesField(5, inlineTensor)
	graph = append(graph, onnxBytesField(5, onnxExternalTensor("initializer.bin"))...)
	subgraph := onnxBytesField(5, onnxExternalTensor("nested.engine"))
	graph = append(graph, onnxBytesField(1, onnxBytesField(5, onnxBytesField(6, subgraph)))...)
	graph = append(graph, onnxBytesField(15, onnxBytesField(1, onnxExternalTensor("sparse.bin")))...)
	model := onnxBytesField(7, graph)
	training := onnxBytesField(2, onnxBytesField(5, onnxExternalTensor("training.bin")))
	model = append(model, onnxBytesField(20, training)...)
	function := onnxBytesField(11, onnxBytesField(5, onnxExternalTensor("function.bin")))
	model = append(model, onnxBytesField(25, function)...)

	reader := &countedONNXReader{Reader: bytes.NewReader(model)}
	locations, err := onnxExternalLocations(context.Background(), reader, int64(len(model)))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"initializer.bin", "nested.engine", "sparse.bin", "training.bin", "function.bin"}
	if !reflect.DeepEqual(locations, want) || reader.read >= 1<<16 {
		t.Fatalf("locations = %v, read %d bytes", locations, reader.read)
	}
}

func TestONNXExternalLocationsRejectsIncompleteMetadata(t *testing.T) {
	for _, scenario := range []struct{ label, model, message string }{
		{"truncated message", "\x3a\x02\x00", "exceeds its message"},
		{"empty location", fixtureONNXExternalData(""), "missing its external data location"},
		{"missing graph", "\x08\x01", "missing its graph"},
	} {
		t.Run(scenario.label, func(t *testing.T) {
			_, err := onnxExternalLocations(context.Background(), strings.NewReader(scenario.model), int64(len(scenario.model)))
			if err == nil || !strings.Contains(err.Error(), scenario.message) {
				t.Fatalf("expected %q, got %v", scenario.message, err)
			}
		})
	}
}
