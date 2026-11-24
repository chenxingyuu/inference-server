package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"path"
	"strings"
)

const (
	maxONNXMessageDepth = 100
	maxONNXMetadataText = 4096
)

type onnxMessage uint8

const (
	onnxModel onnxMessage = iota + 1
	onnxGraph
	onnxNode
	onnxAttribute
	onnxTraining
	onnxFunction
	onnxSparseTensor
	onnxTensor
)

// Only follow ONNX messages that can contain TensorProto values. See onnx/onnx.proto.
var onnxTensorContainers = map[onnxMessage]map[uint64]onnxMessage{
	onnxModel:        {7: onnxGraph, 20: onnxTraining, 25: onnxFunction},
	onnxGraph:        {1: onnxNode, 5: onnxTensor, 15: onnxSparseTensor},
	onnxNode:         {5: onnxAttribute},
	onnxAttribute:    {5: onnxTensor, 6: onnxGraph, 10: onnxTensor, 11: onnxGraph, 22: onnxSparseTensor, 23: onnxSparseTensor},
	onnxTraining:     {1: onnxGraph, 2: onnxGraph},
	onnxFunction:     {7: onnxNode, 11: onnxAttribute},
	onnxSparseTensor: {1: onnxTensor, 2: onnxTensor},
}

// Scan only protobuf tags and external-data metadata. Inline tensor data can be
// large, so skip bytes using a bounded section rather than loading the model.
func onnxExternalLocations(ctx context.Context, file io.ReaderAt, size int64) ([]string, error) {
	if size <= 0 || size > maxExtractBytes {
		return nil, fmt.Errorf("invalid ONNX file size")
	}
	section := io.NewSectionReader(file, 0, size)
	reader := onnxWireReader{
		ctx: ctx, source: section, buffer: bufio.NewReader(section),
		seen: make(map[string]bool),
	}
	if err := reader.scan(onnxModel, size, 0); err != nil {
		return nil, err
	}
	return reader.locations, nil
}

func onnxExternalPath(model, location string) (string, error) {
	if location == "" || path.IsAbs(location) || strings.ContainsAny(location, "\\\x00:") {
		return "", fmt.Errorf("ONNX external data location must be a portable relative path")
	}
	for _, part := range strings.Split(location, "/") {
		if part == ".." {
			return "", fmt.Errorf("ONNX external data location must stay within the model directory")
		}
	}
	name := path.Join(path.Dir(model), location)
	if path.Clean(location) == "." || !validExportPath(name) || name == model {
		return "", fmt.Errorf("ONNX external data location %q cannot be imported", location)
	}
	return name, nil
}

type onnxWireReader struct {
	ctx       context.Context
	source    *io.SectionReader
	buffer    *bufio.Reader
	offset    int64
	seen      map[string]bool
	locations []string
}

func (r *onnxWireReader) ReadByte() (byte, error) {
	value, err := r.buffer.ReadByte()
	if err == nil {
		r.offset++
	}
	return value, err
}

func (r *onnxWireReader) varint(end int64) (uint64, error) {
	value, err := binary.ReadUvarint(r)
	if err != nil || r.offset > end {
		return 0, fmt.Errorf("invalid ONNX protobuf varint")
	}
	return value, nil
}

func (r *onnxWireReader) skip(end int64) error {
	count := end - r.offset
	if count == 0 {
		return nil
	}
	if count <= int64(r.buffer.Buffered()) {
		if _, err := r.buffer.Discard(int(count)); err != nil {
			return err
		}
	} else {
		if _, err := r.source.Seek(end, io.SeekStart); err != nil {
			return err
		}
		r.buffer.Reset(r.source)
	}
	r.offset = end
	return nil
}

type onnxWireField struct {
	number uint64
	wire   uint64
	value  uint64
	end    int64
}

func (r *onnxWireReader) fields(end int64, visit func(onnxWireField) error) error {
	for r.offset < end {
		if err := r.ctx.Err(); err != nil {
			return err
		}
		tag, err := r.varint(end)
		if err != nil {
			return err
		}
		field := onnxWireField{number: tag >> 3, wire: tag & 7}
		if field.number == 0 || field.number > (1<<29)-1 {
			return fmt.Errorf("invalid ONNX protobuf field number")
		}
		var size uint64
		switch field.wire {
		case 0:
			field.value, err = r.varint(end)
		case 1:
			size = 8
		case 2:
			size, err = r.varint(end)
		case 5:
			size = 4
		default:
			return fmt.Errorf("unsupported ONNX protobuf wire type")
		}
		if err != nil {
			return err
		}
		if size > uint64(end-r.offset) {
			return fmt.Errorf("ONNX protobuf field exceeds its message")
		}
		field.end = r.offset + int64(size)
		if err := visit(field); err != nil {
			return err
		}
		if err := r.skip(field.end); err != nil {
			return err
		}
	}
	return nil
}

func (r *onnxWireReader) scan(kind onnxMessage, end int64, depth int) error {
	if depth >= maxONNXMessageDepth {
		return fmt.Errorf("ONNX protobuf is nested too deeply")
	}
	if kind == onnxTensor {
		return r.scanTensor(end)
	}
	hasGraph := false
	err := r.fields(end, func(field onnxWireField) error {
		nested := onnxTensorContainers[kind][field.number]
		if nested == 0 {
			return nil
		}
		if field.wire != 2 {
			return fmt.Errorf("invalid ONNX nested message")
		}
		hasGraph = hasGraph || nested == onnxGraph
		return r.scan(nested, field.end, depth+1)
	})
	if err == nil && kind == onnxModel && !hasGraph {
		return fmt.Errorf("ONNX model is missing its graph")
	}
	return err
}

func (r *onnxWireReader) scanTensor(end int64) error {
	var location string
	var dataLocation uint64
	var hasLocation bool
	err := r.fields(end, func(field onnxWireField) error {
		switch field.number {
		case 13: // TensorProto.external_data
			if field.wire != 2 {
				return fmt.Errorf("invalid ONNX external data entry")
			}
			key, value, err := r.readMetadataEntry(field.end)
			if err != nil {
				return err
			}
			if key == "location" {
				if hasLocation {
					return fmt.Errorf("ONNX tensor declares multiple external data locations")
				}
				location, hasLocation = value, true
			}
		case 14: // TensorProto.data_location
			if field.wire != 0 || field.value > 1 {
				return fmt.Errorf("invalid ONNX tensor data location")
			}
			dataLocation = field.value
		}
		return nil
	})
	if err != nil || dataLocation == 0 {
		return err
	}
	if location == "" {
		return fmt.Errorf("ONNX tensor is missing its external data location")
	}
	if !r.seen[location] {
		if len(r.locations) >= maxArchiveEntries-2 {
			return fmt.Errorf("ONNX model declares too many external data locations")
		}
		r.seen[location] = true
		r.locations = append(r.locations, location)
	}
	return nil
}

func (r *onnxWireReader) readMetadataEntry(end int64) (string, string, error) {
	var key, value string
	err := r.fields(end, func(field onnxWireField) error {
		if field.number != 1 && field.number != 2 {
			return nil
		}
		size := field.end - r.offset
		if field.wire != 2 || size > maxONNXMetadataText {
			return fmt.Errorf("invalid ONNX external data metadata")
		}
		data := make([]byte, int(size))
		n, err := io.ReadFull(r.buffer, data)
		r.offset += int64(n)
		if err != nil {
			return err
		}
		if field.number == 1 {
			key = string(data)
		} else {
			value = string(data)
		}
		return nil
	})
	return key, value, err
}
