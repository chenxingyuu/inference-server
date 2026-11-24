package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	maxExtractBytes   = 4 << 30
	maxArchiveEntries = 4096
)

var errArchiveExtractionLimit = errors.New("archive exceeds the extraction limit")

func extractModelArchive(path, name, dir string) error {
	return extractModelArchiveWithLimit(path, name, dir, maxExtractBytes)
}

func extractModelArchiveWithLimit(path, name, dir string, limit int64) error {
	remaining := limit
	entries := 0
	write := func(name string, directory bool, size int64, open func() (io.ReadCloser, error)) error {
		entries++
		if entries > maxArchiveEntries || size < 0 || size > remaining {
			return errArchiveExtractionLimit
		}
		relative := filepath.Clean(filepath.FromSlash(name))
		if relative == "." && directory {
			return nil
		}
		if !filepath.IsLocal(relative) || relative == "." || strings.Contains(name, "\\") {
			return fmt.Errorf("invalid archive path %q", name)
		}
		for _, part := range strings.Split(relative, string(os.PathSeparator)) {
			if strings.HasPrefix(part, ".") || part == "__MACOSX" {
				return nil
			}
		}
		target := filepath.Join(dir, relative)
		if directory {
			return os.MkdirAll(target, 0o755)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		reader, err := open()
		if err != nil {
			return err
		}
		defer reader.Close()
		file, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return err
		}
		written, err := io.Copy(file, io.LimitReader(reader, size+1))
		closeErr := file.Close()
		if err != nil {
			return err
		}
		if written != size {
			return fmt.Errorf("archive entry size does not match: %q", name)
		}
		remaining -= written
		return closeErr
	}

	if strings.HasSuffix(name, ".zip") {
		z, err := zip.OpenReader(path)
		if err != nil {
			return err
		}
		defer z.Close()
		for _, entry := range z.File {
			if !entry.Mode().IsRegular() && !entry.Mode().IsDir() {
				return fmt.Errorf("unsupported archive entry %q", entry.Name)
			}
			if entry.UncompressedSize64 > uint64(limit) {
				return errArchiveExtractionLimit
			}
			if err := write(entry.Name, entry.Mode().IsDir(), int64(entry.UncompressedSize64), entry.Open); err != nil {
				return err
			}
		}
		return nil
	}
	if !strings.HasSuffix(name, ".tar.gz") && !strings.HasSuffix(name, ".tgz") {
		return fmt.Errorf("archive must be .zip, .tar.gz or .tgz")
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gz.Close()
	// tar.Next also consumes padding, metadata and unread data from skipped entries.
	decoded := &io.LimitedReader{R: gz, N: limit + 1}
	t := tar.NewReader(decoded)
	for {
		header, err := t.Next()
		if decoded.N == 0 {
			return errArchiveExtractionLimit
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if header.Typeflag != tar.TypeReg && header.Typeflag != tar.TypeDir {
			return fmt.Errorf("unsupported archive entry %q", header.Name)
		}
		err = write(header.Name, header.Typeflag == tar.TypeDir, header.Size,
			func() (io.ReadCloser, error) { return io.NopCloser(t), nil })
		if decoded.N == 0 {
			return errArchiveExtractionLimit
		}
		if err != nil {
			return err
		}
	}
}
