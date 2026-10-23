package runner

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func ArchiveMember(z *zip.ReadCloser, root, path, want string) (map[string]string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	matched := map[string]string{}
	seen := map[string]bool{}
	for depth := 0; depth < 32; depth++ {
		rel, err := filepath.Rel(root, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return nil, fmt.Errorf("archive symlink escapes distribution: %s", path)
		}
		member := "libtorch/" + filepath.ToSlash(rel)
		if seen[member] {
			return nil, fmt.Errorf("archive symlink cycle: %s", member)
		}
		seen[member] = true
		var f *zip.File
		for _, candidate := range z.File {
			if candidate.Name == member {
				f = candidate
				break
			}
		}
		if f == nil {
			return nil, fmt.Errorf("distribution archive lacks %s", member)
		}
		info, err := os.Lstat(path)
		if err != nil {
			return nil, err
		}
		reader, err := f.Open()
		if err != nil {
			return nil, err
		}
		if f.Mode()&os.ModeSymlink != 0 {
			if f.UncompressedSize64 > 4096 {
				reader.Close()
				return nil, fmt.Errorf("oversized archive symlink")
			}
			raw, err := io.ReadAll(reader)
			closeErr := reader.Close()
			if err != nil {
				return nil, err
			}
			if closeErr != nil {
				return nil, closeErr
			}
			if info.Mode()&os.ModeSymlink == 0 {
				return nil, fmt.Errorf("extracted member is not the archived symlink: %s", path)
			}
			target, err := os.Readlink(path)
			if err != nil {
				return nil, err
			}
			if target != string(raw) {
				return nil, fmt.Errorf("archive symlink text mismatch: %s", path)
			}
			if filepath.IsAbs(target) {
				return nil, fmt.Errorf("absolute archive symlink rejected: %s", path)
			}
			sum := sha256.Sum256(raw)
			matched[member] = hex.EncodeToString(sum[:])
			path = filepath.Clean(filepath.Join(filepath.Dir(path), target))
			continue
		}
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			reader.Close()
			return nil, err
		}
		if resolved != path {
			reader.Close()
			return nil, fmt.Errorf("unmatched intermediate symlink in distribution path: %s", path)
		}
		if !f.Mode().IsRegular() || !info.Mode().IsRegular() {
			reader.Close()
			return nil, fmt.Errorf("archive target must be regular: %s", path)
		}
		h := sha256.New()
		_, err = io.Copy(h, reader)
		closeErr := reader.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		archiveHash := hex.EncodeToString(h.Sum(nil))
		actual, err := SHA(path)
		if err != nil {
			return nil, err
		}
		if actual != archiveHash || actual != want {
			return nil, fmt.Errorf("extracted distribution differs from pinned archive at %s", path)
		}
		matched[member] = archiveHash
		return matched, nil
	}
	return nil, fmt.Errorf("archive symlink chain exceeds limit")
}
