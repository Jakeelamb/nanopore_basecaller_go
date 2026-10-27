package supgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"syscall"
)

var digestRE = regexp.MustCompile(`^[0-9a-f]{64}$`)

func pin(path string) (Pin, error) {
	p, err := filepath.Abs(path)
	if err != nil {
		return Pin{}, err
	}
	f, err := os.OpenFile(p, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return Pin{}, err
	}
	defer f.Close()
	s, err := f.Stat()
	if err != nil {
		return Pin{}, err
	}
	if !s.Mode().IsRegular() {
		return Pin{}, fmt.Errorf("not a regular file: %s", p)
	}
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return Pin{}, err
	}
	if n != s.Size() {
		return Pin{}, fmt.Errorf("file size changed while reading %s", p)
	}
	return Pin{p, hex.EncodeToString(h.Sum(nil)), n}, nil
}

func verifyPin(base string, p Pin) error {
	if !digestRE.MatchString(p.SHA256) || p.Bytes < 0 {
		return fmt.Errorf("invalid pin %q", p.Path)
	}
	path := p.Path
	if base != "" {
		var err error
		base, err = filepath.Abs(base)
		if err != nil {
			return err
		}
		if filepath.IsAbs(path) || filepath.Clean(path) != path || path == "." || path == ".." {
			return fmt.Errorf("unsafe payload %q", path)
		}
		full := filepath.Join(base, path)
		rel, err := filepath.Rel(base, full)
		if err != nil || rel != path || len(rel) >= 3 && rel[:3] == "../" {
			return fmt.Errorf("escaping payload %q", path)
		}
		actual, err := filepath.EvalSymlinks(full)
		if err != nil {
			return err
		}
		abs, err := filepath.Abs(full)
		if err != nil || actual != abs {
			return fmt.Errorf("symlink payload %q", path)
		}
		path = full
	} else if !filepath.IsAbs(path) {
		return fmt.Errorf("external pin must be absolute: %s", path)
	}
	got, err := pin(path)
	if err != nil {
		return err
	}
	if got.SHA256 != p.SHA256 || got.Bytes != p.Bytes {
		return fmt.Errorf("pin changed: %s", p.Path)
	}
	return nil
}

func readPinnedJSON(path, sha string, out any) error {
	if !digestRE.MatchString(sha) {
		return fmt.Errorf("external frozen SHA256 required")
	}
	p, err := pin(path)
	if err != nil {
		return err
	}
	if p.SHA256 != sha {
		return fmt.Errorf("frozen manifest changed: %s", path)
	}
	if p.Bytes > 64<<20 {
		return fmt.Errorf("manifest exceeds 64 MiB")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return decodeClosed(raw, out)
}

func writeJSON(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(raw, '\n'))
	return err
}
