// Package koiquant retains and validates native SUP QKV int8 arithmetic observations.
package koiquant

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

const Schema = "nanopore-sup-qkv-i8/prepare-v1"
const Elements = 3 * 8 * 64 * 512
const OriginSHA256 = "3faad1885b81146b604eea6e2028767ba204a6f62b71a782add4a268d7b5aae1"
const GenuineSHA256 = "cb27b916510b9bf98576a247db446d842ef7a199c74b54da32bae031258879f0"

//go:embed native/oracle.cpp
var adapter string

type Block struct {
	File, FileSHA256 string
	Start, End       int
	SHA256, Text     string
}

var blockDefinitions = []Block{
	{File: "external/dorado-source/dorado/torch_utils/include/torch_utils/tensor_utils.h", FileSHA256: "4daca69dcca06289f6108340714f19f9f9dbe4a54b36c72eecf9dbe999395bb2", Start: 47, End: 50},
	{File: "external/dorado-source/dorado/torch_utils/tensor_utils.cpp", FileSHA256: "7e535e83d13225e1622cda6d16f842641ea466633009ad03e695d7af7362b018", Start: 291, End: 298},
	{File: "external/dorado-source/dorado/nn/TxModules.cpp", FileSHA256: koiweights.SourceHash, Start: 455, End: 461},
	{File: "verification/koiweights/native/oracle.cpp", FileSHA256: "6f7278497579d9ebf9bc6b33cffc59f18d99a3d09f1ed7e51f74a0e2592e238f", Start: 13, End: 27},
}

var sourceFiles = []string{
	"verification/koiquant/source.go", "verification/koiquant/prepare.go",
	"verification/koiquant/stages.go", "verification/koiquant/fixtures.go",
	"verification/koiquant/oracle_test.go", "verification/koiquant/native/oracle.cpp",
	"verification/koiquant/cmd/main.go",
	"verification/koiquant/capture.go", "verification/koiquant/import.go",
	"verification/koiquant/receipt.go", "verification/koiquant/compare.go",
	"verification/koiquant/capture_test.go", "verification/koiquant/replay.go",
	"verification/tensor/runner/admission.go", "verification/tensor/runner/command.go",
	"verification/tensor/runner/archive.go", "verification/tensor/runner/gotests.go",
	"internal/signal/half.go", "go.mod",
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func checkHash(path, want string) error {
	if len(want) != 64 {
		return fmt.Errorf("missing or invalid SHA256 for %s", path)
	}
	if _, err := hex.DecodeString(want); err != nil {
		return fmt.Errorf("invalid SHA256 for %s", path)
	}
	h, err := koiweights.Hash(path)
	if err != nil {
		return err
	}
	if h != want {
		return fmt.Errorf("SHA256 mismatch for %s: got %s want %s", path, h, want)
	}
	return nil
}

// Extract requires the whole pinned source before selecting its closed ranges.
func Extract(root string) ([]Block, error) {
	blocks := append([]Block(nil), blockDefinitions...)
	for i := range blocks {
		b := &blocks[i]
		path := filepath.Join(root, b.File)
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		if digest(data) != b.FileSHA256 {
			return nil, fmt.Errorf("whole source pin mismatch: %s", b.File)
		}
		lines := strings.Split(string(data), "\n")
		if len(lines) < b.End {
			return nil, fmt.Errorf("missing source range %s", b.File)
		}
		b.Text = strings.Join(lines[b.Start-1:b.End], "\n") + "\n"
		b.SHA256 = digest([]byte(b.Text))
	}
	if blocks[1].SHA256 != "4b749db5735af6f7f29a7619ea4927d0c150d39cb8a79f5806c68fe7a6ac70bf" {
		return nil, fmt.Errorf("quantize_tensor source block mismatch")
	}
	return blocks, nil
}

func generated(blocks []Block) (string, error) {
	if len(blocks) != len(blockDefinitions) {
		return "", fmt.Errorf("missing source blocks")
	}
	out := adapter
	for i, marker := range []string{"// SCALED_TENSOR_BLOCK", "// QUANTIZE_BLOCK", "// CACHE_BLOCK", "// RAW_IO_BLOCK"} {
		if strings.Count(out, marker) != 1 {
			return "", fmt.Errorf("source adapter marker %s", marker)
		}
		out = strings.Replace(out, marker, blocks[i].Text, 1)
	}
	return out, nil
}

func currentSources(root string) (map[string]string, error) {
	out := make(map[string]string, len(sourceFiles))
	for _, p := range sourceFiles {
		h, err := koiweights.Hash(filepath.Join(root, p))
		if err != nil {
			return nil, err
		}
		out[p] = h
	}
	return out, nil
}

func checkSources(root string, saved map[string]string) error {
	current, err := currentSources(root)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(current, saved) {
		return fmt.Errorf("current preparation source pins differ")
	}
	return nil
}
