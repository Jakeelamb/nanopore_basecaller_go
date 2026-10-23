// Package scalingoracle builds a bounded native formula reference, not a Torch
// execution path. It extracts the pinned source expressions without rewriting.
package scalingoracle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const SourceSHA256 = "5d2ac820e21e74d0c5813c5349928685bda4f107f6d18243287f055d5612453c"
const SourcePath = "dorado/read_pipeline/nodes/ScalerNode.cpp"

type Slice struct {
	StartLine, EndLine int
	SHA256, Text       string
}
type Manifest struct {
	SourcePath, SourceSHA256, Qualification string
	Slices                                  []Slice
}

func Extract(source []byte) (Manifest, error) {
	sum := sha256.Sum256(source)
	manifest := Manifest{SourcePath: SourcePath, SourceSHA256: hex.EncodeToString(sum[:]), Qualification: "Verbatim ScalerNode formula and metadata; scalar float32 arithmetic plus compiler _Float16 conversion, not full Torch or SIMD execution"}
	if manifest.SourceSHA256 != SourceSHA256 {
		return manifest, fmt.Errorf("ScalerNode source pin mismatch: %s", manifest.SourceSHA256)
	}
	lines := strings.Split(string(source), "\n")
	for _, bounds := range [][2]int{{202, 203}, {233, 234}} {
		text := strings.Join(lines[bounds[0]-1:bounds[1]], "\n") + "\n"
		hash := sha256.Sum256([]byte(text))
		manifest.Slices = append(manifest.Slices, Slice{bounds[0], bounds[1], hex.EncodeToString(hash[:]), text})
	}
	return manifest, nil
}

func Build(sourceRoot, output string) (string, error) {
	source, err := os.ReadFile(filepath.Join(sourceRoot, SourcePath))
	if err != nil {
		return "", err
	}
	manifest, err := Extract(source)
	if err != nil {
		return "", err
	}
	body := `#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Common { float scale, shift; };
struct Read { float scaling, offset; Common read_common; };
struct Standardisation { float mean, stdev; };
int main(int argc, char** argv) {
 if (argc != 6) return 2;
 Read object{strtof(argv[1], nullptr), strtof(argv[2], nullptr), {}};
 Read* read = &object;
 Standardisation stdn{strtof(argv[3], nullptr), strtof(argv[4], nullptr)};
 float adjustment = strtof(argv[5], nullptr), shift, scale;
` + manifest.Slices[0].Text + manifest.Slices[1].Text + `
 auto emit = [](auto value) { return fwrite(&value, sizeof(value), 1, stdout) == 1; };
 if (!emit(std::bit_cast<uint32_t>(read->read_common.shift)) || !emit(std::bit_cast<uint32_t>(read->read_common.scale))) return 3;
 volatile float adjusted = shift + adjustment;
 for (int value = -32768; value <= 32767; ++value) {
  volatile float difference = float(value) - adjusted;
  volatile float normalized = difference / scale;
  _Float16 half = (_Float16)normalized;
  uint16_t bits; memcpy(&bits, &half, 2);
  if (!emit(bits)) return 3;
 }
 return 0;
}
`
	cpp := filepath.Join(output, "scaling-reference.cpp")
	if err := os.WriteFile(cpp, []byte(body), 0600); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "source-slices.json"), data, 0600); err != nil {
		return "", err
	}
	executable := filepath.Join(output, "scaling-reference")
	command := exec.Command("c++", "-std=c++20", "-O2", "-fno-fast-math", "-ffp-contract=off", cpp, "-o", executable)
	if out, err := command.CombinedOutput(); err != nil {
		return "", fmt.Errorf("compile native reference: %w: %s", err, out)
	}
	return executable, nil
}
