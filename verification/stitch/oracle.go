// Package stitchoracle builds the pinned, untouched Dorado stitching algorithm.
package stitchoracle

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// SourcePins are the original algorithm and non-shim Dorado headers.
// Build rejects changed bytes before invoking the compiler.
var SourcePins = map[string]string{
	"dorado/read_pipeline/base/stitch.cpp":                          "342db2c008aa37643add1f38730c573f47f72aef73932cb28218e4ac6065c931",
	"dorado/read_pipeline/base/include/read_pipeline/base/stitch.h": "d4e8f7ee2af6fa4bf5b7b16e558c0d6a59f7a87077c4dc81dbb172c1336321f6",
	"dorado/utils/include/utils/math_utils.h":                       "da7d19142cc0d858a1d954257a0f5f3d204b14ee3e5d1611da6038fc9d931101",
	"dorado/utils/include/utils/string_utils.h":                     "a46e8f2cb56248590bcf50b7dcbb0d5c1810700c25040979ce1985d51f8c22bb",
}

// VerifySources ensures all compiled Dorado algorithm bytes match the pin.
func VerifySources(source string) error {
	for name, want := range SourcePins {
		data, err := os.ReadFile(filepath.Join(source, name))
		if err != nil {
			return err
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
			return fmt.Errorf("Dorado source hash mismatch: %s: got %s want %s", name, got, want)
		}
	}
	return nil
}

// Build compiles original source with original utility and NVTX include paths.
// The sole substituted header is the test-only ReadCommon container.
func Build(source, output string) (string, error) {
	if err := VerifySources(source); err != nil {
		return "", err
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate oracle adapter")
	}
	dir := filepath.Dir(here)
	if err := os.MkdirAll(output, 0755); err != nil {
		return "", err
	}
	executable := filepath.Join(output, "stitch-oracle")
	command := exec.Command("c++", "-std=c++20", "-O2", "-DNVTX_DISABLE",
		"-I"+filepath.Join(dir, "shim"),
		"-I"+filepath.Join(source, "dorado/read_pipeline/base/include"),
		"-I"+filepath.Join(source, "dorado/utils/include"),
		"-I"+filepath.Join(source, "dorado/3rdparty/NVTX/c/include"),
		filepath.Join(dir, "oracle.cpp"), filepath.Join(source, "dorado/read_pipeline/base/stitch.cpp"),
		"-o", executable)
	if log, err := command.CombinedOutput(); err != nil {
		return "", fmt.Errorf("compile untouched stitch oracle: %w\n%s", err, log)
	}
	return executable, nil
}
