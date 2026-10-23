// Package decodeoracle builds the pinned Dorado float CPU decoder without Torch.
package decodeoracle

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
const SourcePath = "dorado/basecall/decode/beam_search.cpp"
const SourceSHA256 = "80868413877e26a264924263aa3def5ea4fa68877e72aadbf8d5420dad1af07e"

// VerifySources rejects any changed original source before extraction or build.
func VerifySources(root string) error {
	data, err := os.ReadFile(filepath.Join(root, SourcePath))
	if err != nil {
		return err
	}
	return verifyBytes(data)
}

func verifyBytes(data []byte) error {
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != SourceSHA256 {
		return fmt.Errorf("Dorado source hash mismatch: %s: got %s want %s", SourcePath, got, SourceSHA256)
	}
	return nil
}

// Build copies source lines 18..521 verbatim. They contain every statement in
// the helpers and beam_search template. Only dependency includes and the Torch
// tensor dispatch wrapper are omitted. The driver directly instantiates float.
func Build(root, output string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, SourcePath))
	if err != nil {
		return "", err
	}
	if err := verifyBytes(data); err != nil {
		return "", err
	}
	lines := bytes.SplitAfter(data, []byte("\n"))
	slice := bytes.Join(lines[17:521], nil)
	if err = os.MkdirAll(output, 0755); err != nil {
		return "", err
	}
	if err = os.WriteFile(filepath.Join(output, "beam_search.slice.cpp"), slice, 0600); err != nil {
		return "", err
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate native driver")
	}
	driver, err := os.ReadFile(filepath.Join(filepath.Dir(here), "oracle.cpp"))
	if err != nil {
		return "", err
	}
	if err = os.WriteFile(filepath.Join(output, "oracle.cpp"), driver, 0600); err != nil {
		return "", err
	}
	executable := filepath.Join(output, "decode-oracle")
	args := []string{"-std=c++20", "-O2", "-ffp-contract=off", "-DENABLE_NEON_IMPL=0", "-I" + output, filepath.Join(output, "oracle.cpp"), "-o", executable}
	command := exec.Command("c++", args...)
	compiler, compilerErr := exec.Command("c++", "--version").CombinedOutput()
	if compilerErr != nil {
		return "", fmt.Errorf("compiler identity: %w: %s", compilerErr, compiler)
	}
	start := time.Now().UTC()
	log, buildErr := command.CombinedOutput()
	if err = os.WriteFile(filepath.Join(output, "build.log"), log, 0600); err != nil {
		return "", err
	}
	exitCode := -1
	if command.ProcessState != nil {
		exitCode = command.ProcessState.ExitCode()
	}
	receipt := struct {
		SourceCommit, SourcePath, SourceSHA256, SliceSHA256, DriverSHA256, Compiler string
		Args                                                                        []string
		Started, Finished                                                           time.Time
		ExitCode                                                                    int
	}{SourceCommit, SourcePath, SourceSHA256, fmt.Sprintf("%x", sha256.Sum256(slice)), fmt.Sprintf("%x", sha256.Sum256(driver)), string(compiler), args, start, time.Now().UTC(), exitCode}
	receiptBytes, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return "", err
	}
	if err = os.WriteFile(filepath.Join(output, "build-receipt.json"), append(receiptBytes, '\n'), 0600); err != nil {
		return "", err
	}
	if buildErr != nil {
		return "", fmt.Errorf("compile extracted CPU decoder: %w\n%s", buildErr, log)
	}
	return executable, nil
}
