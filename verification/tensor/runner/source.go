package runner

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const CUDAUtilsSHA = "327df4b1f23552c51a57a515aa04aad8fe1fcc242fa94619026bc48763f675c1"

func ExtractCUDA(source, out string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(source, "dorado/torch_utils/cuda_utils.cpp"))
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != CUDAUtilsSHA {
		return "", fmt.Errorf("cuda_utils.cpp source pin mismatch")
	}
	lines := strings.SplitAfter(string(raw), "\n")
	if len(lines) < 394 {
		return "", fmt.Errorf("short CUDA source")
	}
	block := []byte(strings.Join(lines[376:394], ""))
	hash := sha256.Sum256(block)
	if err := os.WriteFile(filepath.Join(out, "cuda_source_block.h"), block, 0644); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash[:]), nil
}
