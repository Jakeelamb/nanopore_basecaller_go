package h02

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// backendBuild is the subset of the ABI4 build manifest this consumer verifies.
type backendBuild struct {
	ABI         int
	Family      string
	Status      string
	Output      string
	OutputSHA   string
	Torch       string
	Inputs      map[string]string
	ManifestSHA string
}

// verifyBackend requires the backend to be the qualified ABI4 CUDA13 build and
// returns its verified manifest. A different ABI, a non-CUDA family, a failed
// build or a changed output is a hard rejection: the replay is only evidence if
// it ran the pinned library.
func verifyBackend(path string) (backendBuild, string, error) {
	var b backendBuild
	manifest := filepath.Join(filepath.Dir(path), "manifest.json")
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return b, manifest, err
	}
	var doc struct {
		ABI       int               `json:"abi_version"`
		Family    string            `json:"cuda_family"`
		Status    string            `json:"status"`
		Output    string            `json:"output"`
		OutputSHA string            `json:"output_sha256"`
		Torch     string            `json:"torch_root"`
		Inputs    map[string]string `json:"input_sha256"`
	}
	if err = json.Unmarshal(raw, &doc); err != nil {
		return b, manifest, err
	}
	b = backendBuild{ABI: doc.ABI, Family: doc.Family, Status: doc.Status,
		Output: doc.Output, OutputSHA: doc.OutputSHA, Torch: doc.Torch, Inputs: doc.Inputs}
	sum := sha256.Sum256(raw)
	b.ManifestSHA = hex.EncodeToString(sum[:])
	if doc.ABI != 4 {
		return b, manifest, fmt.Errorf("backend manifest ABI %d is not 4", doc.ABI)
	}
	if doc.Family != "13.0" {
		return b, manifest, fmt.Errorf("backend CUDA family %q is not 13.0", doc.Family)
	}
	if doc.Status != "built" {
		return b, manifest, fmt.Errorf("backend build status %q is not built", doc.Status)
	}
	abs, err := filepath.Abs(doc.Output)
	if err != nil {
		return b, manifest, err
	}
	got, err := fileSHA(abs)
	if err != nil {
		return b, manifest, err
	}
	if got != doc.OutputSHA {
		return b, manifest, fmt.Errorf("backend %s digest %s does not match the recorded %s", abs, got, doc.OutputSHA)
	}
	return b, manifest, nil
}

func fileSHA(path string) (string, error) { return runner.SHA(path) }
