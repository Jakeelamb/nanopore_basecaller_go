package hacgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// VerifiedBuild is a re-checked oracle build: the manifest, its own digest, and
// the number of declared input pins that still match.
type VerifiedBuild struct {
	Manifest    BuildManifest
	ManifestSHA string
	Path        string
	Inputs      int
	InputsSHA   string
}

// VerifyBuild re-reads an oracle build manifest and requires the binary and every
// declared input pin to still match. A consumer that compares against a capture
// must prove the oracle it names is the one that ran, so a changed translation
// unit, a changed compile line or a rebuilt binary is a hard rejection.
func VerifyBuild(dir string) (VerifiedBuild, error) {
	var v VerifiedBuild
	path := filepath.Join(dir, "manifest.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return v, err
	}
	var m BuildManifest
	if err = json.Unmarshal(raw, &m); err != nil {
		return v, err
	}
	v.Manifest, v.Path = m, path
	sum := sha256.Sum256(raw)
	v.ManifestSHA = hex.EncodeToString(sum[:])
	if m.Status != "built" {
		return v, fmt.Errorf("oracle build status %q is not built", m.Status)
	}
	if m.ABI != 4 {
		return v, fmt.Errorf("oracle build ABI %d is not 4", m.ABI)
	}
	got, err := fileSHA(m.Output)
	if err != nil {
		return v, err
	}
	if got != m.OutputSHA {
		return v, fmt.Errorf("oracle %s digest %s does not match the recorded %s", m.Output, got, m.OutputSHA)
	}
	paths := make([]string, 0, len(m.Inputs))
	for p := range m.Inputs {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	var pins strings.Builder
	for _, p := range paths {
		got, herr := fileSHA(p)
		if herr != nil {
			return v, herr
		}
		if got != m.Inputs[p] {
			return v, fmt.Errorf("declared oracle build pin changed: %s", p)
		}
		fmt.Fprintf(&pins, "%s %s\n", p, got)
	}
	v.Inputs, v.InputsSHA = len(paths), hashString(pins.String())
	return v, nil
}

// Source returns the translation unit the oracle was built from, and its pinned
// digest. The manifest records the absolute path the build used, so the base name
// is what identifies the card's oracle.
func (v VerifiedBuild) Source() (string, string) {
	for p, sha := range v.Manifest.Inputs {
		if strings.Contains(p, string(filepath.Separator)+"verification"+string(filepath.Separator)) &&
			strings.HasSuffix(p, ".cpp") {
			return p, sha
		}
	}
	return "", ""
}

// BackendBuild is the subset of the ABI4 tensor-backend manifest a consumer
// verifies.
type BackendBuild struct {
	ABI         int
	Family      string
	Status      string
	Output      string
	OutputSHA   string
	Torch       string
	Inputs      map[string]string
	ManifestSHA string
	Path        string
	InputCount  int
	InputsSHA   string
}

// VerifyBackend requires the backend to be the qualified ABI4 CUDA13 build and
// returns its verified manifest. A different ABI, a non-CUDA family, a failed
// build or a changed output is a hard rejection: a differential is only evidence
// if it ran the pinned library.
func VerifyBackend(path string) (BackendBuild, error) {
	var b BackendBuild
	manifest := filepath.Join(filepath.Dir(path), "manifest.json")
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return b, err
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
		return b, err
	}
	b = BackendBuild{ABI: doc.ABI, Family: doc.Family, Status: doc.Status,
		Output: doc.Output, OutputSHA: doc.OutputSHA, Torch: doc.Torch,
		Inputs: doc.Inputs, Path: manifest}
	sum := sha256.Sum256(raw)
	b.ManifestSHA = hex.EncodeToString(sum[:])
	if doc.ABI != 4 {
		return b, fmt.Errorf("backend manifest ABI %d is not 4", doc.ABI)
	}
	if doc.Family != "13.0" {
		return b, fmt.Errorf("backend CUDA family %q is not 13.0", doc.Family)
	}
	if doc.Status != "built" {
		return b, fmt.Errorf("backend build status %q is not built", doc.Status)
	}
	abs, err := filepath.Abs(doc.Output)
	if err != nil {
		return b, err
	}
	got, err := fileSHA(abs)
	if err != nil {
		return b, err
	}
	if got != doc.OutputSHA {
		return b, fmt.Errorf("backend %s digest %s does not match the recorded %s", abs, got, doc.OutputSHA)
	}
	paths := make([]string, 0, len(doc.Inputs))
	for p := range doc.Inputs {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	var pins strings.Builder
	for _, p := range paths {
		got, herr := fileSHA(p)
		if herr != nil {
			return b, herr
		}
		if got != doc.Inputs[p] {
			return b, fmt.Errorf("declared backend build pin changed: %s", p)
		}
		fmt.Fprintf(&pins, "%s %s\n", p, got)
	}
	b.InputCount, b.InputsSHA = len(paths), hashString(pins.String())
	return b, nil
}

func hashString(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
