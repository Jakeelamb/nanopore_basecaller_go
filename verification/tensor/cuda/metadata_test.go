package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCUDABuildManifest(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		ok   bool
	}{
		{"abi5_cuda13", `{"abi_version":5,"cuda_family":"13.0","status":"built","output_sha256":"backend-pin","input_sha256":{"oracle":"oracle-pin"},"runtime_sha256":{"library":"runtime-pin"},"torch_root":"/torch"}`, true},
		{"abi4", `{"abi_version":4,"cuda_family":"13.0","status":"built"}`, false},
		{"unknown_abi", `{"abi_version":6,"cuda_family":"13.0","status":"built"}`, false},
		{"missing_abi", `{"cuda_family":"13.0","status":"built"}`, false},
		{"null_abi", `{"abi_version":null,"cuda_family":"13.0","status":"built"}`, false},
		{"string_abi", `{"abi_version":"5","cuda_family":"13.0","status":"built"}`, false},
		{"cpu", `{"abi_version":5,"cuda_family":"none","status":"built"}`, false},
		{"unknown_family", `{"abi_version":5,"cuda_family":"14.0","status":"built"}`, false},
		{"missing_family", `{"abi_version":5,"status":"built"}`, false},
		{"wrong_family_type", `{"abi_version":5,"cuda_family":13,"status":"built"}`, false},
		{"failed", `{"abi_version":5,"cuda_family":"13.0","status":"failed"}`, false},
		{"missing_status", `{"abi_version":5,"cuda_family":"13.0"}`, false},
		{"wrong_status_type", `{"abi_version":5,"cuda_family":"13.0","status":true}`, false},
		{"truncated", `{"abi_version":5`, false},
		{"trailing_json", `{"abi_version":5,"cuda_family":"13.0","status":"built"}{}`, false},
		{"null_manifest", `null`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "manifest.json")
			if err := os.WriteFile(path, []byte(tc.raw), 0600); err != nil {
				t.Fatal(err)
			}
			build, err := readBuildManifest(path)
			if (err == nil) != tc.ok {
				t.Fatalf("metadata acceptance = %t, want %t; error = %v", err == nil, tc.ok, err)
			}
			if tc.ok {
				want := buildManifest{ABI: 5, Family: "13.0", Status: "built", OutputSHA: "backend-pin", Inputs: map[string]string{"oracle": "oracle-pin"}, Runtime: map[string]string{"library": "runtime-pin"}, Torch: "/torch"}
				if !reflect.DeepEqual(build, want) {
					t.Fatalf("metadata changed: got %+v, want %+v", build, want)
				}
			}
		})
	}
	t.Run("missing_manifest", func(t *testing.T) {
		if _, err := readBuildManifest(filepath.Join(t.TempDir(), "missing.json")); !os.IsNotExist(err) {
			t.Fatalf("missing metadata error = %v", err)
		}
	})
}

func TestCUDAEvidenceABI5(t *testing.T) {
	m := newEvidence()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		ABI int `json:"abi_version"`
	}
	if err = json.Unmarshal(raw, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.ABI != 5 || m.Schema != "nanopore-cuda-half-capture/v1" || m.Status != "failed" || m.Started == "" || m.Inputs == nil || m.Runtime == nil || len(m.Commands) != 0 {
		t.Fatalf("unexpected initial receipt: %s", raw)
	}
}

func TestCUDACaptureManifest(t *testing.T) {
	m := newEvidence()
	for _, tc := range []struct {
		name   string
		abi    int
		schema string
		status string
		cases  int
		ok     bool
	}{
		{"abi5", 5, m.Schema, "source_capture_exact_repeat", 37, true},
		{"abi5_mutant_gap", 5, m.Schema, "source_capture_exact_repeat_mutant_gap", 37, true},
		{"historical_abi4", 4, m.Schema, "source_capture_exact_repeat", 37, false},
		{"unknown_abi", 6, m.Schema, "source_capture_exact_repeat", 37, false},
		{"missing_abi", 0, m.Schema, "source_capture_exact_repeat", 37, false},
		{"wrong_schema", 5, "other", "source_capture_exact_repeat", 37, false},
		{"failed", 5, m.Schema, "failed", 37, false},
		{"incomplete", 5, m.Schema, "source_capture_exact_repeat", 36, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prior := evidence{ABI: tc.abi, Schema: tc.schema, Status: tc.status, Cases: make([]savedCase, tc.cases)}
			err := m.checkCaptureManifest(prior)
			if (err == nil) != tc.ok {
				t.Fatalf("capture acceptance = %t, want %t; error = %v", err == nil, tc.ok, err)
			}
			if prior.ABI != tc.abi || m.ABI != 5 || m.Status != "failed" || len(m.Commands) != 0 {
				t.Fatal("metadata validation changed evidence")
			}
		})
	}
}
