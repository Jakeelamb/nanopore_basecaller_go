package main

import (
	"encoding/json"
	"testing"
)

func TestCPUBuildManifest(t *testing.T) {
	for _, tc := range []struct {
		name       string
		manifest   string
		wantOK     bool
		wantStatus string
	}{
		{"abi5_cpu", `{"abi_version":5,"cuda_family":"none"}`, true, "failed"},
		{"abi4_cpu", `{"abi_version":4,"cuda_family":"none"}`, false, "failed"},
		{"unknown_abi", `{"abi_version":999,"cuda_family":"none"}`, false, "failed"},
		{"missing_abi", `{"cuda_family":"none"}`, false, "failed"},
		{"null_abi", `{"abi_version":null,"cuda_family":"none"}`, false, "failed"},
		{"wrong_abi_type", `{"abi_version":"5","cuda_family":"none"}`, false, "failed"},
		{"abi5_cuda", `{"abi_version":5,"cuda_family":"13.0"}`, false, "runtime_held"},
		{"unknown_family", `{"abi_version":5,"cuda_family":"other"}`, false, "runtime_held"},
		{"missing_family", `{"abi_version":5}`, false, "runtime_held"},
		{"null_family", `{"abi_version":5,"cuda_family":null}`, false, "runtime_held"},
		{"wrong_family_type", `{"abi_version":5,"cuda_family":13}`, false, "failed"},
		{"corrupt_json", `{"abi_version":5,"cuda_family":"none"`, false, "failed"},
		{"trailing_json", `{"abi_version":5,"cuda_family":"none"}{}`, false, "failed"},
		{"null_manifest", `null`, false, "failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := evidence{Status: "failed"}
			err := m.checkBuildManifest([]byte(tc.manifest))
			if (err == nil) != tc.wantOK {
				t.Fatalf("metadata acceptance = %v, want %v; error = %v", err == nil, tc.wantOK, err)
			}
			if m.Status != tc.wantStatus {
				t.Errorf("status = %q, want %q", m.Status, tc.wantStatus)
			}
			if tc.name == "corrupt_json" {
				if _, ok := err.(*json.SyntaxError); !ok {
					t.Errorf("corrupt metadata error = %T, want *json.SyntaxError", err)
				}
			}
		})
	}
}
