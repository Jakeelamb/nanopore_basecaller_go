package supgpu

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestReadRuntimePolicyClosedWire(t *testing.T) {
	p, _, _ := metadataGuardFixture(t)
	p.Policy.Allowed = []RuntimeCode{{Object: p.Policy.Audit, SONAME: "inert.so", Needed: []string{}, Tier: "metadata-only", Startup: false}}
	raw, err := json.Marshal(p.Policy)
	if err != nil {
		t.Fatal(err)
	}
	baseline := string(raw)
	for _, tc := range []struct {
		name, body string
	}{
		{"valid", baseline},
		{"duplicate-observer", strings.Replace(baseline, `"observer":`, `"observer":{},"observer":`, 1)},
		{"duplicate-provider", strings.Replace(baseline, `"soname":"inert.so"`, `"soname":"other.so","soname":"inert.so"`, 1)},
		{"wrong-case-observer", strings.Replace(baseline, `"observer":`, `"Observer":`, 1)},
		{"wrong-case-provider", strings.Replace(baseline, `"soname":`, `"SONAME":`, 1)},
		{"missing-provider-scalar", strings.Replace(baseline, `,"startup":false`, "", 1)},
		{"missing-policy-field", strings.Replace(baseline, `"rejected_alternates":[],`, "", 1)},
		{"null", `null`},
		{"null-field", strings.Replace(baseline, `"rejected_alternates":[]`, `"rejected_alternates":null`, 1)},
		{"unknown", strings.Replace(baseline, `"schema":`, `"unknown":true,"schema":`, 1)},
		{"trailing", baseline + ` {}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name != "valid" && tc.body == baseline {
				t.Fatal("wire mutation did not change baseline")
			}
			path := filepath.Join(t.TempDir(), "policy.json")
			if err := os.WriteFile(path, []byte(tc.body), 0600); err != nil {
				t.Fatal(err)
			}
			selected, err := pin(path)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ReadRuntimePolicy(selected)
			if tc.name == "valid" {
				if err != nil || !reflect.DeepEqual(got, p.Policy) {
					t.Fatalf("valid policy changed/rejected: %+v: %v", got, err)
				}
			} else if err == nil {
				t.Fatal("malformed original policy accepted")
			}
		})
	}
}

func TestReadRuntimePolicyAuthenticatesSelectionAndPolicy(t *testing.T) {
	for _, mutation := range []string{"relative", "hash", "zero-bytes", "wrong-bytes", "changed", "lazy", "provider", "directory", "code-bytes"} {
		t.Run(mutation, func(t *testing.T) {
			p, _, _ := metadataGuardFixture(t)
			switch mutation {
			case "lazy":
				p.Policy.Lazy = "unaudited"
			case "provider":
				p.Policy.Allowed = []RuntimeCode{{Object: p.Policy.Audit, SONAME: "inert.so", Needed: []string{}, Tier: ""}}
			case "directory":
				p.Policy.Directories = append(p.Policy.Directories, p.Policy.Directories[0])
			case "code-bytes":
				p.Policy.Observer.Bytes++
			}
			selected := metadataGuardJSON(t, filepath.Join(t.TempDir(), "policy.json"), p.Policy)
			switch mutation {
			case "relative":
				selected.Path = filepath.Base(selected.Path)
			case "hash":
				selected.SHA256 = strings.Repeat("0", 64)
			case "zero-bytes":
				selected.Bytes = 0
			case "wrong-bytes":
				selected.Bytes++
			case "changed":
				if err := os.WriteFile(selected.Path, []byte("{}"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := ReadRuntimePolicy(selected); err == nil {
				t.Fatalf("accepted %s", mutation)
			}
		})
	}
}

// These inert files exercise metadata authentication only. They are not native
// artifacts, loader receipts or evidence that an audited runtime works.
func metadataGuardFixture(t *testing.T) (GuardPlan, PhaseContract, Pin) {
	t.Helper()
	dir := t.TempDir()
	object := filepath.Join(dir, "inert-object")
	if err := os.WriteFile(object, []byte("metadata-only test bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	code, err := pin(object)
	if err != nil {
		t.Fatal(err)
	}
	c := PhaseContract{
		Schema: "nanopore-audited-phase-contract/v1", Role: "observer",
		Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"},
	}
	contract := metadataGuardJSON(t, filepath.Join(dir, "contract.json"), c)
	p := GuardPlan{
		Schema: "nanopore-audited-guard-plan/v1", Contract: contract,
		Policy: RuntimePolicy{
			Schema:    "nanopore-supgpu-selected-runtime/v1",
			Lazy:      "audit-every-loader-object-before-initializers/v1",
			Reference: code, Observer: code, Audit: code,
			Allowed: []RuntimeCode{}, Alternates: []RuntimeCode{},
			Directories: []string{dir},
		},
		Inputs: []Pin{code},
	}
	return p, c, contract
}

func metadataGuardJSON(t *testing.T, path string, value any) Pin {
	t.Helper()
	if err := writeJSON(path, value); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReadGuardPlanMetadata(t *testing.T) {
	p, c, contract := metadataGuardFixture(t)
	selected := metadataGuardJSON(t, filepath.Join(t.TempDir(), "plan.json"), p)
	gotPlan, gotContract, err := ReadGuardPlan(selected, contract)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotPlan, p) || !reflect.DeepEqual(gotContract, c) {
		t.Fatal("metadata reader changed selected records")
	}
	if err := os.WriteFile(contract.Path, []byte("changed contract"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadGuardPlan(selected, contract); err == nil {
		t.Fatal("changed phase bytes accepted")
	}
}

func TestReadGuardPlanRequiresExactByteCounts(t *testing.T) {
	for _, field := range []string{"contract", "reference", "observer"} {
		t.Run(field, func(t *testing.T) {
			p, _, contract := metadataGuardFixture(t)
			switch field {
			case "contract":
				contract.Bytes = 0
				p.Contract = contract
			case "reference":
				p.Policy.Reference.Bytes = 0
			case "observer":
				p.Policy.Observer.Bytes = 0
			}
			selected := metadataGuardJSON(t, filepath.Join(t.TempDir(), "plan.json"), p)
			_, _, err := ReadGuardPlan(selected, contract)
			if err == nil || !strings.Contains(err.Error(), "pin changed") {
				t.Fatalf("zero byte count on a nonempty %s must reject: %v", field, err)
			}
		})
	}
}
