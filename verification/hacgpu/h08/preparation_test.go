package h08

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func inertPreparation(t *testing.T) (CPU13PreparationOptions, Pin) {
	t.Helper()
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "artifacts"), 0700); err != nil {
		t.Fatal(err)
	}
	pin := func(name string) Pin { return hostJSON(t, filepath.Join(dir, name), map[string]string{"inert": name}) }
	o := CPU13PreparationOptions{Schema: "nanopore-h08-cpu13-preparation/v1", ParentBuild: pin("parent"), ParentBuildSelection: pin("parent-selection"), ControllerBuild: pin("target"), ControllerBuildSelection: pin("target-selection"), Fixtures: pin("fixtures"), FakeProof: pin("fake"), Policy: pin("policy"), Backend: closedbuild.ArtifactOptions{Build: pin("backend-build"), Plan: pin("backend-plan"), Producer: pin("backend-producer"), Output: pin("backend-output")}, Directory: dir, MetadataDirectory: filepath.Join(dir, "artifacts", "metadata"), GuardOutput: filepath.Join(dir, "artifacts", "guard"), RequestOutput: filepath.Join(dir, "artifacts", "request")}
	return o, pin("options")
}

func TestCPU13PreparationBoundary(t *testing.T) {
	for _, mutation := range []string{"none", "schema", "partial", "missing", "relative", "duplicate", "collision", "nested", "existing", "outside-artifacts", "input-collision"} {
		t.Run(mutation, func(t *testing.T) {
			o, selected := inertPreparation(t)
			switch mutation {
			case "schema":
				o.Schema = "other"
			case "partial":
				o.ControllerBuildSelection.SHA256 = ""
			case "missing":
				o.Policy = Pin{}
			case "relative":
				o.GuardOutput = "relative"
			case "duplicate":
				o.Policy = o.FakeProof
			case "collision":
				o.GuardOutput = o.RequestOutput
			case "nested":
				o.GuardOutput = filepath.Join(o.MetadataDirectory, "guard")
			case "existing":
				if err := os.Mkdir(o.MetadataDirectory, 0700); err != nil {
					t.Fatal(err)
				}
			case "outside-artifacts":
				o.MetadataDirectory = filepath.Join(o.Directory, "metadata")
			case "input-collision":
				o.GuardOutput = selected.Path
			}
			err := validatePreparation(o, selected)
			if mutation == "none" && err != nil {
				t.Fatal(err)
			}
			if mutation != "none" && err == nil {
				t.Fatalf("accepted %s", mutation)
			}
		})
	}
}

func TestCPU13PreparationOutputContainment(t *testing.T) {
	for _, field := range []string{"MetadataDirectory", "GuardOutput", "RequestOutput"} {
		for _, location := range []string{"checkout", "outside-checkout", "artifacts-root", "artifacts-prefix"} {
			t.Run(field+"/"+location, func(t *testing.T) {
				o, _ := inertPreparation(t)
				outputs := []string{o.MetadataDirectory, o.GuardOutput, o.RequestOutput}
				path := filepath.Join(o.Directory, "stray-output")
				switch location {
				case "outside-checkout":
					path = filepath.Join(t.TempDir(), "stray-output")
				case "artifacts-root":
					path = filepath.Join(o.Directory, "artifacts")
				case "artifacts-prefix":
					parent := filepath.Join(o.Directory, "artifacts-other")
					if err := os.Mkdir(parent, 0700); err != nil {
						t.Fatal(err)
					}
					path = filepath.Join(parent, "stray-output")
				}
				switch field {
				case "MetadataDirectory":
					o.MetadataDirectory = path
				case "GuardOutput":
					o.GuardOutput = path
				case "RequestOutput":
					o.RequestOutput = path
				}
				selected := hostJSON(t, filepath.Join(o.Directory, "selected-options.json"), o)
				if err := validatePreparation(o, selected); err == nil || !strings.Contains(err.Error(), "strict descendant") {
					t.Fatalf("%s boundary accepted %s: %v", field, location, err)
				}
				if _, err := PrepareCPU13Admission(selected); err == nil || !strings.Contains(err.Error(), "strict descendant") {
					t.Fatalf("%s escaped artifacts at %s: %v", field, location, err)
				}
				for _, output := range outputs {
					if _, err := os.Lstat(output); !os.IsNotExist(err) {
						t.Fatalf("boundary rejection created output %s: %v", output, err)
					}
				}
			})
		}
	}
}

func TestCPU13PreparationIndependentOutputParents(t *testing.T) {
	o, selected := inertPreparation(t)
	for i, output := range []*string{&o.MetadataDirectory, &o.GuardOutput, &o.RequestOutput} {
		parent := filepath.Join(o.Directory, "artifacts", []string{"metadata-parent", "guard-parent", "request-parent"}[i])
		if err := os.Mkdir(parent, 0700); err != nil {
			t.Fatal(err)
		}
		*output = filepath.Join(parent, "fresh-output")
	}
	if err := validatePreparation(o, selected); err != nil {
		t.Fatalf("independently selected output parents rejected: %v", err)
	}
	alias := filepath.Join(o.Directory, "artifacts", "alias")
	if err := os.Symlink(filepath.Dir(o.GuardOutput), alias); err != nil {
		t.Fatal(err)
	}
	o.GuardOutput = filepath.Join(alias, "fresh-output")
	if err := validatePreparation(o, selected); err == nil || !strings.Contains(err.Error(), "symlink aliases") {
		t.Fatalf("aliased output parent accepted: %v", err)
	}
}

func TestCPU13PreparationRejectsOriginalPolicyBeforeNormalization(t *testing.T) {
	o, _ := inertPreparation(t)
	target := hostJSON(t, filepath.Join(o.Directory, "inert-code"), map[string]string{"inert": "not a compiler artifact"})
	policy := hostGuardPolicy(t, target, o.Directory)
	policy.Allowed = []supgpu.RuntimeCode{{Object: hostGuardPin(t, target), SONAME: "inert.so", Needed: []string{}, Tier: "metadata-only", Startup: false}}
	raw, err := json.Marshal(policy)
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
		{"wrong-case", strings.Replace(baseline, `"observer":`, `"Observer":`, 1)},
		{"missing-scalar", strings.Replace(baseline, `,"startup":false`, "", 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(o.Directory, "policy-"+tc.name+".json")
			if err := os.WriteFile(path, []byte(tc.body), 0600); err != nil {
				t.Fatal(err)
			}
			original := Pin{Path: path, SHA256: digest([]byte(tc.body))}
			var normalized supgpu.RuntimePolicy
			if err := ReadPreparationJSON(original, &normalized); err != nil {
				t.Fatalf("old generic parser counterexample rejected: %v", err)
			}
			normalizedPin := hostJSON(t, filepath.Join(o.Directory, "normalized-"+tc.name+".json"), normalized)
			if _, err := supgpu.ReadRuntimePolicy(hostGuardPin(t, normalizedPin)); err != nil {
				t.Fatalf("old normalize-then-validate counterexample rejected: %v", err)
			}
			_, strictErr := supgpu.ReadRuntimePolicy(hostGuardPin(t, original))
			if tc.name == "valid" {
				if strictErr != nil {
					t.Fatal(strictErr)
				}
				return
			}
			if tc.body == baseline || strictErr == nil {
				t.Fatal("original malformed wire must reject")
			}
			options := o
			options.Policy = original
			selected := hostJSON(t, filepath.Join(o.Directory, "options-"+tc.name+".json"), options)
			got, err := PrepareCPU13Admission(selected)
			if err == nil || err.Error() != strictErr.Error() || got.RuntimeHeld {
				t.Fatalf("constructor did not reject original policy at shared boundary: %+v: %v; want %v", got, err, strictErr)
			}
			for _, output := range []string{o.MetadataDirectory, o.GuardOutput, o.RequestOutput} {
				if _, err := os.Lstat(output); !os.IsNotExist(err) {
					t.Fatalf("policy rejection created output %s: %v", output, err)
				}
			}
		})
	}
}

func TestCPU13PreparationAuthenticatedRequest(t *testing.T) {
	o, _ := inertPreparation(t)
	p := hostJSON(t, filepath.Join(o.Directory, "request.json"), o)
	if err := os.WriteFile(p.Path, []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareCPU13Admission(p); err == nil || !strings.Contains(err.Error(), "SHA256") {
		t.Fatalf("changed request bytes accepted: %v", err)
	}
	for _, body := range []string{`{"Schema":"nanopore-h08-cpu13-preparation/v1","Unknown":true}`, `{} {}`, `[]`, `null`, `{"Schema":"unknown"}`} {
		if err := os.WriteFile(p.Path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		p.SHA256 = digest([]byte(body))
		if _, err := PrepareCPU13Admission(p); err == nil {
			t.Fatalf("unsupported request accepted: %s", body)
		}
	}
	if _, err := os.Stat(o.MetadataDirectory); !os.IsNotExist(err) {
		t.Fatalf("rejected boundary wrote output: %v", err)
	}
}

func TestReadPreparationJSONFiles(t *testing.T) {
	t.Run("valid-typed-options-and-exact-whitespace", func(t *testing.T) {
		want, _ := inertPreparation(t)
		body, err := json.Marshal(want)
		if err != nil {
			t.Fatal(err)
		}
		body = append([]byte(" \n"), body...)
		body = append(body, '\n', '\t')
		p := Pin{Path: filepath.Join(want.Directory, "selected.json"), SHA256: digest(body)}
		if err := os.WriteFile(p.Path, body, 0600); err != nil {
			t.Fatal(err)
		}
		var got CPU13PreparationOptions
		if err := ReadPreparationJSON(p, &got); err != nil || got != want {
			t.Fatalf("valid typed options differ: %+v: %v", got, err)
		}
		body[0] = '\t'
		if err := os.WriteFile(p.Path, body, 0600); err != nil {
			t.Fatal(err)
		}
		if err := ReadPreparationJSON(p, &got); err == nil || !strings.Contains(err.Error(), "SHA256") {
			t.Fatalf("changed whitespace accepted with old hash: %v", err)
		}
	})
	t.Run("directory", func(t *testing.T) {
		p := Pin{Path: t.TempDir(), SHA256: strings.Repeat("0", 64)}
		var got CPU13PreparationOptions
		if err := ReadPreparationJSON(p, &got); err == nil || !strings.Contains(err.Error(), "regular") {
			t.Fatalf("directory must reject as nonregular: %v", err)
		}
	})
	t.Run("empty", func(t *testing.T) {
		p := Pin{Path: filepath.Join(t.TempDir(), "empty.json"), SHA256: digest(nil)}
		if err := os.WriteFile(p.Path, nil, 0600); err != nil {
			t.Fatal(err)
		}
		var got CPU13PreparationOptions
		if err := ReadPreparationJSON(p, &got); err == nil || !strings.Contains(err.Error(), "typed JSON object") {
			t.Fatalf("empty request accepted: %v", err)
		}
	})
	t.Run("oversize", func(t *testing.T) {
		p := Pin{Path: filepath.Join(t.TempDir(), "oversize.json"), SHA256: strings.Repeat("0", 64)}
		if err := os.WriteFile(p.Path, nil, 0600); err != nil {
			t.Fatal(err)
		}
		// A sparse file exercises the metadata limit without a large payload.
		if err := os.Truncate(p.Path, (64<<20)+1); err != nil {
			t.Fatal(err)
		}
		var got CPU13PreparationOptions
		if err := ReadPreparationJSON(p, &got); err == nil || !strings.Contains(err.Error(), "64 MiB") {
			t.Fatalf("oversize request did not reject at metadata bound: %v", err)
		}
	})
}

func TestPreparationWritesExclusive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "record.json")
	first, err := writePreparationJSON(path, map[string]string{"first": "preserved"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = writePreparationJSON(path, map[string]string{"second": "forbidden"}); err == nil {
		t.Fatal("existing preparation overwritten")
	}
	if _, err = readPin(first); err != nil {
		t.Fatal(err)
	}
}

func TestCPU13PreparationNeverRetargetsPolicy(t *testing.T) {
	dir := t.TempDir()
	target := hostJSON(t, filepath.Join(dir, "target"), map[string]string{"inert": "selected"})
	other := hostJSON(t, filepath.Join(dir, "other"), map[string]string{"inert": "another"})
	selected := hostGuardPin(t, target)
	for _, role := range []string{"reference", "observer"} {
		p := supgpu.RuntimePolicy{Reference: selected, Observer: selected}
		if err := preparationPolicyTarget(p, target); err != nil {
			t.Fatal(err)
		}
		if role == "reference" {
			p.Reference = hostGuardPin(t, other)
		} else {
			p.Observer = hostGuardPin(t, other)
		}
		before := p.Reference
		if err := preparationPolicyTarget(p, target); err == nil {
			t.Fatal("wrong policy target accepted")
		}
		if p.Reference != before {
			t.Fatal("selected policy mutated")
		}
	}
}

// This opt-in consumes real externally selected artifacts. An ordinary skip
// supplies no successful construction, native parity or runtime qualification.
func TestCPU13PreparationGenuineIntegration(t *testing.T) {
	if os.Getenv("H08_CPU13_PREPARE_INTEGRATION") != "1" {
		t.Skip("genuine artifact metadata integration not selected; no positive qualification")
	}
	selected := Pin{Path: os.Getenv("H08_CPU13_PREPARE_OPTIONS"), SHA256: os.Getenv("H08_CPU13_PREPARE_OPTIONS_SHA")}
	expectation := Pin{Path: os.Getenv("H08_CPU13_PREPARE_EXPECTED"), SHA256: os.Getenv("H08_CPU13_PREPARE_EXPECTED_SHA")}
	var expected CPU13Preparation
	if err := ReadPreparationJSON(expectation, &expected); err != nil {
		t.Fatal(err)
	}
	for _, p := range []Pin{expected.Admission, expected.Request, expected.Plan, expected.Contract, expected.Policy} {
		if err := preparationPin(p); err != nil {
			t.Fatal(err)
		}
	}
	if !expected.RuntimeHeld {
		t.Fatal("independent expectation must preserve runtime hold")
	}
	got, err := PrepareCPU13Admission(selected)
	if err != nil {
		t.Fatal(err)
	}
	if got != expected {
		t.Fatalf("independent output expectations differ: got %+v; want %+v", got, expected)
	}
	if _, _, err = ReadCPU13Admission(got.Admission); err != nil {
		t.Fatal(err)
	}
}
