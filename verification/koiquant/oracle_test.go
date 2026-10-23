package koiquant

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, path, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("test source location unavailable")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(path), "../.."))
}

func TestExtractClosedSource(t *testing.T) {
	blocks, err := Extract(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	cpp, err := generated(blocks)
	if err != nil {
		t.Fatal(err)
	}
	for _, block := range blocks {
		if strings.Count(cpp, block.Text) != 1 || block.SHA256 != digest([]byte(block.Text)) {
			t.Fatalf("source block not preserved once: %s", block.File)
		}
	}
	for _, forbidden := range []string{"torch::load", "torch/serialize", "// CACHE_BLOCK", "// QUANTIZE_BLOCK"} {
		if strings.Contains(cpp, forbidden) {
			t.Fatalf("unexpected source in generated raw oracle: %s", forbidden)
		}
	}
	if strings.Count(cpp, "same_bytes(tiled, original.t)") != 1 || strings.Count(cpp, "same_bytes(host, input)") != 1 {
		t.Fatal("missing untouched output or transfer parity")
	}
}

func TestExtractRejectsChangedPinnedSource(t *testing.T) {
	root := repoRoot(t)
	for _, victim := range blockDefinitions {
		t.Run(filepath.Base(victim.File), func(t *testing.T) {
			copyRoot := t.TempDir()
			for _, block := range blockDefinitions {
				b, err := os.ReadFile(filepath.Join(root, block.File))
				if err != nil {
					t.Fatal(err)
				}
				if block.File == victim.File {
					b = append(b, '\n')
				}
				path := filepath.Join(copyRoot, block.File)
				if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				if err = os.WriteFile(path, b, 0644); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := Extract(copyRoot); err == nil {
				t.Fatal("changed whole-file source accepted")
			}
		})
	}
}

func TestFixturesDeterministicAndObservational(t *testing.T) {
	for _, name := range fixtureNames[1:] {
		a, err := fixtureHalf(name)
		if err != nil {
			t.Fatal(err)
		}
		b, err := fixtureHalf(name)
		if err != nil || !reflect.DeepEqual(a, b) || len(a) != Elements {
			t.Fatalf("fixture not deterministic: %s", name)
		}
		if name == "all-half-patterns" {
			var seen [65536]bool
			for _, bits := range a {
				seen[bits] = true
			}
			for bits, ok := range seen {
				if !ok {
					t.Fatalf("half pattern missing: %04x", bits)
				}
			}
		}
	}
	if _, err := fixtureHalf("unknown"); err == nil {
		t.Fatal("unknown fixture accepted")
	}
	if fixtureDomain("genuine") != "genuine-unclassified" {
		t.Fatal("genuine domain was inferred without native arithmetic")
	}
}

func zeroStages(t *testing.T) map[string]koiweights.Data {
	t.Helper()
	dir := t.TempDir()
	out := make(map[string]koiweights.Data)
	for _, spec := range StageSpecs() {
		w, err := width(spec.DType)
		if err != nil {
			t.Fatal(err)
		}
		n := int(w)
		for _, size := range spec.Shape {
			n *= size
		}
		b := make([]byte, n)
		path := filepath.Join(dir, spec.Name+".bin")
		if err := os.WriteFile(path, b, 0644); err != nil {
			t.Fatal(err)
		}
		out[spec.Name] = koiweights.Data{Path: path, DType: spec.DType, Endian: "little", Shape: spec.Shape, Strides: spec.Strides, Bytes: int64(n), SHA256: digest(b)}
	}
	return out
}

func TestValidateStagesRejectsCorruptContracts(t *testing.T) {
	stages := zeroStages(t)
	if err := ValidateStages(stages, stages["input"]); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(map[string]koiweights.Data)
	}{
		{"missing-stage", func(m map[string]koiweights.Data) { delete(m, "hscale") }},
		{"unknown-stage", func(m map[string]koiweights.Data) { m["extra"] = m["abs"] }},
		{"unknown-dtype", func(m map[string]koiweights.Data) { d := m["char"]; d.DType = "uint8"; m["char"] = d }},
		{"missing-dtype", func(m map[string]koiweights.Data) { d := m["char"]; d.DType = ""; m["char"] = d }},
		{"half-width-int8", func(m map[string]koiweights.Data) { d := m["char"]; d.Bytes *= 2; m["char"] = d }},
		{"missing-pin", func(m map[string]koiweights.Data) { d := m["amax"]; d.SHA256 = ""; m["amax"] = d }},
		{"wrong-shape", func(m map[string]koiweights.Data) { d := m["i8tile"]; d.Shape = []int{Elements}; m["i8tile"] = d }},
		{"wrong-strides", func(m map[string]koiweights.Data) { d := m["fscale"]; d.Strides = []int{1}; m["fscale"] = d }},
		{"wrong-endian", func(m map[string]koiweights.Data) { d := m["hround"]; d.Endian = "big"; m["hround"] = d }},
		{"missing-payload", func(m map[string]koiweights.Data) { d := m["abs"]; d.Path += "-missing"; m["abs"] = d }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := make(map[string]koiweights.Data, len(stages))
			for k, d := range stages {
				m[k] = d
			}
			tc.mutate(m)
			if err := ValidateStages(m, stages["input"]); err == nil {
				t.Fatal("invalid observation accepted")
			}
		})
	}
	for _, name := range []string{"transfer", "untouched-i8tile", "untouched-freciprocal"} {
		t.Run(name+"-byte-parity", func(t *testing.T) {
			d := stages[name]
			b, err := os.ReadFile(d.Path)
			if err != nil {
				t.Fatal(err)
			}
			b[0] = 1
			if err := os.WriteFile(d.Path, b, 0644); err != nil {
				t.Fatal(err)
			}
			d.SHA256 = digest(b)
			stages[name] = d
			if err := ValidateStages(stages, stages["input"]); err == nil {
				t.Fatal("changed native parity bytes accepted")
			}
			b[0] = 0
			if err := os.WriteFile(d.Path, b, 0644); err != nil {
				t.Fatal(err)
			}
			d.SHA256 = digest(b)
			stages[name] = d
		})
	}
}

func TestRequireTestPassRejectsFalsePass(t *testing.T) {
	pass := []byte("{\"Action\":\"run\",\"Test\":\"TestRequired\"}\n{\"Action\":\"pass\",\"Test\":\"TestRequired\"}\n{\"Action\":\"pass\"}\n")
	if err := RequireTestPass(pass, []string{"TestRequired"}); err != nil {
		t.Fatal(err)
	}
	for _, raw := range [][]byte{nil, []byte("{\"Action\":\"pass\"}\n"), bytes.ReplaceAll(pass, []byte("TestRequired"), []byte("TestOther")), bytes.ReplaceAll(pass, []byte("\"pass\""), []byte("\"skip\"")), bytes.ReplaceAll(pass, []byte("\"pass\""), []byte("\"fail\"")), append(pass, pass...), []byte("not-json")} {
		if err := RequireTestPass(raw, []string{"TestRequired"}); err == nil {
			t.Fatalf("false pass accepted: %s", raw)
		}
	}
}

func TestSourcePinsCannotDisappear(t *testing.T) {
	root := repoRoot(t)
	pins, err := currentSources(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkSources(root, pins); err != nil {
		t.Fatal(err)
	}
	delete(pins, sourceFiles[0])
	if err := checkSources(root, pins); err == nil {
		t.Fatal("dropped current-source pin accepted")
	}
}

func TestCommandReceiptsBeforeInterpretation(t *testing.T) {
	dir := t.TempDir()
	stdout, stderr := filepath.Join(dir, "stdout"), filepath.Join(dir, "stderr")
	for _, p := range []string{stdout, stderr} {
		if err := os.WriteFile(p, nil, 0644); err != nil {
			t.Fatal(err)
		}
	}
	c := koiweights.Command{Phase: "consumer", Args: []string{"go", "test"}, Started: "2026-10-23T00:00:00Z", Finished: "2026-10-23T00:00:01Z", Stdout: stdout, Stderr: stderr, StdoutSHA256: digest(nil), StderrSHA256: digest(nil)}
	if err := ValidateCommand(c); err != nil {
		t.Fatal(err)
	}
	c.Exit = 1
	if err := ValidateCommand(c); err == nil {
		t.Fatal("nonzero exit accepted")
	}
	c.Exit, c.Finished = 0, "2026-10-22T00:00:01Z"
	if err := ValidateCommand(c); err == nil {
		t.Fatal("reversed clocks accepted")
	}
}

func TestPreparedNativeClaimRejectedBeforeInputs(t *testing.T) {
	for _, m := range []Preparation{
		{Schema: Schema, Status: "captured", NativeAdmission: "admitted", NativeExecution: true},
		{Schema: Schema, Status: "prepared_native_held", Device: "cpu", NativeAdmission: "held", OriginSHA256: OriginSHA256, NativeExecution: true},
	} {
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "manifest.json")
		if err = os.WriteFile(path, b, 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := CheckPrepared("missing-root", path); err == nil || !strings.Contains(err.Error(), "source-only preparation identity") {
			t.Fatalf("native claim not rejected at schema boundary: %v", err)
		}
	}
}

func TestPreparedUnknownFieldsRejectedBeforeInputs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, []byte(`{"Schema":"nanopore-sup-qkv-i8/prepare-v1","UnknownRuntimePin":true}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := CheckPrepared("missing-root", path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown manifest field accepted: %v", err)
	}
}

func TestPrepareRejectsUnknownDeviceBeforeInputs(t *testing.T) {
	out := filepath.Join(t.TempDir(), "attempt-invalid")
	_, err := Prepare(Options{Root: "missing-root", InputCapture: "missing-manifest", Out: out, Device: "gpu"})
	if err == nil || !strings.Contains(err.Error(), "device must be cpu or cuda") {
		t.Fatalf("device was not rejected before input validation: %v", err)
	}
	if _, err := os.Lstat(out); !os.IsNotExist(err) {
		t.Fatal("invalid device created output")
	}
}

func TestOutputAdmissionPreservesExistingAttempts(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module github.com/Jakeelamb/nanopore_basecaller_go\n"), 0644); err != nil {
		t.Fatal(err)
	}
	parent := filepath.Join(root, "artifacts/koiquant")
	if err := os.MkdirAll(parent, 0755); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(parent, "attempt-test")
	if err := admitOutput(root, out); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	if err := admitOutput(root, out); err == nil {
		t.Fatal("existing output accepted")
	}
	if err := admitOutput(root, filepath.Join(root, "elsewhere", "attempt-test")); err == nil {
		t.Fatal("foreign output accepted")
	}
	if err := os.Symlink(t.TempDir(), filepath.Join(parent, "attempt-symlink")); err != nil {
		t.Fatal(err)
	}
	if err := admitOutput(root, filepath.Join(parent, "attempt-symlink")); err == nil {
		t.Fatal("symlink output accepted")
	}
}
