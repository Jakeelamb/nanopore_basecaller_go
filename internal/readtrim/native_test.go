package readtrim_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/readtrim"
)

func TestPinnedNativeMux(t *testing.T) {
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		t.Skip("native qualification requires explicit DORADO_SOURCE; ordinary Go tests do not establish native parity")
	}
	root := os.Getenv("READTRIM_ARTIFACT_ROOT")
	if root == "" {
		root = "../../artifacts/readtrim"
	}
	root, err := filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(root, "attempt-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("native evidence %s", attempt)
	executable, err := oracle.Build(source, filepath.Join(attempt, "native-build"))
	if err != nil {
		t.Fatal(err)
	}
	cases := syntheticCases()
	for i := range cases {
		if _, err := readtrim.MuxChangeTrim(cases[i]); err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
	}
	if err := oracle.Compare(executable, attempt, cases); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{3495262, 3495263, 3495264} {
		dir := filepath.Join(attempt, fmt.Sprintf("binary32-%d", n))
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := oracle.Compare(executable, dir, []readtrim.MuxRead{floatBoundary(n)}); err != nil {
			t.Fatal(err)
		}
	}
	for name := range oracle.SourcePins {
		dir := filepath.Join(attempt, "rejected-source", filepath.Base(name))
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		for other := range oracle.SourcePins {
			data, err := os.ReadFile(filepath.Join(source, other))
			if err != nil {
				t.Fatal(err)
			}
			if name == other {
				data = append(data, '\n')
			}
			path := filepath.Join(dir, other)
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		_, err := oracle.Build(dir, filepath.Join(dir, "build"))
		if err == nil || !strings.Contains(err.Error(), "hash mismatch") {
			t.Fatalf("altered source %s was not rejected: %v", name, err)
		}
	}
	t.Logf("exact synthetic native output for %d ordinary cases and three binary32 cases; four source alterations rejected", len(cases))
	manifests := os.Getenv("READTRIM_CAPTURE_MANIFESTS")
	if manifests == "" {
		t.Run("retained-captures", func(t *testing.T) {
			t.Skip("set READTRIM_CAPTURE_MANIFESTS to qualify genuine non-mux identities; synthetic parity alone does not establish these")
		})
		return
	}
	var retained []readtrim.MuxRead
	receipts := make([]map[string]any, 0)
	for _, path := range filepath.SplitList(manifests) {
		r, receipt := loadRetained(t, path)
		got, err := readtrim.MuxChangeTrim(r)
		if err != nil {
			t.Fatal(err)
		}
		if got.Trimmed || !reflect.DeepEqual(got.Read, r) {
			t.Fatal("genuine non-mux capture changed")
		}
		retained = append(retained, r)
		receipts = append(receipts, receipt)
	}
	dir := filepath.Join(attempt, "retained-captures")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := oracle.Compare(executable, dir, retained); err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(receipts, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "capture-pins.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d genuine signal_positive captures exact identity only; native decoder moves and positive biological MUX tails remain unqualified", len(retained))
}

type capturePayload struct {
	File, SHA256 string
	Bytes        int64
}

func loadRetained(t *testing.T, path string) (readtrim.MuxRead, map[string]any) {
	t.Helper()
	manifestData, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct{ Input, Output, Run capturePayload }
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatal(err)
	}
	pins := map[string]any{"manifest": path, "manifest_sha256": fmt.Sprintf("%x", sha256.Sum256(manifestData))}
	load := func(p capturePayload) []byte {
		t.Helper()
		if filepath.Base(p.File) != p.File {
			t.Fatal("capture payload must be a local basename")
		}
		data, err := os.ReadFile(filepath.Join(filepath.Dir(path), p.File))
		if err != nil {
			t.Fatal(err)
		}
		digest := fmt.Sprintf("%x", sha256.Sum256(data))
		if int64(len(data)) != p.Bytes || digest != p.SHA256 {
			t.Fatalf("capture payload pin mismatch %s", p.File)
		}
		pins[p.File] = p
		return data
	}
	var input struct {
		Metadata struct {
			NumMinknowEvents uint64
			EndReasonName    string
		}
		Normalized        capturePayload
		NormalizedSamples uint64
	}
	if err := json.Unmarshal(load(manifest.Input), &input); err != nil {
		t.Fatal(err)
	}
	if input.Metadata.EndReasonName != "signal_positive" {
		t.Fatal("retained capture must be non-mux signal_positive")
	}
	var output struct {
		Sequence, Qstring string
		Moves             []uint8
	}
	if err := json.Unmarshal(load(manifest.Output), &output); err != nil {
		t.Fatal(err)
	}
	var run struct {
		Spec struct {
			Stride uint32
			Name   string
		}
	}
	if err := json.Unmarshal(load(manifest.Run), &run); err != nil {
		t.Fatal(err)
	}
	data := load(input.Normalized)
	if len(data)%2 != 0 || uint64(len(data)/2) != input.NormalizedSamples {
		t.Fatal("normalized sample count mismatch")
	}
	signal := make([]uint16, len(data)/2)
	for i := range signal {
		signal[i] = binary.LittleEndian.Uint16(data[2*i:])
	}
	pins["end_reason"] = "signal_positive"
	pins["model"] = run.Spec.Name
	return readtrim.MuxRead{Sequence: output.Sequence, Qstring: output.Qstring, Moves: output.Moves, Signal: signal, Stride: run.Spec.Stride, IsMuxChange: false, NumMinknowEvents: input.Metadata.NumMinknowEvents, PreTrimSequenceLength: uint64(len(output.Sequence))}, pins
}
