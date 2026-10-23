package subreadoracle

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func attempt(t *testing.T, prefix string) string {
	t.Helper()
	root := os.Getenv("SUBREAD_ARTIFACT_ROOT")
	if root == "" {
		root = filepath.Join("..", "..", "artifacts", "subread")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.MkdirAll(abs, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(abs, prefix+"-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(dir)
	return dir
}
func equalRead(a, b subread.Read) bool { return bytes.Equal(EncodeRead(a), EncodeRead(b)) }
func compare(cases []Case, got [][]subread.Read) error {
	if len(cases) != len(got) {
		return fmt.Errorf("native case count differs")
	}
	for i, c := range cases {
		want, err := subread.Materialize(c.Read, c.Spacers)
		if err != nil {
			return fmt.Errorf("%s: %w", c.Name, err)
		}
		if len(want) != len(got[i]) {
			return fmt.Errorf("%s child count %d != native %d", c.Name, len(want), len(got[i]))
		}
		for j := range want {
			if !equalRead(want[j], got[i][j]) {
				return fmt.Errorf("%s child %d differs\nGo=%+v\nnative=%+v", c.Name, j, want[j], got[i][j])
			}
		}
	}
	return nil
}
func TestFixtureInputsAndImmutability(t *testing.T) {
	for _, c := range Cases() {
		t.Run(c.Name, func(t *testing.T) {
			before := EncodeRead(c.Read)
			out, err := subread.Materialize(c.Read, c.Spacers)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, EncodeRead(c.Read)) {
				t.Fatal("input mutated")
			}
			for i, r := range out {
				if len(r.Sequence) == 0 && len(c.Read.Sequence) > 0 {
					t.Fatal("empty child emitted")
				}
				if len(c.Spacers) > 0 && len(c.Read.Sequence) > 0 && len(c.Read.Moves) > 0 {
					if r.MeanQStart != 0 || r.NumEvents != 0 || r.ReadNumber != -1 || r.PreTrimSeqLen != uint64(len(r.Sequence)) {
						t.Fatal("child reset differs")
					}
					if r.SubreadID != uint64(i) || r.SplitCount != uint64(len(out)) {
						t.Fatal("child finalization differs")
					}
				}
			}
		})
	}
}
func TestProtocolRejectsMalformedOutput(t *testing.T) {
	for _, data := range [][]byte{nil, {1, 0, 0, 0}, {255, 255, 255, 255}} {
		if _, _, err := DecodeOutput(data, 1, 0); err == nil {
			t.Fatal("bad output accepted")
		}
	}
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(1))
	binary.Write(&b, binary.LittleEndian, uint32(1<<25))
	if _, _, err := DecodeOutput(b.Bytes(), 1, 0); err == nil {
		t.Fatal("oversize output accepted")
	}
}

func TestPinnedNativeMaterialization(t *testing.T) {
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		t.Skip("set DORADO_SOURCE for pinned native materialization; no inference is run")
	}
	dir := attempt(t, "native")
	executable, err := Build(source, filepath.Join(dir, "build"))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("native-clock-boundary", func(t *testing.T) {
		for _, tc := range []struct {
			name, acquisition, roundedOffset, want string
			reject                                 bool
		}{
			{"current-era", "1700000000123", "1", "2023-11-14T22:13:20.124000+00:00\n", false},
			{"last-acquisition", "9223372036854", "0", "2262-04-11T23:47:16.854000+00:00\n", false},
			{"first-acquisition-rejected", "9223372036855", "0", "", true},
			{"last-summed-rounded", "9223372036853", "1", "2262-04-11T23:47:16.854000+00:00\n", false},
			{"first-summed-rounded-rejected", "9223372036854", "1", "", true},
			{"last-trim-and-spacer", "9223372036849", "5", "2262-04-11T23:47:16.854000+00:00\n", false},
			{"first-trim-and-spacer-rejected", "9223372036850", "5", "", true},
			{"last-relative", "0", "9223372036854", "2262-04-11T23:47:16.854000+00:00\n", false},
			{"first-relative-rejected", "0", "9223372036855", "", true},
			{"year-9999-rejected", "253402300799000", "0", "", true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				output, err := Command(dir, "clock-"+tc.name, exec.CommandContext(ctx, executable, "--clock-boundary", tc.acquisition, tc.roundedOffset), nil)
				if tc.reject {
					var exit *exec.ExitError
					if !errors.As(err, &exit) || exit.ExitCode() != 2 || len(output) != 0 || !strings.Contains(err.Error(), "timestamp outside native signed-nanosecond clock domain") {
						t.Fatalf("native clock shim failed to reject before overflowing source conversion: %s, %v", output, err)
					}
					return
				}
				if err != nil || string(output) != tc.want {
					t.Fatalf("native clock boundary differs: %q, %v", output, err)
				}
			})
		}
	})
	cases, ids := Cases(), UUIDCases()
	retained, pins, err := retainedCases()
	if err != nil {
		t.Fatal(err)
	}
	cases = append(cases, retained...)
	if err = SaveJSON(filepath.Join(dir, "cases.json"), cases); err != nil {
		t.Fatal(err)
	}
	input, err := EncodeInput(cases, ids)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	output, err := Command(dir, "native", exec.CommandContext(ctx, executable), input)
	if err != nil {
		t.Fatal(err)
	}
	results, uuids, err := DecodeOutput(output, len(cases), len(ids))
	if err != nil {
		t.Fatal(err)
	}
	if err = compare(cases, results); err != nil {
		t.Fatal(err)
	}
	for i, id := range ids {
		if want := subread.DeriveUUID(id.Parent, id.Index); uuids[i] != want {
			t.Fatalf("UUID %v native %q != %q", id, uuids[i], want)
		}
	}
	for _, c := range retained {
		got, err := subread.Materialize(c.Read, nil)
		if err != nil || len(got) != 1 || !equalRead(got[0], c.Read) {
			t.Fatal("retained no-spacer identity differs")
		}
	}
	if err = SaveJSON(filepath.Join(dir, "native-results.json"), results); err != nil {
		t.Fatal(err)
	}
	var goResults [][]subread.Read
	for _, c := range cases {
		result, e := subread.Materialize(c.Read, c.Spacers)
		if e != nil {
			t.Fatal(e)
		}
		goResults = append(goResults, result)
	}
	if err = SaveJSON(filepath.Join(dir, "go-results.json"), goResults); err != nil {
		t.Fatal(err)
	}
	if err = SaveJSON(filepath.Join(dir, "comparison.json"), map[string]any{"passed": true, "case_count": len(cases), "uuid_cases": ids, "uuid_outputs": uuids, "retained_pins": pins, "native_payload_sha256": Hash(output), "qualification": "all represented read fields and signal bits exact; retained HAC/SUP supplied-no-spacer identity only; no detector, inference, GPU, FASTQ, accuracy or performance claim"}); err != nil {
		t.Fatal(err)
	}
	t.Run("source-mutation-rejected", func(t *testing.T) {
		copyRoot := filepath.Join(dir, "mutated-source")
		for path := range SourcePins {
			data, e := os.ReadFile(filepath.Join(source, path))
			if e != nil {
				t.Fatal(e)
			}
			dst := filepath.Join(copyRoot, path)
			if e = os.MkdirAll(filepath.Dir(dst), 0700); e != nil {
				t.Fatal(e)
			}
			if path == "dorado/utils/uuid_utils.cpp" {
				data = append(data, '\n')
			}
			if e = os.WriteFile(dst, data, 0600); e != nil {
				t.Fatal(e)
			}
		}
		if _, e := Build(copyRoot, filepath.Join(dir, "rejected-source-build")); e == nil {
			t.Fatal("modified native source accepted")
		}
	})
	t.Run("malformed-native-protocol-rejected", func(t *testing.T) {
		wrongMagic := bytes.Clone(input)
		wrongMagic[0] ^= 1
		for name, payload := range map[string][]byte{"wrong-magic": wrongMagic, "truncated": input[:len(input)-1], "trailing": append(bytes.Clone(input), 0)} {
			_, err := Command(dir, "rejected-"+name, exec.CommandContext(ctx, executable), payload)
			if err == nil {
				t.Fatalf("malformed native protocol %s accepted", name)
			}
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() != 2 {
				t.Fatalf("malformed native protocol %s did not return controlled exit", name)
			}
		}
	})
	t.Run("metadata-and-payload-mutations-rejected", func(t *testing.T) {
		index := 1
		for name, change := range map[string]func(*subread.Read){"mean-q-offset": func(r *subread.Read) { r.MeanQStart++ }, "relative-clock": func(r *subread.Read) { r.StartTimeMS++ }, "absolute-clock": func(r *subread.Read) { r.StartTime += "x" }, "root-parent": func(r *subread.Read) { r.ParentID = "wrong" }, "inherited-run": func(r *subread.Read) { r.Inherited.RunID = "wrong" }, "signal-bit": func(r *subread.Read) { r.Signal = append([]uint16(nil), r.Signal...); r.Signal[0] ^= 1 }, "move-bit": func(r *subread.Read) { r.Moves = append([]uint8(nil), r.Moves...); r.Moves[0] ^= 1 }, "end-reason": func(r *subread.Read) { r.EndReason = "wrong" }, "neighbor": func(r *subread.Read) { r.NextID = "wrong" }, "uuid": func(r *subread.Read) { r.ID = "wrong" }, "split-count": func(r *subread.Read) { r.SplitCount++ }} {
			mutated := append([][]subread.Read(nil), results...)
			mutated[index] = append([]subread.Read(nil), results[index]...)
			change(&mutated[index][0])
			if e := compare(cases, mutated); e == nil {
				t.Fatalf("%s mutation accepted", name)
			}
		}
	})
}

type payload struct {
	File, SHA256 string
	Bytes        int64
}

func payloadBytes(dir string, p payload) ([]byte, error) {
	if p.File == "" || filepath.Base(p.File) != p.File || p.Bytes < 0 {
		return nil, fmt.Errorf("invalid capture payload")
	}
	data, err := os.ReadFile(filepath.Join(dir, p.File))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != p.Bytes || Hash(data) != p.SHA256 {
		return nil, fmt.Errorf("capture payload differs: %s", p.File)
	}
	return data, nil
}
func retainedCases() ([]Case, map[string]string, error) {
	root := filepath.Join("..", "..")
	var out []Case
	pins := map[string]string{}
	for _, c := range []struct{ path, sha, name string }{
		{"artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0", "2d37e08b342b94915a987e0905f667e2cfe6ce6f9eaf74e99e66395328681b99", "retained-HAC-no-spacers"},
		{"artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0", "cfb0c39f250ce37450a35457cbdc6b64d7346f0c5c6031c6d9757197ad19f29a", "retained-SUP-no-spacers"},
	} {
		dir := filepath.Join(root, c.path)
		manifest, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
		if err != nil {
			return nil, nil, err
		}
		if Hash(manifest) != c.sha {
			return nil, nil, fmt.Errorf("capture manifest differs: %s", c.path)
		}
		pins[c.path+"/manifest.json"] = Hash(manifest)
		var m struct {
			Schema             int
			Input, Output, Run payload
		}
		if err = json.Unmarshal(manifest, &m); err != nil {
			return nil, nil, err
		}
		if m.Schema != 1 {
			return nil, nil, fmt.Errorf("capture schema differs")
		}
		input, err := payloadBytes(dir, m.Input)
		if err != nil {
			return nil, nil, err
		}
		output, err := payloadBytes(dir, m.Output)
		if err != nil {
			return nil, nil, err
		}
		run, err := payloadBytes(dir, m.Run)
		if err != nil {
			return nil, nil, err
		}
		var in struct {
			Metadata struct {
				StartSample, EndSample, NumSamples, NumMinknowEvents                                                                                        uint64
				SampleRate                                                                                                                                  uint32
				AcquisitionStartTimeMS, ProtocolStartTimeMS                                                                                                 int64
				ReadNumber                                                                                                                                  uint32
				Channel, Well                                                                                                                               uint16
				RunID, FlowCellProductCode, SequencingKit, FlowCellID, PositionID, AcquisitionID, ExperimentID, SampleID, Filename, PoreType, EndReasonName string
			}
			Normalized        payload
			Trimmed           int
			NormalizedSamples uint64
		}
		if err = json.Unmarshal(input, &in); err != nil {
			return nil, nil, err
		}
		var decoded struct {
			ID, Sequence, Qstring string
			Moves                 []uint8
		}
		if err = json.Unmarshal(output, &decoded); err != nil {
			return nil, nil, err
		}
		var model struct {
			Spec struct {
				Stride     int
				Name       string
				MeanQStart uint32
			}
		}
		if err = json.Unmarshal(run, &model); err != nil {
			return nil, nil, err
		}
		normalized, err := payloadBytes(dir, in.Normalized)
		if err != nil {
			return nil, nil, err
		}
		if len(normalized)%2 != 0 || uint64(len(normalized)/2) != in.NormalizedSamples || in.Trimmed < 0 || model.Spec.Stride <= 0 || in.Metadata.AcquisitionStartTimeMS < 0 {
			return nil, nil, fmt.Errorf("capture input invalid")
		}
		signal := make([]uint16, len(normalized)/2)
		for i := range signal {
			signal[i] = binary.LittleEndian.Uint16(normalized[i*2:])
		}
		md := in.Metadata
		relative := md.StartSample * 1000 / uint64(md.SampleRate)
		r := subread.Read{Signal: signal, Sequence: decoded.Sequence, Qstring: decoded.Qstring, Moves: decoded.Moves, Stride: uint64(model.Spec.Stride), ID: decoded.ID, StartSample: md.StartSample, EndSample: md.EndSample, AcquisitionStartMS: uint64(md.AcquisitionStartTimeMS), SampleRate: uint64(md.SampleRate), StartTimeMS: relative, StartTime: time.UnixMilli(md.AcquisitionStartTimeMS + int64(relative)).UTC().Format("2006-01-02T15:04:05.000000+00:00"), LeadingTrim: uint64(in.Trimmed), SplitCount: 1, ReadNumber: int64(md.ReadNumber), NumSamples: md.NumSamples, NumEvents: md.NumMinknowEvents, PreTrimSeqLen: uint64(len(decoded.Sequence)), MeanQStart: model.Spec.MeanQStart, EndReason: md.EndReasonName,
			Inherited: subread.Inherited{RunID: md.RunID, FlowCellProductCode: md.FlowCellProductCode, SequencingKit: md.SequencingKit, FlowcellID: md.FlowCellID, PositionID: md.PositionID, AcquisitionID: md.AcquisitionID, ExperimentID: md.ExperimentID, SampleID: md.SampleID, ProtocolStartMS: md.ProtocolStartTimeMS, ModelName: model.Spec.Name, Mux: uint32(md.Well), Channel: int32(md.Channel), Filename: md.Filename, PoreType: md.PoreType}}
		out = append(out, Case{c.name, r, nil})
		for _, p := range []payload{m.Input, m.Output, m.Run, in.Normalized} {
			pins[c.path+"/"+p.File] = p.SHA256
		}
	}
	return out, pins, nil
}

func TestBehaviorMutations(t *testing.T) {
	if os.Getenv("SUBREAD_MUTATIONS") != "1" {
		t.Skip("run maintained checker for isolated Go behavior mutation controls")
	}
	dir := attempt(t, "mutations")
	for name, mutation := range map[string][2]string{
		"child-mean-q-inherited": {"out.PreTrimSeqLen, out.MeanQStart = uint64(len(out.Sequence)), 0", "out.PreTrimSeqLen, out.MeanQStart = uint64(len(out.Sequence)), in.MeanQStart"},
		"inherited-run-dropped":  {"out := in", "out := in\n\tout.Inherited.RunID = \"\""},
		"child-clock-floored":    {"math.Round(float64(startSample) * 1000.0 / float64(in.SampleRate))", "math.Floor(float64(startSample) * 1000.0 / float64(in.SampleRate))"},
	} {
		root := filepath.Join(dir, name)
		dst := filepath.Join(root, "internal", "subread")
		if err := os.MkdirAll(dst, 0700); err != nil {
			t.Fatal(err)
		}
		for _, file := range []string{"subread.go", "subread_test.go"} {
			data, err := os.ReadFile(filepath.Join("..", "..", "internal", "subread", file))
			if err != nil {
				t.Fatal(err)
			}
			if file == "subread.go" {
				if strings.Count(string(data), mutation[0]) != 1 {
					t.Fatal("mutation anchor differs")
				}
				data = []byte(strings.Replace(string(data), mutation[0], mutation[1], 1))
			}
			if err = os.WriteFile(filepath.Join(dst, file), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module github.com/Jakeelamb/nanopore_basecaller_go\n\ngo 1.24.0\n"), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-count=1", "./internal/subread", "-run", "^TestMaterializedMetadata$")
		cmd.Dir = root
		output, err := Command(root, "mutation", cmd, nil)
		if err == nil || !bytes.Contains(output, []byte("--- FAIL: TestMaterializedMetadata")) {
			t.Fatalf("mutation %s was not rejected by behavior test: %v\n%s", name, err, output)
		}
	}
}
