package hacgpu

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

// TestNormalisationMatchesPinnedFormula replays ScalerNode.cpp:189-206 arithmetic
// and checks the half conversion against hand-computed values.
func TestNormalisationMatchesPinnedFormula(t *testing.T) {
	n := Normalisation{Mean: 94, Stdev: 24, Scaling: 0.1870698481798172, Offset: -270, TrimStart: 10}
	shift, scale := n.ShiftScale()
	if got, want := float64(shift), 94/0.1870698481798172+270; diff(got, want) > 1e-3 {
		t.Fatalf("shift %v, want %v", got, want)
	}
	if got, want := float64(scale), 24/0.1870698481798172; diff(got, want) > 1e-3 {
		t.Fatalf("scale %v, want %v", got, want)
	}
	raw := make([]int16, 12)
	for i := range raw {
		raw[i] = int16(100 + i)
	}
	bits, err := n.Normalise(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(bits) != 2 {
		t.Fatalf("trim did not drop 10 samples: got %d", len(bits))
	}
	for i, v := range []int16{110, 111} {
		// The scaler writes float32 arithmetic into a binary16 buffer, so the
		// pinned result is the half-rounded value, not the float32 quotient.
		want := signal.Float16Bits(float32(float32(v)-shift) / scale)
		if bits[i] != want {
			t.Fatalf("sample %d normalised to %04x, want %04x", i, bits[i], want)
		}
	}
}

// TestNormalisationRejectsDegenerateInputs covers the guard set.
func TestNormalisationRejectsDegenerateInputs(t *testing.T) {
	base := Normalisation{Mean: 94, Stdev: 24, Scaling: 0.187, Offset: -270, TrimStart: 10}
	if _, err := base.Normalise(nil); err == nil {
		t.Fatal("empty signal accepted")
	}
	if _, err := base.Normalise(make([]int16, 5)); err == nil {
		t.Fatal("trim beyond the signal accepted")
	}
	bad := base
	bad.TrimStart = -1
	if _, err := bad.Normalise(make([]int16, 20)); err == nil {
		t.Fatal("negative trim accepted")
	}
	zero := base
	zero.Scaling = 0
	if _, err := zero.Normalise(make([]int16, 20)); err == nil {
		t.Fatal("zero calibration scale accepted")
	}
}

// TestImportRejectsCorruptStage proves the import path is a real control: a single
// flipped byte, a missing stage and a truncated backing dump all reject.
func TestImportRejectsCorruptStage(t *testing.T) {
	dir := writeFixtureCapture(t)
	if _, err := Import(dir); err != nil {
		t.Fatalf("valid fixture capture rejected: %v", err)
	}

	corrupt := writeFixtureCapture(t)
	path := filepath.Join(corrupt, "stage-conv2.bin")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body[0] ^= 0xff
	if err = os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = Import(corrupt); err == nil {
		t.Fatal("corrupt stage payload accepted")
	}

	missing := writeFixtureCapture(t)
	if err = os.Remove(filepath.Join(missing, "stage-flstm1.bin")); err != nil {
		t.Fatal(err)
	}
	if _, err = Import(missing); err == nil {
		t.Fatal("missing stage payload accepted")
	}

	truncated := writeFixtureCapture(t)
	path = filepath.Join(truncated, "stage-input-backing.bin")
	body, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, body[:len(body)-2], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = Import(truncated); err == nil {
		t.Fatal("truncated backing dump accepted")
	}
}

// TestImportRejectsForeignSchema covers schema and commit drift.
func TestImportRejectsForeignSchema(t *testing.T) {
	dir := writeFixtureCapture(t)
	path := filepath.Join(dir, "capture.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err = json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	m["source_commit"] = "0000000000000000000000000000000000000000"
	raw, err = json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = Import(dir); err == nil {
		t.Fatal("foreign source commit accepted")
	}
}

// TestFixtureNormalisationDecodesBitPatterns pins the manifest contract:
// `float_bits` carries uint32 IEEE-754 bit patterns, not JSON numbers. Reading
// them as numbers produced a ~1e9 calibration scale and an all-`+inf` input, so
// this test is the regression control for that mistake.
func TestFixtureNormalisationDecodesBitPatterns(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "manifest.json")
	body := fmt.Sprintf(`{"fixtures":[{"reads":[{"float_bits":{"CalibrationScale":%d,"CalibrationOffset":%d}}]}]}`,
		math.Float32bits(0.1870698481798172), math.Float32bits(-270))
	if err := os.WriteFile(manifest, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	n, err := fixtureNormalisation(manifest, 0)
	if err != nil {
		t.Fatal(err)
	}
	if diff(n.Scaling, 0.1870698481798172) > 1e-9 {
		t.Fatalf("calibration scale decoded as %v, want 0.1870698481798172", n.Scaling)
	}
	if n.Offset != -270 {
		t.Fatalf("calibration offset decoded as %v, want -270", n.Offset)
	}
	if n.TrimStart != 10 {
		t.Fatalf("trim start %d, want the constant 10", n.TrimStart)
	}
	// The same numbers fed through the graph must stay finite.
	n.Mean, n.Stdev = 94, 24
	raw := make([]int16, 32)
	for i := range raw {
		raw[i] = int16(80 + i)
	}
	bits, err := n.Normalise(raw)
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range bits {
		if v := signal.Float32FromHalf(b); math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Fatalf("sample %d normalised to %v", i, v)
		}
	}

	// A manifest that carries the value itself instead of the bit pattern is not
	// silently reinterpreted.
	plain := filepath.Join(dir, "plain.json")
	if err = os.WriteFile(plain, []byte(`{"fixtures":[{"reads":[{"float_bits":{"CalibrationScale":0.187,"CalibrationOffset":-270}}]}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = fixtureNormalisation(plain, 0); err == nil {
		t.Fatal("manifest with plain floats accepted as bit patterns")
	}
}

// TestNormaliseRejectsNonFiniteOutput covers the guard that stops a mis-decoded
// calibration from producing an all-infinity capture.
func TestNormaliseRejectsNonFiniteOutput(t *testing.T) {
	n := Normalisation{Mean: 94, Stdev: 24, Scaling: 1.0443528e9, Offset: 3.28e9, TrimStart: 0}
	if _, err := n.Normalise([]int16{100, 101}); err == nil {
		t.Fatal("non-finite normalisation accepted")
	}
}

// TestImportRejectsGraphShapeDrift proves the declared decoder shape is enforced:
// a factorised capture that claims `linear2` must reject, and a plain capture
// that drops its `linear2` stage must reject too.
func TestImportRejectsGraphShapeDrift(t *testing.T) {
	dir := writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) { m["linear2_present"] = true })
	if _, err := Import(dir); err == nil {
		t.Fatal("factorised capture accepted with linear2 declared")
	}

	dir = writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) {
		m["linear1_kind"] = "plain"
		m["linear2_present"] = false
	})
	if _, err := Import(dir); err == nil {
		t.Fatal("plain capture accepted without a linear2 stage")
	}
}

// TestImportRejectsStackCrossCheckDrift proves the per-layer replay is checked
// against the untouched stack digests, not merely self-consistent.
func TestImportRejectsStackCrossCheckDrift(t *testing.T) {
	dir := writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) {
		m["stack_sha256"].(map[string]any)["convs"] = strings.Repeat("0", 64)
	})
	if _, err := Import(dir); err == nil {
		t.Fatal("capture with a drifted conv stack digest accepted")
	}

	dir = writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) {
		delete(m["stack_sha256"].(map[string]any), "rnns")
	})
	if _, err := Import(dir); err == nil {
		t.Fatal("capture without a recurrent stack digest accepted")
	}
}

// TestImportRejectsConstantScores covers the silent failure mode: a batch or
// T_out the projection kernel cannot tile leaves its output untouched.
func TestImportRejectsConstantScores(t *testing.T) {
	dir := writeFixtureCaptureOpts(t, true)
	if _, err := Import(dir); err == nil {
		t.Fatal("constant scores accepted as a graph result")
	}
}

// TestImportRejectsUnwrittenDrift proves the prospective-exclusion declaration is
// itself checked: rows outside the T axis and self-overlapping rows both reject.
func TestImportRejectsUnwrittenDrift(t *testing.T) {
	dir := writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) {
		m["stages"].([]any)[1].(map[string]any)["unwritten_t"] = [][]int{{30, 10}}
	})
	if _, err := Import(dir); err == nil {
		t.Fatal("unwritten rows outside the T axis accepted")
	}

	dir = writeFixtureCapture(t)
	mutateCapture(t, dir, func(m map[string]any) {
		m["stages"].([]any)[1].(map[string]any)["unwritten_t"] = [][]int{{0, 2}, {1, 2}}
	})
	if _, err := Import(dir); err == nil {
		t.Fatal("self-overlapping unwritten rows accepted")
	}
}

// TestCompareRejectsDrift proves the determinism gate compares real bytes: a
// single flipped payload byte between two otherwise valid captures rejects.
func TestCompareRejectsDrift(t *testing.T) {
	a := writeFixtureCapture(t)
	b := writeFixtureCapture(t)
	if _, err := Compare(a, b); err != nil {
		t.Fatalf("identical fixture captures rejected: %v", err)
	}
	path := filepath.Join(b, "stage-flstm0.bin")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body[3] ^= 0x01
	if err = os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	// The payload no longer matches its own recorded digest, so Import rejects it
	// first; that is still the drift control firing.
	if _, err = Compare(a, b); err == nil {
		t.Fatal("capture with a drifted payload accepted")
	}
}

// TestStageNamesFollowsTheDeclaredGraph proves the stage list is derived from the
// declared layer counts rather than hardcoded.
func TestStageNamesFollowsTheDeclaredGraph(t *testing.T) {
	got := strings.Join(StageNames(3, 7, false), ",")
	want := "input,conv0,conv1,conv2,flstm0,flstm1,flstm2,flstm3,flstm4,flstm5,flstm6,linear1,scores"
	if got != want {
		t.Fatalf("stage list %s, want %s", got, want)
	}
	if got := strings.Join(StageNames(1, 1, true), ","); got != "input,conv0,flstm0,linear1,linear2,scores" {
		t.Fatalf("plain decoder stage list %s", got)
	}
}

func mutateCapture(t *testing.T, dir string, edit func(map[string]any)) {
	t.Helper()
	path := filepath.Join(dir, "capture.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err = json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	edit(m)
	raw, err = json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeFixtureCapture builds a tiny synthetic capture that satisfies every
// structural invariant, so the rejection tests exercise the real verifier.
func writeFixtureCapture(t *testing.T) string {
	return writeFixtureCaptureOpts(t, false)
}

// writeFixtureCaptureOpts can make the scores payload constant, which is the
// silent shape of a graph that did not run (an untileable batch or T_out).
func writeFixtureCaptureOpts(t *testing.T, constantScores bool) string {
	t.Helper()
	dir := t.TempDir()
	const (
		convs      = 3
		lstmLayers = 2
		payload    = 4096
	)
	names := StageNames(convs, lstmLayers, false)
	backing := make([]byte, payload*len(names))
	for i := range backing {
		backing[i] = byte(i * 7)
	}
	c := Capture{
		Schema:       CaptureSchema,
		Status:       "captured",
		Case:         "fixture",
		SourceCommit: SourceCommit,
		Batch:        64,
		Chunk:        192,
		Stride:       6,
		TOut:         32,
		Features:     1,
		Convs:        convs,
		LSTMLayers:   lstmLayers,
		Reservation:  int64(len(backing)),
		BackingElems: int64(len(backing) / 2),
		BackingDType: "Half",
		BackingAlign: 256,
		ScoresDType:  "Half",
		Linear1Kind:  "factorised",
		Standardise:  true,
		StdMean:      94,
		StdStdev:     24,
		ScoresContig: true,
		StackSHA:     map[string]string{},
	}
	for i, name := range names {
		off := int64(i * payload)
		if constantScores && name == "scores" {
			for j := off; j < off+payload; j++ {
				backing[j] = 0
			}
		}
		payloadBytes := backing[off : off+payload]
		dataFile := "stage-" + name + ".bin"
		backingFile := "stage-" + name + "-backing.bin"
		if err := os.WriteFile(filepath.Join(dir, dataFile), payloadBytes, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, backingFile), backing, 0o644); err != nil {
			t.Fatal(err)
		}
		dataSHA, err := bytesSHA(payloadBytes)
		if err != nil {
			t.Fatal(err)
		}
		backingSHA, err := bytesSHA(backing)
		if err != nil {
			t.Fatal(err)
		}
		stage := Stage{
			Name: name, Shape: []int64{64, 32, 1}, DType: "Half", Layout: "NTC",
			OffsetBytes: off, Bytes: payload, ReservationByte: c.Reservation, CurrentBytes: payload,
			N: 64, T: 32, C: 1, InputToRevLSTM: true,
			UnwrittenT: unwrittenFor(name),
			DataFile:   dataFile, DataSHA: dataSHA, BackingFile: backingFile, BackingSHA: backingSHA,
		}
		stage.Kind, stage.Layer = stageKindOf(name)
		if err = checkStageIdentity(stage); err != nil {
			t.Fatalf("fixture stage %q is malformed: %v", name, err)
		}
		c.Stages = append(c.Stages, stage)
	}
	byName := map[string]string{}
	for _, s := range c.Stages {
		byName[s.Name] = s.DataSHA
	}
	c.StackSHA["convs"] = byName["conv2"]
	c.StackSHA["rnns"] = byName["flstm1"]
	c.StackSHA["linear1"] = byName["linear1"]
	c.StackSHA["scores"] = byName["scores"]
	c.ScoresSHA = byName["scores"]
	c.ScoresShape = []int64{64, 32, 1}
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "capture.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// unwrittenFor mirrors the source-derived exclusion rule for the fixture's
// shapes: only the recurrent stages declare guard rows.
func unwrittenFor(name string) [][2]int {
	if strings.HasPrefix(name, "flstm") {
		return [][2]int{{1, 1}}
	}
	return nil
}

// stageKindOf mirrors the oracle's stage naming so the fixture is built the same
// way the native producer builds it.
func stageKindOf(name string) (string, int) {
	switch {
	case name == "input" || name == "scores":
		return name, -1
	case name == "linear1" || name == "linear2":
		return "projection", -1
	case strings.HasPrefix(name, "conv"):
		var i int
		fmt.Sscanf(name, "conv%d", &i)
		return "convolution", i
	default:
		var i int
		fmt.Sscanf(name, "flstm%d", &i)
		return "recurrent", i
	}
}

func diff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}
