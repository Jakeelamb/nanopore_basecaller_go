//go:build cgo

package decode

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	decodeoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/decode"
)

const oracleV2 = "nanopore-decode-oracle/v2"

func resolvedStride(in Input) int {
	if in.ScoreStride == 0 {
		return 4 * in.States
	}
	return in.ScoreStride
}

func writeFixture(w *bytes.Buffer, c fixture, v2 bool) {
	fmt.Fprintf(w, "%d %d %d ", c.in.Blocks, c.in.States, c.opt.BeamWidth)
	if v2 {
		fmt.Fprintf(w, "%d %d ", resolvedStride(c.in), len(c.in.Scores))
	}
	fmt.Fprintf(w, "%d %d %d %d\n", math.Float32bits(c.opt.BeamCut), math.Float32bits(c.opt.BlankScore), math.Float32bits(c.opt.QShift), math.Float32bits(c.opt.QScale))
	for _, tensor := range [][]float32{c.in.Scores, c.in.BackGuides, c.in.Posts} {
		for _, v := range tensor {
			fmt.Fprintf(w, "%d ", math.Float32bits(v))
		}
		w.WriteByte('\n')
	}
}

func writeResult(w *bytes.Buffer, r Result) {
	fmt.Fprintf(w, "%x %x %d", r.Sequence, r.Qstring, len(r.Moves))
	for _, v := range r.Moves {
		fmt.Fprintf(w, " %d", v)
	}
	for _, v := range r.States {
		fmt.Fprintf(w, " %d", v)
	}
	fmt.Fprintf(w, " %d", math.Float32bits(r.FinalScore))
	for _, v := range r.QualData {
		fmt.Fprintf(w, " %d", math.Float32bits(v))
	}
	w.WriteByte('\n')
}

func readResult(reader *bufio.Reader) (Result, error) {
	var seq, q string
	var n int
	if _, err := fmt.Fscan(reader, &seq, &q, &n); err != nil {
		return Result{}, err
	}
	if n < 1 || n > 100000 {
		return Result{}, fmt.Errorf("invalid native blocks %d", n)
	}
	sequence, err := hex.DecodeString(seq)
	if err != nil {
		return Result{}, err
	}
	quality, err := hex.DecodeString(q)
	if err != nil {
		return Result{}, err
	}
	r := Result{Sequence: string(sequence), Qstring: string(quality), Moves: make([]uint8, n), States: make([]int32, n), QualData: make([]float32, 4*n)}
	for i := range r.Moves {
		if _, err = fmt.Fscan(reader, &r.Moves[i]); err != nil {
			return Result{}, err
		}
	}
	for i := range r.States {
		if _, err = fmt.Fscan(reader, &r.States[i]); err != nil {
			return Result{}, err
		}
	}
	var bits uint32
	if _, err = fmt.Fscan(reader, &bits); err != nil {
		return Result{}, err
	}
	r.FinalScore = math.Float32frombits(bits)
	for i := range r.QualData {
		if _, err = fmt.Fscan(reader, &bits); err != nil {
			return Result{}, err
		}
		r.QualData[i] = math.Float32frombits(bits)
	}
	return r, nil
}

func TestPinnedNativeDecode(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE for pinned extracted CPU beam-search differential; does not test Torch or GPU")
	}
	output := os.Getenv("DECODE_ARTIFACT_DIR")
	if output == "" {
		output = t.TempDir()
	}
	if err := os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	executable, err := decodeoracle.Build(root, output)
	if err != nil {
		t.Fatal(err)
	}
	cases := differentialFixtures()
	cases = append(cases, qualityThresholdFixtures(cases[130])...)
	if len(cases) != 1465 {
		t.Fatalf("historical fixture count changed: %d", len(cases))
	}
	pairs := strideFixtures()
	for _, pair := range pairs {
		cases = append(cases, pair.packed, pair.view)
	}
	witness := stridedFixture(constantFixture("stride-control-baseline", 2, 4, 1, 0, -4), 32, 0)
	wrong := witness
	wrong.name = "wrong-contiguous-addressing-mutant"
	wrong.in.ScoreStride = 0
	wrong.in.Scores = append([]float32(nil), witness.in.Scores[:32]...)
	mutated := witness
	mutated.name = "logical-score-mutation"
	mutated.in.Scores = append([]float32(nil), witness.in.Scores...)
	mutated.in.Scores[32] = 12
	witnessIndex := len(cases)
	cases = append(cases, witness, wrong, mutated)
	var input, goOutput bytes.Buffer
	fmt.Fprintln(&input, oracleV2)
	fmt.Fprintln(&goOutput, oracleV2)
	results := make([]Result, len(cases))
	names := make([]string, len(cases))
	for i, c := range cases {
		writeFixture(&input, c, true)
		results[i], err = Decode(c.in, c.opt)
		if err != nil {
			t.Fatalf("case %d %s: %v", i, c.name, err)
		}
		fmt.Fprintf(&goOutput, "%d %d ", resolvedStride(c.in), len(c.in.Scores))
		writeResult(&goOutput, results[i])
		names[i] = fmt.Sprintf("%d %s blocks=%d states=%d beam=%d stride=%d span=%d cut=%g qshift=%g qscale=%g", i, c.name, c.in.Blocks, c.in.States, c.opt.BeamWidth, resolvedStride(c.in), len(c.in.Scores), c.opt.BeamCut, c.opt.QShift, c.opt.QScale)
	}
	for name, data := range map[string][]byte{"inputs.txt": input.Bytes(), "go-output.txt": goOutput.Bytes(), "cases.txt": []byte(strings.Join(names, "\n") + "\n")} {
		if err = os.WriteFile(filepath.Join(output, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	command := exec.Command(executable)
	command.Stdin = bytes.NewReader(input.Bytes())
	var stderr bytes.Buffer
	command.Stderr = &stderr
	started := time.Now().UTC()
	nativeOutput, runErr := command.Output()
	exitCode := -1
	if command.ProcessState != nil {
		exitCode = command.ProcessState.ExitCode()
	}
	receipt := map[string]any{"protocol": oracleV2, "historical_contiguous_cases": 1465, "stride_pairs": len(pairs), "started": started, "finished": time.Now().UTC(), "exit_code": exitCode, "cases": len(cases), "input_sha256": fmt.Sprintf("%x", sha256.Sum256(input.Bytes())), "native_output_sha256": fmt.Sprintf("%x", sha256.Sum256(nativeOutput)), "go_output_sha256": fmt.Sprintf("%x", sha256.Sum256(goOutput.Bytes())), "source_commit": decodeoracle.SourceCommit, "qualification": "extracted float CPU beam_search and generate_sequence, not Torch/GPU or biological evidence"}
	for name, data := range map[string][]byte{"native-output.txt": nativeOutput, "native-stderr.txt": stderr.Bytes()} {
		if err = os.WriteFile(filepath.Join(output, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	defer func() {
		data, e := json.MarshalIndent(receipt, "", "  ")
		if e != nil {
			t.Error(e)
			return
		}
		if e = os.WriteFile(filepath.Join(output, "run-receipt.json"), append(data, '\n'), 0600); e != nil {
			t.Error(e)
		}
	}()
	if runErr != nil {
		t.Fatalf("native decoder: %v: %s", runErr, stderr.Bytes())
	}
	reader := bufio.NewReader(bytes.NewReader(nativeOutput))
	var marker string
	if _, err = fmt.Fscan(reader, &marker); err != nil || marker != oracleV2 {
		t.Fatalf("native v2 marker %q: %v", marker, err)
	}
	nativeResults := make([]Result, len(cases))
	differingFloats := 0
	maxScoreError, maxQualError := 0.0, 0.0
	for i, got := range results {
		var stride, span int
		if _, err := fmt.Fscan(reader, &stride, &span); err != nil || stride != resolvedStride(cases[i].in) || span != len(cases[i].in.Scores) {
			t.Fatalf("case %d native stride/span %d/%d: %v", i, stride, span, err)
		}
		want, err := readResult(reader)
		nativeResults[i] = want
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if got.Sequence != want.Sequence || got.Qstring != want.Qstring || !reflect.DeepEqual(got.Moves, want.Moves) || !reflect.DeepEqual(got.States, want.States) {
			receipt["failed_case"] = names[i]
			t.Fatalf("case %s discrete mismatch\nGo sequence=%q quality=%q moves=%v states=%v\nnative sequence=%q quality=%q moves=%v states=%v", names[i], got.Sequence, got.Qstring, got.Moves, got.States, want.Sequence, want.Qstring, want.Moves, want.States)
		}
		scoreError := math.Abs(float64(got.FinalScore) - float64(want.FinalScore))
		maxScoreError = math.Max(maxScoreError, scoreError)
		if math.Float32bits(got.FinalScore) != math.Float32bits(want.FinalScore) {
			differingFloats++
			receipt["failed_case"] = names[i]
			t.Fatalf("case %s score bits=%08x want %08x", names[i], math.Float32bits(got.FinalScore), math.Float32bits(want.FinalScore))
		}
		for j, p := range got.QualData {
			error := math.Abs(float64(p) - float64(want.QualData[j]))
			maxQualError = math.Max(maxQualError, error)
			if math.Float32bits(p) != math.Float32bits(want.QualData[j]) {
				differingFloats++
				receipt["failed_case"] = names[i]
				t.Fatalf("case %s qual[%d] bits=%08x want %08x", names[i], j, math.Float32bits(p), math.Float32bits(want.QualData[j]))
			}
		}
	}
	var extra string
	if _, err = fmt.Fscan(reader, &extra); err != io.EOF {
		t.Fatalf("unexpected native output %q: %v", extra, err)
	}
	for i := range pairs {
		index := 1465 + 2*i
		if !resultsExact(nativeResults[index], nativeResults[index+1]) {
			t.Fatalf("packed/strided native pair %d differs", i)
		}
	}
	observations := map[string]string{}
	for offset, name := range []string{"baseline", "wrong_contiguous", "logical_mutation"} {
		var line bytes.Buffer
		writeResult(&line, nativeResults[witnessIndex+offset])
		observations[name] = line.String()
	}
	receipt["native_control_observations"] = observations
	if resultsExact(nativeResults[witnessIndex], nativeResults[witnessIndex+1]) {
		t.Fatal("wrong contiguous addressing did not change native result")
	}
	if resultsExact(nativeResults[witnessIndex], nativeResults[witnessIndex+2]) {
		t.Fatal("logical score mutation did not change native result")
	}
	receipt["wrong_contiguous_addressing_detected"] = true
	receipt["logical_score_mutation_detected"] = true
	bridge := constantFixture("legacy-v2-bridge", 2, 4, 7, 0, -4)
	for i := range bridge.in.Scores {
		bridge.in.Scores[i] = float32(i%7 - 3)
	}
	testOracleFraming(t, executable, output, bridge)
	if t.Failed() {
		return
	}
	receipt["framing_controls_passed"] = true
	receipt["discrete_exact"] = true
	receipt["differing_float_values"] = differingFloats
	receipt["max_final_score_absolute_error"] = maxScoreError
	receipt["max_qual_data_absolute_error"] = maxQualError
	receipt["score_comparison"] = "exact float32 bits"
	receipt["qual_data_comparison"] = "exact float32 bits"
	receipt["float_exact"] = true
	t.Logf("%d cases exact sequence/quality/moves/states; float values differing=%d max score absolute error=%g max qual absolute error=%g; oracle artifacts %s", len(cases), differingFloats, maxScoreError, maxQualError, output)
}

func TestNativeSourceMutationRejected(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE for pinned-source mutation rejection")
	}
	if err := decodeoracle.VerifySources(root); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(root, decodeoracle.SourcePath))
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 1
	copyRoot := t.TempDir()
	path := filepath.Join(copyRoot, decodeoracle.SourcePath)
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = decodeoracle.Build(copyRoot, t.TempDir()); err == nil {
		t.Fatal("modified source reached native compilation")
	}
}

// All oracle execution stays under TestPinnedNativeDecode. Each protocol attempt
// retains input, output, stderr, clocks and exit status before any assertion.
func runOracleControl(t *testing.T, executable, output, name, input string) ([]byte, string, int) {
	t.Helper()
	dir := filepath.Join(output, "controls", name)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	command := exec.Command(executable)
	command.Stdin = strings.NewReader(input)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	started := time.Now().UTC()
	stdout, err := command.Output()
	finished := time.Now().UTC()
	code := -1
	if command.ProcessState != nil {
		code = command.ProcessState.ExitCode()
	}
	receipt := map[string]any{
		"protocol": oracleV2, "control": name, "started": started, "finished": finished, "exit_code": code,
		"v2_input_marker": strings.HasPrefix(input, oracleV2+"\n"),
		"input_sha256":    fmt.Sprintf("%x", sha256.Sum256([]byte(input))),
		"stdout_sha256":   fmt.Sprintf("%x", sha256.Sum256(stdout)),
		"stderr_sha256":   fmt.Sprintf("%x", sha256.Sum256(stderr.Bytes())),
	}
	if err != nil {
		receipt["error"] = err.Error()
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	for file, body := range map[string][]byte{"input.txt": []byte(input), "output.txt": stdout, "stderr.txt": stderr.Bytes(), "receipt.json": append(data, '\n')} {
		if err := os.WriteFile(filepath.Join(dir, file), body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return stdout, stderr.String(), code
}

func testOracleFraming(t *testing.T, executable, output string, c fixture) {
	t.Helper()
	encode := func(c fixture, v2 bool) string {
		var data bytes.Buffer
		if v2 {
			fmt.Fprintln(&data, oracleV2)
		}
		writeFixture(&data, c, v2)
		return data.String()
	}
	legacy, v2 := encode(c, false), encode(c, true)
	legacyOutput, stderr, code := runOracleControl(t, executable, output, "legacy-bridge", legacy)
	if code != 0 {
		t.Fatalf("legacy bridge exit=%d: %s", code, stderr)
	}
	v2Output, stderr, code := runOracleControl(t, executable, output, "v2-bridge", v2)
	if code != 0 {
		t.Fatalf("v2 bridge exit=%d: %s", code, stderr)
	}
	prefix := fmt.Sprintf("%s\n%d %d ", oracleV2, resolvedStride(c.in), len(c.in.Scores))
	if !bytes.Equal(v2Output, append([]byte(prefix), legacyOutput...)) {
		t.Fatal("legacy/v2 bridge output differs")
	}
	result, err := Decode(c.in, c.opt)
	if err != nil {
		t.Fatal(err)
	}
	var expected bytes.Buffer
	writeResult(&expected, result)
	if !bytes.Equal(legacyOutput, expected.Bytes()) {
		t.Fatal("legacy bridge differs from packed Go result")
	}
	// Byte equality includes all score and quality bits and the legacy formatting.
	tokens := strings.Fields(legacy)
	rejections := map[string]string{
		"unknown-version":         "nanopore-decode-oracle/v3\n",
		"unknown-token":           "junk\n",
		"marker-only":             oracleV2 + "\n",
		"partial-header":          oracleV2 + "\n2 4 1 32\n",
		"partial-legacy-header":   "2 4\n",
		"mixed-legacy-v2":         legacy + v2,
		"mixed-v2-legacy":         v2 + legacy,
		"repeated-marker":         v2 + v2,
		"trailing-token":          v2 + "junk\n",
		"truncated-v2":            strings.Join(strings.Fields(v2)[:len(strings.Fields(v2))-1], " "),
		"legacy-truncated-floats": strings.Join(tokens[:len(tokens)-1], " "),
	}
	for name, change := range map[string]func(*fixture){
		"short-span":      func(c *fixture) { c.in.Scores = c.in.Scores[:len(c.in.Scores)-1] },
		"long-span":       func(c *fixture) { c.in.Scores = append(c.in.Scores, 0) },
		"negative-stride": func(c *fixture) { c.in.ScoreStride = -1 },
		"overlap-stride":  func(c *fixture) { c.in.ScoreStride = 1 },
		"stride-over-cap": func(c *fixture) { c.in.ScoreStride = maxElements + 1 },
		"stride-max-int":  func(c *fixture) { c.in.ScoreStride = int(^uint(0) >> 1) },
		"span-over-cap":   func(c *fixture) { c.in.Blocks = 3; c.in.ScoreStride = maxElements / 2 },
		"blocks-max-int":  func(c *fixture) { c.in.Blocks = int(^uint(0) >> 1) },
		"logical-nan":     func(c *fixture) { c.in.Scores[0] = math.Float32frombits(0x7fc12345) },
		"logical-inf":     func(c *fixture) { c.in.Scores[0] = float32(math.Inf(1)) },
	} {
		bad := c
		bad.in.Scores = append([]float32(nil), c.in.Scores...)
		change(&bad)
		rejections[name] = encode(bad, true)
	}
	for name, input := range rejections {
		t.Run("framing/"+name, func(t *testing.T) {
			_, stderr, code := runOracleControl(t, executable, output, name, input)
			if code != 2 {
				t.Fatalf("invalid framing exit=%d stderr=%q", code, stderr)
			}
			if name == "legacy-truncated-floats" && stderr != "truncated float bits\n" {
				t.Fatalf("legacy truncation diagnostic changed: %q", stderr)
			}
		})
	}
}
