//go:build cgo

package decode

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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

func writeFixture(w *bytes.Buffer, c fixture) {
	fmt.Fprintf(w, "%d %d %d %d %d %d %d\n", c.in.Blocks, c.in.States, c.opt.BeamWidth, math.Float32bits(c.opt.BeamCut), math.Float32bits(c.opt.BlankScore), math.Float32bits(c.opt.QShift), math.Float32bits(c.opt.QScale))
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
	var input, goOutput bytes.Buffer
	results := make([]Result, len(cases))
	names := make([]string, len(cases))
	for i, c := range cases {
		writeFixture(&input, c)
		results[i], err = Decode(c.in, c.opt)
		if err != nil {
			t.Fatalf("case %d %s: %v", i, c.name, err)
		}
		writeResult(&goOutput, results[i])
		names[i] = fmt.Sprintf("%d %s blocks=%d states=%d beam=%d cut=%g qshift=%g qscale=%g", i, c.name, c.in.Blocks, c.in.States, c.opt.BeamWidth, c.opt.BeamCut, c.opt.QShift, c.opt.QScale)
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
	receipt := map[string]any{"started": started, "finished": time.Now().UTC(), "exit_code": exitCode, "cases": len(cases), "input_sha256": fmt.Sprintf("%x", sha256.Sum256(input.Bytes())), "native_output_sha256": fmt.Sprintf("%x", sha256.Sum256(nativeOutput)), "go_output_sha256": fmt.Sprintf("%x", sha256.Sum256(goOutput.Bytes())), "source_commit": decodeoracle.SourceCommit, "qualification": "extracted float CPU beam_search and generate_sequence, not Torch/GPU or biological evidence"}
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
	differingFloats := 0
	maxScoreError, maxQualError := 0.0, 0.0
	for i, got := range results {
		want, err := readResult(reader)
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
	if _, err = fmt.Fscan(reader, &extra); err == nil {
		t.Fatalf("unexpected native output %q", extra)
	}
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
