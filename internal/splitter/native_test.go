package splitter_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/splitter"
)

func TestPinnedNativeSimplexPA(t *testing.T) {
	c := oracle.Config{Source: os.Getenv("DORADO_SOURCE"), Torch: os.Getenv("SPLITTER_TORCH"), Edlib: os.Getenv("SPLITTER_EDLIB_SOURCE")}
	if c.Source == "" {
		t.Skip("set DORADO_SOURCE, SPLITTER_TORCH and SPLITTER_EDLIB_SOURCE for native parity; ordinary tests do not qualify it")
	}
	if c.Torch == "" || c.Edlib == "" {
		t.Fatal("native oracle requires explicit CPU LibTorch and original edlib source paths")
	}
	root := os.Getenv("SPLITTER_ARTIFACT_ROOT")
	if root == "" {
		root = "../../artifacts/splitter"
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
	t.Logf("native splitter evidence %s", attempt)
	executable, err := oracle.Build(c, filepath.Join(attempt, "native-build"))
	if err != nil {
		t.Fatal(err)
	}
	cases := oracle.Cases()
	inputs := make([]splitter.Input, len(cases))
	names := make([]string, len(cases))
	for i, c := range cases {
		inputs[i], names[i] = c.Input, c.Name
	}
	if err := writeNativeJSON(filepath.Join(attempt, "case-names.json"), names); err != nil {
		t.Fatal(err)
	}
	if err := oracle.Compare(executable, filepath.Join(attempt, "synthetic"), inputs); err != nil {
		t.Fatal(err)
	}
	if err := oracle.CheckHalfTable(executable, filepath.Join(attempt, "half-table")); err != nil {
		t.Fatal(err)
	}
	for _, group := range []struct {
		name, root string
		pins       map[string]string
	}{{"dorado", c.Source, oracle.SourcePins}, {"edlib", c.Edlib, oracle.EdlibPins}} {
		for altered := range group.pins {
			dir := filepath.Join(attempt, "source-rejections", group.name, filepath.Base(altered))
			for path := range group.pins {
				data, err := os.ReadFile(filepath.Join(group.root, path))
				if err != nil {
					t.Fatal(err)
				}
				if path == altered {
					data = append(data, '\n')
				}
				dst := filepath.Join(dir, path)
				if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(dst, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			changed := c
			if group.name == "dorado" {
				changed.Source = dir
			} else {
				changed.Edlib = dir
			}
			if _, err := oracle.Build(changed, filepath.Join(dir, "build")); err == nil || !strings.Contains(err.Error(), "source hash mismatch") {
				t.Fatalf("altered source admitted: %s: %v", altered, err)
			}
		}
	}
	encoded, err := oracle.EncodeInput(inputs[:1])
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string][]byte{"truncated": encoded[:len(encoded)-1], "count-overflow": {1, 0, 0, 4}, "trailing-input": append(append([]byte(nil), encoded...), 0)} {
		dir := filepath.Join(attempt, "protocol-rejections", name)
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "input.bin"), input, 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		_, runErr := oracle.Command(dir, "native", exec.CommandContext(ctx, executable), input)
		cancel()
		if runErr == nil {
			t.Fatalf("invalid protocol %s admitted", name)
		}
		data, err := os.ReadFile(filepath.Join(dir, "native.receipt.json"))
		if err != nil {
			t.Fatal(err)
		}
		var receipt struct {
			Exit int `json:"exit_code"`
		}
		if json.Unmarshal(data, &receipt) != nil || receipt.Exit != 2 {
			t.Fatalf("invalid protocol %s did not exit 2", name)
		}
	}
	if err := oracle.VerifySources(c); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d synthetic cases exact across source clusters, move sums, raw/top candidates, local Q bits, adapter intervals and spacers; 65536 half orderings exact; %d altered source files and three malformed protocols rejected", len(inputs), len(oracle.SourcePins)+len(oracle.EdlibPins))
	paths, receiptPath := os.Getenv("SPLITTER_CAPTURE_MANIFESTS"), os.Getenv("SPLITTER_MOVES_RECEIPT")
	if paths == "" || receiptPath == "" {
		t.Run("genuine-captures", func(t *testing.T) {
			t.Skip("set SPLITTER_CAPTURE_MANIFESTS and SPLITTER_MOVES_RECEIPT for genuine retained no-split evidence")
		})
		return
	}
	var retained []splitter.Input
	var pins []map[string]any
	for _, path := range filepath.SplitList(paths) {
		in, pin := loadSplitCapture(t, path, receiptPath)
		retained = append(retained, in)
		pins = append(pins, pin)
	}
	dir := filepath.Join(attempt, "retained-captures")
	if err := oracle.Compare(executable, dir, retained); err != nil {
		t.Fatal(err)
	}
	for _, in := range retained {
		out, err := splitter.FindSimplexPA(in)
		if err != nil {
			t.Fatal(err)
		}
		if len(out.Spacers) != 0 {
			t.Fatal("retained negative control split")
		}
	}
	if err := writeNativeJSON(filepath.Join(dir, "capture-pins.json"), pins); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d genuine retained normalized captures and source-verified moves exact; no selected split spacers", len(retained))
}

func writeNativeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}
func digestCapture(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

type splitPayload struct {
	File, SHA256 string
	Bytes        int64
}

func loadSplitCapture(t *testing.T, path, receiptPath string) (splitter.Input, map[string]any) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	called, err := caller.Replay(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	manifestData, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	pins := map[string]any{"manifest": path, "manifest_sha256": digestCapture(manifestData)}
	var manifest struct{ Input, Output, Run splitPayload }
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatal(err)
	}
	load := func(p splitPayload) []byte {
		t.Helper()
		if p.File == "" || filepath.Base(p.File) != p.File || p.Bytes < 0 || p.Bytes > 1<<28 {
			t.Fatal("invalid capture payload reference")
		}
		file := filepath.Join(filepath.Dir(path), p.File)
		info, err := os.Lstat(file)
		if err != nil {
			t.Fatal(err)
		}
		if !info.Mode().IsRegular() || info.Size() != p.Bytes {
			t.Fatal("capture payload size/type differs")
		}
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if int64(len(data)) != p.Bytes || digestCapture(data) != p.SHA256 {
			t.Fatal("capture payload hash differs")
		}
		pins[p.File] = p
		return data
	}
	var input struct {
		Normalized        splitPayload
		NormalizedSamples uint64
		Metadata          struct{ EndReasonName string }
	}
	if err := json.Unmarshal(load(manifest.Input), &input); err != nil {
		t.Fatal(err)
	}
	var run struct {
		Spec struct {
			Stride uint64
			Name   string
		}
	}
	if err := json.Unmarshal(load(manifest.Run), &run); err != nil {
		t.Fatal(err)
	}
	load(manifest.Output)
	signalBytes := load(input.Normalized)
	if len(signalBytes)%2 != 0 || uint64(len(signalBytes)/2) != input.NormalizedSamples {
		t.Fatal("capture normalized count differs")
	}
	signal := make([]uint16, len(signalBytes)/2)
	for i := range signal {
		signal[i] = binary.LittleEndian.Uint16(signalBytes[i*2:])
	}
	receiptData, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		Passed   bool
		Captures []struct {
			ManifestSHA256 string
			Stitched       struct {
				SequenceSHA256, QualitySHA256, MovesSHA256 string
				Bases, MoveBytes                           int
			}
		}
	}
	if err := json.Unmarshal(receiptData, &receipt); err != nil {
		t.Fatal(err)
	}
	if !receipt.Passed {
		t.Fatal("move-source receipt did not pass")
	}
	matched := false
	for _, c := range receipt.Captures {
		if c.ManifestSHA256 == digestCapture(manifestData) && c.Stitched.MovesSHA256 == digestCapture(called.Moves) && c.Stitched.SequenceSHA256 == digestCapture([]byte(called.Sequence)) && c.Stitched.QualitySHA256 == digestCapture([]byte(called.Qstring)) && c.Stitched.Bases == len(called.Sequence) && c.Stitched.MoveBytes == len(called.Moves) {
			matched = true
			pins["native_move_endpoint"] = c.Stitched
		}
	}
	if !matched {
		t.Fatal("genuine capture endpoint has no exact source-verified move receipt")
	}
	pins["moves_receipt"] = receiptPath
	pins["moves_receipt_sha256"] = digestCapture(receiptData)
	pins["model"] = run.Spec.Name
	pins["end_reason"] = input.Metadata.EndReasonName
	return splitter.Input{Signal: signal, Moves: called.Moves, Sequence: called.Sequence, Qstring: called.Qstring, Stride: run.Spec.Stride}, pins
}
