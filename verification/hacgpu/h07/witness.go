// Package h07 verifies the fixed factorised HAC output unit without replaying
// convolution or recurrence. Native execution needs separate admission.
package h07

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const H01ReceiptSHA = "405aecc8ff881e488f13555edc2e47ab167fc4e69273b6dfe7e174f11079bfc5"
const H01CaptureSHA = "61dd1db61a7d61234cbc6a8afb852a6cfbde1c5a53ad56a501e2027c1f897149"
const DefaultCapture = "/home/jake/.local/share/opencode/worktree/e3db79/H01/artifacts/tensor/hacgpu-capture-008/stages"
const DownBytes = 524288
const ScoresBytes = 4194304

type Witness struct {
	Capture                                 hacgpu.Capture
	Input, Output                           hacgpu.Stage
	InputParent, OutputParent, Down, Scores []byte
	Pins                                    map[string]string
}

func sum(body []byte) string { h := sha256.Sum256(body); return hex.EncodeToString(h[:]) }
func requirePin(path, want string) error {
	if len(want) != 64 {
		return fmt.Errorf("missing SHA256 for %s", path)
	}
	h, err := runner.SHA(path)
	if err != nil {
		return err
	}
	if h != want {
		return fmt.Errorf("retained SHA256 mismatch: %s", path)
	}
	return nil
}
func readPinned(path, want string) ([]byte, error) {
	if len(want) != 64 {
		return nil, fmt.Errorf("missing SHA256 for %s", path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if sum(b) != want {
		return nil, fmt.Errorf("retained SHA256 mismatch: %s", path)
	}
	return b, nil
}
func ReadWitness(dir string) (w Witness, err error) {
	w.Pins = map[string]string{}
	receiptPath := filepath.Join(filepath.Dir(dir), "receipt.json")
	raw, err := readPinned(receiptPath, H01ReceiptSHA)
	if err != nil {
		return w, err
	}
	w.Pins[receiptPath] = H01ReceiptSHA
	var receipt struct {
		Status string            `json:"status"`
		Inputs map[string]string `json:"input_sha256_all"`
	}
	if err = json.Unmarshal(raw, &receipt); err != nil {
		return w, err
	}
	if receipt.Status != "stage_capture_passed" {
		return w, fmt.Errorf("H01 source witness did not pass")
	}
	metaPath := filepath.Join(dir, "capture.json")
	if _, err = readPinned(metaPath, H01CaptureSHA); err != nil {
		return w, err
	}
	w.Pins[metaPath] = H01CaptureSHA
	w.Capture, err = hacgpu.Import(dir)
	if err != nil {
		return w, err
	}
	c := w.Capture
	if c.Batch != 64 || c.TOut != 32 || c.LSTMSize != 1024 || c.LSTMInnerDim != 128 || c.LSTMLayers != 7 || c.Linear2 || c.Linear1Kind != "factorised" || c.Reservation != 7782400 {
		return w, fmt.Errorf("H07 witness requires the admitted fixed N64/T32 factorised capture")
	}
	w.Input, _ = c.Stage("flstm6")
	w.Output, _ = c.Stage("linear1")
	scores, _ := c.Stage("scores")
	if w.Input.DType != "Char" || w.Input.Layout != "CUTLASS_TNC_I8" || w.Input.InputToRevLSTM ||
		!slices.Equal(w.Input.Shape, []int64{37, 64, 1024}) || w.Input.OffsetBytes != 5357568 || w.Input.Bytes != 2424832 ||
		w.Output.DType != "Half" || w.Output.Layout != "NTC" || !slices.Equal(w.Output.Shape, []int64{64, 32, 1024}) ||
		w.Output.OffsetBytes != DownBytes || w.Output.Bytes != ScoresBytes || scores.DataSHA != w.Output.DataSHA || scores.BackingSHA != w.Output.BackingSHA || len(w.Output.UnwrittenT) != 0 {
		return w, fmt.Errorf("source projection view contract differs")
	}
	byBase := map[string]string{}
	for path, pin := range receipt.Inputs {
		base := filepath.Base(path)
		if before, ok := byBase[base]; ok && before != pin {
			return w, fmt.Errorf("conflicting H01 receipt basename")
		}
		byBase[base] = pin
	}
	for _, s := range []hacgpu.Stage{w.Input, w.Output, scores} {
		for name, pin := range map[string]string{s.DataFile: s.DataSHA, s.BackingFile: s.BackingSHA} {
			if byBase[name] != pin {
				return w, fmt.Errorf("H01 independent receipt does not authenticate %s", name)
			}
			path := filepath.Join(dir, name)
			if _, err = readPinned(path, pin); err != nil {
				return w, err
			}
			w.Pins[path] = pin
		}
	}
	w.InputParent, err = readPinned(filepath.Join(dir, w.Input.BackingFile), w.Input.BackingSHA)
	if err != nil {
		return w, err
	}
	w.OutputParent, err = readPinned(filepath.Join(dir, w.Output.BackingFile), w.Output.BackingSHA)
	if err != nil {
		return w, err
	}
	w.Scores, err = readPinned(filepath.Join(dir, w.Output.DataFile), w.Output.DataSHA)
	if err != nil {
		return w, err
	}
	w.Down = slices.Clone(w.OutputParent[:DownBytes])
	if !bytes.Equal(w.OutputParent[DownBytes:DownBytes+ScoresBytes], w.Scores) {
		return w, fmt.Errorf("source scores do not occupy the declared temp slice")
	}
	if !bytes.Equal(w.InputParent[DownBytes+ScoresBytes:], w.OutputParent[DownBytes+ScoresBytes:]) {
		return w, fmt.Errorf("source projection changed its readonly input or gap")
	}
	return w, nil
}

type WitnessReceipt struct {
	Schema, Status, Started, Finished, Error                             string
	Invocation                                                           []string
	Pins                                                                 map[string]string
	DownSHA, ScoresSHA, Executable, ExecutableSHA                        string
	DownBytes, ScoresBytes                                               int
	ControlFile, ControlOriginalSHA, ControlChangedSHA, ControlRejection string
}

func InspectWitness(capture, out string) (r WitnessReceipt, err error) {
	attempt, err := runner.NewAttempt(out)
	if err != nil {
		return r, err
	}
	r = WitnessReceipt{Schema: "nanopore-hac-output-witness/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Invocation: slices.Clone(os.Args)}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(attempt, "witness.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	w, err := ReadWitness(capture)
	if err != nil {
		return r, err
	}
	r.Pins = w.Pins
	r.DownSHA = sum(w.Down)
	r.ScoresSHA = sum(w.Scores)
	r.DownBytes = len(w.Down)
	r.ScoresBytes = len(w.Scores)
	r.Executable, err = os.Executable()
	if err != nil {
		return r, err
	}
	r.ExecutableSHA, err = runner.SHA(r.Executable)
	if err != nil {
		return r, err
	}
	r.ControlFile = filepath.Join(attempt, "changed-flstm6-parent.bin")
	r.ControlOriginalSHA = sum(w.InputParent)
	copy := slices.Clone(w.InputParent)
	copy[5357568+131072] ^= 1
	r.ControlChangedSHA = sum(copy)
	if err = os.WriteFile(r.ControlFile, copy, 0644); err != nil {
		return r, err
	}
	if _, e := readPinned(r.ControlFile, r.ControlOriginalSHA); e == nil {
		return r, fmt.Errorf("changed genuine input copy accepted")
	} else {
		r.ControlRejection = e.Error()
	}
	for path, pin := range w.Pins {
		if _, err = readPinned(path, pin); err != nil {
			return r, err
		}
	}
	r.Status = "genuine_witness_and_corruption_control_passed"
	return r, nil
}
