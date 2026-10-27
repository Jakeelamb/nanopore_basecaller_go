package decodergpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
)

const ScoresSHA = "56b26176b4272906612eb75690087182a72527f94a9171d9937e5dd80b0d12db"
const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

type Witness struct {
	Scores     []byte
	Pins       map[string]string
	ScoreStage hacgpu.Stage
}

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func pin(pins map[string]string, path string) ([]byte, error) {
	actual, err := filepath.EvalSymlinks(path)
	if err != nil {
		return nil, err
	}
	actual, err = filepath.Abs(actual)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(actual)
	if err != nil {
		return nil, err
	}
	got := digest(b)
	if old, ok := pins[actual]; ok && old != got {
		return nil, fmt.Errorf("input changed: %s", actual)
	}
	pins[actual] = got
	return b, nil
}
func ReadWitness(captureDir, modelDir string) (w Witness, err error) {
	w.Pins = map[string]string{}
	manifest := filepath.Join(captureDir, "capture.json")
	b, err := pin(w.Pins, manifest)
	if err != nil {
		return w, err
	}
	if digest(b) != h07.H01CaptureSHA {
		return w, fmt.Errorf("H01 capture pin changed")
	}
	receiptPath := filepath.Join(filepath.Dir(captureDir), "receipt.json")
	b, err = pin(w.Pins, receiptPath)
	if err != nil {
		return w, err
	}
	if digest(b) != h07.H01ReceiptSHA {
		return w, fmt.Errorf("H01 receipt pin changed")
	}
	var receipt struct {
		Status string            `json:"status"`
		Inputs map[string]string `json:"input_sha256_all"`
	}
	if err = json.Unmarshal(b, &receipt); err != nil {
		return w, err
	}
	if receipt.Status != "stage_capture_passed" {
		return w, fmt.Errorf("H01 producer did not pass")
	}
	config := filepath.Join(modelDir, "config.toml")
	raw, err := pin(w.Pins, config)
	if err != nil {
		return w, err
	}
	expected, ok := receipt.Inputs[config]
	if !ok || digest(raw) != expected {
		return w, fmt.Errorf("HAC config does not match producer pin")
	}
	// The source reads qscore as float and detects explicit clamp sublayers.
	if !strings.Contains(string(raw), "scale = 1.02") || !strings.Contains(string(raw), "bias = -0.3") || strings.Contains(string(raw), `type = "clamp"`) {
		return w, fmt.Errorf("HAC calibration/clamp contract changed")
	}
	capture, err := hacgpu.Import(captureDir)
	if err != nil {
		return w, err
	}
	if capture.SourceCommit != SourceCommit || capture.Batch != N || capture.TOut != T || capture.Chunk != 192 || capture.Convs != 3 || capture.LSTMLayers != 7 || capture.Linear2 || capture.ModelPackage != filepath.Base(modelDir) {
		return w, fmt.Errorf("capture is not the fixed HAC witness")
	}
	stage, ok := capture.Stage("scores")
	if !ok || stage.DType != "Half" || stage.Layout != "NTC" || stage.N != N || stage.T != T || stage.C != C || stage.Bytes != ScoresBytes || stage.OffsetBytes != 524288 || len(stage.Shape) != 3 || stage.Shape[0] != N || stage.Shape[1] != T || stage.Shape[2] != C || len(stage.UnwrittenT) != 0 || stage.DataSHA != ScoresSHA {
		return w, fmt.Errorf("retained score descriptor differs from fixed NTC half contract")
	}
	w.ScoreStage = stage
	w.Scores, err = pin(w.Pins, filepath.Join(captureDir, stage.DataFile))
	if err != nil {
		return w, err
	}
	return w, ValidateScores(w.Scores, ScoresSHA)
}
