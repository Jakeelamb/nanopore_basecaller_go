package h03

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

// Stage records one stage comparison.
type Stage struct {
	Name          string   `json:"name"`
	DType         string   `json:"dtype"`
	Layout        string   `json:"layout"`
	CaptureSHA    string   `json:"capture_payload_sha256"`
	GoSHA         string   `json:"go_payload_sha256"`
	Shape         []int64  `json:"shape"`
	WrittenBytes  int64    `json:"compared_bytes"`
	GoWrittenSHA  string   `json:"go_written_sha256"`
	CapWrittenSHA string   `json:"capture_written_sha256"`
	UnwrittenRows [][2]int `json:"declared_unwritten_t"`
	Equal         bool     `json:"equal"`
}

// compareStage compares the bytes the Go caller produced with the captured stage
// payload, excluding only the rows the source declares it never writes. The
// mask comes from the capture's `unwritten_t`, which Import validated against
// the pinned source layout; it is never derived from observed mismatches.
func compareStage(captureDir string, capture hacgpu.Capture, name string, got *tensor.Storage, s *tensor.Stream) (Stage, error) {
	stage, ok := capture.Stage(name)
	if !ok {
		return Stage{}, fmt.Errorf("capture has no stage %q", name)
	}
	payload, err := capture.RawPayload(captureDir, name)
	if err != nil {
		return Stage{}, err
	}
	observed, err := got.Download(s)
	if err != nil {
		return Stage{}, err
	}
	if int64(len(observed)) != stage.Bytes {
		return Stage{}, fmt.Errorf("stage %q observed %d bytes, captured %d", name, len(observed), stage.Bytes)
	}
	written, err := visitWrittenRows(stage, observed)
	if err != nil {
		return Stage{}, fmt.Errorf("stage %q: %w", name, err)
	}
	expected, err := visitWrittenRows(stage, payload)
	if err != nil {
		return Stage{}, fmt.Errorf("stage %q: %w", name, err)
	}
	rec := Stage{
		Name: name, DType: stage.DType, Layout: stage.Layout,
		CaptureSHA: stage.DataSHA, GoSHA: hashBytes(observed), Shape: stage.Shape,
		WrittenBytes: int64(len(written)),
		GoWrittenSHA: hashBytes(written), CapWrittenSHA: hashBytes(expected),
		UnwrittenRows: stage.UnwrittenT, Equal: bytes.Equal(written, expected),
	}
	if !rec.Equal {
		return Stage{}, fmt.Errorf("stage %q differs from the capture in a written element", name)
	}
	return rec, nil
}

// visitWrittenRows returns the concatenation of every row the source writes, in
// memory order. For NTC the time axis is the middle one and batches are
// contiguous blocks; for a CUTLASS layer the time axis leads
// (WorkingMemory.cpp:87-94). The element size comes from the declared dtype.
func visitWrittenRows(stage hacgpu.Stage, payload []byte) ([]byte, error) {
	elemSize, err := stageElemSize(stage.DType)
	if err != nil {
		return nil, err
	}
	if len(stage.Shape) != 3 {
		return nil, fmt.Errorf("stage shape %v is not rank three", stage.Shape)
	}
	var n, t, c int
	var timeLeading bool
	switch stage.Layout {
	case "NTC":
		n, t, c = int(stage.Shape[0]), int(stage.Shape[1]), int(stage.Shape[2])
	case "CUTLASS_TNC_I8", "CUTLASS_TNC_F16":
		t, n, c = int(stage.Shape[0]), int(stage.Shape[1]), int(stage.Shape[2])
		timeLeading = true
	default:
		return nil, fmt.Errorf("stage layout %q has no declared row order", stage.Layout)
	}
	rowBytes := c * elemSize
	if n <= 0 || t <= 0 || rowBytes <= 0 || stage.Bytes != int64(n)*int64(t)*int64(rowBytes) {
		return nil, fmt.Errorf("stage shape %v, %d bytes and dtype %q do not describe contiguous rows", stage.Shape, stage.Bytes, stage.DType)
	}
	out := make([]byte, 0, len(payload))
	visit := func(offset int) { out = append(out, payload[offset:offset+rowBytes]...) }
	if timeLeading {
		for ti := 0; ti < t; ti++ {
			if stageRowUnwritten(stage, ti) {
				continue
			}
			for ni := 0; ni < n; ni++ {
				visit((ti*n + ni) * rowBytes)
			}
		}
		return out, nil
	}
	for ni := 0; ni < n; ni++ {
		for ti := 0; ti < t; ti++ {
			if stageRowUnwritten(stage, ti) {
				continue
			}
			visit((ni*t + ti) * rowBytes)
		}
	}
	return out, nil
}

func stageElemSize(dtype string) (int, error) {
	switch dtype {
	case "Half":
		return 2, nil
	case "Char":
		return 1, nil
	}
	return 0, fmt.Errorf("stage dtype %q has no declared element size", dtype)
}

func stageRowUnwritten(stage hacgpu.Stage, t int) bool {
	for _, r := range stage.UnwrittenT {
		if t >= r[0] && t < r[0]+r[1] {
			return true
		}
	}
	return false
}

// sameUnwrittenRows proves the plan's source-derived exclusion set equals the
// capture's declaration before execution. A plan that silently excluded a
// different set would make the comparison meaningless.
func sameUnwrittenRows(a [][2]int, b [][2]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hashString(s string) string { return hashBytes([]byte(s)) }

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
