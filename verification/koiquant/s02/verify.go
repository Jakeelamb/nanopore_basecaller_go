// Package s02 compares production Go QKV arithmetic with retained CUDA outputs.
package s02

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

const CUDAImportSHA = "b65fc01a95b46530cfd73f25f8762c2b67752281edc89257278afefbf74e29e8"
const ComparisonSHA = "13466965436bd283913a24587167a9ee1e67d823accd5ceeb5193c1fd3008339"
const RawCUDASHA = "0a79b9a303f284f0030f525f453904a72257dcdec334051b19690eaf402518ac"

type MutationResult struct {
	Name                                            string
	DifferentI8, DifferentInverseBits, RejectedRows int
}

type FixtureResult struct {
	Name                                           string
	Rows, Admitted, Nonfinite, Zero, ScaleOverflow int
	ComparedI8, ComparedInverseBits, ComparedTile  int
	FullTensorRejected                             bool
	OutputSHA256                                   map[string]string
	Mutations                                      []MutationResult
}

type Manifest struct {
	Schema, Status, Started, Finished, Error string
	Invocation                               []string
	CUDA, Comparison, Raw, Binary            koiquant.Pin
	SourceSHA256                             map[string]string
	Fixtures                                 []FixtureResult
	Rows, Admitted, Rejected                 int
	Mutations                                []MutationResult
}

func clock() string          { return time.Now().UTC().Format(time.RFC3339Nano) }
func digest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
func pinned(path, want string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if digest(b) != want {
		return nil, fmt.Errorf("SHA256 mismatch for %s", path)
	}
	return b, nil
}
func classify(row []uint16) model.QKVRowRejection {
	var max uint16
	for _, v := range row {
		v &= 0x7fff
		if v >= 0x7c00 {
			return model.QKVNonfiniteRow
		}
		if v > max {
			max = v
		}
	}
	if max == 0 {
		return model.QKVZeroRow
	}
	if max < 0x1801 {
		return model.QKVScaleOverflowRow
	}
	return ""
}
func tileAt(row, col int) int { return ((row/16*32+col/16)*16+row%16)*16 + col%16 }
func int8Bytes(q []int8) []byte {
	b := make([]byte, len(q))
	for i, v := range q {
		b[i] = byte(v)
	}
	return b
}
func floatBytes(q []float32) []byte {
	b := make([]byte, 4*len(q))
	for i, v := range q {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func verifyFixture(o koiquant.Observation, out string) (r FixtureResult, err error) {
	r = FixtureResult{Name: o.Name, Rows: model.SUPQKVRows, OutputSHA256: map[string]string{}}
	if err = koiquant.ValidateStages(o.Stages, o.Stages["input"]); err != nil {
		return r, err
	}
	for name, device := range o.Devices {
		want := "cuda:0"
		if name == "input" {
			want = "cpu"
		}
		if device != want {
			return r, fmt.Errorf("%s device mismatch", name)
		}
	}
	if len(o.Devices) != len(koiquant.StageSpecs()) {
		return r, fmt.Errorf("missing device observations")
	}
	half, err := koiweights.ReadHalf(o.Stages["input"])
	if err != nil {
		return r, err
	}
	char, err := pinned(o.Stages["char"].Path, o.Stages["char"].SHA256)
	if err != nil {
		return r, err
	}
	inverse, err := pinned(o.Stages["freciprocal"].Path, o.Stages["freciprocal"].SHA256)
	if err != nil {
		return r, err
	}
	tile, err := pinned(o.Stages["i8tile"].Path, o.Stages["i8tile"].SHA256)
	if err != nil {
		return r, err
	}
	sanitized := append([]uint16(nil), half...)
	packed := make([]uint16, 0, len(half))
	indices := make([]int, 0, model.SUPQKVRows)
	mask := make([]byte, model.SUPQKVRows)
	for row := 0; row < model.SUPQKVRows; row++ {
		values := half[row*512 : (row+1)*512]
		reason := classify(values)
		if reason == "" {
			r.Admitted++
			mask[row] = 1
			indices = append(indices, row)
			packed = append(packed, values...)
			continue
		}
		switch reason {
		case model.QKVNonfiniteRow:
			r.Nonfinite++
		case model.QKVZeroRow:
			r.Zero++
		case model.QKVScaleOverflowRow:
			r.ScaleOverflow++
		}
		q, scales, e := model.QuantizeSUPQKVRows(values)
		var rejected *model.QKVRowError
		if !errors.As(e, &rejected) || rejected.Row != 0 || rejected.Reason != reason || q != nil || scales != nil {
			return r, fmt.Errorf("excluded row %d was not explicitly rejected as %s: %v", row, reason, e)
		}
		clear(sanitized[row*512 : (row+1)*512])
		sanitized[row*512] = 0x5800
	}
	if r.Admitted != o.Classification.AdmittedRows || r.Nonfinite != o.Classification.NonfiniteInputRows || r.Zero != o.Classification.ZeroRangeRows || r.ScaleOverflow != o.Classification.OverflowScaleRows {
		return r, fmt.Errorf("row classification differs from accepted CUDA import")
	}
	if r.Admitted != model.SUPQKVRows {
		_, e := model.BuildSUPQKVInt8(half)
		var rejected *model.QKVRowError
		if !errors.As(e, &rejected) {
			return r, fmt.Errorf("original excluded tensor did not fail: %v", e)
		}
		r.FullTensorRejected = true
	}
	cache, err := model.BuildSUPQKVInt8(sanitized)
	if err != nil {
		return r, err
	}
	var q []int8
	var reciprocal, scale []float32
	if r.Admitted > 0 {
		q, reciprocal, err = model.QuantizeSUPQKVRows(packed)
		if err != nil {
			return r, err
		}
		_, scale, err = model.QuantizeFLSTMRows(packed, int64(r.Admitted), 512)
		if err != nil {
			return r, err
		}
	}
	for p, row := range indices {
		wantScale := binary.LittleEndian.Uint32(inverse[row*4:])
		if math.Float32bits(reciprocal[p]) != wantScale || math.Float32bits(cache.InverseScale[row]) != wantScale {
			return r, fmt.Errorf("inverse scale bit mismatch row %d", row)
		}
		r.ComparedInverseBits++
		for c := 0; c < 512; c++ {
			if byte(q[p*512+c]) != char[row*512+c] {
				return r, fmt.Errorf("int8 mismatch row %d column %d", row, c)
			}
			at := tileAt(row, c)
			if byte(cache.Tiled[at]) != tile[at] || byte(q[p*512+c]) != tile[at] {
				return r, fmt.Errorf("tile mismatch row %d column %d", row, c)
			}
			r.ComparedI8++
			r.ComparedTile++
		}
	}
	if r.Admitted == model.SUPQKVRows && (digest(int8Bytes(cache.Tiled)) != o.Stages["i8tile"].SHA256 || digest(floatBytes(cache.InverseScale)) != o.Stages["freciprocal"].SHA256) {
		return r, fmt.Errorf("full tensor output digest mismatch")
	}
	r.Mutations = mutations(packed, q, reciprocal, scale, indices, char, inverse, tile)
	if err = os.Mkdir(out, 0755); err != nil {
		return r, err
	}
	for name, data := range map[string][]byte{"tiled-i8.bin": int8Bytes(cache.Tiled), "inverse-f32le.bin": floatBytes(cache.InverseScale), "admitted-mask.bin": mask} {
		if err = os.WriteFile(filepath.Join(out, name), data, 0644); err != nil {
			return r, err
		}
		r.OutputSHA256[name] = digest(data)
	}
	return r, nil
}

// Verify reads only accepted saved observations. Excluded rows are tested for
// rejection, then replaced with a marked finite row solely to exercise tiling.
func Verify(root, cudaPath, comparisonPath, out string) (m Manifest, err error) {
	if err = os.Mkdir(out, 0755); err != nil {
		return m, fmt.Errorf("fresh output directory required: %w", err)
	}
	m = Manifest{Schema: "nanopore-sup-qkv-i8/go-s02-v1", Status: "failed", Started: clock(), Invocation: os.Args, SourceSHA256: map[string]string{}, Fixtures: []FixtureResult{}}
	defer func() {
		m.Finished = clock()
		if err != nil {
			m.Error = err.Error()
		}
		b, e := json.MarshalIndent(m, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(out, "manifest.json"), append(b, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	m.CUDA = koiquant.Pin{Path: cudaPath, SHA256: CUDAImportSHA}
	m.Comparison = koiquant.Pin{Path: comparisonPath, SHA256: ComparisonSHA}
	b, err := pinned(cudaPath, CUDAImportSHA)
	if err != nil {
		return m, err
	}
	var imported koiquant.Imported
	if err = json.Unmarshal(b, &imported); err != nil {
		return m, err
	}
	if imported.Schema != koiquant.ImportSchema || imported.Status != "complete_observations_validated" || len(imported.Observations) != 31 || imported.Raw.SHA256 != RawCUDASHA {
		return m, fmt.Errorf("accepted CUDA identity mismatch")
	}
	m.Raw = imported.Raw
	if _, err = pinned(m.Raw.Path, m.Raw.SHA256); err != nil {
		return m, err
	}
	b, err = pinned(comparisonPath, ComparisonSHA)
	if err != nil {
		return m, err
	}
	var comparison koiquant.Comparison
	if err = json.Unmarshal(b, &comparison); err != nil {
		return m, err
	}
	if comparison.CUDA.SHA256 != CUDAImportSHA || len(comparison.Stages) != 434 {
		return m, fmt.Errorf("accepted CPU/CUDA comparison mismatch")
	}
	if _, err = koiquant.Extract(root); err != nil {
		return m, err
	}
	for _, name := range []string{"go.mod", "internal/model/koi_qkv_int8.go", "internal/model/hac_gpu_flstm_weights.go", "internal/signal/half.go", "verification/koiquant/s02/verify.go", "verification/koiquant/s02/mutations.go", "verification/koiquant/s02/cmd/main.go"} {
		h, e := koiweights.Hash(filepath.Join(root, name))
		if e != nil {
			return m, e
		}
		m.SourceSHA256[name] = h
	}
	executable, err := os.Executable()
	if err != nil {
		return m, err
	}
	h, err := koiweights.Hash(executable)
	if err != nil {
		return m, err
	}
	m.Binary = koiquant.Pin{Path: executable, SHA256: h}
	for _, o := range imported.Observations {
		r, e := verifyFixture(o, filepath.Join(out, o.Name))
		m.Fixtures = append(m.Fixtures, r)
		if e != nil {
			return m, fmt.Errorf("fixture %s: %w", o.Name, e)
		}
		m.Rows += r.Rows
		m.Admitted += r.Admitted
		m.Rejected += r.Nonfinite + r.Zero + r.ScaleOverflow
		if len(m.Mutations) == 0 {
			for _, v := range r.Mutations {
				m.Mutations = append(m.Mutations, MutationResult{Name: v.Name})
			}
		}
		for i, v := range r.Mutations {
			m.Mutations[i].DifferentI8 += v.DifferentI8
			m.Mutations[i].DifferentInverseBits += v.DifferentInverseBits
			m.Mutations[i].RejectedRows += v.RejectedRows
		}
	}
	if m.Rows != 31*1536 || m.Admitted+m.Rejected != m.Rows {
		return m, fmt.Errorf("incomplete row accounting")
	}
	for _, v := range m.Mutations {
		if v.DifferentI8+v.DifferentInverseBits == 0 {
			return m, fmt.Errorf("mutation %s did not separate saved outputs", v.Name)
		}
	}
	for name, want := range m.SourceSHA256 {
		h, e := koiweights.Hash(filepath.Join(root, name))
		if e != nil || h != want {
			return m, fmt.Errorf("source changed during differential: %s", name)
		}
	}
	m.Status = "bounded_saved_cuda_differential_passed"
	return m, nil
}
