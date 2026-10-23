package caller

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/crf"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
)

type payload struct {
	File, SHA256 string
	Bytes        int64
}
type runPins struct {
	Schema              int
	Spec                model.Spec
	Weights             []model.WeightReceipt
	TensorLibrarySHA256 string
	ModelConfig         payload
	Chunks              chunk.Config
	Decode              decode.Options
}
type inputPins struct {
	Metadata          pod5.Read
	FloatBits         map[string]uint32
	Raw, Normalized   payload
	Trimmed           int
	NormalizedSamples uint64
}
type savedChunk struct {
	Blocks, States                   int
	Decoded                          stitch.Chunk
	Scores, Forward, Backward, Posts payload
}
type manifest struct {
	Schema     int
	Run, Input payload
	Chunks     []savedChunk
	Output     payload
}
type captureRead struct {
	dir      string
	manifest manifest
}

func digest(data []byte) string { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }
func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func save(dir, name string, data []byte) (payload, error) {
	p := payload{File: name, SHA256: digest(data), Bytes: int64(len(data))}
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return payload{}, err
	}
	_, err = f.Write(data)
	err = errors.Join(err, f.Close())
	return p, err
}
func saveJSON(dir, name string, value any) (payload, error) {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return payload{}, err
	}
	return save(dir, name, append(data, '\n'))
}
func startCapture(c Config, s model.Spec, w []model.WeightReceipt, cc chunk.Config, o decode.Options) (payload, error) {
	if err := os.Mkdir(c.CaptureDir, 0700); err != nil {
		return payload{}, fmt.Errorf("capture requires a new directory: %w", err)
	}
	hash, err := fileDigest(c.TensorLibrary)
	if err != nil {
		return payload{}, err
	}
	config, err := os.ReadFile(filepath.Join(c.ModelDir, "config.toml"))
	if err != nil {
		return payload{}, err
	}
	if digest(config) != s.ConfigSHA256 {
		return payload{}, errors.New("model config changed during initialization")
	}
	configFile, err := save(c.CaptureDir, "model-config.toml", config)
	if err != nil {
		return payload{}, err
	}
	return saveJSON(c.CaptureDir, "run.json", runPins{Schema: 1, Spec: s, Weights: w, TensorLibrarySHA256: hash, ModelConfig: configFile, Chunks: cc, Decode: o})
}
func beginRead(root string, run payload, r pod5.Read, n signal.Normalized) (*captureRead, error) {
	dir := filepath.Join(root, r.ID.String())
	if err := os.Mkdir(dir, 0700); err != nil {
		return nil, err
	}
	runData, err := readPayload(root, run)
	if err != nil {
		return nil, err
	}
	var pinsRun runPins
	if err := json.Unmarshal(runData, &pinsRun); err != nil {
		return nil, err
	}
	config, err := readPayload(root, pinsRun.ModelConfig)
	if err != nil {
		return nil, err
	}
	if _, err := save(dir, pinsRun.ModelConfig.File, config); err != nil {
		return nil, err
	}
	localRun, err := save(dir, "run.json", runData)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, 2*len(r.Signal))
	for i, v := range r.Signal {
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(v))
	}
	rawFile, err := save(dir, "raw.i16le", raw)
	if err != nil {
		return nil, err
	}
	half := make([]byte, 2*len(n.Samples))
	for i, v := range n.Samples {
		binary.LittleEndian.PutUint16(half[2*i:], v)
	}
	normalized, err := save(dir, "normalized.f16le", half)
	if err != nil {
		return nil, err
	}
	pins := inputPins{Metadata: r, FloatBits: map[string]uint32{"CalibrationScale": math.Float32bits(r.CalibrationScale), "CalibrationOffset": math.Float32bits(r.CalibrationOffset), "OpenPoreLevel": math.Float32bits(r.OpenPoreLevel), "MedianBefore": math.Float32bits(r.MedianBefore)}, Raw: rawFile, Normalized: normalized, Trimmed: n.Trimmed, NormalizedSamples: uint64(len(n.Samples))}
	pins.Metadata.Signal = nil
	pins.Metadata.CalibrationScale = 0
	pins.Metadata.CalibrationOffset = 0
	pins.Metadata.OpenPoreLevel = 0
	pins.Metadata.MedianBefore = 0
	input, err := saveJSON(dir, "input.json", pins)
	if err != nil {
		return nil, err
	}
	return &captureRead{dir: dir, manifest: manifest{Schema: 1, Run: localRun, Input: input}}, nil
}
func floatBytes(values []float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
func (c *captureRead) addChunk(index int, d stitch.Chunk, r crf.Result) error {
	s := savedChunk{Blocks: r.Blocks, States: r.States, Decoded: d}
	for _, part := range []struct {
		name   string
		values []float32
		dest   *payload
	}{{"scores", r.Scores, &s.Scores}, {"forward", r.Forward, &s.Forward}, {"backward", r.Backward, &s.Backward}, {"posts", r.Posts, &s.Posts}} {
		p, err := save(c.dir, fmt.Sprintf("chunk-%04d-%s.f32le", index, part.name), floatBytes(part.values))
		if err != nil {
			return err
		}
		*part.dest = p
	}
	c.manifest.Chunks = append(c.manifest.Chunks, s)
	// A partial attempt retains each finished chunk even if later computation fails.
	_, err := saveJSON(c.dir, fmt.Sprintf("chunk-%04d.json", index), s)
	return err
}
func (c *captureRead) finish(out CalledRead) error {
	p, err := saveJSON(c.dir, "output.json", out)
	if err != nil {
		return err
	}
	c.manifest.Output = p
	_, err = saveJSON(c.dir, "manifest.json", c.manifest)
	return err
}
func readPayload(dir string, p payload) ([]byte, error) {
	if p.File == "" || filepath.Base(p.File) != p.File || p.Bytes < 0 || p.Bytes > 1<<30 || len(p.SHA256) != 64 {
		return nil, errors.New("invalid capture payload reference")
	}
	path := filepath.Join(dir, p.File)
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() != p.Bytes {
		return nil, errors.New("capture payload size or file type differs")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != p.Bytes || digest(data) != p.SHA256 {
		return nil, fmt.Errorf("capture SHA256 mismatch: %s", p.File)
	}
	return data, nil
}
func readJSON(dir string, p payload, out any) error {
	data, err := readPayload(dir, p)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return err
	}
	if decoder.Decode(new(any)) != io.EOF {
		return errors.New("trailing capture JSON")
	}
	return nil
}
func readFloats(dir string, p payload, count int) ([]float32, error) {
	if count < 0 || count > 1<<27 || p.Bytes != int64(count)*4 {
		return nil, errors.New("captured tensor shape does not match byte count")
	}
	data, err := readPayload(dir, p)
	if err != nil {
		return nil, err
	}
	out := make([]float32, count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

// Replay verifies retained payloads and reruns only Go decoding and stitching.
func Replay(ctx context.Context, path string) (CalledRead, error) {
	if ctx == nil {
		return CalledRead{}, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return CalledRead{}, err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return CalledRead{}, err
	}
	if !info.Mode().IsRegular() || info.Size() > 16<<20 {
		return CalledRead{}, errors.New("invalid capture manifest file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return CalledRead{}, err
	}
	var m manifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&m); err != nil {
		return CalledRead{}, err
	}
	if decoder.Decode(new(any)) != io.EOF {
		return CalledRead{}, errors.New("trailing manifest JSON")
	}
	if m.Schema != 1 || len(m.Chunks) == 0 || len(m.Chunks) > 1000000 {
		return CalledRead{}, errors.New("unsupported or empty capture manifest")
	}
	dir := filepath.Dir(path)
	var run runPins
	var input inputPins
	if err := readJSON(dir, m.Run, &run); err != nil {
		return CalledRead{}, err
	}
	if err := readJSON(dir, m.Input, &input); err != nil {
		return CalledRead{}, err
	}
	config, err := readPayload(dir, run.ModelConfig)
	if err != nil {
		return CalledRead{}, err
	}
	spec, err := model.Identify(config)
	if err != nil {
		return CalledRead{}, err
	}
	if spec != run.Spec {
		return CalledRead{}, errors.New("captured model specification differs from pinned config bytes")
	}
	cc, opt, err := configuration(run.Spec, Config{ChunkSize: run.Chunks.ChunkSize, Overlap: run.Chunks.Overlap})
	if err != nil {
		return CalledRead{}, err
	}
	weightCount := 50
	if spec.Architecture == model.SUP {
		weightCount = 139
	}
	if run.Schema != 1 || cc != run.Chunks || opt != run.Decode || len(run.Weights) != weightCount || len(run.TensorLibrarySHA256) != 64 {
		return CalledRead{}, errors.New("unsupported capture model or configuration pins")
	}
	raw, err := readPayload(dir, input.Raw)
	if err != nil {
		return CalledRead{}, err
	}
	norm, err := readPayload(dir, input.Normalized)
	if err != nil {
		return CalledRead{}, err
	}
	if input.Trimmed != 0 && input.Trimmed != 10 || uint64(len(raw)) != 2*input.Metadata.NumSamples || uint64(len(norm)) != 2*input.NormalizedSamples || input.Metadata.NumSamples != input.NormalizedSamples+uint64(input.Trimmed) {
		return CalledRead{}, errors.New("invalid captured input counts")
	}
	r := input.Metadata
	r.Signal = make([]int16, len(raw)/2)
	for i := range r.Signal {
		r.Signal[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	if len(input.FloatBits) != 4 {
		return CalledRead{}, errors.New("incomplete input float bits")
	}
	for _, key := range []string{"CalibrationScale", "CalibrationOffset", "OpenPoreLevel", "MedianBefore"} {
		if _, ok := input.FloatBits[key]; !ok {
			return CalledRead{}, errors.New("missing input float bits")
		}
	}
	r.CalibrationScale = math.Float32frombits(input.FloatBits["CalibrationScale"])
	r.CalibrationOffset = math.Float32frombits(input.FloatBits["CalibrationOffset"])
	r.OpenPoreLevel = math.Float32frombits(input.FloatBits["OpenPoreLevel"])
	if err := validateRead(r); err != nil {
		return CalledRead{}, err
	}
	expectedNorm, err := signal.Normalize(r.Signal, signal.Calibration{Scale: r.CalibrationScale, Offset: r.CalibrationOffset}, signal.Standardisation{Mean: run.Spec.Mean, Stdev: run.Spec.Stdev}, adjustment(r.FlowCellProductCode, r.OpenPoreLevel, r.CalibrationScale))
	if err != nil {
		return CalledRead{}, err
	}
	if len(expectedNorm.Samples) != len(norm)/2 || expectedNorm.Trimmed != input.Trimmed {
		return CalledRead{}, errors.New("normalized capture length differs")
	}
	for i, v := range expectedNorm.Samples {
		if v != binary.LittleEndian.Uint16(norm[i*2:]) {
			return CalledRead{}, errors.New("normalized capture bits differ")
		}
	}
	offsets, err := chunk.Fixed(input.NormalizedSamples, cc)
	if err != nil {
		return CalledRead{}, err
	}
	if len(offsets) != len(m.Chunks) {
		return CalledRead{}, errors.New("captured chunk count differs")
	}
	chunks := make([]stitch.Chunk, 0, len(m.Chunks))
	for i, s := range m.Chunks {
		if err := ctx.Err(); err != nil {
			return CalledRead{}, err
		}
		if s.States != int(spec.OutputChannels/4) || s.Blocks != int(cc.ChunkSize/cc.Stride) || s.Decoded.Offset != offsets[i] || s.Decoded.RawSize != min(cc.ChunkSize, input.NormalizedSamples-offsets[i]) || s.Decoded.Granularity != cc.ChunkSize {
			return CalledRead{}, errors.New("captured chunk shape or coordinates differ")
		}
		result := crf.Result{Blocks: s.Blocks, States: s.States}
		for _, part := range []struct {
			p     payload
			count int
			dest  *[]float32
		}{{s.Scores, s.Blocks * 4 * s.States, &result.Scores}, {s.Forward, (s.Blocks + 1) * s.States, &result.Forward}, {s.Backward, (s.Blocks + 1) * s.States, &result.Backward}, {s.Posts, (s.Blocks + 1) * s.States, &result.Posts}} {
			values, err := readFloats(dir, part.p, part.count)
			if err != nil {
				return CalledRead{}, err
			}
			*part.dest = values
		}
		for _, v := range result.Forward {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return CalledRead{}, errors.New("nonfinite captured forward guide")
			}
		}
		decoded, err := decode.Decode(result.DecodeInput(), opt)
		if err != nil {
			return CalledRead{}, err
		}
		if decoded.Sequence != s.Decoded.Sequence || decoded.Qstring != s.Decoded.Qstring || !reflect.DeepEqual(decoded.Moves, s.Decoded.Moves) {
			return CalledRead{}, errors.New("captured chunk decode endpoint differs")
		}
		chunks = append(chunks, s.Decoded)
	}
	result, err := stitch.Stitch(stitch.Config{RawSamples: input.NormalizedSamples, Stride: cc.Stride}, chunks)
	if err != nil {
		return CalledRead{}, err
	}
	metadata, err := fastqMetadata(r, run.Spec)
	if err != nil {
		return CalledRead{}, err
	}
	out := CalledRead{ID: r.ID.String(), Sequence: result.Sequence, Qstring: result.Qstring, Moves: result.Moves, NumSamples: r.NumSamples, Trimmed: input.Trimmed, FASTQ: metadata}
	var expected CalledRead
	if err := readJSON(dir, m.Output, &expected); err != nil {
		return CalledRead{}, err
	}
	// Schema 1 captures before FASTQ metadata retain their original output bytes.
	if expected.FASTQ == nil {
		expected.FASTQ = metadata
	}
	if !reflect.DeepEqual(out, expected) {
		return CalledRead{}, errors.New("replayed endpoint differs from capture")
	}
	if err := ctx.Err(); err != nil {
		return CalledRead{}, err
	}
	return out, nil
}
