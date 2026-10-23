package caller

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
)

// GPU capture is the producer-free replay contract for one decoded GPU read. It
// records the runtime that produced the chunks, the exact chunk geometry and the
// normalized signal, so postcall can run without a model, POD5 file or GPU.
const gpuCaptureSchema = "gpu-capture/1"

// RuntimeIdentity pins the producer. A capture with a missing pin or a CPU
// producer cannot be replayed as GPU evidence.
type RuntimeIdentity struct {
	Producer     string `json:"producer"`
	Device       string `json:"device"`
	Compiler     string `json:"compiler"`
	TensorABI    string `json:"tensor_abi"`
	SourceCommit string `json:"source_commit"`
}

var sourceCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

func (r RuntimeIdentity) validate() error {
	if r.Producer != "gpu" {
		return fmt.Errorf("capture producer %q is not gpu", r.Producer)
	}
	if r.Device == "" || r.Compiler == "" || r.TensorABI == "" {
		return errors.New("capture runtime pins are incomplete")
	}
	if !sourceCommitPattern.MatchString(r.SourceCommit) {
		return errors.New("capture source commit is not an exact commit")
	}
	return nil
}

// CapturedChunk is one decoded logical chunk. Granularity is the padding
// alignment used for the model invocation, not a padded length.
type CapturedChunk struct {
	Index       int     `json:"index"`
	Offset      uint64  `json:"offset"`
	RawLength   uint64  `json:"raw_length"`
	Granularity uint64  `json:"granularity"`
	Sequence    string  `json:"sequence"`
	Qstring     string  `json:"qstring"`
	Moves       []uint8 `json:"moves"`
}

// CapturedRead retains exactly the metadata postcall consumes. Signal is the
// normalized half-precision buffer that the GPU model consumed.
type CapturedRead struct {
	ID                     string   `json:"id"`
	NumSamples             uint64   `json:"num_samples"`
	SampleRate             uint64   `json:"sample_rate"`
	StartSample            uint64   `json:"start_sample"`
	EndSample              uint64   `json:"end_sample"`
	Channel                uint16   `json:"channel"`
	Well                   uint8    `json:"well"`
	NumMinknowEvents       uint64   `json:"num_minknow_events"`
	EndReason              uint8    `json:"end_reason"`
	EndReasonName          string   `json:"end_reason_name"`
	RunID                  string   `json:"run_id"`
	FlowCellID             string   `json:"flowcell_id"`
	FlowCellProductCode    string   `json:"flowcell_product_code"`
	SequencingKit          string   `json:"sequencing_kit"`
	PositionID             string   `json:"position_id"`
	AcquisitionID          string   `json:"acquisition_id"`
	ExperimentID           string   `json:"experiment_id"`
	SampleID               string   `json:"sample_id"`
	Filename               string   `json:"filename"`
	PoreType               string   `json:"pore_type"`
	AcquisitionStartTimeMS int64    `json:"acquisition_start_time_ms"`
	ProtocolStartTimeMS    int64    `json:"protocol_start_time_ms"`
	CalibrationScale       float32  `json:"calibration_scale"`
	CalibrationOffset      float32  `json:"calibration_offset"`
	OpenPoreLevel          float32  `json:"open_pore_level"`
	NormalizedSignal       []uint16 `json:"normalized_signal"`
	NormalizedShift        float32  `json:"normalized_shift"`
	NormalizedScale        float32  `json:"normalized_scale"`
	NormalizedTrimmed      int      `json:"normalized_trimmed"`
}

type GPUCapture struct {
	Schema    string          `json:"schema"`
	Runtime   RuntimeIdentity `json:"runtime"`
	Lane      string          `json:"lane"`
	ChunkSize uint64          `json:"chunk_size"`
	BatchSize uint64          `json:"batch_size"`
	Stride    uint64          `json:"stride"`
	Read      CapturedRead    `json:"read"`
	Chunks    []CapturedChunk `json:"chunks"`
	Policy    postcall.Policy `json:"policy"`
	Digest    string          `json:"digest"`
}

// payloadDigest covers every field except Digest itself.
func (c GPUCapture) payloadDigest() (string, error) {
	type plain GPUCapture
	p := plain(c)
	p.Digest = ""
	data, err := json.Marshal(p)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// Seal validates the capture and writes its content digest.
func (c GPUCapture) Seal() (GPUCapture, error) {
	c.Digest = ""
	if err := c.validate(); err != nil {
		return GPUCapture{}, err
	}
	digest, err := c.payloadDigest()
	if err != nil {
		return GPUCapture{}, err
	}
	c.Digest = digest
	return c, nil
}

// DecodeGPUCapture accepts only a closed, sealed GPU capture.
func DecodeGPUCapture(data []byte) (GPUCapture, error) {
	var c GPUCapture
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return GPUCapture{}, err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return GPUCapture{}, errors.New("trailing capture JSON")
	}
	if err := c.validate(); err != nil {
		return GPUCapture{}, err
	}
	want, err := c.payloadDigest()
	if err != nil {
		return GPUCapture{}, err
	}
	if c.Digest == "" || c.Digest != want {
		return GPUCapture{}, errors.New("capture digest differs from payload")
	}
	return c, nil
}

func (c GPUCapture) validate() error {
	if c.Schema != gpuCaptureSchema {
		return fmt.Errorf("unsupported capture schema %q", c.Schema)
	}
	if err := c.Runtime.validate(); err != nil {
		return err
	}
	spec, err := c.spec()
	if err != nil {
		return err
	}
	if c.ChunkSize == 0 || c.BatchSize == 0 || c.Stride == 0 {
		return errors.New("capture geometry is incomplete")
	}
	if uint64(spec.Stride) != c.Stride || c.ChunkSize%uint64(spec.Granularity) != 0 {
		return fmt.Errorf("capture geometry does not match the pinned %s model", c.Lane)
	}
	if err := c.Policy.Validate(c.Read.SequencingKit); err != nil {
		return err
	}
	read, normalized, err := c.readModel()
	if err != nil {
		return err
	}
	if err := validateAdmission(read); err != nil {
		return err
	}
	if _, err := pod5.ParseID(read.ID.String()); err != nil {
		return err
	}
	if len(normalized.Samples) == 0 {
		return errors.New("capture has no normalized signal")
	}
	if len(c.Chunks) == 0 {
		return errors.New("capture has no chunks")
	}
	for i, ch := range c.Chunks {
		if ch.Index != i {
			return fmt.Errorf("chunk %d declares index %d", i, ch.Index)
		}
		if ch.RawLength == 0 || ch.Offset > c.Read.NumSamples || ch.RawLength > c.Read.NumSamples-ch.Offset {
			return fmt.Errorf("chunk %d coordinate is outside the read", i)
		}
		if ch.Granularity == 0 || ch.Granularity%c.Stride != 0 {
			return fmt.Errorf("chunk %d granularity is not stride aligned", i)
		}
		if len(ch.Sequence) != len(ch.Qstring) || len(ch.Moves) != len(ch.Sequence) {
			return fmt.Errorf("chunk %d sequence, quality or moves differ", i)
		}
	}
	return nil
}

func (c GPUCapture) readModel() (pod5.Read, signal.Normalized, error) {
	read := pod5.Read{
		ID: pod5.ID{}, Signal: nil, NumSamples: c.Read.NumSamples,
		CalibrationScale: c.Read.CalibrationScale, CalibrationOffset: c.Read.CalibrationOffset,
		OpenPoreLevel: c.Read.OpenPoreLevel, ReadNumber: 0,
		StartSample: c.Read.StartSample, EndSample: c.Read.EndSample,
		Channel: c.Read.Channel, Well: c.Read.Well, NumMinknowEvents: c.Read.NumMinknowEvents,
		EndReason: c.Read.EndReason, EndReasonName: c.Read.EndReasonName, Filename: c.Read.Filename,
		PoreType: c.Read.PoreType,
	}
	id, err := pod5.ParseID(c.Read.ID)
	if err != nil {
		return pod5.Read{}, signal.Normalized{}, err
	}
	read.ID = id
	read.RunInfo = pod5.RunInfo{
		SampleRate: c.Read.SampleRate, RunID: c.Read.RunID, AcquisitionID: c.Read.AcquisitionID,
		AcquisitionStartTimeMS: c.Read.AcquisitionStartTimeMS, ProtocolStartTimeMS: c.Read.ProtocolStartTimeMS,
		FlowCellID: c.Read.FlowCellID, FlowCellProductCode: c.Read.FlowCellProductCode,
		SequencingKit: c.Read.SequencingKit, PositionID: c.Read.PositionID,
		ExperimentID: c.Read.ExperimentID, SampleID: c.Read.SampleID,
	}
	normalized := signal.Normalized{
		Samples: append([]uint16(nil), c.Read.NormalizedSignal...),
		Shift:   c.Read.NormalizedShift, Scale: c.Read.NormalizedScale, Trimmed: c.Read.NormalizedTrimmed,
	}
	return read, normalized, nil
}

// CaptureSpec returns the pinned model identity a capture replays under. It
// never reads model files.
func CaptureSpec(c GPUCapture) (model.Spec, error) { return c.spec() }

// spec resolves the pinned model identity for the capture lane. It never reads
// model files.
func (c GPUCapture) spec() (model.Spec, error) {
	switch c.Lane {
	case "hac":
		return model.Pinned(model.HAC)
	case "sup":
		return model.Pinned(model.SUP)
	default:
		return model.Spec{}, fmt.Errorf("unsupported capture lane %q", c.Lane)
	}
}

// Assemble restores source chunk order and delegates coverage validation to the
// accepted stitch unit.
func (c GPUCapture) Assemble() (stitch.Result, error) {
	chunks := make([]stitch.Chunk, len(c.Chunks))
	for i, ch := range c.Chunks {
		chunks[i] = stitch.Chunk{Offset: ch.Offset, RawSize: ch.RawLength, Granularity: ch.Granularity,
			Sequence: ch.Sequence, Qstring: ch.Qstring, Moves: ch.Moves}
	}
	return stitch.Stitch(stitch.Config{RawSamples: c.Read.NumSamples, Stride: c.Stride, Variable: c.Lane == "sup"}, chunks)
}

// ReplayGPUCapture runs the accepted postcall pipeline on retained GPU output.
// The requested policy must equal the captured policy; replay cannot reinterpret
// a capture under different trim, split or quality rules.
func ReplayGPUCapture(c GPUCapture, policy postcall.Policy) (Processed, error) {
	if policy != c.Policy {
		return Processed{}, errors.New("replay policy differs from the captured policy")
	}
	spec, err := c.spec()
	if err != nil {
		return Processed{}, err
	}
	assembled, err := c.Assemble()
	if err != nil {
		return Processed{}, err
	}
	read, normalized, err := c.readModel()
	if err != nil {
		return Processed{}, err
	}
	return ProcessGPURead(spec, GPURead{Read: read, Normalized: normalized, Stride: c.Stride, Assembled: assembled}, policy)
}
