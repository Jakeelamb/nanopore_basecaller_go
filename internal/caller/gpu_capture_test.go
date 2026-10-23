package caller

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

func gpuCaptureFixture(t *testing.T) GPUCapture {
	t.Helper()
	read := testRead()
	c := GPUCapture{
		Schema: gpuCaptureSchema,
		Runtime: RuntimeIdentity{Producer: "gpu", Device: "RTX 5070 Laptop", Compiler: "nvcc 13.0.88",
			TensorABI: "abi4", SourceCommit: strings.Repeat("a", 40)},
		Lane: "hac", ChunkSize: 6, BatchSize: 64, Stride: 6,
		Read: CapturedRead{
			ID: read.ID.String(), NumSamples: 36, SampleRate: 5000, Channel: 42, Well: 1,
			RunID: "run", FlowCellID: "flow", FlowCellProductCode: "FLO-PRO114M", SequencingKit: "sqk-lsk114",
			AcquisitionStartTimeMS: 1000, ProtocolStartTimeMS: 900, CalibrationScale: .187, CalibrationOffset: -270,
			NormalizedSignal: make([]uint16, 36), NormalizedShift: 1, NormalizedScale: 2, NormalizedTrimmed: 0,
		},
		Policy: postcall.DefaultPolicy(),
	}
	for i, base := range "ACGTAC" {
		c.Chunks = append(c.Chunks, CapturedChunk{Index: i, Offset: uint64(i * 6), RawLength: 6, Granularity: 6,
			Sequence: string(base), Qstring: "I", Moves: []uint8{1}})
	}
	sealed, err := c.Seal()
	if err != nil {
		t.Fatal(err)
	}
	return sealed
}

func TestGPUCaptureSealDecodeAndReplay(t *testing.T) {
	c := gpuCaptureFixture(t)
	if len(c.Digest) != 64 {
		t.Fatalf("digest = %q", c.Digest)
	}
	again, err := c.Seal()
	if err != nil || again.Digest != c.Digest {
		t.Fatalf("digest is not deterministic: %q %v", again.Digest, err)
	}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeGPUCapture(data)
	if err != nil {
		t.Fatal(err)
	}
	assembled, err := decoded.Assemble()
	if err != nil || assembled.Sequence != "ACGTAC" || len(assembled.Moves) != 6 {
		t.Fatalf("assembled %+v %v", assembled, err)
	}
	out, err := ReplayGPUCapture(decoded, decoded.Policy)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Reads) != 1 || out.Reads[0].Sequence != "ACGTAC" || out.Reads[0].FASTQ.Channel != 42 {
		t.Fatalf("replay output %+v", out.Reads)
	}
}

func TestGPUCaptureRejectsMutants(t *testing.T) {
	base := gpuCaptureFixture(t)
	mutants := map[string]func(*GPUCapture){
		"schema":        func(c *GPUCapture) { c.Schema = "gpu-capture/2" },
		"producer":      func(c *GPUCapture) { c.Runtime.Producer = "cpu" },
		"device":        func(c *GPUCapture) { c.Runtime.Device = "" },
		"compiler":      func(c *GPUCapture) { c.Runtime.Compiler = "" },
		"abi":           func(c *GPUCapture) { c.Runtime.TensorABI = "" },
		"source":        func(c *GPUCapture) { c.Runtime.SourceCommit = "abc" },
		"lane":          func(c *GPUCapture) { c.Lane = "flstm" },
		"geometry":      func(c *GPUCapture) { c.Stride = 0 },
		"chunk-gap":     func(c *GPUCapture) { c.Chunks[1].Index = 2 },
		"chunk-coord":   func(c *GPUCapture) { c.Chunks[1].Offset = 34 },
		"chunk-gran":    func(c *GPUCapture) { c.Chunks[0].Granularity = 4 },
		"chunk-quality": func(c *GPUCapture) { c.Chunks[0].Qstring = "II" },
		"chunk-moves":   func(c *GPUCapture) { c.Chunks[0].Moves = nil },
		"no-chunks":     func(c *GPUCapture) { c.Chunks = nil },
		"no-signal":     func(c *GPUCapture) { c.Read.NormalizedSignal = nil },
		"read-id":       func(c *GPUCapture) { c.Read.ID = "not-a-uuid" },
		"kit":           func(c *GPUCapture) { c.Read.SequencingKit = "SQK-NOPE" },
		"flowcell":      func(c *GPUCapture) { c.Read.FlowCellProductCode = "FLO-NOPE" },
		"policy":        func(c *GPUCapture) { c.Policy.MinQ = 7 },
	}
	for name, mutate := range mutants {
		c := base
		c.Chunks = append([]CapturedChunk(nil), base.Chunks...)
		c.Read.NormalizedSignal = append([]uint16(nil), base.Read.NormalizedSignal...)
		mutate(&c)
		if _, err := c.Seal(); err == nil {
			t.Fatalf("%s mutant sealed", name)
		}
	}
}

func TestGPUCaptureRejectsTamperingAndLooseJSON(t *testing.T) {
	c := gpuCaptureFixture(t)
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	tampered := append([]byte(nil), data...)
	tampered[len(tampered)-3] ^= 1
	if _, err := DecodeGPUCapture(tampered); err == nil {
		t.Fatal("tampered capture accepted")
	}
	unknown := append([]byte(`{"extra":1,`), data[1:]...)
	if _, err := DecodeGPUCapture(unknown); err == nil {
		t.Fatal("unknown capture field accepted")
	}
	if _, err := DecodeGPUCapture(append(data, '{', '}')); err == nil {
		t.Fatal("trailing capture JSON accepted")
	}
	c.Digest = ""
	if _, err := DecodeGPUCapture(mustJSON(t, c)); err == nil {
		t.Fatal("unsealed capture accepted")
	}
	other := c
	other.Digest = strings.Repeat("b", 64)
	if _, err := DecodeGPUCapture(mustJSON(t, other)); err == nil {
		t.Fatal("wrong digest accepted")
	}
}

func TestGPUCaptureReplayRejectsPolicySwap(t *testing.T) {
	c := gpuCaptureFixture(t)
	policy := c.Policy
	policy.Split = false
	if _, err := ReplayGPUCapture(c, policy); err == nil {
		t.Fatal("policy swap accepted")
	}
}

func TestGPUCaptureReplayRejectsBadCoordinates(t *testing.T) {
	c := gpuCaptureFixture(t)
	c.Chunks[1].Offset = 7
	sealed, err := c.Seal()
	if err != nil {
		// Seal may reject unaligned overlap directly; either rejection is valid.
		return
	}
	if _, err := sealed.Assemble(); err == nil {
		t.Fatal("unaligned overlap accepted")
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
