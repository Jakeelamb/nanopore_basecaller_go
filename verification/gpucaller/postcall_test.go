//go:build linux

package gpucaller

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

func writeCapture(t *testing.T, dir string, mutate func(*caller.GPUCapture)) string {
	t.Helper()
	c := caller.GPUCapture{
		Schema: "gpu-capture/1",
		Runtime: caller.RuntimeIdentity{Producer: "gpu", Device: "RTX 5070 Laptop", Compiler: "nvcc 13.0.88",
			TensorABI: "abi4", SourceCommit: strings.Repeat("a", 40)},
		Lane: "hac", ChunkSize: 6, BatchSize: 64, Stride: 6,
		Read: caller.CapturedRead{
			ID: "bb76c31a-36f0-48a0-abd2-107a10e1ade0", NumSamples: 36, SampleRate: 5000, Channel: 42, Well: 1,
			RunID: "run", FlowCellID: "flow", FlowCellProductCode: "FLO-PRO114M", SequencingKit: "sqk-lsk114",
			AcquisitionStartTimeMS: 1000, ProtocolStartTimeMS: 900, CalibrationScale: .187, CalibrationOffset: -270,
			NormalizedSignal: make([]uint16, 36), NormalizedShift: 1, NormalizedScale: 2,
		},
		Policy: postcall.DefaultPolicy(),
	}
	for i, base := range "ACGTAC" {
		c.Chunks = append(c.Chunks, caller.CapturedChunk{Index: i, Offset: uint64(i * 6), RawLength: 6, Granularity: 6,
			Sequence: string(base), Qstring: "I", Moves: []uint8{1}})
	}
	if mutate != nil {
		mutate(&c)
	}
	sealed, err := c.Seal()
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(sealed)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "capture.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPostcallReplayWritesDeterministicReceipt(t *testing.T) {
	dir := t.TempDir()
	input := writeCapture(t, dir, nil)
	out := filepath.Join(dir, "attempt")
	if err := Dispatch(Request{Unit: PostcallCheck, Mode: Replay, Input: input, Output: out}, Handlers{PostcallCheck: PostcallHandlers()}); err != nil {
		t.Fatal(err)
	}
	fastq, err := os.ReadFile(filepath.Join(out, "reads.fastq"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(fastq), "@bb76c31a-36f0-48a0-abd2-107a10e1ade0") || !strings.Contains(string(fastq), "ACGTAC") {
		t.Fatalf("fastq %q", fastq)
	}
	body, err := os.ReadFile(filepath.Join(out, "postcall-receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt PostcallReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != PostcallReceiptSchema || receipt.Lane != "hac" || receipt.Model != "dna_r10.4.1_e8.2_400bps_hac@v6.0.0" ||
		receipt.Children != 1 || receipt.Reads != 1 || !receipt.Deterministic || len(receipt.CaptureDigest) != 64 {
		t.Fatalf("receipt %+v", receipt)
	}
	entries, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			t.Fatalf("temporary file retained: %s", e.Name())
		}
	}
}

func TestPostcallReplayRejectsBadInputs(t *testing.T) {
	dir := t.TempDir()
	input := writeCapture(t, dir, nil)
	existing := filepath.Join(dir, "existing")
	if err := os.Mkdir(existing, 0o755); err != nil {
		t.Fatal(err)
	}
	cases := []Request{
		{Unit: PostcallCheck, Mode: Replay, Input: input, Output: existing},
		{Unit: PostcallCheck, Mode: Replay, Input: filepath.Join(dir, "missing.json"), Output: filepath.Join(dir, "a")},
		{Unit: PostcallCheck, Mode: Native, Input: input, Output: filepath.Join(dir, "b")},
	}
	for i, r := range cases {
		if err := Dispatch(r, Handlers{PostcallCheck: PostcallHandlers()}); err == nil {
			t.Fatalf("case %d accepted", i)
		}
	}
	valid, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	cpu := filepath.Join(dir, "cpu.json")
	if err := os.WriteFile(cpu, bytes.Replace(valid, []byte(`"producer":"gpu"`), []byte(`"producer":"cpu"`), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Dispatch(Request{Unit: PostcallCheck, Mode: Replay, Input: cpu, Output: filepath.Join(dir, "cpu")}, Handlers{PostcallCheck: PostcallHandlers()}); err == nil {
		t.Fatal("CPU capture replayed as GPU evidence")
	}
	tampered := writeCapture(t, dir, nil)
	raw, err := os.ReadFile(tampered)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-3] ^= 1
	if err := os.WriteFile(tampered, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Dispatch(Request{Unit: PostcallCheck, Mode: Replay, Input: tampered, Output: filepath.Join(dir, "tampered")}, Handlers{PostcallCheck: PostcallHandlers()}); err == nil {
		t.Fatal("tampered capture accepted")
	}
}
