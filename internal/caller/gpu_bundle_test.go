package caller

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func TestGPUSourceProfilesAndStrictBoundary(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		spec, _ := model.Pinned(architecture)
		for _, mode := range []Chunking{FixedChunks, VariableChunks} {
			cfg := GPUConfig{BatchSize: 64, ChunkSize: 12288, Overlap: 600, Chunking: mode}
			profiles, err := gpuSourceProfiles(spec, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if architecture == model.SUP && mode == VariableChunks {
				if len(profiles) != 1 || profiles[0].chunkSize != 12288 {
					t.Fatal(profiles)
				}
				continue
			}
			if len(profiles) != 2 || profiles[0].chunkSize != 12288 || profiles[1].chunkSize != 6144 {
				t.Fatal(profiles)
			}
			for j, n := range []uint64{6143, 6144, 6145} {
				want := 0
				if j == 0 {
					want = 1
				}
				if got := gpuSelectProfile(profiles, n); got != want {
					t.Fatalf("%s mode%d normalized%d profile%d want%d", architecture, mode, n, got, want)
				}
			}
		}
	}
}

func TestGPUBundleHasNoFabricatedApproval(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "gpu-bundle.json"), []byte(`{"schema":"unqualified-source-only","guide":"self-hash"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGPUBundle(dir); !errors.Is(err, ErrGPUUnqualified) {
		t.Fatalf("unqualified startup accepted: %v", err)
	}
	if len(approvedGPUBundles) != 0 {
		t.Fatal("production bundle registry requires independent native qualification before this expectation changes")
	}
}

type gpuShortEOFWriter struct{ bytes.Buffer }

func (w *gpuShortEOFWriter) Write(p []byte) (int, error) {
	if len(p) == 28 {
		return len(p) - 1, nil
	}
	return w.Buffer.Write(p)
}

func TestGPUOutputRejectsShortBAMEOF(t *testing.T) {
	writer := &gpuShortEOFWriter{}
	cfg := GPUConfig{Format: BAM, Policy: postcall.DefaultPolicy(), CommandLine: "basecaller --device cuda:0"}
	metadata := pod5.FileMetadata{Runs: []pod5.RunInfo{{RunID: "run", AcquisitionStartTimeMS: 1000, ProtocolStartTimeMS: 900}}}
	w, err := newGPUOutput(writer, cfg, testSpec(), metadata)
	if err != nil {
		t.Fatal(err)
	}
	if err = w.finish(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("BAM EOF short write accepted: %v", err)
	}
	if err = w.finish(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal("output lost sticky EOF failure")
	}
}

type gpuAlwaysShortWriter struct{}

func (gpuAlwaysShortWriter) Write(p []byte) (int, error) { return len(p) / 2, nil }
func TestGPUDestinationRejectsNilErrorShortWrite(t *testing.T) {
	w := &gpuDestination{writer: gpuAlwaysShortWriter{}}
	if _, err := w.Write([]byte("record")); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if n, err := w.Write([]byte("next")); n != 0 || !errors.Is(err, io.ErrShortWrite) {
		t.Fatal("destination resumed after failure")
	}
}

func TestGPUBAMUsesChildLineageAndFloatBits(t *testing.T) {
	rg := bam.ReadGroupID("run", testSpec().Name, "")
	child := postcall.Record{Read: subread.Read{
		ID: "child", ParentID: "parent", SplitPoint: 18, Sequence: "ACGTN", Qstring: "IIIII", Moves: []uint8{1, 1, 1, 1, 1}, Stride: 6,
		NumSamples: 30, ReadNumber: -1, SampleRate: 5000, StartTime: "1970-01-01T00:00:01.000000+00:00", EndReason: "unknown",
		Inherited: subread.Inherited{RunID: "run", ModelName: testSpec().Name, ShiftBits: 0x42bc0001, ScaleBits: 0x41c00001, ScalingMethod: "pa", Mux: 2, Channel: 42, Filename: "reads.pod5"},
	}, Filter: readtrim.FilterResult{Keep: true, MeanQ: 37.5}}
	record, err := gpuBAMRecord(child, rg, true)
	if err != nil {
		t.Fatal(err)
	}
	if record.Name != "child" || record.Flag != 4 || !bytes.Equal(record.Quality, []byte{40, 40, 40, 40, 40}) {
		t.Fatal(record)
	}
	tags := make(map[string]any)
	for _, tag := range record.Aux {
		tags[tag.Name] = tag.Value
	}
	if tags["pi"] != "parent" || tags["sp"] != int32(18) || tags["rn"] != int32(-1) || tags["qs"] != float32(37.5) || tags["me"] != uint32(0) || tags["RG"] != rg {
		t.Fatal(tags)
	}
	if math.Float32bits(tags["sm"].(float32)) != child.Read.Inherited.ShiftBits || math.Float32bits(tags["sd"].(float32)) != child.Read.Inherited.ScaleBits {
		t.Fatal("BAM adapter rounded inherited metadata")
	}
	if !reflect.DeepEqual(tags["mv"], []byte{6, 1, 1, 1, 1, 1}) {
		t.Fatalf("moves changed: %v", tags["mv"])
	}
}

func TestGPUOutputRejectsMixedRunBeforeWriting(t *testing.T) {
	var dst bytes.Buffer
	_, err := newGPUOutput(&dst, GPUConfig{Format: BAM}, testSpec(), pod5.FileMetadata{Runs: []pod5.RunInfo{{RunID: "a"}, {RunID: "b"}}})
	if !errors.Is(err, ErrGPUUnqualified) || dst.Len() != 0 {
		t.Fatalf("mixed-run header was silently flattened: %v", err)
	}
}

func TestGPUOutputHeaderPreservesSourceRunFields(t *testing.T) {
	var dst bytes.Buffer
	policy := postcall.DefaultPolicy()
	policy.Trim = false
	w, err := newGPUOutput(&dst, GPUConfig{Format: BAM, Policy: policy, CommandLine: "basecaller --device cuda:0"}, testSpec(), pod5.FileMetadata{Runs: []pod5.RunInfo{{RunID: "run", SystemName: "instrument", FlowCellID: "flow", ProtocolStartTimeMS: 900, AcquisitionStartTimeMS: 1000, TrackingID: map[string]string{"protocol_group_id": "experiment"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if err = w.finish(); err != nil {
		t.Fatal(err)
	}
	z, err := gzip.NewReader(bytes.NewReader(dst.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	payload, err := io.ReadAll(z)
	if err != nil {
		t.Fatal(err)
	}
	if err = z.Close(); err != nil {
		t.Fatal(err)
	}
	if len(payload) < 8 || string(payload[:4]) != "BAM\x01" {
		t.Fatal("invalid BAM header")
	}
	n := int(binary.LittleEndian.Uint32(payload[4:8]))
	if n > len(payload)-8 {
		t.Fatal("truncated header")
	}
	header := string(payload[8 : 8+n])
	for _, field := range []string{"PN:nanopore_basecaller_go", "CL:basecaller --device cuda:0", "PM:instrument", "DT:1970-01-01T00:00:00.900000+00:00", "acquisition_start_time=1970-01-01T00:00:01.000000+00:00", "experiment_id=experiment", "tm:none"} {
		if !strings.Contains(header, field) {
			t.Fatalf("header lacks %q: %s", field, header)
		}
	}
}

func TestGPUDecodeExtractsOnlyLiveSequenceAndQuality(t *testing.T) {
	d := gpudecode.QualityChunk{Moves: []byte{1, 0, 1, 0}, Sequence: []byte{'A', 'C', 0xff, 0xff}, Q: []byte{'I', 'H', 0xff, 0xff}}
	sequence, q, err := gpuExtractChunk(d)
	if err != nil || sequence != "AC" || q != "IH" {
		t.Fatalf("opaque padded tail escaped: %q %q %v", sequence, q, err)
	}
	d.Moves[1] = 2
	if _, _, err = gpuExtractChunk(d); err == nil {
		t.Fatal("nonbinary move accepted")
	}
}
