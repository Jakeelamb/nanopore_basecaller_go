//go:build cgo

package caller

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/crf"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
)

func syntheticCapture(t *testing.T) (string, CalledRead) {
	t.Helper()
	root := t.TempDir()
	s := testSpec()
	cc, opt, err := configuration(s, Config{ChunkSize: 6})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../model/hac_weights.json")
	if err != nil {
		t.Fatal(err)
	}
	var weights []model.WeightReceipt
	if err := json.Unmarshal(data, &weights); err != nil {
		t.Fatal(err)
	}
	config, err := os.ReadFile("../../verification/caller/testdata/hac-config.toml")
	if err != nil {
		t.Fatal(err)
	}
	configFile, err := save(root, "model-config.toml", config)
	if err != nil {
		t.Fatal(err)
	}
	run, err := saveJSON(root, "run.json", runPins{Schema: 1, Spec: s, Weights: weights, TensorLibrarySHA256: "0000000000000000000000000000000000000000000000000000000000000000", ModelConfig: configFile, Chunks: cc, Decode: opt})
	if err != nil {
		t.Fatal(err)
	}
	r := testRead()
	n, err := signal.Normalize(r.Signal, signal.Calibration{Scale: r.CalibrationScale, Offset: r.CalibrationOffset}, signal.Standardisation{Mean: s.Mean, Stdev: s.Stdev}, 0)
	if err != nil {
		t.Fatal(err)
	}
	c, err := beginRead(root, run, r, n)
	if err != nil {
		t.Fatal(err)
	}
	var chunks []stitch.Chunk
	for i := 0; i < 2; i++ {
		result := crf.Result{Blocks: 1, States: 256, Scores: make([]float32, 1024), Forward: make([]float32, 512), Backward: make([]float32, 512), Posts: make([]float32, 512)}
		for i := range result.Posts {
			result.Posts[i] = 1.0 / 256
		}
		d, err := decode.Decode(result.DecodeInput(), opt)
		if err != nil {
			t.Fatal(err)
		}
		dc := stitch.Chunk{Offset: uint64(i * 6), RawSize: 6, Granularity: 6, Sequence: d.Sequence, Qstring: d.Qstring, Moves: d.Moves}
		chunks = append(chunks, dc)
		if err := c.addChunk(i, dc, result); err != nil {
			t.Fatal(err)
		}
	}
	joined, err := stitch.Stitch(stitch.Config{RawSamples: 12, Stride: 6}, chunks)
	if err != nil {
		t.Fatal(err)
	}
	out := CalledRead{ID: r.ID.String(), Sequence: joined.Sequence, Qstring: joined.Qstring, Moves: joined.Moves, NumSamples: 22, Trimmed: 10}
	out.FASTQ, err = fastqMetadata(r, s)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.finish(out); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(c.dir, "manifest.json"), out
}
func TestReplaySyntheticAndTamper(t *testing.T) {
	path, want := syntheticCapture(t)
	got, err := Replay(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("replay endpoint differs")
	}
	payloadPath := filepath.Join(filepath.Dir(path), "chunk-0000-scores.f32le")
	data, err := os.ReadFile(payloadPath)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	if err := os.WriteFile(payloadPath, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Replay(context.Background(), path); err == nil {
		t.Fatal("tampered scores accepted")
	}
}
func TestReplayManifestShapeAndCancellation(t *testing.T) {
	for _, change := range []string{"shape", "path", "count", "model"} {
		t.Run(change, func(t *testing.T) {
			path, _ := syntheticCapture(t)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var m manifest
			if err := json.Unmarshal(data, &m); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "shape":
				m.Chunks[0].States = 16384
			case "path":
				m.Chunks[0].Scores.File = "../run.json"
			case "count":
				m.Chunks = m.Chunks[:1]
			case "model":
				m.Run.SHA256 = "bad"
			}
			data, err = json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := Replay(context.Background(), path); err == nil {
				t.Fatalf("accepted %s tamper", change)
			}
		})
	}
	path, _ := syntheticCapture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Replay(ctx, path); err == nil {
		t.Fatal("canceled replay accepted")
	}
}
func TestCaptureDoesNotOverwrite(t *testing.T) {
	dir := t.TempDir()
	if _, err := save(dir, "same", []byte("first")); err != nil {
		t.Fatal(err)
	}
	if _, err := save(dir, "same", []byte("second")); err == nil {
		t.Fatal("capture overwrote prior output")
	}
	if _, err := startCapture(Config{CaptureDir: dir}, testSpec(), nil, chunk.Config{}, decode.Options{}); err == nil {
		t.Fatal("capture reused directory")
	}
}

func TestReplayLegacyMetadataAndChangedNewMetadata(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		path, want := syntheticCapture(t)
		dir := filepath.Dir(path)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var m manifest
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatal(err)
		}
		changed := want
		if legacy {
			changed.FASTQ = nil
		} else {
			copied := *want.FASTQ
			copied.Channel++
			changed.FASTQ = &copied
		}
		encoded, err := json.Marshal(changed)
		if err != nil {
			t.Fatal(err)
		}
		m.Output = payload{File: "output.json", SHA256: digest(encoded), Bytes: int64(len(encoded))}
		outputPath := filepath.Join(dir, m.Output.File)
		if err := os.WriteFile(outputPath, encoded, 0600); err != nil {
			t.Fatal(err)
		}
		data, err = json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		got, replayErr := Replay(context.Background(), path)
		if legacy {
			if replayErr != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("legacy metadata derivation: %v", replayErr)
			}
			retained, err := os.ReadFile(outputPath)
			if err != nil {
				t.Fatal(err)
			}
			if string(retained) != string(encoded) {
				t.Fatal("replay rewrote legacy output")
			}
		} else if replayErr == nil {
			t.Fatal("changed captured FASTQ metadata accepted")
		}
	}
}
