//go:build cgo

package caller

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestProcessedManifestPublication(t *testing.T) {
	dir := t.TempDir()
	m := processedManifest{Schema: processedSchema, SourceCommit: postcallSource, InputReads: 1, OutputReads: 2}
	if err := publishProcessedManifest(dir, m); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "processed-manifest.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got processedManifest
	if err := strictJSON(data, &got); err != nil || !reflect.DeepEqual(got, m) {
		t.Fatalf("published manifest differs: %+v, %v", got, err)
	}
	expected, err := json.MarshalIndent(m, "", "  ")
	if err != nil || !bytes.Equal(data, append(expected, '\n')) {
		t.Fatal("published bytes differ", err)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0600 {
		t.Fatal("manifest is not a private regular file", info, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatal("publication left temporary files", entries, err)
	}
	if err := publishProcessedManifest(dir, processedManifest{Schema: "replacement"}); !errors.Is(err, os.ErrExist) {
		t.Fatal("existing final manifest was not rejected", err)
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(after, data) {
		t.Fatal("existing final manifest changed", err)
	}
	entries, err = os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatal("rejected publication left temporary files", entries, err)
	}
}

func TestProcessedManifestConcurrentPublication(t *testing.T) {
	dir := t.TempDir()
	const publishers = 16
	start := make(chan struct{})
	errs := make([]error, publishers)
	var workers sync.WaitGroup
	for i := range errs {
		workers.Add(1)
		go func(i int) {
			defer workers.Done()
			<-start
			errs[i] = publishProcessedManifest(dir, processedManifest{Schema: processedSchema, InputReads: 1, OutputReads: i})
		}(i)
	}
	close(start)
	workers.Wait()
	winner := -1
	for i, err := range errs {
		if err == nil {
			if winner != -1 {
				t.Fatal("multiple publications succeeded")
			}
			winner = i
		} else if !errors.Is(err, os.ErrExist) {
			t.Fatal("unexpected publication failure", err)
		}
	}
	if winner == -1 {
		t.Fatal("no publication succeeded")
	}
	data, err := os.ReadFile(filepath.Join(dir, "processed-manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var got processedManifest
	if err := strictJSON(data, &got); err != nil || got != (processedManifest{Schema: processedSchema, InputReads: 1, OutputReads: winner}) {
		t.Fatal("final manifest is not the complete winner", got, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatal("concurrent publication left temporary files", entries, err)
	}
}

func TestProcessRejectsBeforeInferenceAndClosed(t *testing.T) {
	p := postcall.DefaultPolicy()
	var c Caller
	if _, err := c.Process(context.Background(), testRead(), p); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	r := testRead()
	r.SequencingKit = "SQK-RAD114"
	c.engine = &tensor.Engine{}
	if _, err := c.Process(context.Background(), r, p); err == nil || !strings.Contains(err.Error(), "postprocessing supports") {
		t.Fatal("kit reached inference", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Process(ctx, testRead(), p); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestStagePostMuxBoundaryAndRootWriterMetadata(t *testing.T) {
	r := testRead()
	r.StartSample = 5001
	r.AcquisitionStartTimeMS = 1000
	r.ProtocolStartTimeMS = 900
	r.EndReason = 1
	r.EndReasonName = "mux_change"
	m, err := fastqMetadata(r, testSpec())
	if err != nil {
		t.Fatal(err)
	}
	s := coreStage{Input: r, Stride: 6, Normalized: signal.Normalized{Samples: make([]uint16, 66), Trimmed: 10}, CalledRead: CalledRead{ID: r.ID.String(), Sequence: strings.Repeat("A", 10), Qstring: strings.Repeat("I", 10), Moves: bytes.Repeat([]byte{1}, 10), NumSamples: 76, Trimmed: 10, TailTrimmedSamples: 6, FASTQ: m}}
	p := postcall.DefaultPolicy()
	p.Trim, p.Split = false, false
	out, err := s.process(p)
	if err != nil {
		t.Fatal(err)
	}
	child := out.Postcall.Children[0].Read
	if len(child.Signal) != 60 || child.NumEvents != 0 || child.PreTrimSeqLen != 10 || child.LeadingTrim != 10 || child.StartTimeMS != 1000 || out.Reads[0].FASTQ.StartTimeMS != 2000 {
		t.Fatalf("post-MUX boundary: %+v", child)
	}
	if out.Reads[0].FASTQ.TrimFlags != 0 || m.TrimFlags != 0 {
		t.Fatal("metadata mutated")
	}
	p.Trim = true
	out, err = s.process(p)
	if err != nil {
		t.Fatal(err)
	}
	if out.Reads[0].FASTQ.TrimFlags != TrimAdapter|TrimPrimer || m.TrimFlags != 0 {
		t.Fatal("requested trim flags not isolated")
	}
}

func TestProcessedCaptureRoundTripZeroOutputAndImmutableCore(t *testing.T) {
	core, _ := syntheticCapture(t)
	original, err := os.ReadFile(core)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "post")
	got, err := ReplayProcessed(context.Background(), core, postcall.DefaultPolicy(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Reads) != 0 || len(got.Postcall.Children) != 1 || got.Postcall.Children[0].Filter.Keep {
		t.Fatal("short core not explicitly filtered")
	}
	reads, err := ReplayRecords(context.Background(), filepath.Join(dir, "processed-manifest.json"))
	if err != nil || len(reads) != 0 {
		t.Fatal(reads, err)
	}
	retained, err := os.ReadFile(core)
	if err != nil || !bytes.Equal(retained, original) {
		t.Fatal("core changed", err)
	}
	if _, err := ReplayProcessed(context.Background(), core, postcall.DefaultPolicy(), dir); err == nil {
		t.Fatal("capture reused")
	}
	baseline, err := ReplayRecords(context.Background(), core)
	if err != nil || len(baseline) != 1 {
		t.Fatal("core replay changed", err)
	}
}

func TestProcessedCaptureRejectsTampering(t *testing.T) {
	for _, change := range []string{"missing", "malformed", "payload", "output-hash", "fastq-hash", "policy", "policy-path", "core-path", "core-hash", "core-swapped", "count", "unknown", "trailing", "symlink", "core-symlink"} {
		t.Run(change, func(t *testing.T) {
			core, _ := syntheticCapture(t)
			dir := filepath.Join(t.TempDir(), "post")
			if _, err := ReplayProcessed(context.Background(), core, postcall.DefaultPolicy(), dir); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, "processed-manifest.json")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var m processedManifest
			if err := json.Unmarshal(data, &m); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "missing":
				if err := os.Remove(filepath.Join(dir, m.Output.File)); err != nil {
					t.Fatal(err)
				}
			case "malformed":
				if err := os.WriteFile(path, []byte("{"), 0600); err != nil {
					t.Fatal(err)
				}
			case "payload":
				if err := os.WriteFile(filepath.Join(dir, m.Output.File), []byte("{}"), 0600); err != nil {
					t.Fatal(err)
				}
			case "output-hash":
				m.Output.SHA256 = strings.Repeat("0", 64)
			case "fastq-hash":
				m.FASTQ.SHA256 = strings.Repeat("0", 64)
			case "policy":
				p := postcall.DefaultPolicy()
				p.Trim = false
				encoded, _ := json.Marshal(p)
				if err := os.WriteFile(filepath.Join(dir, m.Policy.File), encoded, 0600); err != nil {
					t.Fatal(err)
				}
				m.Policy.SHA256, m.Policy.Bytes = digest(encoded), int64(len(encoded))
			case "policy-path":
				m.Policy.File = "../processed-policy.json"
			case "core-path":
				m.Core.Path = filepath.Dir(m.Core.Path) + "/../" + filepath.Base(filepath.Dir(m.Core.Path)) + "/manifest.json"
			case "core-hash":
				m.Core.SHA256 = strings.Repeat("0", 64)
			case "core-swapped":
				other, _ := syntheticCapture(t)
				m.Core.Path = other
				m.Core.SHA256 = strings.Repeat("f", 64)
			case "count":
				m.OutputReads = 1
			case "unknown":
				data = append(data[:len(data)-2], []byte(",\"unknown\":true}\n")...)
			case "trailing":
				data = append(data, []byte("{}")...)
			case "symlink":
				target := filepath.Join(dir, m.Output.File)
				if err := os.Rename(target, target+".saved"); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target+".saved", target); err != nil {
					t.Fatal(err)
				}
			case "core-symlink":
				target := m.Core.Path
				if err := os.Rename(target, target+".saved"); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target+".saved", target); err != nil {
					t.Fatal(err)
				}
			}
			if change != "malformed" {
				if change != "unknown" && change != "trailing" {
					data, err = json.Marshal(m)
					if err != nil {
						t.Fatal(err)
					}
				}
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			reads, err := ReplayRecords(context.Background(), path)
			if err == nil || len(reads) != 0 {
				t.Fatalf("accepted %s or returned partial records: %v", change, err)
			}
		})
	}
}

func TestProcessedCaptureRejectsChangedRehashedEndpoint(t *testing.T) {
	core, _ := syntheticCapture(t)
	dir := filepath.Join(t.TempDir(), "post")
	if _, err := ReplayProcessed(context.Background(), core, postcall.DefaultPolicy(), dir); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "processed-manifest.json")
	data, _ := os.ReadFile(path)
	var m processedManifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	outputData, _ := readPayload(dir, m.Output)
	var out Processed
	if err := json.Unmarshal(outputData, &out); err != nil {
		t.Fatal(err)
	}
	out.Postcall.Children[0].Read.NumEvents++
	altered, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, m.Output.File), altered, 0600); err != nil {
		t.Fatal(err)
	}
	m.Output.SHA256, m.Output.Bytes = digest(altered), int64(len(altered))
	data, _ = json.Marshal(m)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReplayRecords(context.Background(), path); err == nil {
		t.Fatal("rehashed changed metadata accepted")
	}
	if reflect.DeepEqual(out.Postcall.Policy, postcall.Policy{}) {
		t.Fatal("policy absent")
	}
}
