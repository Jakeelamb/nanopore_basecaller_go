//go:build cgo && linux

package pod5

import (
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
	"strings"
	"sync"
	"testing"
)

type oracleRead struct {
	ID           string            `json:"id"`
	SignalFile   string            `json:"signal_file"`
	SignalSHA256 string            `json:"signal_sha256"`
	Metadata     map[string]any    `json:"metadata"`
	FloatBits    map[string]uint32 `json:"float_bits"`
	NumSamples   uint64            `json:"num_samples"`
}
type oracleFixture struct {
	Name                string       `json:"name"`
	SourcePath          string       `json:"source_path"`
	SourceSHA256        string       `json:"source_sha256"`
	FileReadCount       int          `json:"file_read_count"`
	FileID              string       `json:"file_id"`
	FileOriginalVersion [3]uint16    `json:"file_original_version"`
	FileIDs             []string     `json:"file_ids"`
	Reads               []oracleRead `json:"reads"`
}
type oracleManifest struct {
	Schema          int             `json:"schema"`
	Producer        string          `json:"producer"`
	NativeProducer  string          `json:"lib_pod5_version"`
	GeneratorSHA256 string          `json:"generator_sha256"`
	Fixtures        []oracleFixture `json:"fixtures"`
}

type sourcePins struct {
	Native struct {
		LibrarySHA256      string `json:"library_sha256"`
		HeaderSHA256       string `json:"c_api_header_sha256"`
		ExportHeaderSHA256 string `json:"export_header_sha256"`
	} `json:"native"`
	Fixtures []oracleFixture `json:"fixtures"`
}

func fileHash(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}
func nativeFixtures(t *testing.T) (string, string, oracleManifest) {
	t.Helper()
	library, dir := os.Getenv("NANOPORE_POD5_LIBRARY"), os.Getenv("POD5_FIXTURE_DIR")
	if library == "" || dir == "" {
		t.Skip("native POD5 parity requires NANOPORE_POD5_LIBRARY and POD5_FIXTURE_DIR; see docs/pod5-reader.md")
	}
	pinData, err := os.ReadFile("../../verification/pod5/source-pin.json")
	if err != nil {
		t.Fatal(err)
	}
	var pins sourcePins
	if err := json.Unmarshal(pinData, &pins); err != nil {
		t.Fatal(err)
	}
	if len(pins.Fixtures) != 2 {
		t.Fatal("unexpected pinned fixture count")
	}
	if got := fileHash(t, library); got != pins.Native.LibrarySHA256 {
		t.Fatalf("native POD5 0.3.39 shared-library pin mismatch: %s", got)
	}
	if fileHash(t, "pod5_format/c_api.h") != pins.Native.HeaderSHA256 || fileHash(t, "pod5_format/pod5_format_export.h") != pins.Native.ExportHeaderSHA256 {
		t.Fatal("vendored POD5 interface header pin mismatch")
	}
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m oracleManifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m.Schema != 1 || m.Producer != "pod5==0.3.49" || m.NativeProducer != "0.3.49" || len(m.Fixtures) != 2 {
		t.Fatalf("unexpected independent oracle manifest: schema %d, %q, fixtures %d", m.Schema, m.Producer, len(m.Fixtures))
	}
	if got := fileHash(t, "../../verification/pod5/generate.py"); got != m.GeneratorSHA256 {
		t.Fatalf("oracle producer source pin mismatch: %s", got)
	}
	for i, f := range m.Fixtures {
		pin := pins.Fixtures[i]
		if f.Name != pin.Name || f.SourceSHA256 != pin.SourceSHA256 || f.FileReadCount != pin.FileReadCount || f.FileID != pin.FileID || f.FileOriginalVersion != pin.FileOriginalVersion || len(f.Reads) != 1 {
			t.Fatalf("fixture differs from source pins: %s", f.Name)
		}
		if len(pin.Reads) != 1 || f.Reads[0].ID != pin.Reads[0].ID || f.Reads[0].SignalSHA256 != pin.Reads[0].SignalSHA256 || !reflect.DeepEqual(f.Reads[0].FloatBits, pin.Reads[0].FloatBits) || f.Reads[0].Metadata["NumSamples"] != float64(pin.Reads[0].NumSamples) {
			t.Fatalf("selected read differs from source pins: %s", f.Name)
		}
		if got := fileHash(t, f.SourcePath); got != f.SourceSHA256 {
			t.Fatalf("source pin mismatch %s: %s", f.Name, got)
		}
	}
	return library, dir, m
}
func openFixture(t *testing.T, library string, f oracleFixture, opts ...Options) *Reader {
	t.Helper()
	r, err := Open(library, f.SourcePath, opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	return r
}
func verifyRead(t *testing.T, read Read, want oracleRead, dir string) {
	t.Helper()
	id, err := ParseID(want.ID)
	if err != nil {
		t.Fatal(err)
	}
	if read.ID != id {
		t.Fatalf("read ID %s != %s", read.ID, id)
	}
	bytes := make([]byte, 2*len(read.Signal))
	for i, x := range read.Signal {
		binary.LittleEndian.PutUint16(bytes[i*2:], uint16(x))
	}
	digest := sha256.Sum256(bytes)
	if hex.EncodeToString(digest[:]) != want.SignalSHA256 {
		t.Fatalf("signal SHA256 %x != %s", digest, want.SignalSHA256)
	}
	expected, err := os.ReadFile(filepath.Join(dir, want.SignalFile))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(bytes, expected) {
		t.Fatal("raw int16 samples differ from independent producer")
	}
	floats := map[string]uint32{"CalibrationScale": math.Float32bits(read.CalibrationScale), "CalibrationOffset": math.Float32bits(read.CalibrationOffset), "OpenPoreLevel": math.Float32bits(read.OpenPoreLevel), "MedianBefore": math.Float32bits(read.MedianBefore)}
	if !reflect.DeepEqual(floats, want.FloatBits) {
		t.Fatalf("float32 bits differ: got %v, want %v", floats, want.FloatBits)
	}
	read.Signal = nil
	read.CalibrationScale = 0
	read.CalibrationOffset = 0
	read.OpenPoreLevel = 0
	read.MedianBefore = 0
	payload, err := json.Marshal(read)
	if err != nil {
		t.Fatal(err)
	}
	var actual map[string]any
	if err := json.Unmarshal(payload, &actual); err != nil {
		t.Fatal(err)
	}
	for key, value := range want.Metadata {
		if !reflect.DeepEqual(actual[key], value) {
			t.Errorf("metadata %s differs: got %v, want %v", key, actual[key], value)
		}
	}
	if read.NumSamples != uint64(len(expected)/2) {
		t.Fatal("num_samples differs from raw signal length")
	}
	t.Logf("read=%s samples=%d raw_i16le_sha256=%x calibration_scale_bits=%08x calibration_offset_bits=%08x open_pore_level_bits=%08x", id, read.NumSamples, digest, floats["CalibrationScale"], floats["CalibrationOffset"], floats["OpenPoreLevel"])
}
func TestNativeParity(t *testing.T) {
	library, dir, m := nativeFixtures(t)
	for _, f := range m.Fixtures {
		t.Run(f.Name, func(t *testing.T) {
			r := openFixture(t, library, f)
			meta, err := r.Metadata(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if meta.ID.String() != f.FileID || meta.OriginalVersion != f.FileOriginalVersion || meta.NumReads != uint64(f.FileReadCount) || meta.Filename != filepath.Base(f.SourcePath) {
				t.Fatalf("file metadata differs: %+v", meta)
			}
			if len(meta.Runs) != 1 {
				t.Fatalf("expected exactly one run in bounded fixture, got %d", len(meta.Runs))
			}
			ids, err := r.IDs(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if len(ids) != f.FileReadCount {
				t.Fatalf("file count %d != %d", len(ids), f.FileReadCount)
			}
			for i, id := range ids {
				if id.String() != f.FileIDs[i] {
					t.Fatalf("storage UUID order mismatch at %d", i)
				}
			}
			for _, want := range f.Reads {
				id, err := ParseID(want.ID)
				if err != nil {
					t.Fatal(err)
				}
				got, err := r.Read(context.Background(), id)
				if err != nil {
					t.Fatal(err)
				}
				verifyRead(t, got, want, dir)
				if !reflect.DeepEqual(got.RunInfo, meta.Runs[0]) {
					t.Fatal("file and read run metadata disagree")
				}
				if got.Filename != filepath.Base(f.SourcePath) {
					t.Fatal("filename mismatch")
				}
				if got.ReadTableVersion < 3 || got.ReadTableVersion > 4 {
					t.Fatalf("unexpected read table version %d", got.ReadTableVersion)
				}
			}
		})
	}
}
func TestNativeLifetimeAndSelection(t *testing.T) {
	library, dir, m := nativeFixtures(t)
	f := m.Fixtures[0]
	ctx := context.Background()
	r := openFixture(t, library, f)
	other := openFixture(t, library, f)
	id, err := ParseID(f.Reads[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := r.IDs(ctx)
	if err != nil {
		t.Fatal(err)
	}
	ids[0] = ID{}
	first, err := r.Read(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	saved := append([]int16(nil), first.Signal...)
	first.Signal[0]++
	first.TrackingID["run_id"] = "mutated"
	second, err := r.Read(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	verifyRead(t, second, f.Reads[0], dir)
	if !reflect.DeepEqual(second.Signal, saved) {
		t.Fatal("read aliases an earlier signal")
	}
	visits := 0
	if err := r.Iterate(ctx, []ID{id, id}, func(read Read) error {
		visits++
		if read.ID != id {
			t.Fatal("selection identity mismatch")
		}
		return nil
	}); err != nil || visits != 2 {
		t.Fatalf("selection order/duplicates: %d %v", visits, err)
	}
	if err := r.Iterate(ctx, []ID{}, func(Read) error { t.Fatal("empty selection visited a read"); return nil }); err != nil {
		t.Fatal(err)
	}
	if err := r.Iterate(ctx, []ID{id, {}}, func(Read) error { t.Fatal("missing UUID should fail before callbacks"); return nil }); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing selection: %v", err)
	}
	if _, err := r.Read(ctx, ID{}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing UUID: %v", err)
	}
	stop := errors.New("callback stop")
	if err := r.Iterate(ctx, nil, func(Read) error { return stop }); !errors.Is(err, stop) {
		t.Fatalf("callback error: %v", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := r.Read(canceled, id); !errors.Is(err, context.Canceled) {
		t.Fatalf("read cancellation: %v", err)
	}
	if _, err := r.IDs(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("IDs cancellation: %v", err)
	}
	if _, err := r.Metadata(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("metadata cancellation: %v", err)
	}
	if err := r.Iterate(canceled, nil, func(Read) error { t.Fatal("canceled iteration callback"); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("iteration cancellation: %v", err)
	}
	between, cancelBetween := context.WithCancel(ctx)
	defer cancelBetween()
	visits = 0
	if err := r.Iterate(between, []ID{id, id}, func(Read) error { visits++; cancelBetween(); return nil }); !errors.Is(err, context.Canceled) || visits != 1 {
		t.Fatalf("between read cancellation: %d %v", visits, err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(ctx, id); !errors.Is(err, ErrClosed) {
		t.Fatalf("stale reader: %v", err)
	}
	if _, err := r.IDs(ctx); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed IDs: %v", err)
	}
	if _, err := r.Metadata(ctx); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed metadata: %v", err)
	}
	// Native run info and batches have been freed, and the first reader is closed.
	verifyRead(t, second, f.Reads[0], dir)
	got, err := other.Read(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	verifyRead(t, got, f.Reads[0], dir)
	if err := other.Iterate(ctx, nil, func(Read) error { return other.Close() }); err != nil {
		t.Fatal(err)
	}
}
func TestNativeRejections(t *testing.T) {
	library, _, m := nativeFixtures(t)
	for _, contents := range [][]byte{nil, []byte("not a POD5 file")} {
		path := filepath.Join(t.TempDir(), "malformed.pod5")
		if err := os.WriteFile(path, contents, 0600); err != nil {
			t.Fatal(err)
		}
		if r, err := Open(library, path); err == nil {
			r.Close()
			t.Fatal("accepted malformed POD5")
		}
	}
	for _, f := range m.Fixtures {
		r := openFixture(t, library, f, Options{MaxSamples: 1})
		if _, err := r.Metadata(context.Background()); err != nil {
			t.Fatalf("metadata must not decode signals: %v", err)
		}
		id, err := ParseID(f.Reads[0].ID)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.Read(context.Background(), id); err == nil || !strings.Contains(err.Error(), "MaxSamples") {
			t.Fatalf("sample bound rejection: %v", err)
		}
		if f.FileReadCount > 1 {
			if r, err := Open(library, f.SourcePath, Options{MaxReads: 1}); err == nil {
				r.Close()
				t.Fatal("accepted file beyond MaxReads")
			}
		}
	}
}
func TestNativeConcurrentClose(t *testing.T) {
	library, _, m := nativeFixtures(t)
	f := m.Fixtures[0]
	r := openFixture(t, library, f)
	id, err := ParseID(f.Reads[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for i := 0; i < 8; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			_, err := r.Read(context.Background(), id)
			if err != nil && !errors.Is(err, ErrClosed) {
				t.Error(err)
			}
		}()
	}
	group.Add(1)
	go func() {
		defer group.Done()
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()
	group.Wait()
}
func TestNativeMissingLibrary(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "missing.so"), "unused.pod5"); err == nil {
		t.Fatal("accepted missing native library")
	}
}
func ExampleID_String() {
	id, err := ParseID("bb76c31a-36f0-48a0-abd2-107a10e1ade0")
	if err != nil {
		panic(err)
	}
	fmt.Println(id.String())
	// Output: bb76c31a-36f0-48a0-abd2-107a10e1ade0
}
