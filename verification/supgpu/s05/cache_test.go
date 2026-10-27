package s05

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func archiveFixture(t *testing.T, change func(map[string][]byte)) (string, model.WeightReceipt) {
	t.Helper()
	values := []float32{1, -0.25, 0, math.Float32frombits(0x80000000)}
	data := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	members := map[string][]byte{
		"data/0":   data,
		"data.pkl": []byte("\x80\x02c__torch__.torch.nn.modules.module\nModule\nctorch\nFloatStorage\n"),
		"code/__torch__/torch/nn/modules/module.py":           []byte("__parameters__ = [\"0\", ]\n__buffers__ = []"),
		"code/__torch__/torch/nn/modules/module.py.debug_pkl": {},
		"constants.pkl": {}, "version": []byte("3\n"), "byteorder": []byte("little"), ".data/serialization_id": []byte("fixture"),
	}
	if change != nil {
		change(members)
	}
	var raw bytes.Buffer
	w := zip.NewWriter(&raw)
	for name, data := range members {
		f, e := w.Create("matrix/" + name)
		if e != nil {
			t.Fatal(e)
		}
		if _, e = f.Write(data); e != nil {
			t.Fatal(e)
		}
	}
	if e := w.Close(); e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(t.TempDir(), "matrix.tensor")
	if e := os.WriteFile(path, raw.Bytes(), 0600); e != nil {
		t.Fatal(e)
	}
	return path, model.WeightReceipt{Name: "matrix.tensor", SHA256: digest(raw.Bytes()), Shape: []int64{2, 2}}
}

func TestPinnedMatrixArchiveHalfBits(t *testing.T) {
	path, p := archiveFixture(t, nil)
	got, e := readHalfArchive(path, p)
	if e != nil {
		t.Fatal(e)
	}
	want := []uint16{0x3c00, 0xb400, 0, 0x8000}
	if len(got) != len(want) {
		t.Fatal(got)
	}
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("element%d=%04x want%04x", i, v, want[i])
		}
	}
	p.SHA256 = strings.Repeat("0", 64)
	if _, e = readHalfArchive(path, p); e == nil {
		t.Fatal("untrusted archive bytes accepted")
	}
}

func TestPinnedMatrixArchiveRejectsOtherProfiles(t *testing.T) {
	cases := map[string]func(map[string][]byte){
		"byte-order": func(m map[string][]byte) { m["byteorder"] = []byte("big") },
		"version":    func(m map[string][]byte) { m["version"] = []byte("4\n") },
		"storage":    func(m map[string][]byte) { m["data.pkl"] = []byte("HalfStorage") },
		"parameter": func(m map[string][]byte) {
			m["code/__torch__/torch/nn/modules/module.py"] = []byte("__parameters__ = [\"other\"]")
		},
		"extra-data":    func(m map[string][]byte) { m["data/1"] = []byte{0} },
		"short-data":    func(m map[string][]byte) { m["data/0"] = m["data/0"][:4] },
		"nonfinite":     func(m map[string][]byte) { binary.LittleEndian.PutUint32(m["data/0"], 0x7fc00000) },
		"half-overflow": func(m map[string][]byte) { binary.LittleEndian.PutUint32(m["data/0"], 0x7f7fffff) },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			path, p := archiveFixture(t, change)
			v, e := readHalfArchive(path, p)
			if e == nil || v != nil {
				t.Fatalf("invalid profile returned data: %x %v", v, e)
			}
		})
	}
	_, e := readHalfArchive("missing", model.WeightReceipt{Name: "missing", SHA256: strings.Repeat("0", 64), Shape: []int64{1 << 23, 2}})
	if e == nil || !strings.Contains(e.Error(), "extent") {
		t.Fatalf("unbounded extent reached file read: %v", e)
	}
}

func TestCacheComparisonRejectsChangedBytesAndEquivalentExtent(t *testing.T) {
	native := []byte{1, 2, 3, 4}
	v := supgpu.Tensor{DType: "Float8_e4m3fn", Shape: []int64{2, 2}, Data: supgpu.Pin{Bytes: 4, SHA256: digest(native)}}
	if e := compareCache(native, []int64{2, 2}, v, native); e != nil {
		t.Fatal(e)
	}
	for _, c := range []struct {
		data   []byte
		shape  []int64
		native []byte
	}{
		{[]byte{1, 2, 3, 5}, []int64{2, 2}, native},
		{native, []int64{1, 4}, native},
		{native, []int64{2, 2}, []byte{1, 2, 3, 5}},
	} {
		if e := compareCache(c.data, c.shape, v, c.native); e == nil {
			t.Fatal("bad comparison accepted")
		}
	}
}

func TestCancelledRunHasNoSideEffects(t *testing.T) {
	out := filepath.Join(t.TempDir(), "untouched")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, e := Run(ctx, Options{Out: out})
	if e == nil || r.Attempt != "" {
		t.Fatalf("%+v %v", r, e)
	}
	if _, e = os.Stat(out); !os.IsNotExist(e) {
		t.Fatal("cancelled run created output")
	}
}

func TestCaptureAnchorNamesTheImportedManifest(t *testing.T) {
	dir := t.TempDir()
	actual := filepath.Join(dir, "capture.json")
	foreign := filepath.Join(dir, "other.json")
	for _, p := range []string{actual, foreign} {
		if e := os.WriteFile(p, []byte("same pinned bytes"), 0600); e != nil {
			t.Fatal(e)
		}
	}
	if got, e := captureDirectory(actual); e != nil || got != dir {
		t.Fatalf("%q %v", got, e)
	}
	if _, e := captureDirectory(foreign); e == nil {
		t.Fatal("different basename authenticated an unconsumed manifest")
	}
	alias := filepath.Join(t.TempDir(), "directory-alias")
	if e := os.Symlink(dir, alias); e != nil {
		t.Fatal(e)
	}
	if got, e := captureDirectory(filepath.Join(alias, "capture.json")); e != nil || got != dir {
		t.Fatalf("directory alias: %q %v", got, e)
	}
	if e := os.Remove(actual); e != nil {
		t.Fatal(e)
	}
	if e := os.Symlink(foreign, actual); e != nil {
		t.Fatal(e)
	}
	if _, e := captureDirectory(actual); e == nil {
		t.Fatal("capture.json symlink authenticated a different manifest")
	}
}
