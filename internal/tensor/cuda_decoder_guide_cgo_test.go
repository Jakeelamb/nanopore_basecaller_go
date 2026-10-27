//go:build cgo && linux

package tensor

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDecoderGuideProgramAndFrameMarshalThroughActualLoader(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	program, err := d.DecoderGuideProgram()
	if err != nil || string(program) != "fake" {
		t.Fatalf("program=%q err=%v", program, err)
	}
	newRows := func(dtype DType, cols int64, data []byte) *Storage {
		v, err := d.NewStorage(s, dtype, []int64{64, cols}, []KoiTag{TagN, TagC}, data)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := v.Close(); err != nil {
				t.Error(err)
			}
		})
		return v
	}
	scoresData := make([]byte, 64*1024*2)
	binary.LittleEndian.PutUint16(scoresData, 0x3400)
	nextData := make([]byte, 64*256*4)
	binary.LittleEndian.PutUint32(nextData, math.Float32bits(1.5))
	scores := newRows(DTypeF16, 1024, scoresData)
	next := newRows(DTypeF32, 256, nextData)
	out := newRows(DTypeF32, 256, make([]byte, 64*256*4))
	if err = d.DecoderGuideFrame(s, scores, next, out, 1.25, -.75); err != nil {
		t.Fatal(err)
	}
	got, err := out.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1.25, -.75, float32(0x3400), 1.5} {
		if binary.LittleEndian.Uint32(got[i*4:]) != math.Float32bits(want) {
			t.Fatalf("marshalled field %d differs", i)
		}
	}
	if err = d.DecoderGuideFrame(s, scores, next, next, 1.25, -.75); err == nil {
		t.Fatal("overlapping writable row accepted")
	}
	if err = out.Close(); err != nil {
		t.Fatal(err)
	}
	if err = d.DecoderGuideFrame(s, scores, next, out, 1.25, -.75); err == nil {
		t.Fatal("closed guide output accepted")
	}
}
