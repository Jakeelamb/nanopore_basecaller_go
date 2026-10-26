//go:build cgo && linux

package tensor_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"math"
	"reflect"
	"testing"
)

func TestDecoderPostGoScanAndQualityAdapterAgainstHostProjectionOnly(t *testing.T) {
	e := tensor.DecoderPostFakeEngineForTest(t)
	d, er := e.NewCUDADevice(0)
	if er != nil {
		t.Fatal(er)
	}
	s, er := d.NewStream()
	if er != nil {
		t.Fatal(er)
	}
	t.Cleanup(func() {
		if e := s.Close(); e != nil {
			t.Error(e)
		}
		if e := d.Close(); e != nil {
			t.Error(e)
		}
	})
	n, er := gpudecode.NewGPUQualityNumbers(d, s, sha256.Sum256([]byte("fake")), 4096)
	if er != nil {
		t.Fatal(er)
	}
	if _, e := gpudecode.NewGPUQualityNumbers(d, s, sha256.Sum256([]byte("wrong")), 4096); e == nil {
		t.Fatal("wrong linked program admitted")
	}
	g, er := gpudecode.PlanGuides(64, 2, 256, 0, 2)
	if er != nil {
		t.Fatal(er)
	}
	scoresRaw := make([]byte, 64*2*1024*2)
	for i := 0; i < len(scoresRaw); i += 2 {
		binary.LittleEndian.PutUint16(scoresRaw[i:], 0x4000)
	}
	guidesRaw := make([]byte, 3*64*256*4)
	newStorage := func(dtype tensor.DType, shape []int64, tags []tensor.KoiTag, data []byte) *tensor.Storage {
		h, e := d.NewStorage(s, dtype, shape, tags, data)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() {
			if e := h.Close(); e != nil {
				t.Error(e)
			}
		})
		return h
	}
	scores := newStorage(tensor.DTypeF16, []int64{64, 2, 1024}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}, scoresRaw)
	guides := newStorage(tensor.DTypeF32, []int64{3, 64, 256}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}, guidesRaw)
	ctx := context.Background()
	working, er := gpudecode.Posteriors(ctx, d, s, scores, guides, g)
	if er != nil {
		t.Fatal(er)
	}
	t.Cleanup(func() {
		if e := working.Close(); e != nil {
			t.Error(e)
		}
	})
	got, er := working.Download(s)
	if er != nil {
		t.Fatal(er)
	}
	for chunk := 0; chunk < 64; chunk++ {
		for row, want := range []uint16{0, 0x2c00, 0x3000} {
			for state := 0; state < 256; state++ {
				if binary.LittleEndian.Uint16(got[chunk*4096+row*512+2*state:]) != want {
					t.Fatal("forward/post-old-alpha/final-row schedule differs")
				}
			}
		}
		if !reflect.DeepEqual(got[chunk*4096+1536:(chunk+1)*4096], scoresRaw[chunk*4096+1536:(chunk+1)*4096]) {
			t.Fatal("posterior score tail changed")
		}
	}
	raw, er := scores.Download(s)
	if er != nil || !reflect.DeepEqual(raw, scoresRaw) {
		t.Fatal("original half scores changed", er)
	}
	f := math.Float32bits
	values := []gpudecode.QualityValues{{Neighbors: [8]uint16{0x3400, 0x3000}, Count: 2, Center: 0x3800}}
	probabilities, er := n.Probabilities(ctx, values)
	if er != nil || !reflect.DeepEqual(probabilities, []uint32{f(.1953125)}) {
		t.Fatal("noncommutative probability marshalling", probabilities, er)
	}
	groups, er := n.Groups(ctx, [][]uint32{{f(.25), f(.5), f(.75)}, {f(.5)}}, f(1.25), f(-.5))
	if er != nil || !reflect.DeepEqual(groups, []gpudecode.QualityGroupNumber{{ErrorBits: f(2.75), Character: 37}, {ErrorBits: f(2.5), Character: 35}}) {
		t.Fatal("ragged group/scalar projection", groups, er)
	}
	summary, er := n.Summaries(ctx, []gpudecode.QualitySummaryRow{{Errors: []uint32{f(.25), f(.75)}, Frames: 4, Emitted: 2}, {Errors: []uint32{f(.25), f(.75)}, Frames: 4, Emitted: 0}})
	if er != nil || len(summary) != 2 || summary[0] != (gpudecode.QualitySummaryNumber{StayBits: f(1.75), MoveBits: f(12), MeanErrorBits: f(-.5)}) || summary[1].MeanErrorBits != 0x7f800000 {
		t.Fatal("summary counters or exceptional bits differ", summary, er)
	}
	if out, e := n.Probabilities(ctx, []gpudecode.QualityValues{{Center: 0x3555}}); e == nil || out != nil {
		t.Fatal("probability success without writes escaped poison")
	}
	if out, e := n.Groups(ctx, [][]uint32{{f(.5)}}, f(-12345), 0); e == nil || out != nil {
		t.Fatal("group success without writes escaped poison")
	}
	if out, e := n.Summaries(ctx, []gpudecode.QualitySummaryRow{{Errors: []uint32{f(-12345)}, Frames: 1, Emitted: 0}}); e == nil || out != nil {
		t.Fatal("padded summary success without writes escaped completion marker")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, e := n.Probabilities(canceled, values); !errors.Is(e, context.Canceled) {
		t.Fatal("cancelled numeric adapter continued", e)
	}
	alphaPoison := append([]byte(nil), scoresRaw...)
	binary.LittleEndian.PutUint16(alphaPoison, 0x3555)
	badScores := newStorage(tensor.DTypeF16, []int64{64, 2, 1024}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}, alphaPoison)
	if out, e := gpudecode.Posteriors(ctx, d, s, badScores, guides, g); e == nil || out != nil {
		t.Fatal("unwritten forward alpha escaped poison")
	}
	postPoison := append([]byte(nil), guidesRaw...)
	binary.LittleEndian.PutUint32(postPoison, f(-12345))
	badGuides := newStorage(tensor.DTypeF32, []int64{3, 64, 256}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}, postPoison)
	if out, e := gpudecode.Posteriors(ctx, d, s, scores, badGuides, g); e == nil || out != nil {
		t.Fatal("unwritten posterior prefix escaped poison")
	}
	other, e2 := e.NewCUDADevice(0)
	if e2 != nil {
		t.Fatal(e2)
	}
	t.Cleanup(func() {
		if e := other.Close(); e != nil {
			t.Error(e)
		}
	})
	foreign, e2 := other.NewStream()
	if e2 != nil {
		t.Fatal(e2)
	}
	t.Cleanup(func() {
		if e := foreign.Close(); e != nil {
			t.Error(e)
		}
	})
	if _, e := gpudecode.NewGPUQualityNumbers(d, foreign, sha256.Sum256([]byte("fake")), 1); e == nil {
		t.Fatal("foreign program stream admitted")
	}
	if e := foreign.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e := other.DecoderPosteriorProgram(foreign); e == nil {
		t.Fatal("closed program stream admitted")
	}
}
