//go:build cgo && linux

package tensor_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestDecoderBeamProductionAdapterAgainstHostArgumentProjectionOnly(t *testing.T) {
	e := tensor.DecoderBeamFakeEngineForTest(t)
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
	n, err := gpudecode.NewGPUBeamNumbers(d, s, sha256.Sum256([]byte("beam")), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gpudecode.NewGPUBeamNumbers(d, s, sha256.Sum256([]byte("wrong")), 4096); err == nil {
		t.Fatal("wrong linked-program hash admitted")
	}
	fb := math.Float32bits
	ctx := context.Background()
	got, err := n.LogCuts(ctx, []uint32{fb(100)})
	if err != nil || !reflect.DeepEqual(got, []uint32{fb(101)}) {
		t.Fatal("log argument projection", got, err)
	}
	row := gpudecode.ExpandRow{Previous: fb(11), Transition: [4]uint16{0x3400, 1, 2, 3}, Guide: [5]uint32{fb(21), fb(22), fb(23), fb(24), fb(25)}}
	expanded, err := n.Expand5(ctx, []gpudecode.ExpandRow{row}, fb(1.25), fb(-.75))
	want := gpudecode.ExpandResult{Scores: [5]uint32{fb(2), fb(11), fb(0x3400), fb(21), fb(1.25)}, OrderedMax: fb(-.75)}
	if err != nil || len(expanded) != 1 || expanded[0] != want {
		t.Fatal("expand argument projection", expanded, err)
	}
	for _, item := range []struct {
		want float32
		call func(context.Context, []gpudecode.F32Pair) ([]uint32, error)
	}{{6, n.LogAddPairs}, {7, n.SubPairs}, {8, n.Midpoints}, {9, n.MaxPairs}} {
		got, err := item.call(ctx, []gpudecode.F32Pair{{First: fb(1), Second: fb(2)}})
		if err != nil || !reflect.DeepEqual(got, []uint32{fb(item.want)}) {
			t.Fatal("pair argument projection", got, err)
		}
	}
	if got, err := n.MaxPairs(ctx, []gpudecode.F32Pair{{First: fb(math.MaxFloat32), Second: fb(math.MaxFloat32)}}); err == nil || got != nil {
		t.Fatal("fake projection poison escaped numeric output validation", got, err)
	}
	if got, err := n.MaxPairs(ctx, []gpudecode.F32Pair{{First: fb(-12345), Second: 0}}); err == nil || got != nil {
		t.Fatal("fake success without output writes escaped poison validation", got, err)
	}
	g, err := gpudecode.PlanGuides(64, 1, 256, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	p, err := gpudecode.PlanBeam(g, 100)
	if err != nil {
		t.Fatal(err)
	}
	scoreBytes, guideBytes := make([]byte, 64*1024*2), make([]byte, 2*64*256*4)
	binary.LittleEndian.PutUint16(scoreBytes, 0x1234)
	binary.LittleEndian.PutUint32(guideBytes[len(guideBytes)-4:], 0x11223344)
	scores, err := d.NewStorage(s, tensor.DTypeF16, []int64{64, 1, 1024}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}, scoreBytes)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := scores.Close(); err != nil {
			t.Error(err)
		}
	})
	guides, err := d.NewStorage(s, tensor.DTypeF32, []int64{2, 64, 256}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}, guideBytes)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := guides.Close(); err != nil {
			t.Error(err)
		}
	})
	in, err := gpudecode.DownloadBeamInput(ctx, p, s, scores, guides)
	if err != nil || len(in.ScoresNTC) != 64*1024 || in.ScoresNTC[0] != 0x1234 || in.GuidesTNS[len(in.GuidesTNS)-1] != 0x11223344 {
		t.Fatal("immutable input handoff differs", err)
	}
	in.ScoresNTC[0] = 0
	in.GuidesTNS[len(in.GuidesTNS)-1] = 0
	a, err := scores.Download(s)
	if err != nil || !reflect.DeepEqual(a, scoreBytes) {
		t.Fatal("score download aliases storage", err)
	}
	b, err := guides.Download(s)
	if err != nil || !reflect.DeepEqual(b, guideBytes) {
		t.Fatal("guide download aliases storage", err)
	}
	if _, err := gpudecode.DownloadBeamInput(ctx, p, s, guides, scores); err == nil {
		t.Fatal("wrong input types admitted")
	}
	other, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := other.Close(); err != nil {
			t.Error(err)
		}
	})
	foreign, err := other.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := foreign.Close(); err != nil {
			t.Error(err)
		}
	})
	if _, err := gpudecode.NewGPUBeamNumbers(d, foreign, sha256.Sum256([]byte("beam")), 1); err == nil {
		t.Fatal("foreign numeric stream owner admitted")
	}
	if err := foreign.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := gpudecode.NewGPUBeamNumbers(other, foreign, sha256.Sum256([]byte("beam")), 1); err == nil {
		t.Fatal("closed numeric stream admitted")
	}
}
