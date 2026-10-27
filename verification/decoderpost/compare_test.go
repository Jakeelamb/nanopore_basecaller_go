package decoderpost

import (
	"encoding/binary"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"testing"
)

func TestD04PosteriorConsumerPreservesTailAndFailureDetails(t *testing.T) {
	p := Layout{64, 2, 256}
	original := make([]byte, 64*2*1024*2)
	candidate := append([]byte(nil), original...)
	candidate[0] = 1
	native := append([]byte(nil), candidate...)
	if r, e := ComparePosteriors(p, original, candidate, native); e != nil || !r.Complete || r.ComparedBytes != len(candidate) {
		t.Fatal("synthetic byte handoff", r, e)
	}
	native[17] = 2
	if r, e := ComparePosteriors(p, original, candidate, native); e == nil || r.Complete || r.Failure == nil || r.Failure.Offset != 17 {
		t.Fatal("mismatch evidence lost", r, e)
	}
	candidate[4095] = 9
	native[4095] = 9
	if r, e := ComparePosteriors(p, original, candidate, native); e == nil || r.Failure.Buffer != "scores-tail" {
		t.Fatal("matching corruption accepted", r, e)
	}
}

func TestD04QualityConsumerUsesSourcePathStrideAndAllRawResultBits(t *testing.T) {
	p := Layout{64, 2, 256}
	b := gpudecode.QualityBatch{Chunks: make([]gpudecode.QualityChunk, 64)}
	for n := range b.Chunks {
		b.Chunks[n] = gpudecode.QualityChunk{Path: []int32{1, 2}, Moves: []byte{1, 0}, Sequence: []byte{'G', 0}, Q: []byte{40, 0}, ResultWords: [8]uint32{uint32(n * 2), 2, 0, 0, 0x3f800000, 0, 0, 0x7fc12345}}
	}
	w, e := EncodeQuality(p, b)
	if e != nil {
		t.Fatal(e)
	}
	if binary.LittleEndian.Uint32(w.Path[8:]) != 1 || binary.LittleEndian.Uint32(w.Path[64*2*4:]) != 0 || binary.LittleEndian.Uint32(w.Results[28:]) != 0x7fc12345 {
		t.Fatal("path stride/initialized tail/raw NaN bits changed")
	}
	if r, e := CompareQuality(p, b, w); e != nil || !r.Complete {
		t.Fatal("synthetic quality handoff", r, e)
	}
	w.Results[8] = 1
	if r, e := CompareQuality(p, b, w); e == nil || r.Complete || r.Failure.Buffer != "results" || r.Failure.Offset != 8 {
		t.Fatal("padded counter mismatch hidden", r, e)
	}
	b.Chunks[0].Q = nil
	if _, e := EncodeQuality(p, b); e == nil {
		t.Fatal("partial candidate accepted")
	}
}
