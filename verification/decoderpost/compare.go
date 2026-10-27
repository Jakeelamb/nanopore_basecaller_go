// Package decoderpost compares D04 bytes after the caller authenticates the
// D01 producer, original inputs, stage custody and selected runtime image.
package decoderpost

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
)

type Layout struct{ N, T, States int64 }

func (p Layout) validate() error { _, e := gpudecode.PlanGuides(p.N, p.T, p.States, 0, 2); return e }

type Failure struct {
	Buffer    string
	Offset    int
	Got, Want byte
	Reason    string
}

func (e *Failure) Error() string {
	return fmt.Sprintf("D04 %s byte%d: %s, got%02x want%02x", e.Buffer, e.Offset, e.Reason, e.Got, e.Want)
}

type Report struct {
	Complete      bool
	ComparedBytes int
	Failure       *Failure
}

func reject(buffer, reason string) (Report, error) {
	e := &Failure{Buffer: buffer, Offset: -1, Reason: reason}
	return Report{Failure: e}, e
}
func compare(buffer string, got, want []byte, prior int) (Report, error) {
	if len(got) != len(want) {
		return reject(buffer, "byte extent differs")
	}
	if !bytes.Equal(got, want) {
		for i, b := range got {
			if b != want[i] {
				e := &Failure{buffer, i, b, want[i], "bits differ"}
				return Report{ComparedBytes: prior + i, Failure: e}, e
			}
		}
	}
	return Report{Complete: true, ComparedBytes: prior + len(got)}, nil
}

// ComparePosteriors compares the entire destructive parent, including the
// unchanged model-score tail. Prefix rows are (T+1)*S half values per chunk.
func ComparePosteriors(p Layout, original, candidate, native []byte) (Report, error) {
	if e := p.validate(); e != nil {
		return reject("scores", e.Error())
	}
	extent := int(p.N * p.T * 4 * p.States * 2)
	if len(original) != extent || len(candidate) != extent || len(native) != extent {
		return reject("scores", "fixed parent extent differs")
	}
	stride, prefix := int(p.T*4*p.States*2), int((p.T+1)*p.States*2)
	for n := 0; n < int(p.N); n++ {
		if !bytes.Equal(candidate[n*stride+prefix:(n+1)*stride], original[n*stride+prefix:(n+1)*stride]) {
			return reject("scores-tail", "candidate changed immutable original tail")
		}
	}
	return compare("scores", candidate, native, 0)
}

type QualityWire struct{ Path, Moves, Sequence, Q, Results []byte }

// EncodeQuality preserves fixed decoder allocation bytes. Chunk paths start at
// n*T, not n*(T+1); the extra N path values retain their initialized zeros.
func EncodeQuality(p Layout, batch gpudecode.QualityBatch) (QualityWire, error) {
	var empty QualityWire
	if e := p.validate(); e != nil {
		return empty, e
	}
	if int64(len(batch.Chunks)) != p.N {
		return empty, fmt.Errorf("fixed quality chunk count differs")
	}
	w := QualityWire{make([]byte, p.N*(p.T+1)*4), make([]byte, p.N*p.T), make([]byte, p.N*p.T), make([]byte, p.N*p.T), make([]byte, p.N*32)}
	for n, c := range batch.Chunks {
		if int64(len(c.Path)) != p.T || int64(len(c.Moves)) != p.T || int64(len(c.Sequence)) != p.T || int64(len(c.Q)) != p.T {
			return empty, fmt.Errorf("complete fixed quality output rows required")
		}
		if c.ResultWords[0] != uint32(int64(n)*p.T) || c.ResultWords[1] != uint32(p.T) {
			return empty, fmt.Errorf("result descriptor start/length differs")
		}
		offset := int64(n) * p.T
		for i, state := range c.Path {
			if state < 0 || state > 3 || c.Moves[i] > 1 {
				return empty, fmt.Errorf("quality path/move exceeds source domain")
			}
			binary.LittleEndian.PutUint32(w.Path[(offset+int64(i))*4:], uint32(state))
		}
		copy(w.Moves[offset:offset+p.T], c.Moves)
		copy(w.Sequence[offset:offset+p.T], c.Sequence)
		copy(w.Q[offset:offset+p.T], c.Q)
		copy(w.Results[n*32:], gpudecode.PackQualityResult(c.ResultWords))
	}
	return w, nil
}

func CompareQuality(p Layout, batch gpudecode.QualityBatch, native QualityWire) (Report, error) {
	w, e := EncodeQuality(p, batch)
	if e != nil {
		return reject("quality", e.Error())
	}
	count := 0
	for _, field := range []struct {
		name      string
		got, want []byte
	}{{"path", w.Path, native.Path}, {"moves", w.Moves, native.Moves}, {"sequence", w.Sequence, native.Sequence}, {"qstring", w.Q, native.Q}, {"results", w.Results, native.Results}} {
		r, e := compare(field.name, field.got, field.want, count)
		if e != nil {
			return r, e
		}
		count = r.ComparedBytes
	}
	return Report{Complete: true, ComparedBytes: count}, nil
}
