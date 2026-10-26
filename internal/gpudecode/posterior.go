package gpudecode

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func posteriorSchedule(ctx context.Context, p GuidePlan, forward func(int64, int, int) error, row func(int64, int) error) error {
	if ctx == nil {
		return fmt.Errorf("posterior context required")
	}
	current := 0
	for t := int64(0); t < p.t; t++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := forward(t, current, current^1); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := row(t, current); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		current ^= 1
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := row(p.t, current); err != nil {
		return err
	}
	return ctx.Err()
}

// Posteriors returns a distinct score parent with the source's destructive
// state-posterior prefix and untouched score tail. Scores and guides are
// borrowed. Go owns alpha initialization, all frame/row choices and swaps.
// The caller must initialize and authenticate DecoderPosteriorProgram first.
func Posteriors(ctx context.Context, d *tensor.Device, s *tensor.Stream, scores, guides *tensor.Storage, p GuidePlan) (result *tensor.Storage, err error) {
	want, e := PlanGuides(p.n, p.t, p.states, p.clamp, p.blank)
	if e != nil || want != p {
		return nil, fmt.Errorf("initialized posterior scan plan required")
	}
	if ctx == nil {
		return nil, fmt.Errorf("posterior context required")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if d == nil || s == nil || scores == nil || guides == nil {
		return nil, tensor.ErrClosed
	}
	meta, err := scores.Meta()
	if err != nil {
		return nil, err
	}
	if meta.DType != tensor.DTypeF16 || !slices.Equal(meta.Shape, []int64{p.n, p.t, 4 * p.states}) || !slices.Equal(meta.Strides, []int64{p.t * 4 * p.states, 4 * p.states, 1}) || !slices.Equal(meta.Tags, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}) || meta.ScalePresent {
		return nil, fmt.Errorf("contiguous half NTC posterior scores required")
	}
	gm, err := guides.Meta()
	if err != nil {
		return nil, err
	}
	if gm.DType != tensor.DTypeF32 || !slices.Equal(gm.Shape, []int64{p.t + 1, p.n, p.states}) || !slices.Equal(gm.Strides, []int64{p.n * p.states, p.states, 1}) || !slices.Equal(gm.Tags, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}) || gm.ScalePresent {
		return nil, fmt.Errorf("contiguous float TNC backward guides required")
	}
	original, err := scores.Download(s)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(original); i += 2 {
		if binary.LittleEndian.Uint16(original[i:])&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("finite original half scores required")
		}
	}
	guideBits, err := guides.Download(s)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(guideBits); i += 4 {
		if !finiteBeamScore(binary.LittleEndian.Uint32(guideBits[i:])) {
			return nil, fmt.Errorf("finite backward guide values required")
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	working, err := d.NewStorage(s, tensor.DTypeF16, meta.Shape, meta.Tags, make([]byte, int(p.n*p.t*4*p.states*2)))
	if err != nil {
		return nil, err
	}
	var alpha [2]*tensor.Storage
	defer func() {
		for i := len(alpha) - 1; i >= 0; i-- {
			if alpha[i] != nil {
				err = errors.Join(err, alpha[i].Close())
			}
		}
		if err != nil {
			err = errors.Join(err, working.Close())
			result = nil
		}
	}()
	if err = scores.CopyTo(s, working); err != nil {
		return nil, err
	}
	for chunk := int64(0); chunk < p.n; chunk++ {
		prefix, e := working.View(uint64(chunk*p.t*4*p.states), []int64{(p.t + 1) * p.states}, []int64{1}, []tensor.KoiTag{tensor.TagC})
		if e != nil {
			return nil, e
		}
		e = errors.Join(prefix.Fill(s, []byte{0, 0x7e}), prefix.Close())
		if e != nil {
			return nil, e
		}
	}
	for i := range alpha {
		alpha[i], err = d.NewStorage(s, tensor.DTypeF32, []int64{p.n, p.states}, []tensor.KoiTag{tensor.TagN, tensor.TagC}, make([]byte, int(p.n*p.states*4)))
		if err != nil {
			return nil, err
		}
	}
	err = posteriorSchedule(ctx, p, func(t int64, previous, next int) (err error) {
		// Earlier posterior stores end at t*S; current source scores start at
		// 4*t*S. Reading the immutable original therefore preserves every input.
		input, err := scores.View(uint64(t*4*p.states), []int64{p.n, 4 * p.states}, []int64{p.t * 4 * p.states, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagC})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, input.Close()) }()
		if err = alpha[next].Fill(s, []byte{0, 0, 0xc0, 0x7f}); err != nil {
			return err
		}
		if err = d.DecoderForwardFrame(s, input, alpha[previous], alpha[next], p.clamp, p.blank); err != nil {
			return err
		}
		bits, e := alpha[next].Download(s)
		if e != nil {
			return e
		}
		for i := 0; i < len(bits); i += 4 {
			if !finiteBeamScore(binary.LittleEndian.Uint32(bits[i:])) {
				return fmt.Errorf("forward alpha contains nonfinite or unwritten output")
			}
		}
		return ctx.Err()
	}, func(t int64, previous int) (err error) {
		beta, err := guides.View(uint64(t*p.n*p.states), []int64{p.n, p.states}, []int64{p.states, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagC})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, beta.Close()) }()
		out, err := working.View(uint64(t*p.states), []int64{p.n, p.states}, []int64{p.t * 4 * p.states, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagC})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, out.Close()) }()
		return d.DecoderPosteriorRow(s, alpha[previous], beta, out)
	})
	if err != nil {
		return nil, err
	}
	packed, err := working.Download(s)
	if err != nil {
		return nil, err
	}
	stride, prefix := int(p.t*4*p.states*2), int((p.t+1)*p.states*2)
	for chunk := 0; chunk < int(p.n); chunk++ {
		row := packed[chunk*stride : (chunk+1)*stride]
		for i := 0; i < prefix; i += 2 {
			if !posteriorProbability(binary.LittleEndian.Uint16(row[i:])) {
				return nil, fmt.Errorf("posterior prefix contains invalid or unwritten probability")
			}
		}
		if !bytes.Equal(row[prefix:], original[chunk*stride+prefix:(chunk+1)*stride]) {
			return nil, fmt.Errorf("posterior changed original score tail")
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return working, nil
}
