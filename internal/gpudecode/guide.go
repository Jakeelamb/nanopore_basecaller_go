package gpudecode

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type GuidePlan struct {
	n, t, states int64
	clamp, blank float32
}

func PlanGuides(n, t, states int64, clamp, blank float32) (GuidePlan, error) {
	if !((states == 256 && n > 0 && n%64 == 0) || (states == 1024 && n > 0 && n%32 == 0)) || t <= 0 || t >= math.MaxInt32 || n > math.MaxInt32 || n > math.MaxInt64/4/states/(t+1) || n*(t+1)*states > 1<<27 {
		return GuidePlan{}, fmt.Errorf("decoder guides require checked fixed HAC/SUP batch geometry")
	}
	if math.IsNaN(float64(clamp)) || math.IsInf(float64(clamp), 0) || clamp < 0 || math.IsNaN(float64(blank)) || math.IsInf(float64(blank), 0) {
		return GuidePlan{}, fmt.Errorf("finite nonnegative clamp and finite blank required")
	}
	return GuidePlan{n, t, states, clamp, blank}, nil
}

func guideSchedule(ctx context.Context, p GuidePlan, zero func() error, frame func(int64) error) error {
	if ctx == nil {
		return fmt.Errorf("guide context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := zero(); err != nil {
		return err
	}
	for t := p.t - 1; t >= 0; t-- {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := frame(t); err != nil {
			return err
		}
	}
	return nil
}

// BackGuides owns terminal initialization and reverse-time scheduling in Go.
// Scores are borrowed half [N,T,4*States]; output owns float32 [T+1,N,States].
// The caller must initialize and authenticate d.DecoderGuideProgram first.
func BackGuides(ctx context.Context, d *tensor.Device, s *tensor.Stream, scores *tensor.Storage, p GuidePlan) (result *tensor.Storage, err error) {
	want, e := PlanGuides(p.n, p.t, p.states, p.clamp, p.blank)
	if e != nil || want != p {
		return result, fmt.Errorf("initialized decoder guide plan required")
	}
	if ctx == nil {
		return result, fmt.Errorf("guide context required")
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if d == nil || s == nil || scores == nil {
		return result, tensor.ErrClosed
	}
	meta, err := scores.Meta()
	if err != nil {
		return result, err
	}
	if meta.DType != tensor.DTypeF16 || len(meta.Shape) != 3 || meta.Shape[0] != p.n || meta.Shape[1] != p.t || meta.Shape[2] != 4*p.states || len(meta.Strides) != 3 || meta.Strides[2] != 1 || meta.Strides[1] != 4*p.states || meta.Strides[0] != p.t*4*p.states || meta.ScalePresent || len(meta.Tags) != 3 || meta.Tags[0] != tensor.TagN || meta.Tags[1] != tensor.TagT || meta.Tags[2] != tensor.TagC {
		return result, fmt.Errorf("contiguous half NTC decoder scores required")
	}
	guides, err := d.NewStorage(s, tensor.DTypeF32, []int64{p.t + 1, p.n, p.states}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}, make([]byte, int((p.t+1)*p.n*p.states*4)))
	if err != nil {
		return result, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, guides.Close())
		}
	}()
	row := func(t int64) (*tensor.Storage, error) {
		return guides.View(uint64(t*p.n*p.states), []int64{p.n, p.states}, []int64{p.states, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagC})
	}
	err = guideSchedule(ctx, p, func() (err error) {
		terminal, err := row(p.t)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, terminal.Close()) }()
		return terminal.Fill(s, []byte{0, 0, 0, 0})
	}, func(t int64) (err error) {
		input, err := scores.View(uint64(t*4*p.states), []int64{p.n, 4 * p.states}, []int64{p.t * 4 * p.states, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagC})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, input.Close()) }()
		next, err := row(t + 1)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, next.Close()) }()
		out, err := row(t)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, out.Close()) }()
		return d.DecoderGuideFrame(s, input, next, out, p.clamp, p.blank)
	})
	if err != nil {
		return result, err
	}
	result = guides
	return result, nil
}
