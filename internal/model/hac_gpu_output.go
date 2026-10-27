package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type outputWeight struct {
	receipt WeightReceipt
	values  []float32
}
type HACOutputWeights struct{ down, up outputWeight }

// LoadHACOutputWeights binds only the two named projection archives through the
// existing verified CPU loader. It does not load or execute a model graph.
func LoadHACOutputWeights(engine *tensor.Engine, directory string) (HACOutputWeights, error) {
	var out HACOutputWeights
	if engine == nil {
		return out, tensor.ErrClosed
	}
	manifest, err := WeightManifest(HAC)
	if err != nil {
		return out, err
	}
	wants := []struct {
		name  string
		shape []int64
		dst   *outputWeight
	}{
		{"11.linear.weight.tensor", []int64{128, 1024}, &out.down},
		{"12.linear.weight.tensor", []int64{1024, 128}, &out.up},
	}
	for _, want := range wants {
		var pin WeightReceipt
		found := 0
		for _, v := range manifest {
			if v.Name == want.name {
				pin = v
				found++
			}
		}
		if found != 1 || !slices.Equal(pin.Shape, want.shape) {
			return HACOutputWeights{}, fmt.Errorf("%w: projection archive role %s", ErrGPUWeights, want.name)
		}
		weight, err := loadWeight(engine, filepath.Join(directory, want.name), pin)
		if err != nil {
			return HACOutputWeights{}, err
		}
		data, readErr := weight.Data()
		err = errors.Join(readErr, weight.Close())
		if err != nil {
			return HACOutputWeights{}, err
		}
		*want.dst = outputWeight{receipt: pin, values: data}
	}
	return out, nil
}
func (w HACOutputWeights) Receipts() []WeightReceipt {
	out := []WeightReceipt{w.down.receipt, w.up.receipt}
	for i := range out {
		out[i].Shape = slices.Clone(out[i].Shape)
	}
	return out
}

type HACOutputCache struct {
	down          []int8
	downScale, up []uint16
}
type HACOutputCacheSnapshot struct {
	Down          []int8
	DownScale, Up []uint16
}

func (c HACOutputCache) Snapshot() HACOutputCacheSnapshot {
	return HACOutputCacheSnapshot{slices.Clone(c.down), slices.Clone(c.downScale), slices.Clone(c.up)}
}
func PackHACOutput(w HACOutputWeights) (HACOutputCache, error) {
	var c HACOutputCache
	if w.down.receipt.Name != "11.linear.weight.tensor" || w.up.receipt.Name != "12.linear.weight.tensor" ||
		!slices.Equal(w.down.receipt.Shape, []int64{128, 1024}) || !slices.Equal(w.up.receipt.Shape, []int64{1024, 128}) ||
		len(w.down.values) != 128*1024 || len(w.up.values) != 1024*128 {
		return c, fmt.Errorf("%w: projection requires named shape-bearing down and up weights", ErrGPUWeights)
	}
	for _, values := range [][]float32{w.down.values, w.up.values} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || signal.Float16Bits(v)&0x7c00 == 0x7c00 {
				return c, fmt.Errorf("%w: projection weight is not finite half", ErrGPUWeights)
			}
		}
	}
	down, scales, err := QuantizeFLSTMRows(CastFLSTMHalf(w.down.values), 128, 1024)
	if err != nil {
		return c, err
	}
	c.down, err = permuteContiguous(down, []int64{8, 2, 2, 2, 2, 1024}, []int{0, 3, 1, 2, 4, 5})
	if err != nil {
		return c, err
	}
	c.downScale, err = permuteContiguous(CastFLSTMHalf(scales), []int64{8, 2, 2, 2, 2}, []int{0, 3, 1, 2, 4})
	if err != nil {
		return c, err
	}
	c.up, err = permuteContiguous(CastFLSTMHalf(w.up.values), []int64{32, 2, 2, 2, 2, 2, 128}, []int{0, 3, 4, 1, 2, 5, 6})
	if err != nil {
		return c, err
	}
	return c, nil
}

// HACOutputTempElements is arithmetic-only and accepts reservation geometry T35.
func HACOutputTempElements(n, t int64) (int64, error) {
	if n <= 0 || t <= 0 {
		return 0, fmt.Errorf("%w: positive projection temp dimensions required", ErrGPUPlanLayout)
	}
	nt, err := gpuMul(n, t)
	if err != nil {
		return 0, err
	}
	elems, err := gpuMul(nt, 128+1024)
	if err != nil {
		return 0, err
	}
	if _, err = gpuMul(elems, 2); err != nil {
		return 0, err
	}
	return elems, nil
}

type outputSpan struct{ offset, bytes int64 }
type HACOutputPlan struct {
	n, t, reservationBytes int64
	full, temp             GPUView
	input, down, out       outputSpan
}

func PlanHACOutput(reservation GPUMemoryReservation, full, temp GPUView) (HACOutputPlan, error) {
	var p HACOutputPlan
	g := full.Geometry
	n, t := g.N, g.T
	if n <= 0 || t <= 0 || n%64 != 0 || t%32 != 0 || n > math.MaxInt32 || t > math.MaxInt32-5 || t+5 > math.MaxInt32/1024/n {
		return p, fmt.Errorf("%w: projection requires checked N64/T32 tiles", ErrGPUPlanLayout)
	}
	want, err := NewGPUStorageGeometry(GPUStorageCUTLASSI8, n, t, 1024)
	if err != nil {
		return p, err
	}
	if g != want || full.Scratch || full.Orientation != GPUOrientationForward {
		return p, fmt.Errorf("%w: projection requires full forward CUTLASS I8", ErrGPUPlanLayout)
	}
	elems, err := HACOutputTempElements(n, t)
	if err != nil {
		return p, err
	}
	need, err := gpuMul(elems, 2)
	if err != nil {
		return p, err
	}
	tempGeometry, err := NewGPUScratchGeometry(GPUDtypeHalf, temp.Geometry.Elems)
	if err != nil || temp.Geometry != tempGeometry {
		return p, fmt.Errorf("%w: inconsistent projection temp geometry", ErrGPUPlanLayout)
	}
	if reservation.Bytes <= 0 || reservation.Bytes%GPUWorkingMemoryAlignment != 0 ||
		!temp.Scratch || temp.Geometry.Layout != GPUStorageRaw || temp.Geometry.Dtype != GPUDtypeHalf || temp.Geometry.LiveBytes < need {
		return p, fmt.Errorf("%w: projection requires reserved half temp", ErrGPUPlanUndersized)
	}
	for _, v := range []GPUView{full, temp} {
		end, e := gpuAdd(v.Offset, v.Geometry.Bytes)
		if e != nil {
			return p, e
		}
		if v.Offset < 0 || v.Offset%GPUWorkingMemoryAlignment != 0 || v.End != end || end > reservation.Bytes {
			return p, fmt.Errorf("%w: projection view escapes reservation", ErrGPUPlanUndersized)
		}
	}
	if full.Offset < temp.End && temp.Offset < full.End {
		return p, fmt.Errorf("%w: projection input/temp overlap", ErrGPUPlanOverlap)
	}
	nt, _ := gpuMul(n, t)
	downBytes, _ := gpuMul(nt, 128*2)
	scoreBytes, _ := gpuMul(nt, 1024*2)
	row, _ := gpuMul(n, 1024)
	start, err := gpuAdd(full.Offset, 2*row)
	if err != nil {
		return p, err
	}
	outputOffset, err := gpuAdd(temp.Offset, downBytes)
	if err != nil {
		return p, err
	}
	p = HACOutputPlan{n: n, t: t, reservationBytes: reservation.Bytes, full: full, temp: temp,
		input: outputSpan{start, nt * 1024}, down: outputSpan{temp.Offset, downBytes}, out: outputSpan{outputOffset, scoreBytes}}
	return p, nil
}

type outputOps interface {
	validate() error
	zero() error
	kernel() error
}

func runHACOutput(ctx context.Context, ops outputOps) error {
	if ctx == nil {
		return fmt.Errorf("nil projection context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ops.validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ops.zero(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ops.kernel(); err != nil {
		return err
	}
	return ctx.Err()
}

type HACOutput struct {
	mu                  sync.Mutex
	device              *tensor.Device
	down, downScale, up *tensor.Storage
	pending             []*tensor.Storage
	closed              bool
}

func NewHACOutput(d *tensor.Device, s *tensor.Stream, c HACOutputCache) (out *HACOutput, err error) {
	if len(c.down) != 128*1024 || len(c.downScale) != 128 || len(c.up) != 1024*128 {
		return nil, fmt.Errorf("%w: invalid packed projection cache", ErrGPUWeights)
	}
	out = &HACOutput{device: d}
	defer func() {
		if err != nil {
			cleanup := out.Close()
			err = errors.Join(err, cleanup)
			if cleanup == nil {
				out = nil
			}
		}
	}()
	i8 := make([]byte, len(c.down))
	for i, v := range c.down {
		i8[i] = byte(v)
	}
	half := func(v []uint16) []byte {
		b := make([]byte, len(v)*2)
		for i, x := range v {
			binary.LittleEndian.PutUint16(b[i*2:], x)
		}
		return b
	}
	out.down, err = d.NewStorage(s, tensor.DTypeI8, []int64{128, 1024}, []tensor.KoiTag{'K', tensor.TagC}, i8)
	if err != nil {
		return out, err
	}
	out.downScale, err = d.NewStorage(s, tensor.DTypeF16, []int64{128}, []tensor.KoiTag{'K'}, half(c.downScale))
	if err != nil {
		return out, err
	}
	out.up, err = d.NewStorage(s, tensor.DTypeF16, []int64{1024, 128}, []tensor.KoiTag{tensor.TagC, 'K'}, half(c.up))
	return out, err
}
func (o *HACOutput) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.closed = true
	var err error
	o.pending, err = closeGPUStorageOwners(o.pending)
	for _, slot := range []**tensor.Storage{&o.up, &o.downScale, &o.down} {
		if *slot != nil {
			if e := (*slot).Close(); e != nil {
				err = errors.Join(err, e)
			} else {
				*slot = nil
			}
		}
	}
	return err
}

type outputCUDAOps struct {
	device   *tensor.Device
	stream   *tensor.Stream
	operands tensor.FactorisedLinearOperands
	geometry tensor.FactorisedLinearGeometry
}

func (o outputCUDAOps) validate() error {
	return o.device.ValidateFactorisedLinear(o.stream, o.operands, o.geometry)
}
func (o outputCUDAOps) zero() error { return o.operands.Down.Fill(o.stream, []byte{0, 0}) }
func (o outputCUDAOps) kernel() error {
	return o.device.FactorisedLinear(o.stream, o.operands, o.geometry)
}

func (o *HACOutput) Run(ctx context.Context, s *tensor.Stream, backing, full *tensor.Storage, p HACOutputPlan) (out *tensor.Storage, err error) {
	if o == nil {
		return nil, tensor.ErrClosed
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed || len(o.pending) != 0 {
		return nil, tensor.ErrClosed
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil projection context")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if p.n <= 0 || p.t <= 0 {
		return nil, fmt.Errorf("%w: uninitialized projection plan", ErrGPUPlanState)
	}
	if backing == nil || full == nil {
		return nil, tensor.ErrClosed
	}
	bm, err := backing.Meta()
	if err != nil {
		return nil, err
	}
	fm, err := full.Meta()
	if err != nil {
		return nil, err
	}
	if bm.DType != tensor.DTypeU8 || !slices.Equal(bm.Shape, []int64{p.reservationBytes}) || !slices.Equal(bm.Strides, []int64{1}) || bm.OffsetElements != 0 ||
		fm.Allocation != bm.Allocation || fm.OffsetElements != uint64(p.full.Offset) || fm.DType != tensor.DTypeI8 ||
		!slices.Equal(fm.Shape, []int64{p.t + 5, p.n, 1024}) || !slices.Equal(fm.Strides, []int64{p.n * 1024, 1024, 1}) ||
		!slices.Equal(fm.Tags, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}) || fm.ScalePresent || fm.Device != bm.Device {
		return nil, fmt.Errorf("%w: projection full input does not match its invocation backing", ErrGPUPlanLayout)
	}
	input, err := full.View(uint64(2*p.n*1024), []int64{p.t, p.n, 1024}, []int64{p.n * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	if err != nil {
		return nil, err
	}
	var down *tensor.Storage
	defer func() {
		failed, cleanup := closeGPUStorageOwners([]*tensor.Storage{input, down})
		o.pending = append(o.pending, failed...)
		err = errors.Join(err, cleanup)
		if err != nil && out != nil {
			if e := out.Close(); e != nil {
				err = errors.Join(err, e)
				o.pending = append(o.pending, out)
			}
			out = nil
		}
	}()
	down, err = backing.Reinterpret(tensor.DTypeF16, uint64(p.down.offset/2), []int64{p.down.bytes / 2}, []int64{1}, []tensor.KoiTag{tensor.TagC})
	if err != nil {
		return nil, err
	}
	out, err = backing.Reinterpret(tensor.DTypeF16, uint64(p.out.offset/2), []int64{p.n, p.t, 1024}, []int64{p.t * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return nil, err
	}
	ops := outputCUDAOps{device: o.device, stream: s, operands: tensor.FactorisedLinearOperands{Input: input, DownWeight: o.down, DownScale: o.downScale, Down: down, UpWeight: o.up, Output: out}, geometry: tensor.FactorisedLinearGeometry{N: p.n, T: p.t}}
	err = runHACOutput(ctx, ops)
	return out, err
}
