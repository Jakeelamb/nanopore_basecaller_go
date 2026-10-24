package h03

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h02"
)

// replay runs the whole declared convolution stack once and compares every
// written byte with the capture. conv0 and conv1 reuse the H02 caller; the last
// convolution goes through the H03 caller.
func replay(r *Receipt, stack *model.HACSmallConvPlan, last *model.HACLastConvPlan, capture hacgpu.Capture, dev model.LastConvDevOptions) error {
	if len(stack.Layers) < 3 {
		return fmt.Errorf("the HAC conv stack needs at least three planned convolutions")
	}
	for i := 0; i < 2; i++ {
		if !stack.Layers[i].Executes {
			return fmt.Errorf("conv%d is planned but not marked executable", i)
		}
	}
	chain := stack.Layers[:2]

	e, err := tensor.Open(r.Backend)
	if err != nil {
		return err
	}
	defer e.Close()
	d, err := e.NewCUDADevice(r.Ordinal)
	if err != nil {
		return err
	}
	s, err := d.NewStream()
	if err != nil {
		return err
	}
	defer s.Close()

	convs, err := h02.ModelConvs(r.Model)
	if err != nil {
		return err
	}
	weights, err := h02.LoadConvWeights(e, r.Model, convs)
	if err != nil {
		return err
	}
	r.Inputs[filepath.Join("internal", "model", "hac_weights.json")] = weights.ManifestSHA

	if r.Packing, err = packAndVerify(d, s, stack.Layers, weights); err != nil {
		return err
	}

	// One backing allocation, exactly the reserved extent, pre-filled with the
	// declared pattern so an unwritten element can be proven untouched. It is
	// I8 because the final convolution's allocation is I8; the F16 views are
	// reinterprets of the same bytes, exactly like `backing_tensor.view(dtype)`.
	if stack.ReservationBytes%2 != 0 {
		return fmt.Errorf("reservation %d is not a whole number of F16 elements", stack.ReservationBytes)
	}
	fill := bytes.Repeat([]byte{FillPattern}, int(stack.ReservationBytes))
	root, err := d.NewStorage(s, tensor.DTypeI8, []int64{stack.ReservationBytes}, []tensor.KoiTag{tensor.TagN}, fill)
	if err != nil {
		return err
	}
	defer root.Close()

	f16, err := root.Reinterpret(tensor.DTypeF16, 0,
		[]int64{stack.ReservationBytes / 2}, []int64{1}, []tensor.KoiTag{tensor.TagN})
	if err != nil {
		return err
	}
	defer f16.Close()

	input, err := f16.View(uint64(stack.InputOffsetBytes/2), []int64{stack.Input.N, stack.Input.T, stack.Input.C},
		[]int64{stack.Input.T * stack.Input.C, stack.Input.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return err
	}
	defer input.Close()
	alloc := make([]*tensor.Storage, len(chain))
	crop := make([]*tensor.Storage, len(chain))
	for i, l := range chain {
		a, aerr := f16.View(uint64(l.OffsetBytes/2), []int64{l.Geometry.N, l.Geometry.Rows, l.Geometry.C},
			[]int64{l.Geometry.Rows * l.Geometry.C, l.Geometry.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
		if aerr != nil {
			return aerr
		}
		defer a.Close()
		alloc[i] = a
		c, cerr := a.View(l.CropOffsetElements(), l.CropShape(), l.CropStrides(),
			[]tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
		if cerr != nil {
			return cerr
		}
		defer c.Close()
		crop[i] = c
	}

	// The last-convolution allocation and the non-contiguous NTC view the
	// primitive writes through (WorkingMemory.cpp:62-80,87-94).
	rowBytes := last.Output.N * last.Output.C
	outAlloc, err := root.View(uint64(last.OutputOffsetBytes),
		[]int64{last.Output.Rows, last.Output.N, last.Output.C},
		[]int64{rowBytes, last.Output.C, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	if err != nil {
		return err
	}
	defer outAlloc.Close()
	outView, err := outAlloc.View(last.OutputViewOffsetElements(), last.OutputViewShape(), last.OutputViewStrides(),
		[]tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return err
	}
	defer outView.Close()

	// The input is the exact bytes the native graph consumed.
	inputPayload, err := capture.RawPayload(r.Capture, "input")
	if err != nil {
		return err
	}
	if int64(len(inputPayload)) != stack.Input.Bytes {
		return fmt.Errorf("captured input is %d bytes, the plan reserves %d", len(inputPayload), stack.Input.Bytes)
	}
	if err = input.Upload(s, inputPayload); err != nil {
		return err
	}

	caches := make([]*model.HACSmallConv, len(chain))
	for i, l := range chain {
		w := weights.Layers[i]
		cache, cerr := model.NewHACSmallConv(d, s, l.Params, w.Weight, w.Bias)
		if cerr != nil {
			return cerr
		}
		defer cache.Close()
		caches[i] = cache
	}

	// conv0: unpadded, so the crop is the whole allocation.
	if err = caches[0].Run(s, input, crop[0], chain[0]); err != nil {
		return fmt.Errorf("conv0: %w", err)
	}
	var serr error
	if r.Chain, serr = appendStage(r.Chain, r.Capture, capture, "conv0", alloc[0], s); serr != nil {
		return serr
	}

	// The conv1 allocation aliases the input buffer the graph has now consumed
	// (WorkingMemory::next reuses the front of the reservation without clearing
	// it). Pre-fill it with the declared pattern so the rows the kernel must not
	// write are observable. Nothing reads this allocation before conv1 writes it.
	if err = alloc[1].Fill(s, []byte{FillPattern, FillPattern}); err != nil {
		return err
	}
	r.DeclaredPreparation = append(r.DeclaredPreparation,
		"conv1-allocation-pre-filled-with-"+fmt.Sprintf("%#02x", FillPattern))

	// conv1: reads conv0's output, writes the cropped view nine rows into its
	// own allocation.
	if err = caches[1].Run(s, alloc[0], crop[1], chain[1]); err != nil {
		return fmt.Errorf("conv1: %w", err)
	}
	if r.Chain, serr = appendStage(r.Chain, r.Capture, capture, "conv1", alloc[1], s); serr != nil {
		return serr
	}
	if err = checkConvPadding(alloc[1], s, chain[1].Params); err != nil {
		return err
	}

	// The conv0 output is about to be partly overwritten by the last
	// convolution, so record it before that happens.
	conv0Before, err := alloc[0].Download(s)
	if err != nil {
		return err
	}

	// The source zeroes the two padding bands of the last convolution's input
	// (ConvStack.cpp:265-266) after conv1 produced it.
	if err = model.ZeroInputPadding(s, alloc[1], last.Input.N, last.Input.Rows, last.Input.C, last.Layer.Params.Padding()); err != nil {
		return err
	}
	if r.InputPadding, err = checkInputPadding(alloc[1], s, last); err != nil {
		return err
	}

	// The last convolution.
	lastConv, err := model.NewHACLastConv(d, s, last.Layer.Params, dev,
		weights.Layers[last.Layer.Index].Weight, weights.Layers[last.Layer.Index].Bias)
	if err != nil {
		return err
	}
	defer lastConv.Close()
	dispatch, err := lastConv.Run(s, alloc[1], outView, last)
	if err != nil {
		return fmt.Errorf("conv%d: %w", last.Layer.Index, err)
	}
	r.Dispatch = Dispatch{
		DevOptions:        dev.String(),
		FastAttempted:     dispatch.FastAttempted,
		FastResult:        dispatch.FastResult.String(),
		FallbackAttempted: dispatch.FallbackAttempted,
		FallbackResult:    dispatch.FallbackResult.String(),
		Path:              string(dispatch.Path),
	}

	// The whole backing, so the written rows, the aliased rows and the fill
	// spans all come from one consistent read.
	whole, err := root.Download(s)
	if err != nil {
		return err
	}
	// The stage record is assigned before the error is checked, so a failed
	// attempt keeps the observed digest.
	r.Stages, err = appendStage(r.Stages, r.Capture, capture, fmt.Sprintf("conv%d", last.Layer.Index), outAlloc, s)
	if err != nil {
		return err
	}

	if len(r.LastConv.Aliases) != 1 {
		return fmt.Errorf("the declared case needs exactly one allocation alias, the plan recorded %d", len(r.LastConv.Aliases))
	}
	alias := r.LastConv.Aliases[0]
	if alias.Layer != 0 {
		return fmt.Errorf("the last allocation aliases conv%d, not conv0", alias.Layer)
	}
	if r.Unwritten, err = checkUnwritten(whole, conv0Before, alias, last, last.Output.Bytes); err != nil {
		return err
	}
	// The two alias claims are now observed, not merely planned.
	expected := 0
	for _, span := range alias.UntouchedRows {
		expected += span[1]
	}
	r.LastConv.Aliases[0].Preserved = len(r.Unwritten.AliasedRows) == expected
	r.LastConv.Aliases[0].Changed = true
	if r.Lifetime, err = checkFill(whole, stack, last); err != nil {
		return err
	}
	return nil
}

// appendStage compares the bytes the Go caller produced with the captured stage
// payload, excluding only the rows the source declares it never writes. The mask
// comes from the capture's `unwritten_t`, which Import validated against the
// pinned source layout; it is never derived from observed mismatches.
func appendStage(stages []Stage, captureDir string, capture hacgpu.Capture, name string, got *tensor.Storage, s *tensor.Stream) ([]Stage, error) {
	observed, err := got.Download(s)
	if err != nil {
		return stages, err
	}
	return appendStageBytes(stages, captureDir, capture, name, observed)
}

func appendStageBytes(stages []Stage, captureDir string, capture hacgpu.Capture, name string, observed []byte) ([]Stage, error) {
	stage, ok := capture.Stage(name)
	if !ok {
		return stages, fmt.Errorf("capture has no stage %q", name)
	}
	payload, err := capture.RawPayload(captureDir, name)
	if err != nil {
		return stages, err
	}
	if int64(len(observed)) != stage.Bytes {
		return stages, fmt.Errorf("stage %q observed %d bytes, captured %d", name, len(observed), stage.Bytes)
	}
	written, err := visitWrittenRows(stage, observed)
	if err != nil {
		return stages, fmt.Errorf("stage %q: %w", name, err)
	}
	expected, err := visitWrittenRows(stage, payload)
	if err != nil {
		return stages, fmt.Errorf("stage %q: %w", name, err)
	}
	rec := Stage{
		Name: name, DType: stage.DType, Layout: stage.Layout,
		CaptureSHA: stage.DataSHA, GoSHA: hashBytes(observed), Shape: stage.Shape,
		WrittenBytes: int64(len(written)),
		GoWrittenSHA: hashBytes(written), CapWrittenSHA: hashBytes(expected),
		UnwrittenRows: stage.UnwrittenT, Equal: bytes.Equal(written, expected),
	}
	// The record is kept even when it differs: a failed attempt is evidence, and
	// a mismatch that leaves no digest behind cannot be diagnosed later.
	stages = append(stages, rec)
	if !rec.Equal {
		return stages, fmt.Errorf("stage %q differs from the capture in a written element (%d written bytes, Go %s, capture %s)",
			name, rec.WrittenBytes, rec.GoWrittenSHA, rec.CapWrittenSHA)
	}
	return stages, nil
}

// packAndVerify packs every layer's weights in Go and then proves the Go bytes
// against the verification-only native packing oracle that reproduces
//
//	conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(kF16)
//	conv->bias.to(kF16)
//
// (ConvStack.cpp:230-234). The oracle is never on the caller path. The first two
// layers pack to the H02 [W*C_in, C_out] layout; the last layer packs to
// [C_out, W*C_in], which the same oracle proves by transposition. No new native
// packing symbol is introduced.
func packAndVerify(d *tensor.Device, s *tensor.Stream, layers []model.HACSmallConvLayer, weights h02.ConvWeights) ([]Packing, error) {
	var out []Packing
	for _, l := range layers {
		w := weights.Layers[l.Index]
		if w.Index != l.Index {
			return nil, fmt.Errorf("weight layer %d does not match the plan's conv%d", w.Index, l.Index)
		}
		lastLayout := l.Params.OutputLayout == model.GPUStorageCUTLASSI8 || l.Params.OutputLayout == model.GPUStorageCUTLASSF16
		var goWeight []uint16
		var err error
		layout := "W*C_in,C_out"
		if lastLayout {
			goWeight, err = model.PackLastConvWeights(w.Weight, l.Params.COut, l.Params.CIn, l.Params.WinLen)
			layout = "C_out,W*C_in"
		} else {
			goWeight, err = model.PackSmallConvWeights(w.Weight, l.Params.COut, l.Params.CIn, l.Params.WinLen)
		}
		if err != nil {
			return nil, err
		}
		goBias, err := model.PackSmallConvBias(w.Bias)
		if err != nil {
			return nil, err
		}

		src, err := d.NewStorage(s, tensor.DTypeF32,
			[]int64{l.Params.COut, l.Params.CIn, l.Params.WinLen},
			[]tensor.KoiTag{tensor.TagW, tensor.TagC, tensor.TagD}, float32Bytes(w.Weight))
		if err != nil {
			return nil, err
		}
		defer src.Close()
		biasSrc, err := d.NewStorage(s, tensor.DTypeF32, []int64{l.Params.COut},
			[]tensor.KoiTag{tensor.TagC}, float32Bytes(w.Bias))
		if err != nil {
			return nil, err
		}
		defer biasSrc.Close()
		packed, err := d.NewStorage(s, tensor.DTypeF16,
			[]int64{l.Params.WinLen * l.Params.CIn, l.Params.COut},
			[]tensor.KoiTag{tensor.TagW, tensor.TagC}, make([]byte, len(goWeight)*2))
		if err != nil {
			return nil, err
		}
		defer packed.Close()
		packedBias, err := d.NewStorage(s, tensor.DTypeF16, []int64{l.Params.COut},
			[]tensor.KoiTag{tensor.TagC}, make([]byte, len(goBias)*2))
		if err != nil {
			return nil, err
		}
		defer packedBias.Close()

		if err = d.ReferencePackSmallConvF16(s, src, biasSrc, packed, packedBias); err != nil {
			return nil, fmt.Errorf("conv%d reference packing: %w", l.Index, err)
		}
		refWeight, err := packed.Download(s)
		if err != nil {
			return nil, err
		}
		refBias, err := packedBias.Download(s)
		if err != nil {
			return nil, err
		}
		goWeightBytes := model.HalfBytes(goWeight)
		goBiasBytes := model.HalfBytes(goBias)
		if lastLayout {
			refWeight = transposeHalf(refWeight, l.Params.WinLen*l.Params.CIn, l.Params.COut)
		}
		rec := Packing{
			Index: l.Index, Layout: layout,
			WeightFileSHA: w.WeightSHA, WeightShape: w.Shape, BiasFileSHA: w.BiasSHA,
			GoWeightSHA: hashBytes(goWeightBytes), ReferenceSHA: hashBytes(refWeight),
			GoBiasSHA: hashBytes(goBiasBytes), ReferenceBiasSHA: hashBytes(refBias),
			Transposed: lastLayout,
			Equal:      bytes.Equal(goWeightBytes, refWeight) && bytes.Equal(goBiasBytes, refBias),
		}
		if !rec.Equal {
			return nil, fmt.Errorf("conv%d Go packing differs from the native reference", l.Index)
		}
		out = append(out, rec)
	}
	return out, nil
}

// transposeHalf turns a row-major [rows, cols] half matrix into [cols, rows].
// It is the exact byte relation between the H02 packing and the H03 packing, so
// the H03 packer is proven against the same native bytes, not a second oracle.
func transposeHalf(src []byte, rows, cols int64) []byte {
	out := make([]byte, len(src))
	for r := int64(0); r < rows; r++ {
		for c := int64(0); c < cols; c++ {
			copy(out[(c*rows+r)*2:(c*rows+r)*2+2], src[(r*cols+c)*2:(r*cols+c)*2+2])
		}
	}
	return out
}

// checkConvPadding proves the rows outside a padded convolution's crop are still
// the declared fill pattern: the kernel wrote only its cropped view.
func checkConvPadding(alloc *tensor.Storage, s *tensor.Stream, params model.SmallConvParams) error {
	if params.OutputTPadding == 0 {
		return nil
	}
	observed, err := alloc.Download(s)
	if err != nil {
		return err
	}
	rowBytes := int(params.COut * 2)
	total := int64(len(observed)) / int64(rowBytes)
	for _, band := range [][2]int64{{0, params.OutputTPadding}, {total - params.OutputTPadding, total}} {
		for t := band[0]; t < band[1]; t++ {
			for _, b := range observed[t*int64(rowBytes) : (t+1)*int64(rowBytes)] {
				if b != FillPattern {
					return fmt.Errorf("conv padding row %d was written (byte %#02x)", t, b)
				}
			}
		}
	}
	return nil
}

// checkInputPadding proves the two bands the source zeroes are exactly zero and
// nothing else changed. It re-reads the allocation rather than trusting the
// write, so a fill that silently no-ops is caught.
func checkInputPadding(alloc *tensor.Storage, s *tensor.Stream, last *model.HACLastConvPlan) (Padding, error) {
	observed, err := alloc.Download(s)
	if err != nil {
		return Padding{}, err
	}
	padding := last.Layer.Params.Padding()
	elem := int64(1)
	if last.Input.Dtype == model.GPUDtypeHalf {
		elem = 2
	}
	rowBytes := last.Input.C * elem
	rec := Padding{Index: last.Layer.Index, Batch: last.Input.N, Rows: padding, AllZero: true}
	if padding == 0 {
		return rec, nil
	}
	for n := int64(0); n < last.Input.N; n++ {
		for _, start := range []int64{0, last.Input.Rows - padding} {
			off := (n*last.Input.Rows + start) * rowBytes
			for _, b := range observed[off : off+padding*rowBytes] {
				if b != 0 {
					rec.AllZero = false
					return rec, fmt.Errorf("input padding band [%d,%d) of batch %d holds %#02x", start, start+padding, n, b)
				}
			}
		}
	}
	rec.Bytes = 2 * last.Input.N * padding * rowBytes
	return rec, nil
}

// checkUnwritten accounts for the last allocation's unwritten rows from one
// consistent read of the backing:
//
//   - row 0 is still the declared fill pattern;
//   - rows the earlier allocation aliases and the view does not cover still hold
//     that earlier allocation's output, byte for byte;
//   - rows the view does cover no longer hold it.
//
// None of these is compared with the capture: they are declared-unwritten
// regions, so the evidence is the declared preparation and the plan's interval
// arithmetic.
func checkUnwritten(whole, earlier []byte, alias Alias, last *model.HACLastConvPlan, outBytes int64) (Unwritten, error) {
	elem := int64(1)
	if last.Output.Dtype == model.GPUDtypeHalf {
		elem = 2
	}
	rowBytes := last.Output.N * last.Output.C * elem
	rec := Unwritten{Index: last.Layer.Index, Rows: last.UnwrittenRows()}
	if int64(len(whole)) < last.OutputOffsetBytes+outBytes {
		return rec, fmt.Errorf("backing is %d bytes, the last allocation ends at %d", len(whole), last.OutputOffsetBytes+outBytes)
	}
	alloc := whole[last.OutputOffsetBytes : last.OutputOffsetBytes+outBytes]
	startRow := int64(last.OutputViewOffsetElements() / uint64(rowBytes))
	for _, r := range rec.Rows {
		rec.Bytes += int64(r[1]) * rowBytes
	}
	if startRow > 0 {
		for _, b := range alloc[:startRow*rowBytes] {
			if b != FillPattern {
				return rec, fmt.Errorf("last allocation row 0 was written (byte %#02x)", b)
			}
		}
		rec.RowZeroFill = true
	}
	if int64(len(earlier)) != alias.EarlierBytes {
		return rec, fmt.Errorf("the earlier allocation is %d bytes, the plan recorded %d", len(earlier), alias.EarlierBytes)
	}
	first := alias.FirstSharedRow
	if first < 0 || (first+alias.SharedRows)*rowBytes > outBytes {
		return rec, fmt.Errorf("the alias rows [%d,%d) fall outside the last allocation", first, first+alias.SharedRows)
	}
	written := startRow
	writtenEnd := written + last.Layer.TOut
	// A declared-unwritten row is one of two things, and the plan says which
	// before execution: still fill, or an earlier allocation's output.
	for _, r := range rec.Rows {
		for t := int64(r[0]); t < int64(r[0]+r[1]); t++ {
			if t >= written && t < writtenEnd {
				return rec, fmt.Errorf("row %d is declared unwritten but the view covers it", t)
			}
			row := alloc[t*rowBytes : (t+1)*rowBytes]
			if t >= first && t < first+alias.SharedRows {
				srcRow := t - first
				if !bytes.Equal(row, earlier[srcRow*rowBytes:(srcRow+1)*rowBytes]) {
					return rec, fmt.Errorf("unwritten row %d does not still hold the earlier allocation's output", t)
				}
				rec.AliasedRows = append(rec.AliasedRows, [2]int{int(t), 1})
				rec.AliasedBytes += rowBytes
				continue
			}
			for _, b := range row {
				if b != FillPattern {
					return rec, fmt.Errorf("unwritten row %d is outside the alias and was written (byte %#02x)", t, b)
				}
			}
		}
	}
	for _, span := range alias.OverwrittenRows {
		for t := int64(span[0]); t < int64(span[0]+span[1]); t++ {
			srcRow := t - first
			if srcRow < 0 || (srcRow+1)*rowBytes > int64(len(earlier)) {
				continue
			}
			if bytes.Equal(alloc[t*rowBytes:(t+1)*rowBytes], earlier[srcRow*rowBytes:(srcRow+1)*rowBytes]) {
				return rec, fmt.Errorf("written row %d still holds the earlier allocation's output", t)
			}
		}
	}
	rec.StillFill = true
	return rec, nil
}

// checkFill proves every byte outside the ranges the graph legitimately wrote is
// still the declared fill pattern. The allowed ranges are derived from the plan,
// not from the observed bytes.
func checkFill(whole []byte, stack *model.HACSmallConvPlan, last *model.HACLastConvPlan) (Lifetime, error) {
	rec := Lifetime{}
	if int64(len(whole)) != stack.ReservationBytes {
		return rec, fmt.Errorf("backing is %d bytes, the plan reserves %d", len(whole), stack.ReservationBytes)
	}
	elem := int64(1)
	if last.Output.Dtype == model.GPUDtypeHalf {
		elem = 2
	}
	rowBytes := last.Output.N * last.Output.C * elem
	written := [][2]int64{
		{stack.InputOffsetBytes, stack.InputOffsetBytes + stack.Input.Bytes},
	}
	for i := 0; i < 2; i++ {
		l := stack.Layers[i]
		written = append(written, [2]int64{l.OffsetBytes, l.OffsetBytes + l.Geometry.Bytes})
	}
	start := last.OutputOffsetBytes + int64(last.OutputViewOffsetElements()/uint64(rowBytes))*rowBytes
	written = append(written, [2]int64{start, start + last.Layer.TOut*rowBytes})
	sort.Slice(written, func(i, j int) bool { return written[i][0] < written[j][0] })
	for _, span := range subtract(0, stack.ReservationBytes, written) {
		for _, b := range whole[span[0]:span[1]] {
			if b != FillPattern {
				return rec, fmt.Errorf("backing span [%d,%d) was written but no declared view covers it", span[0], span[1])
			}
		}
		rec.FillSpans = append(rec.FillSpans, span)
	}
	rec.BackingStillFill = true
	return rec, nil
}

// subtract removes every covered interval from [start,end) and returns the
// remaining spans in ascending order.
func subtract(start, end int64, covered [][2]int64) [][2]int64 {
	spans := [][2]int64{{start, end}}
	for _, c := range covered {
		var next [][2]int64
		for _, s := range spans {
			if c[1] <= s[0] || s[1] <= c[0] {
				next = append(next, s)
				continue
			}
			if s[0] < c[0] {
				next = append(next, [2]int64{s[0], c[0]})
			}
			if c[1] < s[1] {
				next = append(next, [2]int64{c[1], s[1]})
			}
		}
		spans = next
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i][0] < spans[j][0] })
	return spans
}

func float32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// runtimePins records the exact shared objects this process mapped, so the
// receipt names the libraries that ran rather than the sonames the link line
// asked for.
func runtimePins() (map[string]string, error) {
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	families := map[string]bool{}
	for _, prefix := range []string{"libtorch_cpu.so", "libc10.so", "libtorch_cuda.so", "libc10_cuda.so",
		"libcudart.so", "libcublas.so", "libcuda.so", "libkoi"} {
		families[prefix] = true
	}
	seen := map[string]bool{}
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !filepath.IsAbs(path) || seen[path] {
			continue
		}
		base := filepath.Base(path)
		matched := false
		for prefix := range families {
			if strings.HasPrefix(base, prefix) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		seen[path] = true
		sum, herr := fileSHA(path)
		if herr != nil {
			return nil, herr
		}
		out[path] = sum
	}
	if len(out) < 6 {
		names := make([]string, 0, len(out))
		for path := range out {
			names = append(names, path)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("only %d pinned runtime libraries mapped: %v", len(out), names)
	}
	return out, nil
}
