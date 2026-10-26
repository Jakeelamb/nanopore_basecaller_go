// Command guards exercises the H03 native last-convolution guards directly.
//
// It exists so a rejection control can prove the *native* guard fires: every
// call goes through Device.LastConvKoiUnchecked or Device.LastConvLinearUnchecked,
// which bypass the Go pre-checks, so an observed rejection can only come from the
// boundary. The control script removes one native guard at a time, rebuilds the
// backend and re-runs this command; a changed outcome is the evidence that the
// guard was load-bearing.
//
// Every case uses the declared case's geometry (batch 64, T_in 192, C_in 16,
// C_out 1024, window 19, stride 6, padding 9, T_out 32). A scaled-down geometry
// was tried first and the primitive refuses it with KOI_NOT_SUPPORTED, which
// would have made "the kernel ran" indistinguishable from "the kernel refused
// the geometry". See artifacts/tensor/h03-controls-001.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}

func run() error {
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	backend := flag.String("backend", "", "ABI4 CUDA13 tensor backend")
	ordinal := flag.Int("ordinal", 0, "visible CUDA ordinal")
	admitted := flag.Bool("runtime-admitted", false, "coordinator released this bounded native invocation")
	flag.Parse()
	if *backend == "" {
		return fmt.Errorf("-backend required")
	}
	if !*admitted {
		return fmt.Errorf("guard probe requires explicit coordinator admission")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return err
	}
	defer lock.Close()

	e, err := tensor.Open(*backend)
	if err != nil {
		return err
	}
	defer e.Close()
	d, err := e.NewCUDADevice(*ordinal)
	if err != nil {
		return err
	}
	s, err := d.NewStream()
	if err != nil {
		return err
	}

	// The declared case's geometry, in elements.
	const (
		n, tIn, cIn, cOut   = 64, 192, 16, 1024
		window, stride      = 19, 6
		padding             = window / 2
		tAlloc              = tIn + 2*padding
		tOut                = tIn / stride
		allocRows           = tOut + 2*padding // 37: the CUTLASS_TNC_I8 allocation
		viewOffsetElements  = n * cOut         // one time row: narrow(0, 1, T_out)
		rowBytes            = n * cOut
		inNStride           = tAlloc * cIn
		outNStride          = cOut
		outTStride          = n * cOut
		convAllocationBytes = allocRows * rowBytes
	)

	nct := []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}
	storage := func(dtype tensor.DType, shape []int64, tags []tensor.KoiTag) *tensor.Storage {
		count := int64(1)
		for _, v := range shape {
			count *= v
		}
		st, serr := d.NewStorage(s, dtype, shape, tags, make([]byte, count*int64(dtype.Size())))
		if serr != nil {
			panic(serr)
		}
		return st
	}

	in := storage(tensor.DTypeF16, []int64{n, tAlloc, cIn}, nct)
	w := storage(tensor.DTypeF16, []int64{cOut, window * cIn}, []tensor.KoiTag{tensor.TagC, tensor.TagW})
	b := storage(tensor.DTypeF16, []int64{cOut}, []tensor.KoiTag{tensor.TagC})
	root := storage(tensor.DTypeI8, []int64{convAllocationBytes}, []tensor.KoiTag{tensor.TagT})
	out, err := root.View(viewOffsetElements, []int64{n, tOut, cOut}, []int64{outNStride, outTStride, 1}, nct)
	if err != nil {
		return err
	}
	// A one-row view: with T_in 6 the output shape still agrees, so only the
	// allocation-rows guard can reject that case.
	oneRow, err := root.View(viewOffsetElements, []int64{n, 1, cOut}, []int64{outNStride, outTStride, 1}, nct)
	if err != nil {
		return err
	}
	// A contiguous view of the same extent, for the output-strides case.
	contigOut, err := root.View(viewOffsetElements, []int64{n, tOut, cOut}, []int64{tOut * cOut, cOut, 1}, nct)
	if err != nil {
		return err
	}
	defer in.Close()
	defer w.Close()
	defer b.Close()
	defer root.Close()
	defer out.Close()
	defer oneRow.Close()
	defer contigOut.Close()

	// F32 input for the dtype guard.
	f32in := storage(tensor.DTypeF32, []int64{n, tAlloc, cIn}, nct)
	defer f32in.Close()

	// A shorter weight whose strides match its own shape, so the weight-shape
	// guard is the one that rejects it. The parent carries one extra row so the
	// mutant's read of the full 304-element window stays inside the allocation.
	shortWRoot := storage(tensor.DTypeF16, []int64{cOut * (window*cIn + cIn)}, []tensor.KoiTag{tensor.TagC})
	defer shortWRoot.Close()
	shortW, err := shortWRoot.View(0, []int64{cOut, window*cIn - 1}, []int64{window*cIn - 1, 1},
		[]tensor.KoiTag{tensor.TagC, tensor.TagW})
	if err != nil {
		return err
	}
	defer shortW.Close()

	// An F16 output of the same shape for the output-type guard. An I8 request
	// writes fewer bytes than the allocation holds, so the mutant stays inside.
	f16Root := storage(tensor.DTypeF16, []int64{convAllocationBytes}, []tensor.KoiTag{tensor.TagT})
	defer f16Root.Close()
	f16Out, err := f16Root.View(viewOffsetElements, []int64{n, tOut, cOut}, []int64{outNStride, outTStride, 1}, nct)
	if err != nil {
		return err
	}
	defer f16Out.Close()

	// The overlap guard needs the input and the output to be views of one
	// allocation with intersecting byte intervals. The 2097152-byte I8 output
	// starts at byte 65536 and fits inside the parent.
	overlapRoot := storage(tensor.DTypeI8, []int64{3000000}, []tensor.KoiTag{tensor.TagN})
	defer overlapRoot.Close()
	overlapF16, err := overlapRoot.Reinterpret(tensor.DTypeF16, 0, []int64{1500000}, []int64{1}, []tensor.KoiTag{tensor.TagN})
	if err != nil {
		return err
	}
	defer overlapF16.Close()
	overlapIn, err := overlapF16.View(0, []int64{n, tAlloc, cIn}, []int64{inNStride, cIn, 1}, nct)
	if err != nil {
		return err
	}
	defer overlapIn.Close()
	overlapOut, err := overlapRoot.View(viewOffsetElements, []int64{n, tOut, cOut}, []int64{outNStride, outTStride, 1}, nct)
	if err != nil {
		return err
	}
	defer overlapOut.Close()
	// The old element-endpoint comparison accepted this offset because 215040
	// F16 elements precede 262144 I8 elements. Their byte intervals overlap by
	// 167936 bytes. Both dispatch paths must reject it before enqueueing.
	shiftedOverlapOut, err := overlapRoot.View(262144, []int64{n, tOut, cOut}, []int64{outNStride, outTStride, 1}, nct)
	if err != nil {
		return err
	}
	defer shiftedOverlapOut.Close()

	// A contiguous allocation too short for the fallback's last window.
	tinyIn := storage(tensor.DTypeF16, []int64{n, 20, cIn}, nct)
	defer tinyIn.Close()

	base := tensor.LastConvGeometry{
		N: n, TIn: tIn, CIn: cIn, COut: cOut, Window: window, Stride: stride, Padding: padding,
		InNStride: inNStride, OutNStride: outNStride, OutTStride: outTStride,
		Activation: tensor.KoiTanh, InType: tensor.KoiF16, OutType: tensor.KoiI8, AccumType: tensor.KoiF16,
	}
	paddingCase := base
	paddingCase.Padding = padding - 1
	rowsCase := base
	rowsCase.TIn = 6
	strideCase := base
	strideCase.InNStride = inNStride - 1
	outStridesCase := base
	outStridesCase.OutTStride = tOut * cOut
	activationCase := base
	activationCase.Activation = tensor.KoiIdentity
	accumCase := base
	accumCase.AccumType = tensor.KoiI8
	linearWindowCase := base
	linearWindowCase.InNStride = 20 * cIn

	type probe struct {
		name   string
		linear bool
		in, w  *tensor.Storage
		b      *tensor.Storage
		out    *tensor.Storage
		g      tensor.LastConvGeometry
	}
	cases := []probe{
		{"valid", false, in, w, b, out, base},
		{"padding", false, in, w, b, out, paddingCase},
		{"input-rows", false, in, w, b, oneRow, rowsCase},
		{"input-strides", false, in, w, b, out, strideCase},
		{"output-strides", false, in, w, b, contigOut, outStridesCase},
		{"weight-shape", false, in, shortW, b, out, base},
		{"activation", false, in, w, b, out, activationCase},
		{"input-type", false, f32in, w, b, out, base},
		{"output-type", false, in, w, b, f16Out, base},
		{"accumulation", false, in, w, b, out, accumCase},
		{"overlap", false, overlapIn, w, b, overlapOut, base},
		{"shifted-overlap", false, overlapIn, w, b, shiftedOverlapOut, base},
		{"fallback-valid", true, in, w, b, out, base},
		{"fallback-shifted-overlap", true, overlapIn, w, b, shiftedOverlapOut, base},
		{"fallback-weight-shape", true, in, shortW, b, out, base},
		{"fallback-window", true, tinyIn, w, b, out, linearWindowCase},
	}
	for _, c := range cases {
		var res tensor.KoiResult
		var err error
		if c.linear {
			res, err = d.LastConvLinearUnchecked(s, c.in, c.w, c.b, c.out, c.g)
		} else {
			res, err = d.LastConvKoiUnchecked(s, c.in, c.w, c.b, c.out, c.g)
		}
		switch {
		case err != nil && errors.Is(err, tensor.ErrClosed):
			fmt.Printf("%s: closed: %v\n", c.name, err)
		case err != nil:
			fmt.Printf("%s: rejected: %s\n", c.name, strings.TrimSpace(err.Error()))
		case res != tensor.KoiSuccess:
			fmt.Printf("%s: koi %s\n", c.name, res)
		default:
			fmt.Printf("%s: ok\n", c.name)
		}
	}
	return nil
}
