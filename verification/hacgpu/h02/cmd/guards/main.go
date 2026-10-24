// Command guards exercises the H02 native small-convolution guards directly.
//
// It exists so a rejection control can prove the *native* guard fires: every
// call goes through Device.SmallConvF16Unchecked, which bypasses the Go
// pre-checks, so an observed rejection can only come from the boundary. The
// control script removes one native guard at a time, rebuilds the backend and
// re-runs this command; a changed outcome is the evidence that the guard was
// load-bearing.
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

	// The declared conv0 geometry: input [64,192,1], weights [5,16], bias [16],
	// output [64,192,16], all F16.
	f16 := func(shape []int64) *tensor.Storage {
		n := int64(1)
		for _, v := range shape {
			n *= v
		}
		tags := make([]tensor.KoiTag, len(shape))
		for i := range tags {
			tags[i] = tensor.TagC
		}
		st, err := d.NewStorage(s, tensor.DTypeF16, shape, tags, make([]byte, n*2))
		if err != nil {
			panic(err)
		}
		return st
	}
	in := f16([]int64{64, 192, 1})
	w := f16([]int64{5, 16})
	b := f16([]int64{16})
	out := f16([]int64{64, 192, 16})
	defer in.Close()
	defer w.Close()
	defer b.Close()
	defer out.Close()

	f32Tags := []tensor.KoiTag{tensor.TagW, tensor.TagC}
	f32w, err := d.NewStorage(s, tensor.DTypeF32, []int64{5, 16}, f32Tags, make([]byte, 5*16*4))
	if err != nil {
		return err
	}
	defer f32w.Close()

	base := tensor.SmallConvGeometry{
		N: 64, CIn: 1, COut: 16, TIn: 192, Window: 5, Stride: 1,
		Padding: 2, OutNStride: 192 * 16, Activation: tensor.KoiSwish,
	}
	// The padding control deliberately also carries a wrong output stride, so
	// that removing the padding guard advances the request to the next guard
	// instead of reaching the kernel with an out-of-bounds read.
	paddingCase := base
	paddingCase.Padding = 0
	paddingCase.OutNStride = 192*16 - 16
	strideCase := base
	strideCase.OutNStride = 192*16 - 16
	multipleCase := base
	multipleCase.TIn = 193
	multipleCase.Stride = 2
	activationCase := base
	activationCase.Activation = tensor.KoiTanhX5

	cases := []struct {
		name string
		g    tensor.SmallConvGeometry
		w    *tensor.Storage
	}{
		{"valid", base, w},
		{"padding", paddingCase, w},
		{"stride", strideCase, w},
		{"stride-multiple", multipleCase, w},
		{"activation", activationCase, w},
		{"dtype", base, f32w},
	}
	for _, c := range cases {
		err := d.SmallConvF16Unchecked(s, in, c.w, b, out, c.g)
		switch {
		case err == nil:
			fmt.Printf("%s: ok\n", c.name)
		case errors.Is(err, tensor.ErrClosed):
			fmt.Printf("%s: closed: %v\n", c.name, err)
		default:
			fmt.Printf("%s: rejected: %s\n", c.name, strings.TrimSpace(err.Error()))
		}
	}
	return nil
}
