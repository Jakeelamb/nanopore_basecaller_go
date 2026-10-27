// Command hacgpu builds the verification-only HAC stage oracle and performs the
// admitted stage capture. It never runs without explicit coordinator admission.
package main

import (
	"flag"
	"fmt"
	"os"

	hacmodel "github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h02"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h03"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h04"
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
	unit := flag.String("unit", "", "build, capture, verify, compare, sources, small-convs, last-conv, flstm-weights, flstm-weights-replay or flstm-weights-consumer")
	buildDir := flag.String("build-dir", "", "qualified GCC15 build tree")
	dorado := flag.String("dorado-source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout")
	build := flag.String("build", "", "oracle build manifest")
	source := flag.String("source", "", "oracle translation unit for -unit build (default: the H01 stage oracle)")
	model := flag.String("model", "", "pinned HAC model directory")
	fixture := flag.String("fixture", "", "pinned POD5 fixture manifest")
	read := flag.Int("read", 0, "fixture read index")
	signal := flag.String("signal", "", "raw int16 signal file")
	batch := flag.Int("batch", 64, "fixed batch size, a multiple of 64")
	chunk := flag.Int("chunk", 0, "fixed chunk length in samples")
	out := flag.String("out", "", "fresh artifacts directory")
	compare := flag.String("compare", "", "second capture directory for -unit compare")
	captureDir := flag.String("capture", "", "admitted H01 capture directory for -unit small-convs")
	backend := flag.String("backend", "", "ABI4 CUDA13 tensor backend for GPU units or flstm-weights-consumer")
	input := flag.String("input", "", "pinned raw f16 case input for -unit flstm-weights")
	receipt := flag.String("receipt", "", "retained producer receipt for -unit flstm-weights-replay")
	receiptSHA := flag.String("receipt-sha256", "", "independently retained producer receipt SHA256")
	ordinal := flag.Int("ordinal", 0, "CUDA ordinal")
	admitted := flag.Bool("runtime-admitted", false, "coordinator released this bounded native invocation")
	koiConv := flag.Bool("koi-conv", true, "source dev option koi_conv for -unit last-conv")
	koiConvF32 := flag.Bool("koi-conv-f32", false, "source dev option koi_conv_f32 for -unit last-conv")
	flag.Parse()

	switch *unit {
	case "flstm-weights-consumer":
		r, err := h04.VerifyConsumer(h04.ConsumerOptions{ReplayOptions: h04.ReplayOptions{Receipt: *receipt, ReceiptSHA256: *receiptSHA, Out: *out}, Backend: *backend, Admitted: *admitted})
		if err != nil {
			return err
		}
		fmt.Printf("%s: %d fresh Go/native pairs, %d final caches, %d arithmetic mutants\n", r.Status, r.NativePayloads, r.FinalCaches, len(r.Controls))
		return nil
	case "flstm-weights-replay":
		r, err := h04.Replay(h04.ReplayOptions{Receipt: *receipt, ReceiptSHA256: *receiptSHA, Out: *out})
		if err != nil {
			return err
		}
		fmt.Printf("%s: %d native payloads, %d final caches, %d changed-copy controls\n", r.Status, r.NativePayloads, r.FinalCaches, len(r.Controls))
		return nil
	case "sources":
		if err := hacgpu.VerifyStageSources(*dorado); err != nil {
			return err
		}
		fmt.Println("pinned HAC stage sources verified")
		return nil
	case "build":
		if *buildDir == "" || *out == "" {
			return fmt.Errorf("-build-dir and -out required")
		}
		m, err := hacgpu.BuildOracle(hacgpu.BuildOptions{BuildDir: *buildDir, DoradoSource: *dorado, Out: *out, Source: *source})
		if err != nil {
			return err
		}
		fmt.Printf("%s\n%s\n", m.Output, m.OutputSHA)
		return nil
	case "capture":
		if *build == "" || *model == "" || *fixture == "" || *signal == "" || *out == "" {
			return fmt.Errorf("-build, -model, -fixture, -signal and -out required")
		}
		if *chunk <= 0 {
			return fmt.Errorf("-chunk required")
		}
		r, err := hacgpu.RunCapture(hacgpu.CaptureOptions{
			Build: *build, Dorado: *dorado, Model: *model, Fixture: *fixture,
			Read: *read, Signal: *signal, Batch: *batch, Chunk: *chunk, Out: *out,
			Ordinal: *ordinal, Admitted: *admitted,
		})
		if err != nil {
			return err
		}
		fmt.Printf("%s %s\n", r.Status, r.Capture)
		return nil
	case "verify":
		if *out == "" {
			return fmt.Errorf("-out required")
		}
		c, err := hacgpu.Import(*out)
		if err != nil {
			return err
		}
		fmt.Printf("capture verified: %d stages, %s, reservation %d\n", len(c.Stages), c.StageList(), c.Reservation)
		return nil
	case "small-convs":
		if *captureDir == "" || *backend == "" || *model == "" || *out == "" {
			return fmt.Errorf("-capture, -backend, -model and -out required")
		}
		r, err := h02.Run(h02.Options{
			Capture: *captureDir, Model: *model, Backend: *backend, Out: *out,
			Ordinal: *ordinal, Admitted: *admitted,
		})
		if err != nil {
			return err
		}
		fmt.Printf("%s %s\n", r.Status, r.Attempt)
		return nil
	case "last-conv":
		if *captureDir == "" || *backend == "" || *model == "" || *out == "" {
			return fmt.Errorf("-capture, -backend, -model and -out required")
		}
		dev := hacmodel.DefaultLastConvDevOptions()
		dev.KoiConv = *koiConv
		dev.KoiConvF32 = *koiConvF32
		r, err := h03.Run(h03.Options{
			Capture: *captureDir, Model: *model, Backend: *backend, Out: *out,
			Ordinal: *ordinal, Admitted: *admitted, DevOptions: &dev,
		})
		if err != nil {
			return err
		}
		fmt.Printf("%s %s\n", r.Status, r.Attempt)
		return nil
	case "flstm-weights":
		if *build == "" || *backend == "" || *model == "" || *input == "" || *out == "" {
			return fmt.Errorf("-build, -backend, -model, -input and -out required")
		}
		if *chunk <= 0 {
			return fmt.Errorf("-chunk required")
		}
		r, err := h04.Run(h04.Options{
			Build: *build, Backend: *backend, Model: *model, Input: *input, Out: *out,
			Ordinal: *ordinal, Batch: *batch, Chunk: *chunk, Admitted: *admitted,
		})
		if err != nil {
			return err
		}
		fmt.Printf("%s %s\n", r.Status, r.Attempt)
		return nil
	case "compare":
		if *out == "" || *compare == "" {
			return fmt.Errorf("-out and -compare required")
		}
		n, err := hacgpu.Compare(*out, *compare)
		if err != nil {
			return err
		}
		fmt.Printf("captures identical: %d stages, every payload and backing dump matches\n", n)
		return nil
	default:
		return fmt.Errorf("unknown unit %q", *unit)
	}
}
