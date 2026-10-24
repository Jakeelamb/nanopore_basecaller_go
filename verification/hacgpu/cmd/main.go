// Command hacgpu builds the verification-only HAC stage oracle and performs the
// admitted stage capture. It never runs without explicit coordinator admission.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
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
	unit := flag.String("unit", "", "build, capture, verify, compare or sources")
	buildDir := flag.String("build-dir", "", "qualified GCC15 build tree")
	dorado := flag.String("dorado-source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout")
	build := flag.String("build", "", "oracle build manifest")
	model := flag.String("model", "", "pinned HAC model directory")
	fixture := flag.String("fixture", "", "pinned POD5 fixture manifest")
	read := flag.Int("read", 0, "fixture read index")
	signal := flag.String("signal", "", "raw int16 signal file")
	batch := flag.Int("batch", 64, "fixed batch size, a multiple of 64")
	chunk := flag.Int("chunk", 0, "fixed chunk length in samples")
	out := flag.String("out", "", "fresh artifacts directory")
	compare := flag.String("compare", "", "second capture directory for -unit compare")
	ordinal := flag.Int("ordinal", 0, "CUDA ordinal")
	admitted := flag.Bool("runtime-admitted", false, "coordinator released this bounded native invocation")
	flag.Parse()

	switch *unit {
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
		m, err := hacgpu.BuildOracle(hacgpu.BuildOptions{BuildDir: *buildDir, DoradoSource: *dorado, Out: *out})
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
