package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h05"
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
	var o h05.Options
	flag.StringVar(&o.Capture, "capture", "", "retained H01 stages directory")
	flag.StringVar(&o.Caches, "caches", "", "retained H04 native caches directory")
	flag.StringVar(&o.Backend, "backend", "", "ABI4 CUDA13 backend")
	flag.StringVar(&o.Out, "out", "", "fresh artifacts/tensor attempt directory")
	flag.IntVar(&o.Ordinal, "ordinal", 0, "visible CUDA ordinal")
	flag.BoolVar(&o.Admitted, "runtime-admitted", false, "coordinator admitted this bounded native execution")
	flag.BoolVar(&o.Controls, "controls", false, "run separately admitted native guard and cache-order controls after positive acceptance")
	inspect := flag.Bool("inspect", false, "verify retained input pins without native loading or a host lock")
	flag.Parse()
	if *inspect {
		if err := h05.Inspect(o); err != nil {
			return err
		}
		fmt.Println("retained H01/H04 input pins and prospective H05 row masks verified; native execution not run")
		return nil
	}
	if o.Backend == "" || o.Out == "" {
		return fmt.Errorf("-backend and -out required for native replay")
	}
	r, err := h05.Run(context.Background(), o)
	b, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		return e
	}
	fmt.Println(string(b))
	return err
}
