package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu/s05"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	o := s05.Options{}
	for _, v := range []struct {
		name string
		p    *s05.Anchor
	}{{"capture", &o.Capture}, {"prepared", &o.Prepared}, {"build", &o.Build}, {"runtime", &o.Runtime}} {
		flag.StringVar(&v.p.Path, v.name, "", "independently pinned native artifact")
		flag.StringVar(&v.p.SHA256, v.name+"-sha256", "", "external artifact SHA256")
	}
	flag.StringVar(&o.Model, "model", "/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0", "fixed source parameter directory")
	flag.StringVar(&o.Out, "out", "", "fresh ignored result directory")
	flag.Parse()
	r, e := s05.Run(context.Background(), o)
	b, m := json.MarshalIndent(r, "", "  ")
	if m == nil {
		fmt.Println(string(b))
	}
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(runner.ExitCode(e))
	}
	if m != nil {
		fmt.Fprintln(os.Stderr, m)
		os.Exit(1)
	}
}
