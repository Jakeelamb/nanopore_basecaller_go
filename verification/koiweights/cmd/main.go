package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

func main() {
	var o koiweights.Options
	var admissionControl, mutants bool
	flag.BoolVar(&admissionControl, "admission-control", false, "prove held stock lock blocks capture before commands")
	flag.BoolVar(&mutants, "mutants", false, "require behavioral Go mutants to fail without native execution")
	flag.StringVar(&o.Source, "source", "", "pinned Dorado source root")
	flag.StringVar(&o.TensorManifest, "tensor-manifest", "", "qualified CPU13 tensor manifest")
	flag.StringVar(&o.Weight, "weight", "", "pinned SUP first-layer QKV archive")
	flag.StringVar(&o.Out, "out", "", "fresh output attempt directory")
	flag.StringVar(&o.Replay, "replay", "", "retained capture manifest; launches no native producer")
	flag.Parse()
	var err error
	o.Root, err = os.Getwd()
	if err == nil {
		if admissionControl {
			err = koiweights.AdmissionControl(o)
		} else if mutants {
			err = koiweights.Mutants(o)
		} else if o.Replay != "" {
			err = koiweights.Replay(o)
		} else {
			err = koiweights.Capture(o)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		if errors.Is(err, koiweights.ErrAdmissionBlocked) {
			os.Exit(3)
		}
		os.Exit(1)
	}
	if admissionControl {
		fmt.Println("verified held stock lock blocks capture before compiler or producer commands")
	} else if mutants {
		fmt.Println("verified six Go behavior mutants are rejected")
	} else {
		fmt.Println("verified SUP QKV FP16 source-block parity; no GPU or caller acceptance")
	}
}
