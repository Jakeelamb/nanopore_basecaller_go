package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/gpuprotocol"
)

func main() {
	plan := flag.String("plan", "provenance/gpu-acceptance-plan.json", "prospective protocol")
	retained := flag.Bool("verify-retained", false, "read pinned retained manifests and memberships only")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "unexpected arguments")
		os.Exit(2)
	}
	f, err := os.Open(*plan)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	p, err := gpuprotocol.Decode(f)
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err == nil {
		err = gpuprotocol.Check(p)
	}
	if err == nil && *retained {
		err = gpuprotocol.VerifyPins(p)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println("PROTOCOL_VALID prospective-unarmed; runtime acceptance pending")
}
