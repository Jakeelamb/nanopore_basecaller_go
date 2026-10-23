package main

import (
	"flag"
	"fmt"
	"os"

	modeloracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/model"
)

func main() {
	directory := flag.String("model", "", "read-only stock SUP model directory")
	flag.Parse()
	if *directory == "" {
		fmt.Fprintln(os.Stderr, "-model is required")
		os.Exit(2)
	}
	if err := modeloracle.WriteSUPWeightManifest(*directory, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
