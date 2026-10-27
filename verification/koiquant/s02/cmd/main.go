package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant/s02"
)

func main() {
	cuda := flag.String("cuda", "artifacts/koiquant/attempt-resume-replay-cuda-20261024-001/manifest.json", "accepted saved CUDA import")
	comparison := flag.String("comparison", "artifacts/koiquant/attempt-resume-comparison-20261024-001/manifest.json", "accepted saved CPU/CUDA comparison")
	out := flag.String("out", "", "fresh host differential output directory")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "-out is required")
		os.Exit(1)
	}
	m, err := s02.Verify(".", *cuda, *comparison, *out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("%s: %d fixtures, %d admitted rows, %d rejected rows\n", m.Status, len(m.Fixtures), m.Admitted, m.Rejected)
	for _, v := range m.Mutations {
		fmt.Printf("%s: changed int8=%d inverse-bits=%d rejected-rows=%d\n", v.Name, v.DifferentI8, v.DifferentInverseBits, v.RejectedRows)
	}
}
