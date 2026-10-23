package main

import (
	"encoding/json"
	"flag"
	backend "github.com/Jakeelamb/nanopore_basecaller_go/verification/backend"
	"os"
	"strings"
)

func main() {
	source := flag.String("source", "external/dorado-source", "pinned Dorado source checkout")
	dependencies := flag.String("dependencies", "external/backend", "archive directory plus torch, koi and pod5 development roots")
	cuda := flag.String("cuda", "/opt/cuda", "CUDA toolkit root")
	family := flag.String("cuda-family", "12.8", "pinned dependency family, 12.8 or 13.0")
	acquire := flag.Bool("acquire", false, "download and extract pinned packages into dependency root with 30 GiB reserve")
	packages := flag.String("packages", "pod5,koi,torch", "comma-separated acquisition package names")
	flag.Parse()
	if *acquire {
		receipt, err := backend.AcquireSelected(*dependencies, *family, strings.Split(*packages, ","))
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		_ = encoder.Encode(receipt)
		if err != nil {
			println(err.Error())
			os.Exit(1)
		}
		return
	}
	report := backend.Inspect(*source, *dependencies, *cuda, *family)
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		os.Exit(2)
	}
	if report.MissingOrInvalid > 0 {
		os.Exit(1)
	}
}
