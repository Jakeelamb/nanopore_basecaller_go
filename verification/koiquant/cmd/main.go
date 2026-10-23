package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant"
)

func main() {
	root := flag.String("root", ".", "canonical Go module root")
	input := flag.String("input-capture", "artifacts/koiweights/attempt-qkv-f16-002/manifest.json", "pinned existing QKV half capture")
	out := flag.String("out", "", "fresh artifacts/koiquant/attempt-* output")
	device := flag.String("device", "cpu", "planned capture tier, cpu or cuda; does not initialize a device")
	check := flag.String("check", "", "validate a prepared source-only manifest without native execution")
	flag.Parse()
	if flag.NArg() != 0 {
		fail(fmt.Errorf("unexpected positional arguments"))
	}
	absRoot, err := filepath.Abs(*root)
	if err != nil {
		fail(err)
	}
	if *check != "" {
		if *out != "" {
			fail(fmt.Errorf("-check and -out are mutually exclusive"))
		}
		_, err = koiquant.CheckPrepared(absRoot, *check)
		if err != nil {
			fail(err)
		}
		fmt.Println("prepared source and inputs verified; native execution remains held")
		return
	}
	if *out == "" {
		fail(fmt.Errorf("-out is required to prepare a fresh attempt"))
	}
	_, err = koiquant.Prepare(koiquant.Options{Root: absRoot, InputCapture: *input, Out: *out, Device: *device, Invocation: os.Args})
	if err != nil {
		fail(err)
	}
	fmt.Println(filepath.Join(*out, "manifest.json"))
	fmt.Println("source-only preparation complete; native execution remains held")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
