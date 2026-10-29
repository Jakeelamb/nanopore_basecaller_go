// The source-only command has no build, execution, admission, or device modes.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/packed"
)

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("mode required: prepare or generate")
	}
	f := flag.NewFlagSet(args[0], flag.ContinueOnError)
	out := f.String("out", "", "fresh absolute output directory")
	switch args[0] {
	case "prepare":
		request := f.String("request", "", "canonical request JSON path")
		sha := f.String("request-sha", "", "independent SHA256 of request")
		size := f.Int64("request-bytes", 0, "independent request extent")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		if f.NArg() != 0 {
			return fmt.Errorf("unexpected positional arguments")
		}
		_, err := packed.Prepare(packed.Pin{Path: *request, SHA256: *sha, Bytes: *size}, *out)
		return err
	case "generate":
		source := f.String("source-root", "", "canonical pinned Dorado source root")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		if f.NArg() != 0 {
			return fmt.Errorf("unexpected positional arguments")
		}
		_, err := packed.Generate(*source, *out)
		return err
	default:
		return fmt.Errorf("unknown source-only mode %q", args[0])
	}
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
