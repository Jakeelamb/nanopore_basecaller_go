package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu/graph"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type options struct {
	mode                        string
	admission, request, receipt nativeproof.Pin
}

func parse(args []string) (o options, err error) {
	f := flag.NewFlagSet("sup-graph", flag.ContinueOnError)
	f.SetOutput(io.Discard)
	f.StringVar(&o.mode, "mode", "", "consumer, guarded-child, replay")
	f.StringVar(&o.admission.Path, "admission", "", "independent admission path")
	f.StringVar(&o.admission.SHA256, "admission-sha", "", "independent admission SHA256")
	f.StringVar(&o.request.Path, "request", "", "private request path")
	f.StringVar(&o.request.SHA256, "request-sha", "", "private request SHA256")
	f.StringVar(&o.receipt.Path, "receipt", "", "wrapper receipt path")
	f.StringVar(&o.receipt.SHA256, "receipt-sha", "", "wrapper receipt SHA256")
	if err = f.Parse(args); err != nil {
		return
	}
	if f.NArg() != 0 {
		return o, fmt.Errorf("positional arguments forbidden")
	}
	seen := map[string]bool{}
	for _, arg := range args {
		if len(arg) > 0 && arg[0] == '-' {
			name := arg
			for len(name) > 0 && name[0] == '-' {
				name = name[1:]
			}
			for i, c := range name {
				if c == '=' {
					name = name[:i]
					break
				}
			}
			if seen[name] {
				return o, fmt.Errorf("duplicate flag %s", name)
			}
			seen[name] = true
		}
	}
	allowed := map[string]bool{"mode": true}
	switch o.mode {
	case "consumer":
		allowed["admission"] = true
		allowed["admission-sha"] = true
	case "guarded-child":
		allowed["request"] = true
		allowed["request-sha"] = true
	case "replay":
		allowed["admission"] = true
		allowed["admission-sha"] = true
		allowed["receipt"] = true
		allowed["receipt-sha"] = true
	default:
		return o, fmt.Errorf("unknown SUP mode")
	}
	for name := range seen {
		if !allowed[name] {
			return o, fmt.Errorf("flag %s invalid for %s", name, o.mode)
		}
	}
	for name := range allowed {
		if !seen[name] {
			return o, fmt.Errorf("required flag %s", name)
		}
	}
	pins := []nativeproof.Pin{o.admission}
	if o.mode == "guarded-child" {
		pins = []nativeproof.Pin{o.request}
	}
	if o.mode == "replay" {
		pins = append(pins, o.receipt)
	}
	for _, p := range pins {
		if p.Path == "" || p.SHA256 == "" {
			return o, fmt.Errorf("complete independent pins required")
		}
	}
	return
}
func run(ctx context.Context, args []string, out io.Writer) error {
	o, e := parse(args)
	if e != nil {
		return e
	}
	var r graph.Result
	switch o.mode {
	case "consumer":
		r, e = graph.Consume(ctx, o.admission)
	case "guarded-child":
		r, e = graph.RunGuardedChild(ctx, o.request)
	case "replay":
		r, e = graph.Replay(o.admission, o.receipt)
	}
	if e != nil {
		return e
	}
	return json.NewEncoder(out).Encode(r)
}
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if e := run(ctx, os.Args[1:], os.Stdout); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(runner.ExitCode(e))
	}
}
