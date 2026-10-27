package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h04"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	receipt := flag.String("receipt", "", "retained H04 producer receipt")
	sha := flag.String("receipt-sha256", "", "independently retained producer receipt SHA256")
	out := flag.String("out", "", "fresh replay directory under artifacts/tensor")
	verifyConsumer := flag.Bool("verify-consumer", false, "run Go arithmetic with the existing native CPU archive loader")
	backend := flag.String("backend", "", "accepted ABI4 backend for CPU archive loading")
	admitted := flag.Bool("runtime-admitted", false, "coordinator released this bounded CPU loader invocation")
	flag.Parse()
	o := h04.ReplayOptions{Receipt: *receipt, ReceiptSHA256: *sha, Out: *out}
	if *verifyConsumer {
		r, err := h04.VerifyConsumer(h04.ConsumerOptions{ReplayOptions: o, Backend: *backend, Admitted: *admitted})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(runner.ExitCode(err))
		}
		fmt.Printf("%s: %d fresh Go/native pairs, %d final caches, %d arithmetic mutants\n", r.Status, r.NativePayloads, r.FinalCaches, len(r.Controls))
		return
	}
	r, err := h04.Replay(o)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
	fmt.Printf("%s: %d native payloads, %d final caches, %d changed-copy controls\n", r.Status, r.NativePayloads, r.FinalCaches, len(r.Controls))
}
