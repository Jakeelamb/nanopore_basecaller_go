package main

import (
	"context"
	"flag"
	"fmt"
	replaymoves "github.com/Jakeelamb/nanopore_basecaller_go/verification/replaymoves"
	"os"
)

func main() {
	source := flag.String("source", "", "pinned Dorado source checkout")
	output := flag.String("output", "", "new artifact directory")
	flag.Parse()
	r, e := replaymoves.Replay(context.Background(), replaymoves.Config{Source: *source, Output: *output, Captures: flag.Args()})
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	fmt.Printf("PASS %d captures; receipt %s/receipt.json; %s\n", len(r.Captures), *output, r.Qualification)
}
