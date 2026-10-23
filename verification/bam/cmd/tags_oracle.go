//go:build cgo

package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	bamverify "github.com/Jakeelamb/nanopore_basecaller_go/verification/bam"
)

func init() { extraCommands["tags-oracle"] = tagsOracleCmd }

// tagsOracleCmd needs cgo because the Go side of the comparison recomputes qs
// with the ported float libm decode path.
func tagsOracleCmd(args []string) error {
	fs := flag.NewFlagSet("tags-oracle", flag.ContinueOnError)
	src := fs.String("dorado-source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout")
	out := fs.String("out", "", "fresh output directory")
	cxx := fs.String("cxx", "g++", "C++ compiler")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *src == "" || *out == "" {
		return fmt.Errorf("-dorado-source and -out are required")
	}
	receipt, err := bamverify.VerifyTagsOracle(*src, *out, *cxx)
	for _, c := range receipt.Cases {
		fmt.Printf("%-24s ported=%-5v passed=%-5v native=%v go=%v %s\n",
			c.Name, c.Ported, c.Passed, c.NativeTags, c.GoTags, c.Detail)
	}
	if err != nil {
		return err
	}
	fmt.Printf("receipt=%s passed=%v cases=%d\n", filepath.Join(*out, "tags-receipt.json"), receipt.Passed, len(receipt.Cases))
	return nil
}
