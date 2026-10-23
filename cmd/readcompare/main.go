// readcompare compares exact canonical SAM-record multisets, excluding headers.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/recordcompare"
	"io"
	"os"
)

func run(args []string, out io.Writer) int {
	encoder := json.NewEncoder(out)
	encoder.SetIndent("", "  ")
	if len(args) != 2 {
		encoder.Encode(map[string]string{"error": "usage: readcompare LEFT RIGHT (SAM text or BAM; BAM detected by magic or .bam extension)"})
		return 1
	}
	report, err := recordcompare.ComparePaths(args[0], args[1])
	if err != nil {
		encoder.Encode(map[string]string{"error": err.Error()})
		return 1
	}
	if err := encoder.Encode(report); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if !report.CanonicalEqual {
		return 1
	}
	return 0
}
func main() { os.Exit(run(os.Args[1:], os.Stdout)) }
