// Command bamverify derives a writer manifest from a retained SAM and verifies
// a Go-written unaligned BAM against it. It is host-only: no GPU, model or
// native producer is used.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	bamverify "github.com/Jakeelamb/nanopore_basecaller_go/verification/bam"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "manifest":
		err = manifestCmd(os.Args[2:])
	case "verify":
		err = verifyCmd(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  bamverify manifest -sam SAM -out NEW_DIR [-drop-pg PREFIXES]
  bamverify verify -plan PLAN -attempt ATTEMPT_DIR -out NEW_DIR`)
	os.Exit(2)
}

func manifestCmd(args []string) error {
	fs := flag.NewFlagSet("manifest", flag.ContinueOnError)
	sam := fs.String("sam", "", "retained SAM oracle")
	out := fs.String("out", "", "fresh output directory")
	drop := fs.String("drop-pg", "samtools", "comma-separated @PG ID prefixes to exclude")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *sam == "" || *out == "" {
		return fmt.Errorf("-sam and -out are required")
	}
	data, err := os.ReadFile(*sam)
	if err != nil {
		return err
	}
	var prefixes []string
	for _, p := range strings.Split(*drop, ",") {
		if p = strings.TrimSpace(p); p != "" {
			prefixes = append(prefixes, p)
		}
	}
	parsed, err := bamverify.ParseSAM(strings.NewReader(string(data)), bamverify.ParseOptions{
		SourceSAM: *sam, SourceHash: bamverify.HashBytes(data), DropProgramPrefixes: prefixes})
	if err != nil {
		return err
	}
	body, err := parsed.Manifest.Encode()
	if err != nil {
		return err
	}
	if err := os.Mkdir(*out, 0o755); err != nil {
		return fmt.Errorf("fresh output required: %w", err)
	}
	path := filepath.Join(*out, "manifest.json")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		return err
	}
	header, err := parsed.Manifest.Header.Text()
	if err != nil {
		return err
	}
	summary := map[string]any{
		"schema":            parsed.Manifest.Schema,
		"manifest":          path,
		"manifest_sha256":   bamverify.HashBytes(body),
		"source_sam":        parsed.Manifest.SourceSAM,
		"source_sam_sha256": parsed.Manifest.SourceSAMHash,
		"records":           len(parsed.Manifest.Records),
		"aux_order":         parsed.Manifest.Records[0].AuxNames(),
		"header_lines":      strings.Count(header, "\n"),
		"dropped_programs":  parsed.DroppedPrograms,
		"raw_bam_identity":  "unclaimed",
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(summary)
}

func verifyCmd(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	plan := fs.String("plan", "", "verification plan JSON")
	attempt := fs.String("attempt", "", "attempt directory containing manifest.json")
	out := fs.String("out", "", "fresh output directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *plan == "" || *attempt == "" || *out == "" {
		return fmt.Errorf("-plan, -attempt and -out are required")
	}
	receipt, err := bamverify.Verify(*plan, *attempt, *out)
	if err != nil {
		return err
	}
	summary := map[string]any{
		"receipt":            filepath.Join(*out, "receipt.json"),
		"bam":                filepath.Join(*out, "unaligned.bam"),
		"bam_sha256":         receipt.BAMSAH256,
		"canonical_equal":    receipt.Canonical.CanonicalEqual,
		"raw_byte_identical": receipt.RawBAM.ByteIdentical,
		"raw_bam_identity":   "unclaimed",
		"controls":           len(receipt.Controls),
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(summary)
}
