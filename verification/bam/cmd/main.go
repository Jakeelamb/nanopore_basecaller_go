// Command bamverify derives a writer manifest from a retained native BAM and
// verifies a Go-written unaligned BAM against it. It is host-only: no GPU, model
// or native producer is used.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

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
	case "tags-oracle":
		err = tagsOracleCmd(os.Args[2:])
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
  bamverify manifest -bam ORACLE.bam -out NEW_DIR
  bamverify verify -plan PLAN.json -attempt ATTEMPT_DIR -out NEW_DIR
  bamverify tags-oracle -dorado-source ROOT -out NEW_DIR [-cxx g++]`)
	os.Exit(2)
}

func manifestCmd(args []string) error {
	fs := flag.NewFlagSet("manifest", flag.ContinueOnError)
	bamPath := fs.String("bam", "", "retained native unaligned BAM")
	out := fs.String("out", "", "fresh output directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *bamPath == "" || *out == "" {
		return fmt.Errorf("-bam and -out are required")
	}
	data, err := os.ReadFile(*bamPath)
	if err != nil {
		return err
	}
	parsed, err := bamverify.ParseBAM(data, bamverify.ParseOptions{
		SourceBAM: *bamPath, SourceHash: bamverify.HashBytes(data)})
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
	summary := map[string]any{
		"schema":                 parsed.Manifest.Schema,
		"manifest":               path,
		"manifest_sha256":        bamverify.HashBytes(body),
		"source_bam":             parsed.Manifest.SourceBAM,
		"source_bam_sha256":      parsed.Manifest.SourceBAMHash,
		"header_text_sha256":     parsed.HeaderSHA256,
		"payload_sha256":         parsed.PayloadSHA256,
		"payload_bytes":          parsed.PayloadBytes,
		"bgzf_members":           len(parsed.Members),
		"records":                len(parsed.Manifest.Records),
		"aux_order":              parsed.Manifest.Records[0].AuxNames(),
		"payload_identity":       "required",
		"compressed_identity":    "unclaimed",
		"bgzf_topology_identity": "unclaimed",
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
		"receipt":              filepath.Join(*out, "receipt.json"),
		"bam":                  filepath.Join(*out, "unaligned.bam"),
		"go_bam_sha256":        receipt.GoBAMHash,
		"go_payload_sha256":    receipt.GoPayloadHash,
		"payload_identical":    receipt.Payload.Identical,
		"compressed_identical": receipt.Compressed.Identical,
		"bgzf_members_match":   receipt.BGZF.MembersMatch,
		"canonical_equal":      receipt.Canonical.CanonicalEqual,
		"controls":             len(receipt.Controls),
		"faults":               len(receipt.Faults),
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(summary)
}

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
