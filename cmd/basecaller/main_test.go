package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
)

func TestCLIRejectsImplicitOrUnsupportedWork(t *testing.T) {
	for _, args := range [][]string{nil, {"--device", "cuda", "--no-trim", "--disable-read-splitting"}, {"--model", "x"}, {"--model", "x", "--pod5", "p", "--tensor-library", "t", "--pod5-library", "l", "--no-trim", "--disable-read-splitting"}, {"--all", "--read-id", "bb76c31a-36f0-48a0-abd2-107a10e1ade0", "--no-trim", "--disable-read-splitting", "--model", "m", "--pod5", "p", "--tensor-library", "t", "--pod5-library", "l"}, {"replay"}, {"--replay", "missing", "--model", "x"}, {"--read-id", "invalid"}} {
		var out, errout bytes.Buffer
		if err := run(context.Background(), args, &out, &errout); err == nil {
			t.Fatalf("accepted args %v", args)
		}
		if out.Len() != 0 {
			t.Fatal("rejection polluted FASTQ stdout")
		}
	}
	var out, errout bytes.Buffer
	if err := run(context.Background(), []string{"--help"}, &out, &errout); !errors.Is(err, flag.ErrHelp) {
		t.Fatal(err)
	}
}

func TestCLIGPURejectsUnqualifiedStartupWithoutSuccess(t *testing.T) {
	previous, set := os.LookupEnv("DORADO_EXTRA_CHUNK_SIZES")
	if err := os.Unsetenv("DORADO_EXTRA_CHUNK_SIZES"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if set {
			_ = os.Setenv("DORADO_EXTRA_CHUNK_SIZES", previous)
		}
	})
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "gpu-bundle.json"), []byte(`{"source_only":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	var out, diagnostics bytes.Buffer
	err := run(context.Background(), []string{"--device", "cuda:0", "--native-bundle", dir, "--model", "unopened-model", "--pod5", "unopened.pod5", "--all", "--chunking", "fixed"}, &out, &diagnostics)
	if !errors.Is(err, caller.ErrGPUUnqualified) || out.Len() != 0 || strings.Contains(diagnostics.String(), "processed input_reads") {
		t.Fatalf("unqualified GPU startup reported success: %v stdout=%q stderr=%q", err, out.String(), diagnostics.String())
	}
}

func TestCLIGPURejectsAmbiguousModesAndLimits(t *testing.T) {
	for _, extra := range [][]string{{"--chunking", "invented"}, {"--format", "sam"}, {"--pending-chunks", "0"}, {"--live-samples", "1"}, {"--tensor-library", "unbound"}, {"--capture", "unbound"}} {
		args := append([]string{"--device", "cuda:0", "--model", "m", "--pod5", "p", "--native-bundle", "bundle", "--all"}, extra...)
		var out, diagnostics bytes.Buffer
		if err := run(context.Background(), args, &out, &diagnostics); err == nil || out.Len() != 0 || strings.Contains(diagnostics.String(), "processed input_reads") {
			t.Fatalf("GPU rejection lost: %v", args)
		}
	}
}
func TestUUIDFlag(t *testing.T) {
	var ids idsFlag
	const id = "bb76c31a-36f0-48a0-abd2-107a10e1ade0"
	if err := ids.Set(id); err != nil {
		t.Fatal(err)
	}
	if err := ids.Set(id); err == nil {
		t.Fatal("duplicate UUID accepted")
	}
}
