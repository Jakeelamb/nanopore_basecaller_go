package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"testing"
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
