package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.sam"), filepath.Join(dir, "b.sam")
	record := "r\t4\t*\t0\t0\t*\t*\t0\t0\tAC\tII\n"
	if e := os.WriteFile(a, []byte(record), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(b, []byte(record), 0600); e != nil {
		t.Fatal(e)
	}
	var out bytes.Buffer
	if code := run([]string{a, b}, &out); code != 0 {
		t.Fatalf("%d %s", code, out.String())
	}
	var report map[string]any
	if e := json.Unmarshal(out.Bytes(), &report); e != nil {
		t.Fatal(e)
	}
	if report["canonical_equal"] != true {
		t.Fatal(report)
	}
	if e := os.WriteFile(b, []byte(""), 0600); e != nil {
		t.Fatal(e)
	}
	out.Reset()
	if run([]string{a, b}, &out) != 1 {
		t.Fatal("mismatch did not fail")
	}
	out.Reset()
	if run([]string{a}, &out) != 1 {
		t.Fatal("usage did not fail")
	}
	out.Reset()
	if run([]string{a, filepath.Join(dir, "missing.sam")}, &out) != 1 {
		t.Fatal("missing file did not fail")
	}
}
