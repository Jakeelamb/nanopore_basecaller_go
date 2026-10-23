package cudafixture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSavedCorpusAndRejections(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cases")
	if err := Generate(dir); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 37 {
		t.Fatalf("cases %d: %v", len(entries), err)
	}
	for _, entry := range entries {
		path := filepath.Join(dir, entry.Name(), "case.json")
		c, bits, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if c.Name == "contiguous" {
			for i, want := range []uint16{0, 0x8000, 0x3c00, 0xbc00} {
				if bits["a"][i] != want || bits["b"][i] != want {
					t.Fatal("GEMM corpus lacks positive, negative and signed zero operands")
				}
			}
		}
		if c.Name == "all-half-bits" {
			for i, b := range bits["all"] {
				if b != uint16(i) {
					t.Fatal(i)
				}
			}
		}
	}
	path := filepath.Join(dir, "rectangular", "case.json")
	c, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Case){func(c *Case) { c.DType = "f32" }, func(c *Case) { c.Endian = "big" }, func(c *Case) { v := c.Views["a"]; v.Offset = 1 << 63; c.Views["a"] = v }, func(c *Case) { v := c.Views["a"]; v.Strides = []int64{-1, 1}; c.Views["a"] = v }, func(c *Case) { v := c.Views["a"]; v.Shape = []int64{3, 4}; c.Views["a"] = v }} {
		var altered Case
		raw, _ := json.Marshal(c)
		json.Unmarshal(raw, &altered)
		change(&altered)
		if altered.Validate() == nil {
			t.Fatal("accepted corrupt metadata")
		}
	}
	file := filepath.Join(filepath.Dir(path), c.Allocations["a"].File)
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	raw[0] ^= 1
	os.WriteFile(file, raw, 0600)
	if _, _, err = Load(path); err == nil {
		t.Fatal("accepted corrupt bytes")
	}
	os.WriteFile(file, raw[:len(raw)-2], 0600)
	if _, _, err = Load(path); err == nil {
		t.Fatal("accepted short bytes")
	}
}
