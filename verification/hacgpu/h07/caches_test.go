package h07

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func cacheFixture(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	body := "nanopore-hac-output-caches/v1\ndown\tChar\t128,1024\t131072\tdown.bin\ndown-scale\tHalf\t128\t256\tdown-scale.bin\nup\tHalf\t1024,128\t262144\tup.bin\n"
	if err := os.WriteFile(filepath.Join(dir, "caches.tsv"), []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	for name, n := range map[string]int{"down.bin": 131072, "down-scale.bin": 256, "up.bin": 262144} {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, n), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir, body
}
func TestOutputCacheClosedDescriptors(t *testing.T) {
	dir, body := cacheFixture(t)
	cache, err := ReadCacheDescriptors(dir)
	if err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, c := range cache {
		total += c.Bytes
		if len(c.SHA256) != 64 {
			t.Fatal("missing actual payload hash")
		}
	}
	if total != 393472 {
		t.Fatalf("native cache bytes %d", total)
	}
	for _, changed := range []string{
		strings.Replace(body, "Char", "Half", 1), strings.Replace(body, "128,1024", "1024,128", 1),
		strings.Replace(body, "131072", "131071", 1), strings.Replace(body, "down.bin", "../down.bin", 1),
		strings.Replace(body, "down-scale\tHalf", "down-scale\tFloat", 1), body + "extra\n",
		strings.Replace(body, "up\tHalf\t1024,128\t262144\tup.bin\n", "", 1),
	} {
		if err := os.WriteFile(filepath.Join(dir, "caches.tsv"), []byte(changed), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadCacheDescriptors(dir); err == nil {
			t.Fatal("malformed cache publication accepted")
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "caches.tsv"), []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(filepath.Join(dir, "up.bin"), 262143); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadCacheDescriptors(dir); err == nil {
		t.Fatal("truncated cache payload accepted")
	}
}
func TestPinnedPayloadRejectsSelfRehashedCopy(t *testing.T) {
	dir, _ := cacheFixture(t)
	path := filepath.Join(dir, "down.bin")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	pin := sum(body)
	if _, err := readPinned(path, pin); err != nil {
		t.Fatal(err)
	}
	body[17] ^= 1
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(path, pin); err == nil {
		t.Fatal("changed copy accepted against independent pin")
	}
}
