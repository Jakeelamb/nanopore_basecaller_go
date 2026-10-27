package h07

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestOutputOracleLinkerScriptTransitiveInputs(t *testing.T) {
	got, err := LinkerScriptInputs("/* GNU ld script */ OUTPUT_FORMAT(elf64-x86-64) GROUP ( /usr/lib/libm.so.6 AS_NEEDED ( /usr/lib/libmvec.so.1 ) )")
	want := []string{"/usr/lib/libm.so.6", "/usr/lib/libmvec.so.1"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("inputs %v %v", got, err)
	}
	got, err = LinkerScriptInputs(`INPUT ( "library with space.so" -lgcc )`)
	if err != nil || !reflect.DeepEqual(got, []string{"library with space.so", "-lgcc"}) {
		t.Fatalf("quoted inputs %v %v", got, err)
	}
	for _, body := range []string{"GROUP ( missing.so", `INPUT ( "unterminated )`} {
		if _, err := LinkerScriptInputs(body); err == nil {
			t.Fatal("malformed script accepted")
		}
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "libm.so")
	nested := filepath.Join(dir, "libnested.so")
	vector := filepath.Join(dir, "libmvec.so.1")
	for path, body := range map[string][]byte{script: []byte("GROUP ( libnested.so )"), nested: []byte("INPUT ( libmvec.so.1 )"), vector: []byte("\x7fELFfrozen")} {
		if err := os.WriteFile(path, body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	pins := map[string]string{}
	if err := pinFile(pins, script); err != nil {
		t.Fatal(err)
	}
	resolve := func(input, from string) (string, error) { return filepath.Join(filepath.Dir(from), input), nil }
	if err := freezeLinkerScripts(pins, resolve); err != nil {
		t.Fatal(err)
	}
	if len(pins) != 3 {
		t.Fatalf("transitive frozen input count %d", len(pins))
	}
	loaderCache := filepath.Join(dir, "ld.so.cache")
	if err := os.WriteFile(loaderCache, []byte("glibc-ld.so.cache\x00("), 0600); err != nil {
		t.Fatal(err)
	}
	if err := pinFile(pins, loaderCache); err != nil {
		t.Fatal(err)
	}
	if err := freezeLinkerScripts(pins, resolve); err != nil {
		t.Fatalf("loader data was parsed as a linker script: %v", err)
	}
	if err := VerifyLinkTrace(script+"\n"+nested+"\n"+vector+"\n", dir, pins, nil); err != nil {
		t.Fatal(err)
	}
	delete(pins, vector)
	if err := VerifyLinkTrace(vector+"\n", dir, pins, nil); err == nil {
		t.Fatal("effective unknown-link-input guard was bypassed")
	}
}
