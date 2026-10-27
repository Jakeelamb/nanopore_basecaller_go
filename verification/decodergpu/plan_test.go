package decodergpu

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
)

func TestLinkReferencesSeeWrappedDSOAndLibraries(t *testing.T) {
	files, libs, err := LinkReferences([]string{"a.a", "-Wl,--no-as-needed,/tmp/liba.so,-lfoo,-rpath,/tmp/runtime", "-l:exact.so", "-Wl,-Map,map.txt,-L,/search", "-Wl,b.o"})
	if err != nil || !reflect.DeepEqual(files, []string{"a.a", "/tmp/liba.so", "b.o"}) || !reflect.DeepEqual(libs, []string{"-lfoo", "-l:exact.so"}) {
		t.Fatalf("link inputs%v %v %v", files, libs, err)
	}
	for _, args := range [][]string{{"unclassified"}, {"-Wl,-rpath"}, {"-Wl,-l"}} {
		if _, _, err := LinkReferences(args); err == nil {
			t.Fatal("unclassified link input accepted")
		}
	}
}
func TestTransitiveLinkScriptsAndPredeclaredTrace(t *testing.T) {
	dir := t.TempDir()
	top := filepath.Join(dir, "libtop.so")
	nested := filepath.Join(dir, "nested.ld")
	leaf := filepath.Join(dir, "libleaf.so.1")
	for path, body := range map[string]string{top: "GROUP ( AS_NEEDED ( nested.ld ) )", nested: "INPUT ( libleaf.so.1 )", leaf: "\x7fELFfake"} {
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	pins := map[string]string{}
	if err := filePin(pins, top); err != nil {
		t.Fatal(err)
	}
	if err := ExpandLinkerScripts(pins, func(name, from string) (string, error) { return filepath.Join(filepath.Dir(from), name), nil }); err != nil {
		t.Fatal(err)
	}
	if len(pins) != 3 {
		t.Fatalf("transitive closure%d", len(pins))
	}
	if err := h07.VerifyLinkTrace(top+"\n"+leaf+"\n", dir, pins, nil); err != nil {
		t.Fatal(err)
	}
	delete(pins, leaf)
	if err := h07.VerifyLinkTrace(leaf+"\n", dir, pins, nil); err == nil {
		t.Fatal("post hoc linker input accepted")
	}
}
