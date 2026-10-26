package cgobuild

import (
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func specification() Plan {
	cc := Pin{Path: "/qualified/gcc-15", SHA256: strings.Repeat("a", 64)}
	cxx := Pin{Path: "/qualified/g++-15", SHA256: strings.Repeat("b", 64)}
	p := Plan{CC: cc, CXX: cxx, Inputs: map[string]string{}, Compiler: closedbuild.Compiler{Tools: []closedbuild.Tool{{Name: "gcc-driver", Executable: cc}, {Name: "driver", Executable: cxx}}, Plugins: []Pin{{Path: "/qualified/liblto_plugin.so", SHA256: strings.Repeat("c", 64)}}}}
	for _, name := range resourceNames {
		q := Pin{Path: "/resource/" + name, SHA256: strings.Repeat("d", 64)}
		p.Resources = append(p.Resources, Resource{Name: name, Input: q})
		p.Inputs[q.Path] = q.SHA256
	}
	return p
}

func TestNativeSpecificationRejectsMissingPluginCCAndLinkResourceBeforeChild(t *testing.T) {
	p := specification()
	if e := ValidateSpecification(p); e != nil {
		t.Fatal(e)
	}
	for _, c := range []struct {
		Name   string
		Mutate func(*Plan)
	}{
		{"plugin", func(p *Plan) { p.Compiler.Plugins = nil }},
		{"CC path", func(p *Plan) { p.CC.Path = "/usr/bin/gcc" }},
		{"pthread archive", func(p *Plan) {
			p.Resources = slices.DeleteFunc(p.Resources, func(r Resource) bool { return r.Name == "libpthread.a" })
		}},
		{"dl archive", func(p *Plan) { delete(p.Inputs, "/resource/libdl.a") }},
		{"CRT", func(p *Plan) {
			p.Resources = slices.DeleteFunc(p.Resources, func(r Resource) bool { return r.Name == "crt1.o" })
		}},
	} {
		t.Run(c.Name, func(t *testing.T) {
			copy := p
			copy.Resources = slices.Clone(p.Resources)
			copy.Inputs = maps.Clone(p.Inputs)
			c.Mutate(&copy)
			called := false
			if ValidateSpecification(copy) == nil {
				called = true
			}
			if called {
				t.Fatal("invalid specification reached a prospective child")
			}
		})
	}
}

func TestStaticSourceSelectionIncludesActualCgoAndABISourcesWithoutGoOrCC(t *testing.T) {
	root, e := filepath.Abs("../../../..")
	if e != nil {
		t.Fatal(e)
	}
	packages, inputs, e := SelectSources(root, runtime.GOROOT())
	if e != nil {
		t.Fatal(e)
	}
	var cgo []string
	for _, p := range packages {
		if len(p.Cgo) > 0 {
			cgo = append(cgo, p.ImportPath)
		}
	}
	if !slices.Equal(cgo, []string{Module + "/internal/tensor", "runtime/cgo"}) {
		t.Fatalf("Cgo source family: %v", cgo)
	}
	for _, file := range []string{"internal/tensor/native_cgo.go", "internal/tensor/cuda_cgo.go", "internal/tensor/loader.c", "internal/tensor/loader.h", "internal/tensor/abi.h", "internal/model/hac_weights.json", "internal/model/sup_weights.json"} {
		if inputs[filepath.Join(root, file)] == "" {
			t.Fatalf("omitted actual source %s", file)
		}
	}
}

func TestGeneratedFilesArePinnedBeforeUseAndCannotBeOmittedOrChanged(t *testing.T) {
	root := t.TempDir()
	objects := filepath.Join(root, "go-work", "b001")
	if e := os.MkdirAll(objects, 0700); e != nil {
		t.Fatal(e)
	}
	p := Package{ImportPath: Module + "/internal/tensor", Cgo: []string{"native_cgo.go", "cuda_cgo.go"}}
	producer := filepath.Join(root, "cgo-receipt.json")
	if e := os.WriteFile(producer, []byte("actual producer receipt fixture"), 0600); e != nil {
		t.Fatal(e)
	}
	pr, e := closedbuild.PinFile(producer)
	if e != nil {
		t.Fatal(e)
	}
	for _, name := range RequiredGenerated(p) {
		if e = os.WriteFile(filepath.Join(objects, name), []byte(name), 0600); e != nil {
			t.Fatal(e)
		}
	}
	g, e := CaptureGenerated(p, "translation", root, objects, filepath.Join(root, "snapshots"), pr)
	if e != nil {
		t.Fatal(e)
	}
	if e = CheckGeneratedLive(g); e != nil {
		t.Fatal(e)
	}
	missing := maps.Clone(g.Files)
	delete(missing, "_cgo_export.h")
	if CheckGenerated(p, "translation", missing) == nil {
		t.Fatal("omitted generated header accepted")
	}
	if e = os.WriteFile(filepath.Join(objects, "native_cgo.cgo2.c"), []byte("changed after capture"), 0600); e != nil {
		t.Fatal(e)
	}
	if CheckGeneratedLive(g) == nil {
		t.Fatal("changed generated C input accepted before child")
	}
}

func TestGeneratedSymlinkEscapeAndUnknownRoleRejected(t *testing.T) {
	p := Package{ImportPath: Module + "/internal/tensor", Cgo: []string{"native_cgo.go"}}
	if CheckGenerated(p, "unknown", nil) == nil {
		t.Fatal("unknown generated stage accepted")
	}
	if CheckGenerated(Package{ImportPath: "foreign/package"}, "translation", nil) == nil {
		t.Fatal("foreign generated package accepted")
	}
	root, outside := t.TempDir(), t.TempDir()
	objects := filepath.Join(root, "b001")
	if e := os.Mkdir(objects, 0700); e != nil {
		t.Fatal(e)
	}
	for _, name := range RequiredGenerated(p) {
		if e := os.WriteFile(filepath.Join(objects, name), []byte(name), 0600); e != nil {
			t.Fatal(e)
		}
	}
	name := "native_cgo.cgo2.c"
	if e := os.WriteFile(filepath.Join(outside, name), []byte(name), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.Remove(filepath.Join(objects, name)); e != nil {
		t.Fatal(e)
	}
	if e := os.Symlink(filepath.Join(outside, name), filepath.Join(objects, name)); e != nil {
		t.Fatal(e)
	}
	producer, e := closedbuild.PinFile(filepath.Join(outside, name))
	if e != nil {
		t.Fatal(e)
	}
	if _, e = CaptureGenerated(p, "translation", root, objects, filepath.Join(root, "snapshots"), producer); e == nil {
		t.Fatal("generated source outside work root accepted")
	}
}

func TestExplicitNestedEmbedStaysWithinPackageRegularFiles(t *testing.T) {
	dir := t.TempDir()
	native := filepath.Join(dir, "native")
	if e := os.Mkdir(native, 0700); e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(native, "audit.c")
	if e := os.WriteFile(path, []byte("audit source"), 0600); e != nil {
		t.Fatal(e)
	}
	q, e := explicitEmbed(dir, "native/audit.c")
	if e != nil || q.Path != path {
		t.Fatalf("explicit nested regular embed: %+v %v", q, e)
	}
	outside := filepath.Join(t.TempDir(), "audit.c")
	if e = os.WriteFile(outside, []byte("outside"), 0600); e != nil {
		t.Fatal(e)
	}
	if e = os.Symlink(outside, filepath.Join(native, "escape.c")); e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"native/*.c", "native", "../audit.c", outside, "native/escape.c", "native/../native/audit.c"} {
		t.Run(name, func(t *testing.T) {
			if _, e := explicitEmbed(dir, name); e == nil {
				t.Fatal("nonregular/unbounded embed admitted")
			}
		})
	}
}
