package supgpu

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestCopiedQuotedHeaderNeedsOriginalSourceParent(t *testing.T) {
	dir := t.TempDir()
	copied := filepath.Join(dir, "copy")
	original := filepath.Join(dir, "original")
	for _, p := range []string{copied, original} {
		if err := os.Mkdir(p, 0700); err != nil {
			t.Fatal(err)
		}
	}
	source := filepath.Join(copied, "TxModel.cpp")
	header := filepath.Join(original, "TxModel.h")
	for p, body := range map[string]string{source: "#include \"TxModel.h\"\n", header: "// original header\n"} {
		if err := os.WriteFile(p, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	c := BuildCommand{Directory: dir, Argv: []string{"/must/not/run/compiler", "-DNDEBUG", "-iquote", original, "-c", source, "-o", filepath.Join(dir, "unit.o")}}
	got, err := quotedHeaderInputs(c)
	if err != nil || len(got) != 1 || got[0].Path != header {
		t.Fatalf("quote lookup: %+v %v", got, err)
	}
	c.Argv = []string{"/must/not/run/compiler", "-DNDEBUG", "-I" + copied, "-c", source, "-o", filepath.Join(dir, "unit.o")}
	if _, err = quotedHeaderInputs(c); err == nil || !strings.Contains(err.Error(), "TxModel.h") {
		t.Fatal("missing stock quote directory accepted")
	}
}

func TestLinkTraceAndMapRejectUnfrozenOrChangedInput(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "frozen.a")
	b := filepath.Join(dir, "other.so")
	for _, p := range []string{a, b} {
		if err := os.WriteFile(p, []byte("fake bytes, never loaded"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	frozen, err := pin(a)
	if err != nil {
		t.Fatal(err)
	}
	for _, mapMode := range []bool{false, true} {
		body := a + "\n"
		if mapMode {
			body = "LOAD " + a + "\n"
		}
		if err = verifyLinkInputs(body, dir, []Pin{frozen}, nil, mapMode); err != nil {
			t.Fatal(err)
		}
		bad := b + "\n"
		if mapMode {
			bad = "LOAD " + b + "\n"
		}
		if err = verifyLinkInputs(bad, dir, []Pin{frozen}, nil, mapMode); err == nil {
			t.Fatal("unfrozen actual linker input accepted")
		}
	}
	if err = os.WriteFile(a, []byte("changed frozen archive"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = verifyLinkInputs(a+"\n", dir, []Pin{frozen}, nil, false); err == nil {
		t.Fatal("changed frozen linker input accepted")
	}
}

func TestLinkerScriptGroupInputAndAsNeeded(t *testing.T) {
	body := `/* GNU ld script */ OUTPUT_FORMAT(elf64-x86-64) GROUP ( /one/libm.so.6 AS_NEEDED ( /two/libmvec.so.1 ) ) INPUT ( "a file.so" -lgcc )`
	got, err := linkerScriptInputs(body)
	want := []string{"/one/libm.so.6", "/two/libmvec.so.1", "a file.so", "-lgcc"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("%q %v", got, err)
	}
	for _, bad := range []string{"GROUP ( missing.so", `INPUT ( "unterminated )`, "SEARCH_DIR(/escape)"} {
		if _, err := linkerScriptInputs(bad); err == nil {
			t.Fatalf("invalid script accepted %q", bad)
		}
	}
}

func TestWrappedLibrariesAndLookupBindingsAreProspective(t *testing.T) {
	dir := t.TempDir()
	usr := filepath.Join(dir, "usr")
	gcc := filepath.Join(usr, "lib", "gcc", "triple", "15")
	if err := os.MkdirAll(gcc, 0700); err != nil {
		t.Fatal(err)
	}
	write := func(path string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("\x7fELFfake"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"libstdc++fs.a", "Scrt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "libstdc++.so", "libc.so", "libc_nonshared.a", "libm.so"} {
		write(filepath.Join(gcc, name))
	}
	wrapped := filepath.Join(dir, "torch", "libtorch_cpu.so")
	write(wrapped)
	libs, bindings, queries, err := linkClosure([]string{"-Wl,--no-as-needed," + wrapped, "-lstdc++fs"}, dir, filepath.Join(usr, "bin", "g++-15"))
	if err != nil {
		t.Fatal(err)
	}
	if libs[1] != filepath.Join(gcc, "libstdc++fs.a") || len(queries) == 0 {
		t.Fatal("lookup was not converted into a bound object")
	}
	found := false
	for _, b := range bindings {
		if b.Input.Path == wrapped {
			found = true
		}
	}
	if !found {
		t.Fatal("wrapped DSO was omitted")
	}
	for _, q := range queries {
		if err = sameQueryResult(q.Expected+"\n", q, dir); err != nil {
			t.Fatal(err)
		}
	}
	// GNU -l searches each directory before advancing to the next directory.
	// An archive in an earlier -L directory wins over a later shared object.
	early := filepath.Join(dir, "early")
	late := filepath.Join(dir, "late")
	write(filepath.Join(early, "libchoice.a"))
	write(filepath.Join(late, "libchoice.so"))
	chosen, _, _, err := linkClosure([]string{"-L" + early, "-L" + late, "-lchoice"}, dir, filepath.Join(usr, "bin", "g++-15"))
	if err != nil {
		t.Fatal(err)
	}
	if chosen[2] != filepath.Join(early, "libchoice.a") {
		t.Fatal("-l directory search order changed")
	}
	if err = sameQueryResult(wrapped, ToolQuery{[]string{"unused"}, "unresolved.so"}, dir); err == nil {
		t.Fatal("changed library lookup accepted")
	}
}

func TestReadOnlyTransitiveELFClosureAndUnknownRuntimeControl(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "libm.so")
	if err := os.WriteFile(script, []byte("/* GNU ld script */ GROUP ( libm.so.6 )\n"), 0600); err != nil {
		t.Fatal(err)
	}
	binary, err := isELF(script)
	if err != nil || binary {
		t.Fatal("linker script was mistaken for a runtime DSO")
	}
	leaf := metadataELF(t, filepath.Join(dir, "libc.so.6"), "", "")
	middle := metadataELF(t, filepath.Join(dir, "libmvec.so.1"), filepath.Base(leaf.Path), "")
	seed := metadataELF(t, filepath.Join(dir, "libm.so.6"), filepath.Base(middle.Path), "")
	if binary, err := isELF(seed.Path); err != nil || !binary {
		t.Fatalf("ELF fixture rejected: %v", err)
	}
	nodes, inputs, err := elfClosure([]Pin{seed}, []string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(inputs, []Pin{seed, middle, leaf}) || len(nodes) != len(inputs) {
		t.Fatalf("transitive ELF inputs differ: %+v", inputs)
	}
	if !reflect.DeepEqual(nodes[0].Needed, []string{filepath.Base(middle.Path)}) || !reflect.DeepEqual(nodes[1].Needed, []string{filepath.Base(leaf.Path)}) || len(nodes[2].Needed) != 0 {
		t.Fatal("transitive ELF dependencies were omitted")
	}
	if err = requireRuntimeMappings([]Pin{seed}, inputs); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "unknown.so")
	if err = os.WriteFile(path, []byte("unfrozen runtime, not loaded"), 0600); err != nil {
		t.Fatal(err)
	}
	unknown, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = requireRuntimeMappings([]Pin{unknown}, inputs); err == nil {
		t.Fatal("unknown runtime mapping accepted")
	}
	changed := seed
	changed.SHA256 = strings.Repeat("0", 64)
	if err = requireRuntimeMappings([]Pin{changed}, inputs); err == nil {
		t.Fatal("changed mapped runtime bytes accepted")
	}
	scriptPin, err := pin(script)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = elfClosure([]Pin{scriptPin}, []string{dir}); err == nil || !strings.Contains(err.Error(), "runtime seed is not ELF") {
		t.Fatalf("linker script accepted as a runtime seed: %v", err)
	}
	if err = os.Remove(leaf.Path); err != nil {
		t.Fatal(err)
	}
	if _, _, err = elfClosure([]Pin{seed}, []string{dir}); err == nil || !strings.Contains(err.Error(), "unresolved prospective DT_NEEDED libc.so.6") {
		t.Fatalf("missing transitive dependency accepted: %v", err)
	}
}

func syntheticController(t *testing.T) (GoController, func(GoController) Pin, string, string) {
	t.Helper()
	t.Chdir(filepath.Join("..", ".."))
	dir := t.TempDir()
	root := filepath.Join(dir, "goroot")
	tools := filepath.Join(root, "pkg", "tool", "linux_amd64")
	includes := filepath.Join(root, "pkg", "include")
	for _, p := range []string{tools, includes, filepath.Join(root, "bin")} {
		if err := os.MkdirAll(p, 0700); err != nil {
			t.Fatal(err)
		}
	}
	source := filepath.Join(dir, "runner.go")
	tool := filepath.Join(tools, "compile")
	for _, path := range []string{source, filepath.Join(dir, "profile.go"), filepath.Join(dir, "main.go"), filepath.Join(dir, "observer.h"), filepath.Join(dir, "graph.cpp"), tool, filepath.Join(root, "bin", "go"), filepath.Join(includes, "textflag.h")} {
		if err := os.WriteFile(path, []byte("synthetic closed input"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	goexe := filepath.Join(root, "bin", "go")
	env := goEnvironment{GOROOT: root, GOTOOLDIR: tools, GOVERSION: "synthetic", GOOS: "linux", GOARCH: "amd64", GOAMD64: "v1", CGO_ENABLED: "0", GOENV: "", GOWORK: "off"}
	envRaw, _ := json.Marshal(env)
	pkg := goPackage{Dir: dir, GoFiles: []string{"runner.go", "profile.go", "main.go"}, EmbedFiles: []string{"observer.h", "graph.cpp"}}
	deps, _ := json.Marshal(pkg)
	paths, ng, ne, _, err := selectedGoInputs(deps, envRaw, goexe)
	if err != nil {
		t.Fatal(err)
	}
	put := func(name string, raw []byte) Pin {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		p, err := pin(path)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := pin(exe)
	if err != nil {
		t.Fatal(err)
	}
	c := GoController{Schema: "nanopore-supgpu-go-controller/v1", Binary: binary, Dependencies: put("deps.json", deps), Environment: put("env.json", envRaw), Snapshots: []Snapshot{}, Commands: []runner.CommandReceipt{}, ProjectGoFiles: ng, ProjectEmbeds: ne, GoVersion: "synthetic"}
	var manifest strings.Builder
	for i, path := range paths {
		p, err := pin(path)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		copy := put(fmt.Sprintf("copy-%d", i), raw)
		c.Snapshots = append(c.Snapshots, Snapshot{path, copy})
		fmt.Fprintf(&manifest, "%s  %s\n", p.SHA256, path)
	}
	c.Before = put("before.sha256", []byte(manifest.String()))
	log := put("log", nil)
	for i := 0; i < 3; i++ {
		c.Commands = append(c.Commands, runner.CommandReceipt{Args: []string{goexe}, Environment: hostGoEnvironment(), Started: "2026-10-25T01:00:00Z", Finished: "2026-10-25T01:00:01Z", Exit: 0, Stdout: log.Path, Stderr: log.Path, StdoutSHA: log.SHA256, StderrSHA: log.SHA256})
	}
	c.Commands[0].Args = []string{goexe, "env", "-json", "GOROOT", "GOTOOLDIR", "GOVERSION", "GOOS", "GOARCH", "GOAMD64", "CGO_ENABLED", "GOFLAGS", "GOWORK", "GOENV"}
	c.Commands[0].Stdout = c.Environment.Path
	c.Commands[0].StdoutSHA = c.Environment.SHA256
	c.Commands[1].Args = []string{goexe, "list", "-deps", "-json", "./verification/supgpu/cmd"}
	c.Commands[1].Stdout = c.Dependencies.Path
	c.Commands[1].StdoutSHA = c.Dependencies.SHA256
	c.Commands[2].Args = []string{goexe, "build", "-buildvcs=false", "-o", binary.Path, "./verification/supgpu/cmd"}
	serial := 0
	save := func(v GoController) Pin {
		t.Helper()
		serial++
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return put(fmt.Sprintf("controller-%d.json", serial), raw)
	}
	return c, save, source, tool
}

func TestActualGoSelectionAndSourceToolBinaryDriftGuards(t *testing.T) {
	c, save, source, tool := syntheticController(t)
	if _, err := readController(save(c), true); err != nil {
		t.Fatal(err)
	}
	missing := c
	missing.Snapshots = append([]Snapshot{}, c.Snapshots[1:]...)
	if _, err := readController(save(missing), false); err == nil {
		t.Fatal("omitted selected Go input accepted")
	}
	for _, path := range []string{source, tool} {
		original, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, []byte("actual changed source/tool bytes"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = readController(save(c), true); err == nil {
			t.Fatal("source/tool drift accepted")
		}
		if err = os.WriteFile(path, original, 0600); err != nil {
			t.Fatal(err)
		}
	}
	bad := c
	bad.Binary.SHA256 = strings.Repeat("0", 64)
	if _, err := readController(save(bad), true); err == nil {
		t.Fatal("controller byte drift accepted")
	}
}
