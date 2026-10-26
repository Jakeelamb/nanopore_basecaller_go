package supgpu

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// These inert ELF metadata fixtures have no machine code or program segments.
// No compiler or loader creates or executes them.
func metadataELF(t *testing.T, path, dependency, interpreter string) Pin {
	t.Helper()
	sections := []elf.Section64{{}}
	data := new(bytes.Buffer)
	data.Write(make([]byte, 64))
	names := "\x00.shstrtab\x00.dynstr\x00.dynamic\x00.interp\x00"
	add := func(name string, typ elf.SectionType, body []byte, link uint32, entry uint64) {
		sections = append(sections, elf.Section64{Name: uint32(strings.Index(names, name)), Type: uint32(typ), Off: uint64(data.Len()), Size: uint64(len(body)), Link: link, Addralign: 1, Entsize: entry})
		data.Write(body)
	}
	add(".shstrtab", elf.SHT_STRTAB, []byte(names), 0, 0)
	add(".dynstr", elf.SHT_STRTAB, []byte("\x00"+dependency+"\x00"), 0, 0)
	dynamic := new(bytes.Buffer)
	if dependency != "" {
		if err := binary.Write(dynamic, binary.LittleEndian, elf.Dyn64{Tag: int64(elf.DT_NEEDED), Val: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := binary.Write(dynamic, binary.LittleEndian, elf.Dyn64{}); err != nil {
		t.Fatal(err)
	}
	add(".dynamic", elf.SHT_DYNAMIC, dynamic.Bytes(), 2, 16)
	if interpreter != "" {
		add(".interp", elf.SHT_PROGBITS, []byte(interpreter+"\x00"), 0, 0)
	}
	h := elf.Header64{Type: uint16(elf.ET_DYN), Machine: uint16(elf.EM_X86_64), Version: 1, Shoff: uint64(data.Len()), Ehsize: 64, Shentsize: 64, Shnum: uint16(len(sections)), Shstrndx: 1}
	copy(h.Ident[:], []byte{0x7f, 'E', 'L', 'F', byte(elf.ELFCLASS64), byte(elf.ELFDATA2LSB), 1})
	if err := binary.Write(data, binary.LittleEndian, sections); err != nil {
		t.Fatal(err)
	}
	header := new(bytes.Buffer)
	if err := binary.Write(header, binary.LittleEndian, h); err != nil {
		t.Fatal(err)
	}
	copy(data.Bytes(), header.Bytes())
	if err := os.WriteFile(path, data.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCompilerDSOAndInterpreterMustBeFrozenBeforeLaunch(t *testing.T) {
	dir := t.TempDir()
	loader := metadataELF(t, filepath.Join(dir, "ld.so"), "", "")
	math := metadataELF(t, filepath.Join(dir, "libcompiler_math.so"), "", "")
	tool := metadataELF(t, filepath.Join(dir, "cc1plus"), filepath.Base(math.Path), loader.Path)
	cachePath := filepath.Join(dir, "loader-cache")
	if err := os.WriteFile(cachePath, []byte("synthetic loader cache"), 0600); err != nil {
		t.Fatal(err)
	}
	cache, err := pin(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	c, err := compilerRuntimeForTools([]CompilerTool{{"cc1plus", tool}}, []Pin{}, []string{dir}, cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Nodes) != 3 || !reflect.DeepEqual(c.Inputs, []Pin{tool, math, loader, cache}) || c.Nodes[0].Interpreter != loader.Path || !reflect.DeepEqual(c.Nodes[0].Needed, []string{filepath.Base(math.Path)}) {
		t.Fatalf("compiler dependency or interpreter omitted: %+v", c)
	}
	if err = requireCompilerInputs(c, c.Inputs); err != nil {
		t.Fatal(err)
	}
	for _, q := range compilerQueries(c) {
		if err = sameQueryResult(q.Expected+"\n", q, dir); err != nil {
			t.Fatal(err)
		}
		if err = sameQueryResult(loader.Path+"\n", q, dir); err == nil {
			t.Fatal("a different compiler-selected program was accepted")
		}
	}
	for _, omitted := range []Pin{tool, math, loader, cache} {
		inputs := []Pin{}
		for _, p := range c.Inputs {
			if p != omitted {
				inputs = append(inputs, p)
			}
		}
		if err = requireCompilerInputs(c, inputs); err == nil || !strings.Contains(err.Error(), omitted.Path) {
			t.Fatalf("omitted compiler-only input accepted: %s: %v", omitted.Path, err)
		}
	}
	if err = os.WriteFile(math.Path, []byte("changed compiler-only DSO"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = requireCompilerInputs(c, c.Inputs); err == nil {
		t.Fatal("changed compiler-only DSO accepted")
	}
}

func TestQualifiedCompilerRuntimeStaticMetadata(t *testing.T) {
	_, c := qualifiedCompilerRuntime(t)
	if len(c.Tools) != 10 || len(c.Plugins) != 4 {
		t.Fatal("compiler program family is incomplete")
	}
	for _, name := range []string{"libisl.so.", "libmpc.so.", "libmpfr.so.", "libgmp.so.", "ld-linux-x86-64.so."} {
		found := false
		for _, p := range c.Inputs {
			found = found || strings.HasPrefix(filepath.Base(p.Path), name)
		}
		if !found {
			t.Fatalf("actual compiler runtime omission: %s", name)
		}
	}
	if err := requireCompilerInputs(c, c.Inputs); err != nil {
		t.Fatal(err)
	}
	for _, q := range compilerQueries(c) {
		if err := sameQueryResult(q.Expected+"\n", q, t.TempDir()); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("read-only compiler audit: %d tools, %d plugins, %d ELF nodes, %d pins; no native execution", len(c.Tools), len(c.Plugins), len(c.Nodes), len(c.Inputs))
}

func TestCompilerProgramQueryUsesOnlyDeclaredPATH(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	q := ToolQuery{[]string{"not-executed", "-print-prog-name=as"}, "/usr/bin/as"}
	if err := sameQueryResult("as\n", q, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if err := sameQueryResult("ld\n", q, t.TempDir()); err == nil {
		t.Fatal("a different compiler-selected program was accepted")
	}
	if err := sameQueryResult("missing-supgpu-tool\n", q, t.TempDir()); err == nil {
		t.Fatal("an unresolved compiler-selected program was accepted")
	}
}

func TestCompilerRuntimeGatePrecedesFirstQuery(t *testing.T) {
	driver, c := qualifiedCompilerRuntime(t)
	p := BuildPlan{Compiler: driver, CompilerRuntime: c, Inputs: []Pin{}}
	for _, input := range c.Inputs {
		if !strings.Contains(filepath.Base(input.Path), "libisl.so.") {
			p.Inputs = append(p.Inputs, input)
		}
	}
	assertCompilerStoppedBeforeQuery(t, p, "compiler runtime input absent", "libisl.so.")
}

func qualifiedCompilerRuntime(t *testing.T) (Pin, CompilerRuntime) {
	t.Helper()
	if os.Getenv("NANOPORE_QUALIFIED_HOST_TESTS") != "1" {
		t.Skip("qualified host audit unavailable by default; set NANOPORE_QUALIFIED_HOST_TESTS=1 to require the GCC15 runtime")
	}
	compiler := "/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/toolchain-gcc15/usr/bin/g++-15"
	c, err := planCompilerRuntime(compiler)
	if err != nil {
		t.Fatal(err)
	}
	driver, err := pin(compiler)
	if err != nil {
		t.Fatal(err)
	}
	return driver, c
}

func TestUnavailableCompilerRuntimeStopsBeforeFirstQuery(t *testing.T) {
	p := BuildPlan{Compiler: Pin{Path: filepath.Join(t.TempDir(), "absent", "bin", "g++-15")}}
	assertCompilerStoppedBeforeQuery(t, p, "qualified GCC program directory is ambiguous", "")
}

func assertCompilerStoppedBeforeQuery(t *testing.T, p BuildPlan, want, detail string) {
	t.Helper()
	out := t.TempDir()
	_, err := executeCompiler(p, out, out, "must-not-start", []string{"/must-not-execute-supgpu-query"})
	if err == nil || !strings.Contains(err.Error(), want) || !strings.Contains(err.Error(), detail) {
		t.Fatalf("compiler runtime violation did not stop launch: %v", err)
	}
	files, err := os.ReadDir(out)
	if err != nil || len(files) != 0 {
		t.Fatalf("command boundary was entered before closure: %v %v", files, err)
	}
}
