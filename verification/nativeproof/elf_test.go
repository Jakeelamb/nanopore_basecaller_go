package nativeproof

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type fixture struct {
	needed              []string
	rpath, runpath      []string
	interpreter, soname string
	kind                elf.Type
	machine             elf.Machine
	extraDynamic        int
}

// A real ELF64 header, file-backed read-only PT_LOAD, PT_DYNAMIC, optional
// PT_INTERP, and matching runtime/section string tables. There is no code.
func writeELF(t *testing.T, path string, v fixture) Pin {
	t.Helper()
	if v.kind == 0 {
		v.kind = elf.ET_DYN
	}
	if v.machine == 0 {
		v.machine = elf.EM_X86_64
	}
	body := make([]byte, 64)
	copy(body, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(body[16:], uint16(v.kind))
	binary.LittleEndian.PutUint16(body[18:], uint16(v.machine))
	binary.LittleEndian.PutUint32(body[20:], 1)
	binary.LittleEndian.PutUint16(body[52:], 64)
	binary.LittleEndian.PutUint16(body[54:], 56)
	binary.LittleEndian.PutUint16(body[58:], 64)
	binary.LittleEndian.PutUint16(body[60:], 4)
	binary.LittleEndian.PutUint16(body[62:], 1)
	programs := 2
	if v.interpreter != "" {
		programs++
	}
	binary.LittleEndian.PutUint64(body[32:], 64)
	binary.LittleEndian.PutUint16(body[56:], uint16(programs))
	body = append(body, make([]byte, 56*programs)...)
	if v.interpreter != "" {
		p := body[64+2*56 : 64+3*56]
		binary.LittleEndian.PutUint32(p, uint32(elf.PT_INTERP))
		binary.LittleEndian.PutUint64(p[8:], uint64(len(body)))
		binary.LittleEndian.PutUint64(p[32:], uint64(len(v.interpreter)+1))
		binary.LittleEndian.PutUint64(p[40:], uint64(len(v.interpreter)+1))
		body = append(body, append([]byte(v.interpreter), 0)...)
	}
	shstr := []byte("\x00.shstrtab\x00.dynstr\x00.dynamic\x00")
	shstrAt := len(body)
	body = append(body, shstr...)
	dynstr := []byte{0}
	dynamic := []byte{}
	add := func(tag elf.DynTag, text string) {
		entry := make([]byte, 16)
		binary.LittleEndian.PutUint64(entry, uint64(tag))
		binary.LittleEndian.PutUint64(entry[8:], uint64(len(dynstr)))
		dynamic = append(dynamic, entry...)
		dynstr = append(dynstr, append([]byte(text), 0)...)
	}
	for _, name := range v.needed {
		add(elf.DT_NEEDED, name)
	}
	for _, value := range v.rpath {
		add(elf.DT_RPATH, value)
	}
	for _, value := range v.runpath {
		add(elf.DT_RUNPATH, value)
	}
	if v.soname != "" {
		add(elf.DT_SONAME, v.soname)
	}
	dynstrAt := len(body)
	body = append(body, dynstr...)
	for len(body)%8 != 0 {
		body = append(body, 0)
	}
	dynamicAt := len(body)
	value := func(tag elf.DynTag, val uint64) {
		entry := make([]byte, 16)
		binary.LittleEndian.PutUint64(entry, uint64(tag))
		binary.LittleEndian.PutUint64(entry[8:], val)
		dynamic = append(dynamic, entry...)
	}
	value(elf.DT_STRTAB, 0x400000+uint64(dynstrAt))
	value(elf.DT_STRSZ, uint64(len(dynstr)))
	for i := 0; i < v.extraDynamic; i++ {
		value(elf.DT_DEBUG, 0)
	}
	value(elf.DT_NULL, 0)
	body = append(body, dynamic...)
	for len(body)%8 != 0 {
		body = append(body, 0)
	}
	binary.LittleEndian.PutUint64(body[40:], uint64(len(body)))
	body = append(body, make([]byte, 64)...)
	section := func(name string, kind elf.SectionType, at, size int, link uint32) {
		h := make([]byte, 64)
		binary.LittleEndian.PutUint32(h, uint32(bytes.Index(shstr, []byte(name))))
		binary.LittleEndian.PutUint32(h[4:], uint32(kind))
		if name != ".shstrtab" {
			binary.LittleEndian.PutUint64(h[8:], uint64(elf.SHF_ALLOC))
			binary.LittleEndian.PutUint64(h[16:], 0x400000+uint64(at))
		}
		binary.LittleEndian.PutUint64(h[24:], uint64(at))
		binary.LittleEndian.PutUint64(h[32:], uint64(size))
		binary.LittleEndian.PutUint32(h[40:], link)
		if kind == elf.SHT_DYNAMIC {
			binary.LittleEndian.PutUint64(h[56:], 16)
		}
		body = append(body, h...)
	}
	section(".shstrtab", elf.SHT_STRTAB, shstrAt, len(shstr), 0)
	section(".dynstr", elf.SHT_STRTAB, dynstrAt, len(dynstr), 0)
	section(".dynamic", elf.SHT_DYNAMIC, dynamicAt, len(dynamic), 2)
	for index, kind := range []elf.ProgType{elf.PT_LOAD, elf.PT_DYNAMIC} {
		p := body[64+index*56 : 64+(index+1)*56]
		binary.LittleEndian.PutUint32(p, uint32(kind))
		binary.LittleEndian.PutUint32(p[4:], uint32(elf.PF_R))
		at, size := 0, len(body)
		if kind == elf.PT_DYNAMIC {
			at, size = dynamicAt, len(dynamic)
		}
		binary.LittleEndian.PutUint64(p[8:], uint64(at))
		binary.LittleEndian.PutUint64(p[16:], 0x400000+uint64(at))
		binary.LittleEndian.PutUint64(p[32:], uint64(size))
		binary.LittleEndian.PutUint64(p[40:], uint64(size))
	}
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(body)
	return Pin{path, hex.EncodeToString(h[:])}
}

func TestInspectELFMetadataAndCanonicalPin(t *testing.T) {
	dir := t.TempDir()
	p := writeELF(t, filepath.Join(dir, "tool"), fixture{kind: elf.ET_EXEC, needed: []string{"liba.so", "libb.so"}, rpath: []string{"${ORIGIN}/lib"}, interpreter: "/test/loader", soname: "tool.so"})
	alias := filepath.Join(dir, "alias")
	if err := os.Symlink(p.Path, alias); err != nil {
		t.Fatal(err)
	}
	m, err := InspectELF(alias)
	if err != nil {
		t.Fatal(err)
	}
	if m.Pin != p || m.Class != elf.ELFCLASS64 || m.Data != elf.ELFDATA2LSB || m.Machine != elf.EM_X86_64 || m.Type != elf.ET_EXEC || m.Interpreter != "/test/loader" || m.SONAME != "tool.so" || !reflect.DeepEqual(m.Needed, []string{"liba.so", "libb.so"}) || !reflect.DeepEqual(m.RPath, []string{"${ORIGIN}/lib"}) {
		t.Fatalf("metadata differs: %+v", m)
	}
}

func TestCloseELFInterpreterInheritedRPathAndDeterminism(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "lib")
	if err := os.Mkdir(lib, 0700); err != nil {
		t.Fatal(err)
	}
	loader := writeELF(t, filepath.Join(dir, "loader"), fixture{soname: "loader"})
	leaf := writeELF(t, filepath.Join(lib, "libleaf.so"), fixture{soname: "libleaf.so"})
	writeELF(t, filepath.Join(lib, "libmid.so"), fixture{needed: []string{"libleaf.so"}, soname: "libmid.so"})
	root := writeELF(t, filepath.Join(dir, "tool"), fixture{kind: elf.ET_EXEC, needed: []string{"libmid.so"}, rpath: []string{"$ORIGIN/lib"}, interpreter: loader.Path})
	a, err := CloseELF([]Pin{root, leaf}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := CloseELF([]Pin{leaf, root, root}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) || len(a.Nodes) != 4 || len(a.Inputs) != 4 || len(a.Bindings) != 3 {
		t.Fatalf("unstable or incomplete closure: %+v %+v", a, b)
	}
	if _, err = CloseELF([]Pin{root}, []string{lib, filepath.Join(dir, "lib")}); err != nil {
		t.Fatal(err)
	}
}

func TestCloseELFRunPathIsDirectOnly(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "lib")
	if err := os.Mkdir(lib, 0700); err != nil {
		t.Fatal(err)
	}
	writeELF(t, filepath.Join(lib, "libleaf.so"), fixture{})
	writeELF(t, filepath.Join(lib, "libmid.so"), fixture{needed: []string{"libleaf.so"}})
	root := writeELF(t, filepath.Join(dir, "tool"), fixture{needed: []string{"libmid.so"}, runpath: []string{"$ORIGIN/lib"}})
	if _, err := CloseELF([]Pin{root}, nil); err == nil {
		t.Fatal("RUNPATH incorrectly inherited by grandchild")
	}
	writeELF(t, filepath.Join(lib, "libmid.so"), fixture{needed: []string{"libleaf.so"}, runpath: []string{"${ORIGIN}"}})
	if _, err := CloseELF([]Pin{root}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestCloseELFRejectsSeedIdentityBeforeParsing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "not-elf")
	if err := os.WriteFile(path, []byte("not ELF"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := CloseELF(nil, nil); err == nil {
		t.Fatal("nil seeds accepted")
	}
	for _, p := range []Pin{{path, ""}, {path, strings.Repeat("0", 64)}, {filepath.Join(dir, "missing"), strings.Repeat("0", 64)}} {
		if _, err := CloseELF([]Pin{p}, nil); err == nil {
			t.Fatalf("invalid seed accepted: %+v", p)
		}
	}
	if _, err := CloseELF([]Pin{{path, strings.Repeat("0", 64)}}, nil); err == nil || !strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("seed parsed before pin verification: %v", err)
	}
	for _, path := range []string{dir, path, filepath.Join(dir, "missing")} {
		if _, err := InspectELF(path); err == nil {
			t.Fatalf("unsupported input accepted: %s", path)
		}
	}
	p := writeELF(t, filepath.Join(dir, "object"), fixture{kind: elf.ET_REL})
	if _, err := InspectELF(p.Path); err == nil {
		t.Fatal("relocatable object admitted as loadable ELF")
	}
}

func TestCloseELFAmbiguityCanonicalAliasesAndMissingBindings(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a")
	b := filepath.Join(dir, "b")
	for _, p := range []string{a, b} {
		if err := os.Mkdir(p, 0700); err != nil {
			t.Fatal(err)
		}
	}
	root := writeELF(t, filepath.Join(dir, "root"), fixture{needed: []string{"libsame.so"}})
	if _, err := CloseELF([]Pin{root}, []string{a}); err == nil {
		t.Fatal("missing dependency accepted")
	}
	writeELF(t, filepath.Join(a, "libsame.so"), fixture{})
	alias := filepath.Join(dir, "alias")
	if err := os.Symlink(a, alias); err != nil {
		t.Fatal(err)
	}
	if _, err := CloseELF([]Pin{root}, []string{a, alias}); err != nil {
		t.Fatal(err)
	}
	writeELF(t, filepath.Join(b, "libsame.so"), fixture{})
	if _, err := CloseELF([]Pin{root}, []string{a, b}); err == nil {
		t.Fatal("distinct files with identical bytes silently bound by search order")
	}
}

func TestCloseELFRejectsUnsupportedPathsCyclesAndABI(t *testing.T) {
	dir := t.TempDir()
	for _, value := range []string{"$LIB", "$ORIGIN_BAD", "", ":/usr/lib", "relative", filepath.Join(dir, "missing")} {
		p := writeELF(t, filepath.Join(dir, "bad-path"), fixture{runpath: []string{value}})
		if _, err := CloseELF([]Pin{p}, nil); err == nil {
			t.Fatalf("unsupported RUNPATH accepted: %q", value)
		}
	}
	p := writeELF(t, filepath.Join(dir, "bad-needed"), fixture{needed: []string{"../libfoo.so"}})
	if _, err := CloseELF([]Pin{p}, nil); err == nil {
		t.Fatal("working-directory-relative dependency accepted")
	}
	p = writeELF(t, filepath.Join(dir, "root"), fixture{needed: []string{"libb.so"}, rpath: []string{"$ORIGIN"}})
	writeELF(t, filepath.Join(dir, "libb.so"), fixture{needed: []string{"root"}, rpath: []string{"$ORIGIN"}})
	if _, err := CloseELF([]Pin{p}, nil); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("cycle not rejected: %v", err)
	}
	writeELF(t, filepath.Join(dir, "libb.so"), fixture{machine: elf.EM_AARCH64})
	if _, err := CloseELF([]Pin{p}, nil); err == nil || !strings.Contains(err.Error(), "ABI") {
		t.Fatalf("incompatible ABI not rejected: %v", err)
	}
	for _, dirs := range [][]string{{""}, {"relative"}, {filepath.Join(dir, "missing")}} {
		if _, err := CloseELF([]Pin{p}, dirs); err == nil {
			t.Fatalf("invalid search dirs accepted: %v", dirs)
		}
	}
}

func TestCloseELFRejectsGlobalSONAMECollision(t *testing.T) {
	dir := t.TempDir()
	a := writeELF(t, filepath.Join(dir, "a"), fixture{soname: "libsame.so"})
	b := writeELF(t, filepath.Join(dir, "b"), fixture{soname: "libsame.so", runpath: []string{"$ORIGIN"}})
	if _, err := CloseELF([]Pin{a, b}, nil); err == nil {
		t.Fatal("two distinct global SONAME bindings accepted")
	}
}

func TestInspectCurrentGoExecutableWithoutExecution(t *testing.T) {
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	m, err := InspectELF(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.Pin.Path == "" || len(m.Pin.SHA256) != 64 {
		t.Fatalf("missing actual executable metadata: %+v", m)
	}
}

func TestInspectELFRejectsHiddenOrInvalidDynamicDependencies(t *testing.T) {
	dir := t.TempDir()
	for _, mutation := range []string{"bad-string-offset", "sectionless-dynamic", "filter"} {
		p := writeELF(t, filepath.Join(dir, mutation), fixture{needed: []string{"libhidden.so"}, interpreter: "/test/loader"})
		body, err := os.ReadFile(p.Path)
		if err != nil {
			t.Fatal(err)
		}
		sections := int(binary.LittleEndian.Uint64(body[40:]))
		dynamic := int(binary.LittleEndian.Uint64(body[sections+3*64+24:]))
		switch mutation {
		case "bad-string-offset":
			binary.LittleEndian.PutUint64(body[dynamic+8:], ^uint64(0))
		case "sectionless-dynamic":
			binary.LittleEndian.PutUint32(body[sections+3*64+4:], uint32(elf.SHT_PROGBITS))
		case "filter":
			binary.LittleEndian.PutUint64(body[dynamic:], uint64(elf.DT_FILTER))
		}
		if err = os.WriteFile(p.Path, body, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = InspectELF(p.Path); err == nil {
			t.Fatalf("incomplete dynamic dependency metadata silently accepted: %s", mutation)
		}
	}
}
