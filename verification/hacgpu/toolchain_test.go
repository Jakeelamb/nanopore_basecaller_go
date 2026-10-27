package hacgpu

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestSplitCompileCommandLiteralArguments(t *testing.T) {
	for _, tt := range []struct {
		command string
		want    []string
	}{
		{`g++ -DNAME="some value" 'path with spaces.cpp'`, []string{"g++", "-DNAME=some value", "path with spaces.cpp"}},
		{`g++ ab" cd"'ef'`, []string{"g++", "ab cdef"}},
		{`g++ "" ''`, []string{"g++", "", ""}},
		{`g++ dir\ name/file.cpp`, []string{"g++", "dir name/file.cpp"}},
		{`g++ "a\q" "\$HOME" '$HOME'`, []string{"g++", `a\q`, "$HOME", "$HOME"}},
		{"g++ a\\\nb", []string{"g++", "ab"}},
		{`g++ 'a'\''b'`, []string{"g++", "a'b"}},
	} {
		got, err := splitCommand(tt.command)
		if err != nil || !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("%q: arguments %q, %v; want %q", tt.command, got, err, tt.want)
		}
	}
	for _, command := range []string{`g++ "missing`, `g++ 'missing`, "g++ trailing\\"} {
		if _, err := splitCommand(command); err == nil {
			t.Fatalf("malformed compile command %q accepted", command)
		}
	}
}

func TestOracleToolchainReadsRetainedMetadata(t *testing.T) {
	dir := t.TempDir()
	entries := []compileEntry{
		{File: "source/basecall/model/CRFModel.cpp", Command: "/must/not/execute/g++ -I/retained/source -isystem /retained/torch/include -DMODE=1 -O3 -c file.cpp"},
		{File: "source/nn/TxModules.cpp", Command: `/must/not/execute/clang++ -I/retained/tx -DTX=1 -DFMT_BEGIN_NAMESPACE="namespace fmt { inline namespace ont {" -DFMT_END_NAMESPACE="}}" -c tx.cpp`},
	}
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "compile_commands.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	ninja := "build bin/dorado: CXX_EXECUTABLE_LINKER__dorado_Release\n  LINK_LIBRARIES = first.a -Wl,--no-as-needed /retained/libtorch.so\n  LINK_PATH = -L/retained\n"
	if err := os.WriteFile(filepath.Join(dir, "build.ninja"), []byte(ninja), 0600); err != nil {
		t.Fatal(err)
	}
	want := OracleBuildToolchain{Compiler: "/must/not/execute/g++", Includes: []string{"-I/retained/source", "-isystem", "/retained/torch/include"}, Defines: []string{"-DMODE=1"}, Libraries: []string{"first.a", "-Wl,--no-as-needed", "/retained/libtorch.so"}}
	got, err := OracleToolchain(dir)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("toolchain %+v, %v; want %+v", got, err, want)
	}
	got.Includes[0] = "changed"
	again, err := OracleToolchain(dir)
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatalf("a caller changed later toolchain reads: %+v, %v", again, err)
	}
	tx, err := OracleToolchainForSource(dir, "nn/TxModules.cpp")
	if err != nil || tx.Compiler != "/must/not/execute/clang++" || !reflect.DeepEqual(tx.Includes, []string{"-I/retained/tx"}) || !reflect.DeepEqual(tx.Defines, []string{"-DTX=1", "-DFMT_BEGIN_NAMESPACE=namespace fmt { inline namespace ont {", "-DFMT_END_NAMESPACE=}}"}) || !reflect.DeepEqual(tx.Libraries, want.Libraries) {
		t.Fatalf("wrong source toolchain selected: %+v, %v", tx, err)
	}
	for _, suffix := range []string{"", "nn/missing.cpp"} {
		if _, err := OracleToolchainForSource(dir, suffix); err == nil {
			t.Fatalf("missing source selector %q accepted", suffix)
		}
	}
	duplicate, err := json.Marshal(append(entries, entries[1]))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "compile_commands.json"), duplicate, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := OracleToolchainForSource(dir, "nn/TxModules.cpp"); err == nil {
		t.Fatal("ambiguous source selector accepted")
	}
	if err := os.Remove(filepath.Join(dir, "build.ninja")); err != nil {
		t.Fatal(err)
	}
	if _, err := OracleToolchain(dir); err == nil {
		t.Fatal("missing retained link metadata accepted")
	}
}
