package closedbuild

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestCompileAdmissionBeforeReadOrOutput(t *testing.T) {
	out := filepath.Join(t.TempDir(), "must-not-create")
	_, e := Compile(BuildOptions{Plan: Pin{Path: "/must/not/open"}, Out: out})
	if !errors.Is(e, runner.ErrAdmissionBlocked) || runner.ExitCode(e) != 75 {
		t.Fatalf("admission gate: %v", e)
	}
	if _, e = os.Stat(out); !os.IsNotExist(e) {
		t.Fatalf("unadmitted output exists: %v", e)
	}
}
func TestProfilesAreExplicitAndFinite(t *testing.T) {
	for _, profiles := range [][]string{nil, {""}, {"all"}, {"backend", "backend"}, {"cuda-oracle"}} {
		if _, e := PlanBuild(PlanOptions{Profiles: profiles}); e == nil {
			t.Fatalf("accepted implicit/unsupported profiles %v", profiles)
		}
	}
}
func TestSelectedGoOwnsAssemblyEmbedAndModule(t *testing.T) {
	value := goPackage{Dir: "/source", GoFiles: []string{"main.go"}, HFiles: []string{"defs.h"}, SFiles: []string{"entry.s"}, SysoFiles: []string{"extra.syso"}, EmbedFiles: []string{"data/value.json"}, Module: &struct{ GoMod string }{"/source/go.mod"}}
	body, e := json.Marshal(value)
	if e != nil {
		t.Fatal(e)
	}
	actual, e := selectedGo(body)
	if e != nil {
		t.Fatal(e)
	}
	want := []string{"/source/data/value.json", "/source/defs.h", "/source/entry.s", "/source/extra.syso", "/source/go.mod", "/source/main.go"}
	if !slices.Equal(actual, want) {
		t.Fatalf("source custody: %v", actual)
	}
	for _, bad := range []string{`{"Dir":"/source","GoFiles":["../other.go"]}`, `{"Dir":"/source","CgoFiles":["bridge.go"]}`, `{"Dir":"relative","GoFiles":["main.go"]}`, `{"Dir":"/source","Error":{"Err":"broken selection"}}`} {
		if _, e = selectedGo([]byte(bad)); e == nil {
			t.Fatalf("bad selection accepted: %s", bad)
		}
	}
}
func TestPinnedBytesAndCanonicalAlias(t *testing.T) {
	d := t.TempDir()
	path := filepath.Join(d, "file")
	if e := os.WriteFile(path, []byte("first"), 0600); e != nil {
		t.Fatal(e)
	}
	p, e := PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	alias := filepath.Join(d, "alias")
	if e = os.Symlink(path, alias); e != nil {
		t.Fatal(e)
	}
	q, e := PinFile(alias)
	if e != nil || q != p {
		t.Fatalf("alias pin: %+v %v", q, e)
	}
	if e = os.WriteFile(path, []byte("second"), 0600); e != nil {
		t.Fatal(e)
	}
	if verify(p) == nil {
		t.Fatal("changed bytes accepted")
	}
}

func TestChangedIncludeRootAliasRejected(t *testing.T) {
	d := t.TempDir()
	a := filepath.Join(d, "original")
	b := filepath.Join(d, "replacement")
	for _, p := range []string{a, b} {
		if e := os.Mkdir(p, 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(filepath.Join(p, "header.h"), []byte("identical bytes"), 0600); e != nil {
			t.Fatal(e)
		}
	}
	alias := filepath.Join(d, "include")
	if e := os.Symlink(a, alias); e != nil {
		t.Fatal(e)
	}
	bindings, e := pathBindings([]string{alias}, d)
	if e != nil {
		t.Fatal(e)
	}
	if e = checkPathBindings(bindings); e != nil {
		t.Fatal(e)
	}
	if e = os.Remove(alias); e != nil {
		t.Fatal(e)
	}
	if e = os.Symlink(b, alias); e != nil {
		t.Fatal(e)
	}
	if checkPathBindings(bindings) == nil {
		t.Fatal("changed include root accepted despite matching old canonical file bytes")
	}
}

func TestPthreadArchiveCannotBeOmitted(t *testing.T) {
	if _, e := PlanBuild(PlanOptions{Profiles: []string{"backend"}}); e == nil || !strings.Contains(e.Error(), "explicit pthread archive pin required") {
		t.Fatalf("plan omitted driver-added input: %v", e)
	}
	d := t.TempDir()
	path := filepath.Join(d, "libpthread.a")
	if e := os.WriteFile(path, []byte("!<arch>\n"), 0600); e != nil {
		t.Fatal(e)
	}
	p, e := PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	inputs := map[string]string{}
	q, e := closePthreadArchive(p, "/selected/g++-15", inputs)
	if e != nil {
		t.Fatal(e)
	}
	if p.SHA256 != PthreadArchiveSHA || inputs[p.Path] != PthreadArchiveSHA || !slices.Equal(q.Argv, []string{"/selected/g++-15", "-print-file-name=libpthread.a"}) || q.Expected != p.Path {
		t.Fatal("pthread input/query was not closed before invocation")
	}
	trace := path + "\n"
	if e = h07.VerifyLinkTrace(trace, d, map[string]string{}, nil); e == nil || !strings.Contains(e.Error(), "absent from before-compile closure") {
		t.Fatalf("omitted archive not rejected: %v", e)
	}
	if e = h07.VerifyLinkTrace(trace, d, inputs, nil); e != nil {
		t.Fatal(e)
	}
	if e = queryResult(path+"\n", q, d, inputs); e != nil {
		t.Fatal(e)
	}
	if e = queryResult(path+"\n", q, d, map[string]string{}); e == nil {
		t.Fatal("selected query resource absent from prior inputs accepted")
	}
	otherDir := filepath.Join(d, "other")
	if e = os.Mkdir(otherDir, 0700); e != nil {
		t.Fatal(e)
	}
	other := filepath.Join(otherDir, "libpthread.a")
	if e = os.WriteFile(other, []byte("!<arch>\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if e = queryResult(other+"\n", q, d, inputs); e == nil {
		t.Fatal("same bytes at a different compiler-selected path accepted")
	}
	if e = os.WriteFile(path, []byte("!<arch>\nchanged"), 0600); e != nil {
		t.Fatal(e)
	}
	changed, e := PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = closePthreadArchive(changed, "/selected/g++-15", map[string]string{}); e == nil {
		t.Fatal("self-rehashed nonempty archive accepted")
	}
}
