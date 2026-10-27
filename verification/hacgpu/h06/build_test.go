package h06

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func testBuildFile(t *testing.T, path string, body []byte) nativeproof.Pin {
	t.Helper()
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	p, e := buildPin(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}

func TestH06CompilerAdmissionPrecedesAnyInputOrChild(t *testing.T) {
	if _, err := BuildOracle(BuildOptions{Plan: nativeproof.Pin{Path: "must-not-read"}, Out: "must-not-create"}); err == nil || !strings.Contains(err.Error(), "admission") || runner.ExitCode(err) != 75 {
		t.Fatalf("unadmitted compiler boundary: %v", err)
	}
	if _, err := BuildController(ControllerBuildOptions{Out: "must-not-create"}); err == nil {
		t.Fatal("unadmitted Go builder accepted")
	}
	calls := 0
	reject := errors.New("changed compiler plugin")
	_, err := checkedBuildCommand(func() error { return reject }, func() (runner.CommandReceipt, error) { calls++; return runner.CommandReceipt{}, nil })
	if !errors.Is(err, reject) || calls != 0 {
		t.Fatalf("child launched before closure: calls=%d err=%v", calls, err)
	}
	checks := 0
	_, err = checkedBuildCommand(func() error {
		checks++
		if checks == 2 {
			return reject
		}
		return nil
	}, func() (runner.CommandReceipt, error) { calls++; return runner.CommandReceipt{Exit: 0}, nil })
	if !errors.Is(err, reject) || checks != 2 || calls != 1 {
		t.Fatalf("post-child closure drift accepted: %d %d %v", checks, calls, err)
	}
}

func TestH06CompilerDomainsRejectOmissionsAndDrift(t *testing.T) {
	dir := t.TempDir()
	body := make([]byte, 64)
	copy(body, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(body[16:], 2)
	binary.LittleEndian.PutUint16(body[18:], 62)
	binary.LittleEndian.PutUint32(body[20:], 1)
	binary.LittleEndian.PutUint16(body[52:], 64)
	binary.LittleEndian.PutUint16(body[54:], 56)
	binary.LittleEndian.PutUint16(body[58:], 64)
	pin := testBuildFile(t, filepath.Join(dir, "compiler-plugin"), body)
	closure, err := nativeproof.CloseELF([]nativeproof.Pin{pin}, []string{dir})
	if err != nil {
		t.Fatal(err)
	}
	cache := testBuildFile(t, filepath.Join(dir, "cache"), []byte("cache"))
	c := compilerClosure{Tools: []selectedTool{{Name: "driver", Executable: pin}}, SearchDirs: []string{dir}, LoaderCache: cache, Domains: []compilerDomain{{Name: "driver", Root: pin, ELF: closure}}, Inputs: map[string]string{pin.Path: pin.SHA256, cache.Path: cache.SHA256}}
	for _, name := range []string{"cc1plus", "collect2", "as", "ld", "nm", "lto-wrapper", "lto1", "cc1", "gcc-driver"} {
		c.Tools = append(c.Tools, selectedTool{Name: name, Executable: pin})
		c.Domains = append(c.Domains, compilerDomain{Name: name, Root: pin, ELF: closure})
	}
	inputs := map[string]string{pin.Path: pin.SHA256, cache.Path: cache.SHA256}
	if err = checkCompilerDomains(c, inputs); err != nil {
		t.Fatal(err)
	}
	delete(inputs, pin.Path)
	if err = checkCompilerDomains(c, inputs); err == nil {
		t.Fatal("omitted selected compiler/plugin input accepted")
	}
	inputs[pin.Path] = pin.SHA256
	c.Domains[0].ELF.Inputs = nil
	if err = checkCompilerDomains(c, inputs); err == nil {
		t.Fatal("self-consistent unobserved closure accepted")
	}
	c.Domains[0].ELF = closure
	changed := append([]byte(nil), body...)
	changed[24] = 1
	if err = os.WriteFile(pin.Path, changed, 0600); err != nil {
		t.Fatal(err)
	}
	if err = checkCompilerDomains(c, inputs); err == nil {
		t.Fatal("changed selected compiler/plugin bytes accepted")
	}
}

func TestH06NativeLimits(t *testing.T) {
	if err := checkBuildLimits("200000 100000", "3221225472", "0"); err != nil {
		t.Fatal(err)
	}
	for _, v := range [][3]string{{"max 100000", "3221225472", "0"}, {"200001 100000", "3221225472", "0"}, {"200000 100000", "3221225473", "0"}, {"200000 100000", "3221225472", "1"}, {"0 100000", "1", "0"}, {"1 0", "1", "0"}} {
		if err := checkBuildLimits(v[0], v[1], v[2]); err == nil {
			t.Fatalf("unbounded process accepted: %v", v)
		}
	}
}

func TestH06ExactDependenciesAndLinkChoice(t *testing.T) {
	if err := checkBuildDependencies([]string{"a.h", "source.cpp"}, []string{"a.h", "source.cpp"}); err != nil {
		t.Fatal(err)
	}
	if err := checkBuildDependencies([]string{"a.h", "source.cpp"}, []string{"a.h", "new.h", "source.cpp"}); err == nil {
		t.Fatal("new actual header admitted after compile")
	}
	wm := "archive/libdorado_nn.a(WorkingMemory.cpp.o)"
	flstm := "archive/libdorado_nn.a(FLSTMStack.cpp.o)"
	if err := checkHalfLinkMap(wm+"\n"+flstm, "reference"); err != nil {
		t.Fatal(err)
	}
	if err := checkHalfLinkMap(wm, "observed"); err != nil {
		t.Fatal(err)
	}
	if err := checkHalfLinkMap(wm, "reference"); err == nil {
		t.Fatal("reference without stock implementation accepted")
	}
	if err := checkHalfLinkMap(wm+"\n"+flstm, "observed"); err == nil {
		t.Fatal("second archived observed implementation accepted")
	}
}

func TestH06DriverQueryRequiresClosedResource(t *testing.T) {
	dir := t.TempDir()
	pin := testBuildFile(t, filepath.Join(dir, "lib.a"), []byte("archive"))
	q := linkQuery{Argv: []string{"driver", "-print-file-name=lib.a"}, Expected: pin.Path}
	if err := verifyBuildQuery(pin.Path+"\n", q, dir, map[string]string{pin.Path: pin.SHA256}); err != nil {
		t.Fatal(err)
	}
	if err := verifyBuildQuery(pin.Path, q, dir, map[string]string{}); err == nil {
		t.Fatal("query resolved an undeclared input")
	}
	if err := verifyBuildQuery("lib.a", q, dir, map[string]string{pin.Path: pin.SHA256}); err == nil {
		t.Fatal("required library left unresolved")
	}
	q.Expected = "libmissing.so"
	q.Argv[1] = "-print-file-name=libmissing.so"
	if err := verifyBuildQuery("libmissing.so\n", q, dir, nil); err != nil {
		t.Fatal(err)
	}
}

func TestH06ClosedLinkTemplateHasNoNewSearch(t *testing.T) {
	got, err := closedLinkFiles([]string{"/archive/lib.a", "-Wl,-rpath,/lib:/opt/lib:", "-Wl,--no-as-needed,/torch/libtorch.so", "-Wl,--as-needed"})
	if err != nil || len(got) != 2 || got[0] != "/archive/lib.a" || got[1] != "/torch/libtorch.so" {
		t.Fatalf("closed files %v: %v", got, err)
	}
	for _, args := range [][]string{{"-lmissing"}, {"relative.a"}, {"-L/new/search"}, {"-Wl,-rpath"}, {"-Wl,--script,new.ld"}, {"-Wl,--whole-archive"}} {
		if _, err = closedLinkFiles(args); err == nil {
			t.Fatalf("new linker lookup/control accepted: %v", args)
		}
	}
}

func TestH06ClosedResourceAuthenticatesLexicalSymlinkPin(t *testing.T) {
	dir := t.TempDir()
	pin := testBuildFile(t, filepath.Join(dir, "libactual.so.1"), []byte("accepted library"))
	alias := filepath.Join(dir, "libactual.so")
	if err := os.Symlink(pin.Path, alias); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]string{}
	if err := addClosedQueryInput(alias, map[string]string{alias: pin.SHA256}, inputs); err != nil {
		t.Fatal(err)
	}
	if inputs[pin.Path] != pin.SHA256 {
		t.Fatal("canonical input not bound to the accepted lexical pin")
	}
	if err := addClosedQueryInput(alias, nil, map[string]string{}); err == nil {
		t.Fatal("unattested resource admitted by a fresh hash")
	}
	if err := os.WriteFile(pin.Path, []byte("changed library"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := addClosedQueryInput(alias, map[string]string{alias: pin.SHA256}, inputs); err == nil {
		t.Fatal("changed symlink target bytes accepted")
	}
}

func TestH06QuerySelectionDoesNotAdmitUnusedAlternatives(t *testing.T) {
	dir := t.TempDir()
	used := testBuildFile(t, filepath.Join(dir, "libm.so"), []byte("selected link script"))
	unused := testBuildFile(t, filepath.Join(dir, "libm.a"), []byte("unused static alternative"))
	inputs := map[string]string{used.Path: used.SHA256}
	queries := []linkQuery{{Argv: []string{"driver", "-print-file-name=libm.so"}, Expected: used.Path}, {Argv: []string{"driver", "-print-file-name=libm.a"}, Expected: unused.Path}, {Argv: []string{"driver", "-print-file-name=absent.a"}, Expected: "absent.a"}}
	got, err := bindSelectedQueries(queries, map[string]string{used.Path: used.SHA256}, inputs)
	if err != nil || len(got) != 1 || got[0].Expected != used.Path {
		t.Fatalf("selected queries %v: %v", got, err)
	}
	if _, ok := inputs[unused.Path]; ok {
		t.Fatal("unused lookup alternative was promoted into an input")
	}
}
