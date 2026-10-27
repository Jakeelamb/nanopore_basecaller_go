package cgobuild

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Normalized from build003/rejected/request-3769517875/rejection.json.
// Only the package/object paths are replaced with this test's captured context.
const retainedPrefixedSeedProbe = `["-I","<package>","-fPIC","-m64","-pthread","-Wl,--no-gc-sections","-fmessage-length=0","-ffile-prefix-map=<object>=/tmp/go-build","-gno-record-gcc-switches","-frandom-seed=1","-O2","-g","-c","-x","c","-","-o","/dev/null"]`

func supportFixture(t *testing.T, family string) (Plan, Session, []GeneratedSet, string, []string) {
	t.Helper()
	p, s, lease, log := fakeFixture(t)
	derived, e := derivedRead(s)
	if e != nil || len(derived) != 1 {
		t.Fatalf("captured context: %v", e)
	}
	if family == "runtime/cgo" {
		p.Packages[0].ImportPath = family
		derived[0].Package = family
		body, e := json.Marshal(derived[0])
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(s.Directory, "derived", "tensor.json"), body, 0600); e != nil {
			t.Fatal(e)
		}
	}
	var args []string
	if e = json.Unmarshal([]byte(retainedPrefixedSeedProbe), &args); e != nil {
		t.Fatal(e)
	}
	for i, a := range args {
		args[i] = strings.ReplaceAll(strings.ReplaceAll(a, "<package>", p.Packages[0].Directory), "<object>", lease.ObjectDir)
	}
	t.Chdir(filepath.Dir(lease.ObjectDir))
	return p, s, derived, log, args
}

func TestPrefixedCompilerSupportTrialsRetainCapturedCgoContext(t *testing.T) {
	for _, family := range []string{"runtime/cgo", TensorPackage} {
		t.Run(family, func(t *testing.T) {
			p, s, derived, log, retained := supportFixture(t, family)
			for _, trial := range []string{"-frandom-seed=1", "-fdebug-prefix-map=a=b", "-ffile-prefix-map=a=b"} {
				t.Run(trial, func(t *testing.T) {
					args := slices.Clone(retained)
					args[len(args)-9] = trial
					before := spawns(log)
					op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
					if e != nil || op.Role != "go-flag-probe" || op.Request.Support == nil || op.Request.Support.Package != family || op.Request.Support.Trial != trial || spawns(log) != before+2 {
						t.Fatalf("source-derived full-prefix trial: %v", e)
					}
					origin := op.Request.Support
					if op.Inputs[origin.Producer.Path] != origin.Producer.SHA256 || op.Inputs[origin.Types.Path] != origin.Types.SHA256 {
						t.Fatal("captured producer/types absent from before inputs")
					}
					dir := filepath.Dir(op.Before.Path)
					dir = filepath.Dir(dir)
					if e = verifyNativeOperation(p, s, op, dir, derived); e != nil {
						t.Fatal(e)
					}
					missing := op
					request := *op.Request
					request.Support = nil
					missing.Request = &request
					if verifyNativeOperation(p, s, missing, dir, derived) == nil {
						t.Fatal("reader accepted omitted support context")
					}
				})
			}
		})
	}
}

func TestPrefixedSupportRejectsForeignFormsAndKeepsNativeSeedStrict(t *testing.T) {
	p, s, derived, log, retained := supportFixture(t, TensorPackage)
	for _, kind := range []string{"foreign source directory", "foreign object map", "foreign map target", "unknown trial", "unknown prefix", "prefix order", "stdin", "output", "no captured context"} {
		t.Run(kind, func(t *testing.T) {
			args := slices.Clone(retained)
			var stdin []byte
			for i, a := range args {
				switch kind {
				case "foreign source directory":
					if i == 1 {
						args[i] = "/foreign/package"
					}
				case "foreign object map":
					if strings.HasPrefix(a, "-ffile-prefix-map=") {
						args[i] = "-ffile-prefix-map=/foreign/object=/tmp/go-build"
					}
				case "foreign map target":
					if strings.HasPrefix(a, "-ffile-prefix-map=") {
						args[i] = strings.TrimSuffix(a, "/tmp/go-build") + "/foreign"
					}
				}
			}
			switch kind {
			case "unknown trial":
				args[len(args)-9] = "-funsafe-foreign=1"
			case "unknown prefix":
				args = append(args[:5:5], append([]string{"-foreign-prefix"}, args[5:]...)...)
			case "prefix order":
				args[2], args[3] = args[3], args[2]
			case "stdin":
				stdin = []byte("int foreign;")
			case "output":
				args[len(args)-1] = filepath.Join(p.Options.WorkRoot, "foreign-output")
			case "no captured context":
				if e := os.Remove(filepath.Join(s.Directory, "derived", "tensor.json")); e != nil {
					t.Fatal(e)
				}
				defer func() {
					body, _ := json.Marshal(derived[0])
					if e := os.WriteFile(filepath.Join(s.Directory, "derived", "tensor.json"), body, 0600); e != nil {
						t.Error(e)
					}
				}()
			}
			before := spawns(log)
			if _, e := interposeCC(s, p, args, stdin, nil, false, fakeBoundary); e == nil || spawns(log) != before {
				t.Fatalf("foreign prefixed probe reached compiler: %v", e)
			}
		})
	}
	for _, seed := range []string{"1", strings.Repeat("a", 20)} {
		t.Run("native-seed-"+seed, func(t *testing.T) {
			pkg := p.Packages[0]
			source := filepath.Join(pkg.Directory, pkg.C[0])
			obj := filepath.Dir(derived[0].Files["_cgo_gotypes.go"].Path)
			args := []string{"-I", pkg.Directory, "-I", obj, "-fPIC", "-m64", "-pthread", "-O2", "-g", "-frandom-seed=" + seed, "-o", filepath.Join(obj, "seed.o"), "-c", source}
			before := spawns(log)
			op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
			if seed == "1" {
				if e == nil || spawns(log) != before {
					t.Fatal("seed1 was admitted for a real TU")
				}
			} else if e != nil || op.Role != "native-tu" || op.Request.Support != nil {
				t.Fatalf("real20-character TU seed rejected: %v", e)
			}
		})
	}
}
