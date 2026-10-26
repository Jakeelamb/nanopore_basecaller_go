package cgobuild

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func targetFixture(t *testing.T) Plan {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                          "module " + Module + "\n\ngo 1.27.3\n",
		"internal/tensor/tensor.go":       "package tensor\nfunc Value() int { return 1 }\n",
		"internal/tensor/tensor_test.go":  "package tensor\nimport \"testing\"\nfunc TestAlpha(t *testing.T) {}\nfunc TestBeta(t *testing.T) {}\n",
		"internal/adapter/adapter.go":     "package adapter\nimport \"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor\"\nfunc Value() int { return tensor.Value() }\n",
		"internal/tensor/adapter_test.go": "package tensor_test\nimport (\"testing\";\"github.com/Jakeelamb/nanopore_basecaller_go/internal/adapter\")\nfunc TestExternalAdapter(t *testing.T) { if adapter.Value()!=1 { t.Fatal(\"adapter\") } }\n",
		"verification/hacgpu/h08/host.go": "package h08\nimport _ \"embed\"\n//go:embed cpu13/testmain.go.txt\nvar Source string\nfunc CPU13TestMain(run func() int) int { return run() }\n",
	}
	template, e := os.ReadFile("../cpu13/testmain.go.txt")
	if e != nil {
		t.Fatal(e)
	}
	files["verification/hacgpu/h08/cpu13/testmain.go.txt"] = string(template)
	for name, body := range files {
		path := filepath.Join(root, name)
		if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(path, []byte(body), 0600); e != nil {
			t.Fatal(e)
		}
	}
	p := Plan{Schema: "nanopore-h08-cgo-build-plan/v2", Role: CPU13Tests, Options: Options{Role: CPU13Tests, Root: root, GoRoot: runtime.GOROOT(), WorkRoot: filepath.Join(root, "work")}}
	p.Packages, p.Inputs, e = selectSources(root, runtime.GOROOT(), CPU13Tests)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = pinned(filepath.Join(runtime.GOROOT(), "src/cmd/go/internal/load/test.go"), p.Inputs); e != nil {
		t.Fatal(e)
	}
	p.CPU13, e = cpu13Profile(p)
	if e != nil {
		t.Fatal(e)
	}
	p.BuildArgv = plannedBuildArgs(p)
	return p
}

func TestFixedRoleReadersRejectWrongRoleBeforeOpeningPlan(t *testing.T) {
	for _, role := range []BuildRole{H08Command, CPU13Tests, "foreign-command"} {
		path := filepath.Join(t.TempDir(), "receipt.json")
		if e := write(path, Receipt{Role: role, Plan: Pin{Path: "/must/not/open"}}); e != nil {
			t.Fatal(e)
		}
		q, _ := closedPin(path)
		if role != H08Command {
			if _, e := ReadBuild(q); e == nil || !strings.Contains(e.Error(), "wrong fixed build role") {
				t.Fatalf("ordinary reader accepted/tested another role: %v", e)
			}
		}
		if role != CPU13Tests {
			if _, e := ReadCPU13Build(q); e == nil || !strings.Contains(e.Error(), "wrong fixed build role") {
				t.Fatalf("test reader accepted/tested another role: %v", e)
			}
		}
	}
}

func TestCPU13FixedProfileBindsOverlayCopiesAndExactArgv(t *testing.T) {
	p := targetFixture(t)
	if e := validateTarget(p); e != nil {
		t.Fatal(e)
	}
	if !slices.Contains(p.BuildArgv, "test") || !slices.Contains(p.BuildArgv, "-c") || p.BuildArgv[len(p.BuildArgv)-1] != "./internal/tensor" || outputPath(p) != filepath.Join(p.Options.WorkRoot, "tensor.test") {
		t.Fatal("CPU13 did not select the fixed compile-only test command")
	}
	if !knownPackage(p, TensorPackage+testSuffix) || !knownPackage(p, TensorPackage+"_test"+testSuffix) || !knownPackage(p, TensorPackage+".test") || knownPackage(p, "fmt"+testSuffix) || knownPackage(p, TensorPackage+" [foreign.test]") {
		t.Fatal("test package description boundary differs")
	}
	for _, kind := range []string{"ordinary overlay", "foreign mapping", "foreign package", "changed generated main", "missing add-only target"} {
		t.Run(kind, func(t *testing.T) {
			copy := p
			c := *p.CPU13
			copy.CPU13 = &c
			copy.BuildArgv = slices.Clone(p.BuildArgv)
			switch kind {
			case "ordinary overlay":
				copy.Role, copy.Options.Role = H08Command, H08Command
				copy.BuildArgv = plannedBuildArgs(copy)
			case "foreign mapping":
				copy.CPU13.OverlayJSON = `{"Replace":{"foreign":"foreign"}}`
			case "foreign package":
				copy.BuildArgv[len(copy.BuildArgv)-1] = "./cmd/foreign"
			case "changed generated main":
				copy.CPU13.MainSource += "\nfunc forged(){}\n"
			case "missing add-only target":
				copy.CPU13.OverlayTarget = filepath.Join(copy.Options.Root, "internal/tensor/tensor_test.go")
			}
			if validateTarget(copy) == nil {
				t.Fatal("mutated fixed profile accepted")
			}
		})
	}
	if e := os.WriteFile(p.CPU13.OverlayTarget, []byte("package tensor_test"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := cpu13Profile(p); e == nil {
		t.Fatal("overlay replaced an existing source file")
	}
}

func TestCPU13StaticClosureKeepsMaintainedTestsAndGoGenerator(t *testing.T) {
	root, e := filepath.Abs("../../../..")
	if e != nil {
		t.Fatal(e)
	}
	p := Plan{Role: CPU13Tests, Options: Options{Role: CPU13Tests, Root: root, GoRoot: runtime.GOROOT(), WorkRoot: filepath.Join(t.TempDir(), "work")}}
	p.Packages, p.Inputs, e = selectSources(root, runtime.GOROOT(), CPU13Tests)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = pinned(filepath.Join(runtime.GOROOT(), "src/cmd/go/internal/load/test.go"), p.Inputs); e != nil {
		t.Fatal(e)
	}
	p.CPU13, e = cpu13Profile(p)
	if e != nil {
		t.Fatal(e)
	}
	pkg, ok := sourcePackage(p, TensorPackage)
	if !ok || len(pkg.TestGo) < 10 {
		t.Fatal("maintained tensor tests omitted")
	}
	for _, name := range []string{"TestNativeTensorOperations", "TestNativeLoadWeight", "TestNativeLoadedRuntimePins", "TestNativeLoaderABIRejections", "TestNativeCloseFailureRetainsLibrary", "TestNativeABIBoundary"} {
		if !strings.Contains(p.CPU13.MainSource, `"`+name+`"`) {
			t.Fatalf("maintained native test missing from generated main: %s", name)
		}
	}
	if !slices.Contains(p.CPU13.TestCopies, Module+"/verification/hacgpu/h08") || !strings.Contains(p.CPU13.MainSource, "_xtest.TestMain(m)") {
		t.Fatal("external TestMain/parent dependency test-copy omitted")
	}
}

func TestGeneratedTestmainMatchesRealGo0GeneratorAndRejectsByteChanges(t *testing.T) {
	p := targetFixture(t)
	for _, dir := range []string{p.Options.WorkRoot, filepath.Join(p.Options.WorkRoot, "cache"), filepath.Join(p.Options.WorkRoot, "tmp"), filepath.Join(p.Options.WorkRoot, "logs")} {
		if e := os.MkdirAll(dir, 0700); e != nil {
			t.Fatal(e)
		}
	}
	if e := os.WriteFile(p.CPU13.OverlayPath, []byte(p.CPU13.OverlayJSON), 0400); e != nil {
		t.Fatal(e)
	}
	args := listArgs(p)
	env := []string{"PATH=/usr/bin:/bin", "CGO_ENABLED=0", "CC=/must/not/execute", "CXX=/must/not/execute", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off", "GOMAXPROCS=1", "GOCACHE=" + filepath.Join(p.Options.WorkRoot, "cache"), "GOTMPDIR=" + filepath.Join(p.Options.WorkRoot, "tmp"), "GOPATH=" + filepath.Join(p.Options.WorkRoot, "gopath")}
	logDir := filepath.Join(p.Options.WorkRoot, "logs")
	if retain := os.Getenv("NANOPORE_CGOBUILD_TEST_EVIDENCE"); retain != "" {
		var e error
		logDir, e = os.MkdirTemp(retain, "go0-testmain-")
		if e != nil {
			t.Fatal(e)
		}
	}
	command, _, e := runProgram(context.Background(), logDir, p.Options.Root, args, env, nil, false, true)
	stdout, _ := os.ReadFile(command.Stdout)
	stderr, _ := os.ReadFile(command.Stderr)
	t.Logf("CGO0 fake-module generator argv=%q env=%q exit=%d stdout_sha=%s stderr_sha=%s retained=%s", args, env, command.Exit, command.StdoutSHA, command.StderrSHA, logDir)
	if e != nil {
		t.Fatalf("generator failed: %v; raw stderr=%s", e, stderr)
	}
	t.Logf("CGO0 fake-module Go generator argv=%q exit=%d stdout_sha=%s stderr_sha=%s", args, command.Exit, command.StdoutSHA, command.StderrSHA)
	f, e := os.Open(command.Stdout)
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	d := json.NewDecoder(f)
	var generated string
	for d.More() {
		var x struct {
			ImportPath string
			GoFiles    []string
		}
		if e = d.Decode(&x); e != nil {
			t.Fatal(e)
		}
		if x.ImportPath == TensorPackage+".test" {
			if len(x.GoFiles) != 1 {
				t.Fatal("real Go main source omitted")
			}
			generated = x.GoFiles[0]
		}
	}
	q, e := closedPin(generated)
	if e := os.WriteFile(filepath.Join(logDir, "expected-testmain.txt"), []byte(p.CPU13.MainSource), 0400); e != nil {
		t.Fatal(e)
	}
	actual, e := os.ReadFile(q.Path)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(logDir, "actual-testmain.txt"), actual, 0400); e != nil {
		t.Fatal(e)
	}
	if q.SHA256 != p.CPU13.MainSHA256 {
		t.Fatalf("source-derived testmain differs from real Go generator: %+v", q)
	}
	if e = verifyActualSelection(p, stdout); e != nil {
		t.Fatalf("actual CGO0 fake-module test graph differs from prospective fixed profile: %v", e)
	}
	mainBoundaryControls(t, p, actual)
	for _, mutation := range []string{"missing external file", "foreign external file", "foreign external import"} {
		t.Run(mutation, func(t *testing.T) {
			decoder := json.NewDecoder(strings.NewReader(string(stdout)))
			var changed strings.Builder
			found := false
			for decoder.More() {
				var row map[string]any
				if e := decoder.Decode(&row); e != nil {
					t.Fatal(e)
				}
				if row["ImportPath"] == TensorPackage+"_test"+testSuffix {
					found = true
					switch mutation {
					case "missing external file":
						row["GoFiles"] = []string{OverlayName}
					case "foreign external file":
						row["GoFiles"] = []string{"foreign_test.go", OverlayName}
					case "foreign external import":
						row["Imports"] = []string{"foreign/import"}
					}
				}
				if e := json.NewEncoder(&changed).Encode(row); e != nil {
					t.Fatal(e)
				}
			}
			if !found || verifyActualSelection(p, []byte(changed.String())) == nil {
				t.Fatal("changed actual external test selection accepted")
			}
		})
	}
}

func mainBoundaryControls(t *testing.T, p Plan, body []byte) {
	t.Helper()
	file := filepath.Join(p.Options.WorkRoot, "go-work/b001/_testmain.go")
	if e := os.MkdirAll(filepath.Dir(file), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(file, body, 0400); e != nil {
		t.Fatal(e)
	}
	q, e := closedPin(file)
	if e != nil {
		t.Fatal(e)
	}
	if e = validateMainInputs(p, TensorPackage+".test", []string{q.Path}, map[string]string{q.Path: q.SHA256}); e != nil {
		t.Fatal(e)
	}
	if validateMainInputs(p, TensorPackage+".test", []string{q.Path}, map[string]string{q.Path: digest([]byte("changed main"))}) == nil {
		t.Fatal("changed generated bytes reached compile boundary")
	}
	if validateMainInputs(p, TensorPackage+".test", nil, nil) == nil {
		t.Fatal("missing generated main reached compile boundary")
	}
	if validateMainInputs(p, TensorPackage+"_test"+testSuffix, []string{"/foreign/main.go"}, map[string]string{}) == nil {
		t.Fatal("foreign external TestMain reached compile boundary")
	}
}

func TestCPU13ExternalTestsBindSourceImportsAndRejectAnotherTestMain(t *testing.T) {
	p := targetFixture(t)
	pkg, ok := sourcePackage(p, TensorPackage)
	if !ok || !slices.Equal(pkg.XTestGo, []string{"adapter_test.go"}) || !slices.Contains(pkg.XTestImports, Module+"/internal/adapter") {
		t.Fatal("external test files or imports omitted")
	}
	if p.Inputs[filepath.Join(pkg.Directory, "adapter_test.go")] == "" || !slices.Contains(p.CPU13.TestCopies, Module+"/internal/adapter") || !strings.Contains(p.CPU13.MainSource, "_xtest.TestExternalAdapter") {
		t.Fatal("external source, tensor dependency copy, or generated test entry omitted")
	}
	path := filepath.Join(pkg.Directory, "adapter_test.go")
	if e := os.WriteFile(path, []byte("package tensor_test\nimport \"testing\"\nfunc TestMain(m *testing.M) {}\n"), 0600); e != nil {
		t.Fatal(e)
	}
	var e error
	p.Packages, p.Inputs, e = selectSources(p.Options.Root, p.Options.GoRoot, CPU13Tests)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = pinned(filepath.Join(p.Options.GoRoot, "src/cmd/go/internal/load/test.go"), p.Inputs); e != nil {
		t.Fatal(e)
	}
	if _, e = cpu13Profile(p); e == nil {
		t.Fatal("source TestMain bypassed the fixed phase overlay")
	}
}
