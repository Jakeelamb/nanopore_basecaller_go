package h08

import (
	"bufio"
	_ "embed"
	"flag"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const qualifiedArchiveSHA = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"

//go:embed cpu13/testmain.go.txt
var CPU13OverlaySource string

var cpu13Names = []string{"TestNativeTensorOperations", "TestNativeOwnershipAndCopies", "TestNativeMalformedInputs", "TestNativeLoadWeight", "TestNativeConcurrentClose", "TestNativeMissingLibrary", "TestNativeLoaderABIRejections", "TestNativeLoadedRuntimePins", "TestNativeDecoderPrimitives", "TestNativeCloseFailureRetainsLibrary", "TestNativeAttentionPrimitives", "TestNativeAttentionRejections", "TestNativeABIBoundary"}
var fakeNames = []string{"TestNativeLoaderABIRejections", "TestNativeCloseFailureRetainsLibrary"}

func genuineNames() []string {
	names := []string{}
	for _, name := range cpu13Names {
		if !slices.Contains(fakeNames, name) {
			names = append(names, name)
		}
	}
	return names
}

var fixtureNames = []string{"float32.tensor", "int64.tensor", "multiple.tensor", "empty.tensor", "primitive-reference.json", "argsort-reference.json", "attention-reference.json", "real-weight.json"}

type FixtureView struct {
	Schema, Directory                        string
	SourceManifest, Archive, Backend, Weight Pin
	Files                                    map[string]Pin
	Manifest, Runtime                        Pin
}

// FixtureOptions selects retained files and a new view; no producer is run.
type FixtureOptions struct {
	SourceManifest, Archive, Backend, Weight Pin
	Directory                                string
	Runtime                                  map[string]string
}

func ReadFixtureView(pin Pin) (v FixtureView, err error) {
	if err = pinnedJSON(pin, &v); err != nil {
		return
	}
	if v.Schema != "nanopore-h08-cpu-fixture-view/v1" || !filepath.IsAbs(v.Directory) || v.Archive.SHA256 != qualifiedArchiveSHA || len(v.Files) != len(fixtureNames) {
		return v, fmt.Errorf("CPU fixture view/schema/distribution differs")
	}
	var source struct {
		ArchiveSHA string            `json:"archive_sha256"`
		Inputs     map[string]string `json:"input_sha256"`
	}
	if err = pinnedJSON(v.SourceManifest, &source); err != nil {
		return
	}
	if source.ArchiveSHA != qualifiedArchiveSHA {
		return v, fmt.Errorf("fixture producer distribution differs")
	}
	for _, p := range []Pin{v.Archive, v.Backend, v.Weight, v.Manifest, v.Runtime} {
		if _, err = readPin(p); err != nil {
			return
		}
	}
	weights, e := model.WeightManifest(model.HAC)
	if e != nil {
		return v, e
	}
	weightSHA := ""
	for _, w := range weights {
		if w.Name == "0.conv.weight.tensor" {
			weightSHA = w.SHA256
		}
	}
	if filepath.Base(v.Weight.Path) != "0.conv.weight.tensor" || weightSHA != v.Weight.SHA256 {
		return v, fmt.Errorf("CPU real weight is not pinned HAC conv0")
	}
	var manifest struct {
		Schema string            `json:"schema"`
		Inputs map[string]string `json:"input_sha256"`
	}
	if err = pinnedJSON(v.Manifest, &manifest); err != nil {
		return
	}
	if manifest.Schema != "nanopore-h08-fixture-only-manifest/v1" || v.Manifest.Path != filepath.Join(v.Directory, "manifest.json") {
		return v, fmt.Errorf("fixture-only manifest identity differs")
	}
	expected := map[string]string{v.Weight.Path: v.Weight.SHA256}
	for _, name := range fixtureNames {
		p, ok := v.Files[name]
		original := filepath.Join(filepath.Dir(v.SourceManifest.Path), name)
		if !ok || p.Path != filepath.Join(v.Directory, name) || p.SHA256 != source.Inputs[original] {
			return v, fmt.Errorf("fixture differs from retained producer: %s", name)
		}
		if _, err = readPin(p); err != nil {
			return
		}
		expected[p.Path] = p.SHA256
	}
	if !maps.Equal(expected, manifest.Inputs) || source.Inputs[v.Weight.Path] != v.Weight.SHA256 {
		return v, fmt.Errorf("fixture-only manifest/real-weight witness differs")
	}
	link := filepath.Join(v.Directory, filepath.Base(v.Backend.Path))
	target, e := filepath.EvalSymlinks(link)
	if e != nil || target != v.Backend.Path {
		return v, fmt.Errorf("fixture backend symlink differs")
	}
	return
}

func PrepareFixtureView(o FixtureOptions) (v FixtureView, err error) {
	var source struct {
		ArchiveSHA string            `json:"archive_sha256"`
		Inputs     map[string]string `json:"input_sha256"`
	}
	if err = pinnedJSON(o.SourceManifest, &source); err != nil {
		return
	}
	if source.ArchiveSHA != qualifiedArchiveSHA || o.Archive.SHA256 != qualifiedArchiveSHA || !filepath.IsAbs(o.Directory) {
		return v, fmt.Errorf("unqualified fixture distribution/view")
	}
	for _, p := range []Pin{o.Archive, o.Backend, o.Weight} {
		if _, err = readPin(p); err != nil {
			return
		}
	}
	if err = checkPins(o.Runtime); err != nil {
		return
	}
	if err = os.Mkdir(o.Directory, 0755); err != nil {
		return
	}
	v = FixtureView{Schema: "nanopore-h08-cpu-fixture-view/v1", Directory: o.Directory, SourceManifest: o.SourceManifest, Archive: o.Archive, Backend: o.Backend, Weight: o.Weight, Files: map[string]Pin{}}
	inputs := map[string]string{o.Weight.Path: o.Weight.SHA256}
	for _, name := range fixtureNames {
		p := Pin{Path: filepath.Join(filepath.Dir(o.SourceManifest.Path), name), SHA256: source.Inputs[filepath.Join(filepath.Dir(o.SourceManifest.Path), name)]}
		var b []byte
		if b, err = readPin(p); err != nil {
			return
		}
		path := filepath.Join(o.Directory, name)
		if err = os.WriteFile(path, b, 0400); err != nil {
			return
		}
		v.Files[name] = Pin{Path: path, SHA256: digest(b)}
		inputs[path] = digest(b)
	}
	if err = os.Symlink(o.Backend.Path, filepath.Join(o.Directory, filepath.Base(o.Backend.Path))); err != nil {
		return
	}
	path := filepath.Join(o.Directory, "manifest.json")
	if err = writeJSONFile(path, struct {
		Schema string            `json:"schema"`
		Inputs map[string]string `json:"input_sha256"`
	}{"nanopore-h08-fixture-only-manifest/v1", inputs}); err != nil {
		return
	}
	if v.Manifest, err = closedbuild.PinFile(path); err != nil {
		return
	}
	path = filepath.Join(o.Directory, "runtime.json")
	if err = writeJSONFile(path, map[string]any{"runtime_sha256": o.Runtime, "output": o.Backend.Path, "output_sha256": o.Backend.SHA256}); err != nil {
		return
	}
	if v.Runtime, err = closedbuild.PinFile(path); err != nil {
		return
	}
	path = filepath.Join(o.Directory, "view.json")
	if err = writeJSONFile(path, v); err != nil {
		return
	}
	p, e := closedbuild.PinFile(path)
	if e != nil {
		return v, e
	}
	return ReadFixtureView(p)
}

type CPU13Request struct {
	Schema                               string
	Fixtures, ControllerBuild, FakeProof Pin
	ControllerBuildSelection             Pin
	Backend                              closedbuild.ArtifactOptions
	Out                                  string
}
type CPU13Receipt struct {
	Schema, Status string
	Admission      Pin
	Run            supgpu.GuardedRun
	Tests          map[string]string
}

func cpu13Command(target, request Pin) []string {
	return []string{target.Path, "-h08-request", request.Path, "-h08-request-sha", request.SHA256, "-test.v", "-test.count=1", "-test.run", "^(" + strings.Join(genuineNames(), "|") + ")$"}
}

func readCPU13Admission(pin Pin) (a RuntimeAdmission, q CPU13Request, err error) {
	if err = pinnedJSON(pin, &a); err != nil {
		return
	}
	if a.Schema != "nanopore-h08-runtime-admission/v2" {
		return a, q, fmt.Errorf("H08 runtime admission schema differs")
	}
	if err = pinnedJSON(a.Request, &q); err != nil {
		return
	}
	if q.Schema != "nanopore-h08-cpu13-request/v1" {
		return a, q, fmt.Errorf("CPU13 request schema differs")
	}
	if _, e := cgobuild.ReadH08Artifact(a.ParentBuild, a.ParentBuildSelection); e != nil {
		return a, q, e
	}
	b, e := cgobuild.ReadCPU13Artifact(q.ControllerBuild, q.ControllerBuildSelection)
	if e != nil {
		return a, q, e
	}
	v, e := ReadFixtureView(q.Fixtures)
	if e != nil {
		return a, q, e
	}
	if v.Backend != q.Backend.Output {
		return a, q, fmt.Errorf("CPU fixtures selected another backend")
	}
	backend, e := AuthenticateBackend(q.Backend, v.Backend.Path)
	if e != nil {
		return a, q, e
	}
	if e = validateBackendPolicy(a, backend); e != nil {
		return a, q, e
	}
	if _, e = readFakeProof(q.FakeProof, b.Output(), q.ControllerBuild, q.ControllerBuildSelection); e != nil {
		return a, q, e
	}
	if !slices.Equal(a.Command.Args, cpu13Command(b.Output(), a.Request)) || !filepath.IsAbs(q.Out) || a.Command.Output == q.Out {
		return a, q, fmt.Errorf("CPU13 exact test target/argv/output differs")
	}
	err = validateGuardInputs(a, b.Output(), a.Request, []Pin{a.ParentBuild, a.ParentBuildSelection, q.ControllerBuild, q.ControllerBuildSelection, q.FakeProof, q.Fixtures, q.Backend.Build, q.Backend.Plan, q.Backend.Producer, q.Backend.Output}, "hac-cpu13")
	return
}

func testStatuses(raw []byte, names []string) (map[string]string, error) {
	status := map[string]string{}
	runs := map[string]int{}
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "=== RUN   ") {
			name := strings.TrimPrefix(line, "=== RUN   ")
			if !strings.Contains(name, "/") {
				runs[name]++
			}
		}
		if strings.HasPrefix(strings.TrimSpace(line), "--- SKIP:") || strings.HasPrefix(strings.TrimSpace(line), "--- FAIL:") {
			return nil, fmt.Errorf("CPU13 skip/failure recorded")
		}
		if strings.HasPrefix(line, "--- PASS: ") {
			fields := strings.Fields(line)
			if len(fields) < 3 || status[fields[2]] != "" {
				return nil, fmt.Errorf("CPU13 duplicate/malformed result")
			}
			status[fields[2]] = "pass"
		}
	}
	if e := scanner.Err(); e != nil {
		return nil, e
	}
	if len(status) != len(names) || len(runs) != len(names) {
		return nil, fmt.Errorf("CPU13 exact test set incomplete")
	}
	for _, name := range names {
		if status[name] != "pass" || runs[name] != 1 {
			return nil, fmt.Errorf("CPU13 required test missing/repeated: %s", name)
		}
	}
	return status, nil
}

func ReadCPU13(pin Pin, backend closedbuild.ArtifactOptions) (r CPU13Receipt, err error) {
	if err = pinnedJSON(pin, &r); err != nil {
		return
	}
	if r.Schema != "nanopore-h08-cpu13/v1" || r.Status != "cpu13_weight_passed" {
		return r, fmt.Errorf("genuine CPU13 qualification absent")
	}
	a, q, e := readCPU13Admission(r.Admission)
	if e != nil {
		return r, e
	}
	if q.Backend != backend {
		return r, fmt.Errorf("CPU13 tested another selected backend")
	}
	if _, e = supgpu.ReplayGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command, r.Run); e != nil {
		return r, e
	}
	var b []byte
	if b, err = readPin(Pin{Path: r.Run.Command.Stdout, SHA256: r.Run.Command.StdoutSHA}); err != nil {
		return
	}
	actual, e := testStatuses(b, genuineNames())
	if e != nil {
		return r, e
	}
	target, e := closedbuild.PinFile(a.Command.Args[0])
	if e != nil {
		return r, e
	}
	fake, fe := readFakeProof(q.FakeProof, target, q.ControllerBuild, q.ControllerBuildSelection)
	if fe != nil {
		return r, fe
	}
	maps.Copy(actual, fake.Tests)
	if e != nil || !reflect.DeepEqual(actual, r.Tests) {
		return r, fmt.Errorf("CPU13 raw test results differ: %v", e)
	}
	return
}

// CPU13TestMain is called only by the explicitly source-bound external test
// overlay. Its fixture selection precedes m.Run; no tests are copied here.
func CPU13TestMain(run func() int) int {
	profile := flag.String("h08-profile", "genuine", "genuine or fake maintained tensor tests")
	request := flag.String("h08-request", "", "independently pinned CPU13 request")
	sha := flag.String("h08-request-sha", "", "independent request SHA256")
	flag.Parse()
	if *profile == "fake" {
		if flag.Lookup("test.run").Value.String() != "^("+strings.Join(fakeNames, "|")+")$" {
			fmt.Fprintln(os.Stderr, "fake profile requires exact two-test selector")
			return 1
		}
		return runFakeTests(run)
	}
	if *profile != "genuine" || flag.Lookup("test.run").Value.String() != "^("+strings.Join(genuineNames(), "|")+")$" {
		fmt.Fprintln(os.Stderr, "genuine CPU13 profile requires exact eleven-test selector")
		return 1
	}
	e := func() error {
		phases, e := supgpu.OpenPhaseClient()
		if e != nil {
			return e
		}
		defer phases.Close()
		var q CPU13Request
		if e = pinnedJSON(Pin{Path: *request, SHA256: *sha}, &q); e != nil {
			return e
		}
		if q.Schema != "nanopore-h08-cpu13-request/v1" {
			return fmt.Errorf("CPU13 request schema differs")
		}
		b, e := cgobuild.ReadCPU13Artifact(q.ControllerBuild, q.ControllerBuildSelection)
		if e != nil {
			return e
		}
		v, e := ReadFixtureView(q.Fixtures)
		if e != nil {
			return e
		}
		executable, e := os.Executable()
		if e != nil {
			return e
		}
		actual, e := closedbuild.PinFile(executable)
		if e != nil || actual != b.Output() || v.Backend != q.Backend.Output {
			return fmt.Errorf("CPU13 actual target/backend differs")
		}
		if _, e = AuthenticateBackend(q.Backend, v.Backend.Path); e != nil {
			return e
		}
		for key, value := range map[string]string{"NANOPORE_TENSOR_BACKEND": v.Backend.Path, "NANOPORE_TENSOR_FIXTURES": v.Directory, "NANOPORE_TENSOR_WEIGHT": v.Weight.Path, "NANOPORE_TENSOR_RUNTIME_MANIFEST": v.Runtime.Path} {
			if e = os.Setenv(key, value); e != nil {
				return e
			}
		}
		if e = phases.Checkpoint("pre-model"); e != nil {
			return e
		}
		if e = phases.Checkpoint("pre-forward"); e != nil {
			return e
		}
		code := run()
		if e = phases.Checkpoint("post-forward"); e != nil {
			return e
		}
		if code != 0 {
			return fmt.Errorf("maintained CPU13 tests exited %d", code)
		}
		return nil
	}()
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		return 1
	}
	return 0
}

func RunCPU13(admission Pin) (r CPU13Receipt, err error) {
	a, _, err := readCPU13Admission(admission)
	if err != nil {
		return r, err
	}
	if err = actualController(a.ParentBuild, a.ParentBuildSelection); err != nil {
		return r, err
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = CPU13Receipt{Schema: "nanopore-h08-cpu13/v1", Status: "failed", Admission: admission}
	defer func() {
		if err != nil {
			r.Status = "failed"
		}
		if info, e := os.Stat(a.Command.Output); e == nil && info.IsDir() {
			if e = writeJSONFile(filepath.Join(a.Command.Output, "cpu13.json"), r); err == nil {
				err = e
			}
		}
	}()
	r.Run, err = supgpu.ExecuteGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command)
	if err != nil {
		return r, err
	}
	if _, err = supgpu.ReplayGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command, r.Run); err != nil {
		return r, err
	}
	b, err := readPin(Pin{Path: r.Run.Command.Stdout, SHA256: r.Run.Command.StdoutSHA})
	if err != nil {
		return r, err
	}
	if r.Tests, err = testStatuses(b, genuineNames()); err != nil {
		return r, err
	}
	var q CPU13Request
	if err = pinnedJSON(a.Request, &q); err != nil {
		return r, err
	}
	target, e := closedbuild.PinFile(a.Command.Args[0])
	if e != nil {
		return r, e
	}
	fake, e := readFakeProof(q.FakeProof, target, q.ControllerBuild, q.ControllerBuildSelection)
	if e != nil {
		return r, e
	}
	maps.Copy(r.Tests, fake.Tests)
	r.Status = "cpu13_weight_passed"
	err = writeJSONFile(filepath.Join(a.Command.Output, "cpu13.json"), r)
	return
}
