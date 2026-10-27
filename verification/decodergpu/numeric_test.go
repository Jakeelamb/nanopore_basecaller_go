package decodergpu

import (
	"encoding/json"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestNumericEnvironmentContractsPermitOnlyDeclaredLibraryPath(t *testing.T) {
	load := numericEnvironment(supgpu.RuntimePolicy{Audit: supgpu.Pin{Path: "/audit"}, Directories: []string{"/driver"}})
	numeric := numericEnvironment(supgpu.RuntimePolicy{Audit: supgpu.Pin{Path: "/audit"}, Directories: []string{"/cudart", "/driver"}})
	if err := compareNumericEnvironments(load, numeric); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"extra-cache", "changed-audit", "changed-threads", "duplicate", "missing", "control-prefix"} {
		t.Run(kind, func(t *testing.T) {
			got := slices.Clone(numeric)
			switch kind {
			case "extra-cache":
				got = append(got, "CUDA_CACHE_DISABLE=1")
			case "changed-audit":
				got[5] = "LD_AUDIT=/other"
			case "changed-threads":
				got[3] = "OMP_NUM_THREADS=1"
			case "duplicate":
				got[7] = got[0]
			case "missing":
				got = got[:7]
			case "control-prefix":
				got[0] = "NVIDIA_TF32_OVERRIDE=0"
			}
			if compareNumericEnvironments(load, got) == nil {
				t.Fatal("undeclared difference accepted")
			}
		})
	}
}
func TestNumericSharedProvidersRequireExactCode(t *testing.T) {
	a := supgpu.RuntimePolicy{Allowed: []supgpu.RuntimeCode{{SONAME: "libcuda.so.1", Object: supgpu.Pin{Path: "/cuda", SHA256: "driver"}}, {SONAME: "libcupti.so.13", Object: supgpu.Pin{Path: "/cupti", SHA256: "cupti"}}, {SONAME: "libnvidia-gpucomp.so.615", Object: supgpu.Pin{Path: "/jit", SHA256: "jit"}}}}
	b := a
	b.Allowed = append(slices.Clone(a.Allowed), supgpu.RuntimeCode{SONAME: "libcudart.so.13", Object: supgpu.Pin{Path: "/cudart", SHA256: "runtime"}})
	if err := compareNumericProviders(a, b); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"changed", "missing", "duplicate", "unknown-jit"} {
		t.Run(kind, func(t *testing.T) {
			bad := b
			bad.Allowed = slices.Clone(b.Allowed)
			switch kind {
			case "changed":
				bad.Allowed[0].Object.SHA256 = "other"
			case "missing":
				bad.Allowed = bad.Allowed[1:]
			case "duplicate":
				bad.Allowed = append(bad.Allowed, bad.Allowed[0])
			case "unknown-jit":
				bad.Allowed = append(bad.Allowed, supgpu.RuntimeCode{SONAME: "libnvidia-other.so"})
			}
			if compareNumericProviders(a, bad) == nil {
				t.Fatal("provider change accepted")
			}
		})
	}
}
func numericIdentityFixture(t *testing.T) (NumericIdentity, ImageReport, LineageContext, []string) {
	t.Helper()
	_, modules, functions, launches := imageFixture(t)
	modules[0].DeliveredStage = 1
	env := numericEnvironment(supgpu.RuntimePolicy{Audit: supgpu.Pin{Path: "/audit"}, Directories: []string{"/runtime"}})
	ref := LineageContext{UUID: strings.Repeat("1", 32), Major: 12, Minor: 0, DriverVersion: 13040}
	row := NumericIdentity{Stage: "decode", PID: 100, DeviceOrdinal: 0, UUID: ref.UUID, Major: 12, Minor: 0, DriverAPI: 13040, RuntimeAPI: 13040, KernelVersion: "kernel\n", Environment: slices.Clone(env), Contexts: []NumericContext{{Pointer: 1, UID: 9, DeviceOrdinal: 0}}}
	return row, ImageReport{Modules: modules, Functions: functions, Launches: launches}, ref, env
}
func TestNumericIdentityRequiresContemporaneousCompleteBinding(t *testing.T) {
	row, images, ref, env := numericIdentityFixture(t)
	check := func(r NumericIdentity, i ImageReport) error {
		return validateNumericIdentity(r, 4, i, ref, digest([]byte("kernel\n")), env, 100)
	}
	if err := check(row, images); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"uuid", "uuid-malformed", "sm", "pid", "device", "driver", "runtime", "kernel", "environment-extra", "environment-change", "missing-context", "wrong-context", "wrong-uid", "wrong-context-device", "duplicate-context", "launch-uid", "stage"} {
		t.Run(kind, func(t *testing.T) {
			r := row
			r.Environment = slices.Clone(row.Environment)
			r.Contexts = slices.Clone(row.Contexts)
			i := images
			i.Launches = slices.Clone(images.Launches)
			switch kind {
			case "uuid":
				r.UUID = strings.Repeat("2", 32)
			case "uuid-malformed":
				r.UUID = "bad"
			case "sm":
				r.Major = 11
			case "pid":
				r.PID++
			case "device":
				r.DeviceOrdinal++
			case "driver":
				r.DriverAPI++
			case "runtime":
				r.RuntimeAPI = 0
			case "kernel":
				r.KernelVersion += "changed"
			case "environment-extra":
				r.Environment = append(r.Environment, "CUDA_CACHE_PATH=/new")
			case "environment-change":
				r.Environment[6] = "LD_LIBRARY_PATH=/other"
			case "missing-context":
				r.Contexts = nil
			case "wrong-context":
				r.Contexts[0].Pointer++
			case "wrong-uid":
				r.Contexts[0].UID++
			case "wrong-context-device":
				r.Contexts[0].DeviceOrdinal++
			case "duplicate-context":
				r.Contexts = append(r.Contexts, r.Contexts[0])
			case "launch-uid":
				i.Launches[0].UID++
			case "stage":
				r.Stage = "inputs"
			}
			if check(r, i) == nil {
				t.Fatal("missing/changed contemporaneous evidence accepted")
			}
		})
	}
	for stage, name := range []string{"inputs", "back-guide", "beam-search", "posterior", "decode"} {
		r := row
		r.Stage = name
		if stage == 0 {
			r.Contexts = nil
		}
		if err := validateNumericIdentity(r, stage, images, ref, digest([]byte("kernel\n")), env, 100); err != nil {
			t.Fatal(stage, err)
		}
	}
}
func TestNumericVersionBranchCannotRelaxLegacyCapture(t *testing.T) {
	p := ClosedBuildPlan{Schema: "nanopore-decoder-closed-build/v2"}
	b := CompilerPhase{Output: File{SHA256: decoderBinarySHA}}
	r := RuntimePlan{Schema: "nanopore-decoder-runtime-plan/v3", Build: File{SHA256: decoderBuildSHA}}
	if !numericRuntimeIdentity(r, b, p) || !captureSchemaMatches(CaptureReceipt{Schema: "nanopore-decoder-native-capture/v1"}, p) {
		t.Fatal("legacy exact identity rejected")
	}
	bad := r
	bad.NumericLineage = File{SHA256: acceptedNumericLineageSHA}
	if numericRuntimeIdentity(bad, b, p) {
		t.Fatal("numeric policy smuggled into legacy")
	}
	p.Schema = "nanopore-decoder-closed-build/v3"
	r.Schema = "nanopore-decoder-runtime-plan/v4"
	r.Guard.Path = "/new/guard.json"
	r.KernelVersion.Path = filepath.Join("/new", "driver-version.txt")
	r.NumericLineage.SHA256 = acceptedNumericLineageSHA
	if !numericRuntimeIdentity(r, b, p) || !captureSchemaMatches(CaptureReceipt{Schema: "nanopore-decoder-native-capture/v2"}, p) {
		t.Fatal("new numeric identity rejected")
	}
	if captureSchemaMatches(CaptureReceipt{Schema: "nanopore-decoder-native-capture/v1"}, p) {
		t.Fatal("legacy capture accepted for required contemporaneous observations")
	}
	r.NumericLineage.SHA256 = "other"
	if numericRuntimeIdentity(r, b, p) {
		t.Fatal("unaccepted lineage selected")
	}
}
func TestNumericLineageRejectsUnacceptedProofBeforeReading(t *testing.T) {
	if _, err := readNumericLineage(File{Path: "/not-read", SHA256: "self-authored"}); err == nil || err.Error() != "numeric lineage requires independently accepted sm120a replay" {
		t.Fatal(err)
	}
}
func TestNumericFinalizedBindingDoesNotChangeOriginalPolicy(t *testing.T) {
	original, modules, functions, launches := imageFixture(t)
	delivered := testFile(t, t.TempDir(), "delivered.cubin", "fake delivered image")
	modules[0].Payload = delivered
	if _, err := bindSelectedImages(modules, functions, launches, original); err == nil {
		t.Fatal("old original-image policy accepted different whole image")
	}
	selected := original.Images[0]
	selected.Image = delivered
	selected.Architecture = "sm_120a/reproduced-finalized"
	bindings, err := bindSelectedImages(modules, functions, launches, ImagePolicy{Images: []DeviceImage{selected}})
	if err != nil || len(bindings) != 4 {
		t.Fatal(bindings, err)
	}
	if original.Images[0].Image == delivered {
		t.Fatal("original policy mutated")
	}
}

func TestNumericHistoricalSourceRootUsesAuthenticatedModule(t *testing.T) {
	t.Chdir(filepath.Clean("../.."))
	dir := t.TempDir()
	originalRoot := filepath.Join(dir, "producer")
	module := filepath.Join(originalRoot, "go.mod")
	env := testFile(t, dir, "go-env.json", `{ "GOMOD": "`+module+`" }`)
	g := GoClosure{Commands: []runner.CommandReceipt{{}, {Stdout: env.Path, StdoutSHA: env.SHA256}, {}}, Selection: map[string]string{module: module}, Pins: map[string]string{module: strings.Repeat("1", 64)}}
	names := []string{"oracle.cpp", "sha256.inc"}
	bindings := map[string]string{}
	for _, name := range names {
		bindings[filepath.Join(originalRoot, "verification/decodergpu/native", name)] = "snapshot"
	}
	if err := verifyNativeSourceMembership(bindings, g, names, false); err != nil {
		t.Fatal(err)
	}
	current := map[string]string{}
	for _, name := range names {
		p, e := canonical(filepath.Join("verification/decodergpu/native", name))
		if e != nil {
			t.Fatal(e)
		}
		current[p] = "current snapshot"
	}
	if err := verifyNativeSourceMembership(current, GoClosure{}, names, true); err != nil {
		t.Fatal(err)
	}
	if verifyNativeSourceMembership(bindings, g, names, true) == nil {
		t.Fatal("current mode accepted foreign producer root")
	}
	if verifyNativeSourceMembership(current, g, names, false) == nil {
		t.Fatal("historical mode substituted current root")
	}
	for _, kind := range []string{"missing-env", "missing-GOMOD", "relative-GOMOD", "missing-selection", "missing-content-pin", "relative-selection", "foreign-root", "rehashed-unbound-module", "tampered-env"} {
		t.Run(kind, func(t *testing.T) {
			copy := g
			copy.Commands = slices.Clone(g.Commands)
			copy.Selection = map[string]string{module: module}
			copy.Pins = map[string]string{module: strings.Repeat("1", 64)}
			body := map[string]string{"GOMOD": module}
			switch kind {
			case "missing-env":
				copy.Commands = nil
			case "missing-GOMOD":
				delete(body, "GOMOD")
			case "relative-GOMOD":
				body["GOMOD"] = "go.mod"
			case "missing-selection":
				delete(copy.Selection, module)
			case "missing-content-pin":
				delete(copy.Pins, module)
			case "relative-selection":
				copy.Selection[module] = "relative/go.mod"
			case "foreign-root":
				foreign := filepath.Join(dir, "foreign", "go.mod")
				copy.Selection[module] = foreign
				copy.Pins[foreign] = strings.Repeat("2", 64)
			case "rehashed-unbound-module":
				body["GOMOD"] = filepath.Join(dir, "unbound", "go.mod")
			}
			raw, e := json.Marshal(body)
			if e != nil {
				t.Fatal(e)
			}
			file := testFile(t, dir, kind+".json", string(raw))
			if len(copy.Commands) == 3 {
				copy.Commands[1].Stdout = file.Path
				copy.Commands[1].StdoutSHA = file.SHA256
			}
			if kind == "tampered-env" {
				if e = os.WriteFile(file.Path, []byte("changed"), 0600); e != nil {
					t.Fatal(e)
				}
			}
			if verifyNativeSourceMembership(bindings, copy, names, false) == nil {
				t.Fatal("invalid historical module source accepted")
			}
		})
	}
}

func TestNumericPreparationPinsEveryRequiredNativeSource(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	names, err := nativeSourceNames("nanopore-decoder-closed-build/v3")
	if err != nil || len(names) != 5 {
		t.Fatal(names, err)
	}
	directory := filepath.Join(root, "verification/decodergpu/native")
	if err = os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		if err = os.WriteFile(filepath.Join(directory, name), []byte("prospective source "+name), 0600); err != nil {
			t.Fatal(err)
		}
	}
	pins, err := nativePreparationSources("nanopore-decoder-closed-build/v3")
	if err != nil || len(pins) != 5 {
		t.Fatal(pins, err)
	}
	if err = requireNativePreparationSources(pins, "nanopore-decoder-closed-build/v3"); err != nil {
		t.Fatal(err)
	}
	include := filepath.Join(directory, "numeric_identity.inc")
	delete(pins, include)
	if err = requireNativePreparationSources(pins, "nanopore-decoder-closed-build/v3"); err == nil || !strings.Contains(err.Error(), "numeric_identity.inc") {
		t.Fatal("omitted new required include accepted", err)
	}
}

func TestNumericHistoricalNativeAndLineageBindingsAcrossForeignRoots(t *testing.T) {
	temp := t.TempDir()
	producerRoot := filepath.Join(temp, "old-producer")
	module := filepath.Join(producerRoot, "go.mod")
	environment := testFile(t, temp, "recorded-env.json", `{"GOMOD":"`+module+`"}`)
	g := GoClosure{Commands: []runner.CommandReceipt{{}, {Stdout: environment.Path, StdoutSHA: environment.SHA256}, {}}, Selection: map[string]string{module: module}, Pins: map[string]string{module: strings.Repeat("1", 64)}}
	p := LineagePlan{Directory: filepath.Join(temp, "accepted-lineage"), Go: g, SourceBindings: map[string]string{}, SnapshotPins: map[string]string{}, Inputs: map[string]string{}}
	for i, source := range lineageSourcePaths {
		original := filepath.Join(producerRoot, source)
		sha := strings.Repeat(string(rune('a'+i)), 64)
		copy := filepath.Join(p.Directory, "native-inputs", sha+filepath.Ext(source))
		alias := filepath.Join(p.Directory, "native-inputs", filepath.Base(source))
		p.SourceBindings[original] = copy
		p.SnapshotPins[copy] = sha
		p.SnapshotPins[alias] = sha
		p.Inputs[original] = sha
	}
	nativeNames, err := nativeSourceNames("nanopore-decoder-closed-build/v2")
	if err != nil {
		t.Fatal(err)
	}
	native := map[string]string{}
	for _, name := range nativeNames {
		native[filepath.Join(producerRoot, "verification/decodergpu/native", name)] = "original snapshot"
	}
	for _, name := range []string{"foreign-worktree-a", "foreign-worktree-b"} {
		t.Run(name, func(t *testing.T) {
			directory := filepath.Join(temp, name)
			if err := os.Mkdir(directory, 0700); err != nil {
				t.Fatal(err)
			}
			t.Chdir(directory)
			if err := verifyNativeSourceMembership(native, g, nativeNames, false); err != nil {
				t.Fatal(err)
			}
			if err := verifyLineageSourceSnapshots(p, false); err != nil {
				t.Fatal(err)
			}
			if verifyNativeSourceMembership(native, g, nativeNames, true) == nil {
				t.Fatal("current native gate accepted foreign root")
			}
			if verifyLineageSourceSnapshots(p, true) == nil {
				t.Fatal("current lineage gate accepted foreign root")
			}
		})
	}
	for _, kind := range []string{"source-name", "source-root", "foreign-snapshot", "snapshot-hash", "missing-alias", "alias-hash", "missing-GOMOD", "rehashed-unbound-GOMOD"} {
		t.Run(kind, func(t *testing.T) {
			bad := p
			bad.SourceBindings = maps.Clone(p.SourceBindings)
			bad.SnapshotPins = maps.Clone(p.SnapshotPins)
			bad.Inputs = maps.Clone(p.Inputs)
			bad.Go = p.Go
			bad.Go.Commands = slices.Clone(g.Commands)
			source := filepath.Join(producerRoot, lineageSourcePaths[0])
			copy := bad.SourceBindings[source]
			alias := filepath.Join(p.Directory, "native-inputs", filepath.Base(source))
			switch kind {
			case "source-name":
				delete(bad.SourceBindings, source)
				bad.SourceBindings[source+"-wrong"] = copy
			case "source-root":
				delete(bad.SourceBindings, source)
				bad.SourceBindings[filepath.Join(temp, "foreign", lineageSourcePaths[0])] = copy
			case "foreign-snapshot":
				bad.SourceBindings[source] = filepath.Join(temp, "foreign-snapshot.inc")
			case "snapshot-hash":
				bad.SnapshotPins[copy] = strings.Repeat("f", 64)
			case "missing-alias":
				delete(bad.SnapshotPins, alias)
			case "alias-hash":
				bad.SnapshotPins[alias] = strings.Repeat("f", 64)
			case "missing-GOMOD", "rehashed-unbound-GOMOD":
				body := `{}`
				if kind == "rehashed-unbound-GOMOD" {
					body = `{"GOMOD":"` + filepath.Join(temp, "unbound", "go.mod") + `"}`
				}
				file := testFile(t, temp, kind+"-env.json", body)
				bad.Go.Commands[1].Stdout = file.Path
				bad.Go.Commands[1].StdoutSHA = file.SHA256
			}
			if verifyLineageSourceSnapshots(bad, false) == nil {
				t.Fatal("historical lineage membership accepted", kind)
			}
		})
	}
}

func TestNumericAcceptedLineageProofFromForeignWorktree(t *testing.T) {
	anchor := os.Getenv("D01_NUMERIC_REAL_LINEAGE_PROOF")
	if anchor == "" {
		t.Skip("explicit retained real-proof read-only host opt-in")
	}
	if !filepath.IsAbs(anchor) {
		t.Fatal("absolute independently accepted lineage proof required")
	}
	t.Chdir(filepath.Clean("../.."))
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	proof, err := readNumericLineage(File{Path: anchor, SHA256: acceptedNumericLineageSHA})
	if err != nil {
		t.Fatal(err)
	}
	if cwd == proof.Runtime.WorkingDirectory {
		t.Fatal("real historical reader was not exercised from a foreign worktree")
	}
	if proof.Report.Status != "load_only_observation_qualified" || proof.Report.Binding.Input.SHA256 != "8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40" || proof.Report.Binding.Output.SHA256 != lineageOutputSHA || proof.Context.UUID != "807d7ae76a99c79512c223fc801d38e7" || proof.Context.DriverVersion != 13040 || proof.Report.Binding.Context != proof.Context.Current || proof.Report.Binding.ContextUID != proof.Context.UID {
		t.Fatal("accepted whole-image/input/device/context binding differs")
	}
	out := os.Getenv("D01_NUMERIC_PROOF_READBACK")
	if !filepath.IsAbs(out) {
		t.Fatal("fresh absolute retained proof readback required")
	}
	if err = writeJSON(out, map[string]any{"status": "accepted_saved_lineage_proof_read_from_foreign_worktree", "consumer_working_directory": cwd, "producer_working_directory": proof.Runtime.WorkingDirectory, "accepted_report": File{Path: anchor, SHA256: acceptedNumericLineageSHA}, "binding": proof.Report.Binding, "context": proof.Context, "code": proof.Report.Code, "input_pins": proof.Inputs, "native_producer_executed": false, "historical_capture002_numerical_qualification": false}); err != nil {
		t.Fatal(err)
	}
	if err = checkPins(proof.Inputs); err != nil {
		t.Fatal(err)
	}
	t.Logf("accepted real lineage proof: %d input pins, UUID %s, original %s, delivered %s", len(proof.Inputs), proof.Context.UUID, proof.Report.Binding.Input.SHA256, proof.Report.Binding.Output.SHA256)
}
