package gpuprofile

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func target(t *testing.T) Target {
	t.Helper()
	f, err := os.Open("../../provenance/gpu-target.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := DecodeTarget(f)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func observed(t Target, f16, e4m3 int) Observation {
	return Observation{RuntimeFamily: t.RuntimeFamily, DeviceUUID: t.DeviceUUID, CUDAVisibleDevices: t.CUDAVisibleDevices, ComputeMajor: 12, SMCount: 36, F16Status: &f16, E4M3Status: &e4m3}
}

func request(model string) Request {
	hash, batch := HACConfig, 64
	if model == SUP {
		hash, batch = SUPConfig, 32
	}
	return Request{Model: model, ConfigSHA256: hash, Batch: batch, Output: "fastq", Kit: "SQK-LSK114", DevOptions: DefaultDevOptions()}
}

func TestUnknownCapabilitiesFailClosed(t *testing.T) {
	p := target(t)
	o := observed(p, 0, 0)
	o.F16Status = nil
	if _, err := Bind(p, o); err == nil {
		t.Fatal("unobserved F16 accepted")
	}
	o = observed(p, 0, 0)
	o.E4M3Status = nil
	if _, err := Bind(p, o); err == nil {
		t.Fatal("unobserved E4M3 accepted")
	}
	if _, err := Prepare(BackendIdentity{}, request(HAC)); err == nil {
		t.Fatal("zero identity accepted")
	}
}

func TestFrozenIdentityNegativeControls(t *testing.T) {
	p := target(t)
	for _, mutate := range []func(*Target){
		func(p *Target) { p.RuntimeFamily = "official-cuda12" },
		func(p *Target) { p.CUDAVisibleDevices.Present = true },
		func(p *Target) { p.Models[0].ConfigSHA256 = strings.Repeat("0", 64) },
		func(p *Target) { p.DevOptions.KoiUseF8 = false },
		func(p *Target) { p.DevOptions.KoiUseI8 = false },
		func(p *Target) { p.DevOptions.RemoveBits = 0 },
	} {
		changed := p
		mutate(&changed)
		if changed.Validate() == nil {
			t.Fatal("mutated frozen target accepted")
		}
	}
	for _, mutate := range []func(*Observation){
		func(o *Observation) { o.RuntimeFamily = "official-cuda12" },
		func(o *Observation) { o.CUDAVisibleDevices.Present = true },
		func(o *Observation) { o.DeviceUUID = "different" },
	} {
		o := observed(p, 0, 0)
		mutate(&o)
		if _, err := Bind(p, o); err == nil {
			t.Fatal("identity mismatch accepted")
		}
	}
}

func TestCapabilityChangesAlterDispatch(t *testing.T) {
	p := target(t)
	b, err := Bind(p, observed(p, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	d, err := Prepare(b, request(SUP))
	if err != nil {
		t.Fatal(err)
	}
	if !d.Selection().FP8 || !d.Selection().Int8 || !d.Selection().VariableChunks {
		t.Fatal("defaults not selected")
	}
	changed, err := Bind(p, observed(p, 0, 1))
	if err != nil {
		t.Fatal(err)
	}
	other, err := Prepare(changed, request(SUP))
	if err != nil {
		t.Fatal(err)
	}
	if b == changed || other.Selection().FP8 {
		t.Fatal("capability change did not change identity and precision")
	}
	generic, err := Bind(p, observed(p, 1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Prepare(generic, request(SUP)); err == nil {
		t.Fatal("generic SUP accepted without cards")
	}
	for _, status := range []int{2, 3, -1, 4} {
		if _, err := Bind(p, observed(p, 0, status)); err == nil {
			t.Fatal("query error accepted")
		}
	}
}

func TestTrimSplitFixedVCSMatrix(t *testing.T) {
	p := target(t)
	b, err := Bind(p, observed(p, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []string{HAC, SUP} {
		for _, fixed := range []bool{false, true} {
			for _, trim := range []bool{false, true} {
				for _, split := range []bool{false, true} {
					r := request(model)
					r.FixedChunks, r.DisableTrim, r.DisableSplit = fixed, trim, split
					d, err := Prepare(b, r)
					if err != nil {
						t.Fatal(err)
					}
					if d.Request() != r || d.Selection().VariableChunks == fixed {
						t.Fatal("request policy lost")
					}
					copy := d.Request()
					copy.DevOptions.KoiUseF8 = false
					if d.Request() != r {
						t.Fatal("dispatch mutated through returned value")
					}
				}
			}
		}
	}
}

func TestRejectUnsupportedRequests(t *testing.T) {
	p := target(t)
	for _, mutate := range []func(*Request){
		func(r *Request) { r.Model = "rna" },
		func(r *Request) { r.ConfigSHA256 = "wrong" },
		func(r *Request) { r.Batch = 0 },
		func(r *Request) { r.Batch = 1 },
		func(r *Request) { r.Output = "aligned-bam" },
		func(r *Request) { r.Kit = "other" },
		func(r *Request) { r.DevOptions.KoiUseF8 = false },
	} {
		r := request(SUP)
		mutate(&r)
		if CheckRequest(p, r) == nil {
			t.Fatal("unsupported request accepted")
		}
	}
}

func TestSourcePredicates(t *testing.T) {
	if CanRunFLSTM(7, 100) || CanRunFLSTM(8, 8) || !CanRunFLSTM(8, 9) {
		t.Fatal("FLSTM predicate differs")
	}
	for _, c := range []int{128, 129, 1152} {
		if VariableChunksSupported(Shape{Kind: "lstm", C: c}, 8, 9, true, false, 1) {
			t.Fatal("invalid LSTM VCS accepted")
		}
	}
	s := Shape{Kind: "flstm", C: 1024, K: 128}
	if !VariableChunksSupported(s, 8, 9, true, false, 1) {
		t.Fatal("FLSTM VCS rejected")
	}
	if VariableChunksSupported(s, 8, 9, false, false, 1) || VariableChunksSupported(s, 8, 9, true, true, 1) || VariableChunksSupported(s, 8, 9, true, false, 0) {
		t.Fatal("unsupported VCS platform accepted")
	}
}

func TestCaptureRejectsAdmissionAndQualificationBeforeLock(t *testing.T) {
	p := target(t)
	out := t.TempDir() + "/must-not-exist"
	for _, admitted := range []bool{false, true} {
		m, err := CaptureProfile(ProbeOptions{Target: p, Admitted: admitted, Basic: "missing-basic", Build: "missing-build", Out: out})
		if err == nil || m.Schema != "" || m.Status != "" || m.Observation != nil || m.Commands != nil {
			t.Fatal("early rejection allocated receipt or admitted")
		}
		if !admitted && !strings.Contains(err.Error(), "admission required") {
			t.Fatal(err)
		}
		if admitted && !strings.Contains(err.Error(), "missing-basic") {
			t.Fatal("qualification must reject before lock", err)
		}
		if _, e := os.Stat(out); !os.IsNotExist(e) {
			t.Fatal("early rejection allocated attempt")
		}
	}
}

func successfulRaw() RawProbe {
	return RawProbe{DeviceUUID: DeviceUUID, DeviceCount: 1, Major: 12, SMs: 36, Driver: 13040, Runtime: 13040, F16: 0, E4M3: 0, Predicates: NativePredicates{HACFused: true, SUPTiled: true, FP8: true, Int8: true, HACVCS: true, SUPVCS: true, DefaultVCS: true, RemoveBits: 4}}
}

func TestClosedProbeRejectsMissingNullUnknownDuplicateAndTrailing(t *testing.T) {
	p := successfulRaw()
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var decoded RawProbe
	if err = decodeClosed(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		strings.Replace(string(raw), `"f16_status":0,`, "", 1),
		strings.Replace(string(raw), `"f16_status":0`, `"f16_status":null`, 1),
		strings.Replace(string(raw), `"f16_status":0`, `"f16_status":0,"unknown":false`, 1),
		strings.Replace(string(raw), `"f16_status":0`, `"f16_status":0,"f16_status":1`, 1),
		strings.Replace(string(raw), `"f16_status":0`, `"f16_status":0,"F16_STATUS":0`, 1),
		string(raw) + "{}",
	} {
		if decodeClosed([]byte(bad), &decoded) == nil {
			t.Fatal("invalid closed payload accepted", bad)
		}
	}
}

func TestNativePredicatesMustAgree(t *testing.T) {
	p := target(t)
	raw := successfulRaw()
	o, matrix, err := ValidateProbe(p, raw)
	if err != nil || o.F16Status == nil || len(matrix) != 16 {
		t.Fatal("valid observation rejected", err)
	}
	for _, mutate := range []func(*RawProbe){
		func(p *RawProbe) { p.Predicates.HACFused = false },
		func(p *RawProbe) { p.Predicates.DefaultVCS = false },
		func(p *RawProbe) { p.Predicates.FP8 = false },
		func(p *RawProbe) { p.Predicates.RemoveBits = 0 },
		func(p *RawProbe) { p.F16 = 2 },
		func(p *RawProbe) { p.Runtime = 12080 },
		func(p *RawProbe) { p.CVDPresent = true },
	} {
		changed := raw
		mutate(&changed)
		if _, _, err = ValidateProbe(p, changed); err == nil {
			t.Fatal("native disagreement accepted")
		}
	}
	raw.E4M3 = 1
	raw.Predicates.FP8 = false
	if _, _, err = ValidateProbe(p, raw); err != nil {
		t.Fatal("observed unsupported FP8 rejected", err)
	}
}

func TestFrozenEvidenceRejectsValidCapabilityMutant(t *testing.T) {
	p := target(t)
	raw := successfulRaw()
	o, matrix, err := ValidateProbe(p, raw)
	if err != nil {
		t.Fatal(err)
	}
	frozen := ProbeReceipt{Status: "predicate_observation_passed", Target: p, Raw: &raw, Observation: &o, Matrix: matrix}
	changed := raw
	changed.E4M3 = 1
	changed.Predicates.FP8 = false
	other, matrix2, err := ValidateProbe(p, changed)
	if err != nil {
		t.Fatal("mutant must be independently valid", err)
	}
	candidate := frozen
	candidate.Raw = &changed
	candidate.Observation = &other
	candidate.Matrix = matrix2
	if VerifyFrozen(frozen, candidate) == nil {
		t.Fatal("valid capability mutant bypassed frozen evidence")
	}
	if err = VerifyFrozen(frozen, frozen); err != nil {
		t.Fatal(err)
	}
}

func TestPinnedPredicateExtraction(t *testing.T) {
	header, err := PredicateHeader("../../external/dorado-source")
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{"prop->multiProcessorCount >= 9", "koi_tc_is_available(KOI_F16)", "koi_tc_is_available(KOI_E4M3)", "check_variable_chunk_sizes_supported", "source_default_vcs", "remove_bits"} {
		if !strings.Contains(string(header), fragment) {
			t.Fatal("missing source predicate", fragment)
		}
	}
	root := t.TempDir()
	if _, err = PredicateHeader(root); err == nil {
		t.Fatal("missing pinned source accepted")
	}
}

func TestImportRejectsChangedManifestAndMissingEvidence(t *testing.T) {
	path := t.TempDir() + "/profile.json"
	if err := os.WriteFile(path, []byte(`{}`), 0644); err != nil {
		t.Fatal(err)
	}
	sha, err := runner.SHA(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = ImportProfile(path, sha); err == nil {
		t.Fatal("missing receipt fields accepted")
	}
	if err = os.WriteFile(path, []byte(`{"status":"predicate_observation_passed"}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = ImportProfile(path, sha); err == nil || !strings.Contains(err.Error(), "pin mismatch") {
		t.Fatal("changed frozen manifest accepted", err)
	}
}

func TestCorruptQualificationDoesNotAllocateReceipt(t *testing.T) {
	basic := t.TempDir() + "/basic.json"
	if err := os.WriteFile(basic, []byte(`{"abi_version":4,"status":"source_differential_passed"}`), 0644); err != nil {
		t.Fatal(err)
	}
	m, err := CaptureProfile(ProbeOptions{Target: target(t), Admitted: true, Basic: basic, Build: "not-reached"})
	if err == nil || !strings.Contains(err.Error(), "pin mismatch") || m.Status != "" || m.Raw != nil || m.Observation != nil {
		t.Fatal("corrupt qualification reached capture", err)
	}
}

// This fixture reaches importer metadata guards without claiming ABI4/native evidence.
func retainedCompileMetadata(t *testing.T) ProbeReceipt {
	t.Helper()
	root := t.TempDir()
	c := ProbeCompile{Compiler: filepath.Join(root, "compiler"), KoiRoot: filepath.Join(root, "koi"), CUDARoot: filepath.Join(root, "cuda"), Directory: root}
	paths := []string{"verification/gpuprofile/native/probe.cpp", c.header(), c.koiHeader(), filepath.Join(c.CUDARoot, "include/cuda_runtime_api.h"), filepath.Join(root, "extra-header.h")}
	if err := os.WriteFile(c.depfile(), []byte(c.binary()+": "+strings.Join(paths, " ")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	p := target(t)
	raw := successfulRaw()
	o, matrix, err := ValidateProbe(p, raw)
	if err != nil {
		t.Fatal(err)
	}
	m := ProbeReceipt{Schema: ProfileSchema, Status: "predicate_observation_passed", Started: "2026-10-23T17:00:00Z", Finished: "2026-10-23T17:00:01Z", Invocation: []string{"retained-metadata-test"}, Target: p, Compile: c, Environment: map[string]Environment{}, Inputs: map[string]string{}, Runtime: map[string]string{"synthetic-runtime": strings.Repeat("a", 64)}, Raw: &raw, Observation: &o, Matrix: matrix}
	for _, args := range [][]string{{c.Compiler, "--version"}, c.argv(), {c.binary(), filepath.Join(root, "probe.maps")}} {
		m.Commands = append(m.Commands, runner.CommandReceipt{Args: args, Environment: []string{}, Started: m.Started, Finished: m.Finished})
	}
	mandatory := append([]string{}, consumerSources...)
	mandatory = append(mandatory, paths...)
	mandatory = append(mandatory, c.Compiler, c.binary(), c.depfile(), c.koiLibrary(), filepath.Join(c.CUDARoot, "include/cuda.h"), filepath.Join(root, "probe.maps"))
	for _, path := range mandatory {
		m.Inputs[path] = strings.Repeat("a", 64)
	}
	m.Inputs[c.koiHeader()] = KoiHeaderSHA
	m.Inputs[c.koiLibrary()] = KoiLibrarySHA
	m.Inputs[c.depfile()], err = runner.SHA(c.depfile())
	if err != nil {
		t.Fatal(err)
	}
	if err = validateCompileClosure(m); err != nil {
		t.Fatal("baseline must traverse the complete compile guard", err)
	}
	requireMetadataImportRejection(t, m, "missing required input pin "+BasicSHA)
	return m
}

func cloneMetadata(t *testing.T, m ProbeReceipt) ProbeReceipt {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var out ProbeReceipt
	if err = json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func requireMetadataImportRejection(t *testing.T, m ProbeReceipt, guard string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mutated-profile.json")
	if err := jsonFile(path, m); err != nil {
		t.Fatal(err)
	}
	// The caller deliberately designates the mutated metadata hash for this test.
	// Rejection must come from receipt semantics, not from a stale frozen hash.
	hash, err := runner.SHA(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = ImportProfile(path, hash); err == nil || !strings.Contains(err.Error(), guard) {
		t.Fatalf("wanted targeted guard %q, got %v", guard, err)
	}
}

func TestRetainedMetadataRequiresConsumerProbeAndDependencyPins(t *testing.T) {
	m := retainedCompileMetadata(t)
	paths := append([]string{}, consumerSources...)
	paths = append(paths, m.Compile.header(), m.Compile.depfile(), m.Compile.binary())
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			changed := cloneMetadata(t, m)
			delete(changed.Inputs, path)
			requireMetadataImportRejection(t, changed, "missing mandatory consumer/probe pin "+path)
		})
	}
	changed := cloneMetadata(t, m)
	delete(changed.Inputs, filepath.Join(m.Compile.Directory, "extra-header.h"))
	requireMetadataImportRejection(t, changed, "missing compiler dependency pin")
}

func TestRetainedMetadataBindsExactCompileArgv(t *testing.T) {
	m := retainedCompileMetadata(t)
	mutants := [][]string{{m.Compile.Compiler, "--version"}}
	for _, token := range []string{"-std=c++20", "-DDORADO_CUDA_BUILD=1", "-I" + m.Compile.Directory, "-MD", m.Compile.depfile(), "verification/gpuprofile/native/probe.cpp", m.Compile.koiLibrary(), "-L" + filepath.Join(m.Compile.CUDARoot, "lib64"), "-lcudart", m.Compile.binary()} {
		args := append([]string{}, m.Commands[1].Args...)
		found := false
		for i, arg := range args {
			if arg == token {
				args[i] = "changed-input-or-option"
				found = true
				break
			}
		}
		if !found {
			t.Fatal("mutant did not target compile argument", token)
		}
		mutants = append(mutants, args)
	}
	for _, args := range mutants {
		changed := cloneMetadata(t, m)
		changed.Commands[1].Args = args
		requireMetadataImportRejection(t, changed, "compile argv does not bind pinned probe")
	}
}

func TestRetainedMetadataReparsesDepfileTargetAndRequiredHeaders(t *testing.T) {
	for _, mode := range []string{"wrong-target", "omit-generated-header"} {
		t.Run(mode, func(t *testing.T) {
			m := retainedCompileMetadata(t)
			raw, err := os.ReadFile(m.Compile.depfile())
			if err != nil {
				t.Fatal(err)
			}
			text := string(raw)
			guard := "invalid probe dependency target or syntax"
			if mode == "wrong-target" {
				text = strings.Replace(text, m.Compile.binary()+":", m.Compile.binary()+"-other:", 1)
			} else {
				text = strings.Replace(text, m.Compile.header()+" ", "", 1)
				guard = "depfile omits required probe dependency"
			}
			if err = os.WriteFile(m.Compile.depfile(), []byte(text), 0644); err != nil {
				t.Fatal(err)
			}
			m.Inputs[m.Compile.depfile()], err = runner.SHA(m.Compile.depfile())
			if err != nil {
				t.Fatal(err)
			}
			requireMetadataImportRejection(t, m, guard)
		})
	}
}
