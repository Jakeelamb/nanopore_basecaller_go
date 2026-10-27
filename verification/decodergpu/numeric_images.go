package decodergpu

import (
	"debug/elf"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

const acceptedNumericLineageSHA = "5b83a359ee98c6c052f672b56c09942516bb11af02685554d58817964f982bc5"

type numericLineage struct {
	Report  LineageReplayReport
	Runtime LineageRuntime
	Context LineageContext
	Inputs  map[string]string
}

func numericBuild(p ClosedBuildPlan) bool { return p.Schema == "nanopore-decoder-closed-build/v3" }
func numericRuntimeIdentity(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan) bool {
	if numericBuild(p) {
		return r.Schema == "nanopore-decoder-runtime-plan/v4" && r.NumericLineage.SHA256 == acceptedNumericLineageSHA && r.KernelVersion.Path == filepath.Join(filepath.Dir(r.Guard.Path), "driver-version.txt")
	}
	return r.Schema == "nanopore-decoder-runtime-plan/v3" && r.Build.SHA256 == decoderBuildSHA && b.Output.SHA256 == decoderBinarySHA && p.Schema == "nanopore-decoder-closed-build/v2" && r.NumericLineage == (File{}) && r.KernelVersion == (File{})
}
func captureSchemaMatches(c CaptureReceipt, p ClosedBuildPlan) bool {
	if numericBuild(p) {
		return c.Schema == "nanopore-decoder-native-capture/v2"
	}
	return c.Schema == "nanopore-decoder-native-capture/v1"
}

func readNumericLineage(anchor File) (proof numericLineage, err error) {
	if anchor.SHA256 != acceptedNumericLineageSHA {
		return proof, fmt.Errorf("numeric lineage requires independently accepted sm120a replay")
	}
	if err = readJSON(anchor, &proof.Report); err != nil {
		return proof, err
	}
	report := proof.Report
	if report.Schema != "nanopore-decoder-load-replay/v1" || report.Status != "load_only_observation_qualified" || report.Error != "" || report.Scope != "contemporaneous-load-only" || report.SourceStatus != "failed" || report.SourceError != lineageEnvironmentError || report.NativeExit != 0 || report.SourceControllerExit != 1 {
		return proof, fmt.Errorf("numeric lineage accepted report status differs")
	}
	r, b, _, consumer, err := readLineageRuntime(report.Runtime, false)
	if err != nil {
		return proof, err
	}
	if r.CaseIndex != 1 || r.Build != report.NativeBuild || b.Output != report.NativeExecutable {
		return proof, fmt.Errorf("numeric lineage producer binding differs")
	}
	var saved LineageLoadReceipt
	if err = readJSON(report.SourceLoad, &saved); err != nil {
		return proof, err
	}
	if err = validateLineageSavedLoad(saved, report.Runtime, lineageEnvironmentReaderFailed); err != nil {
		return proof, err
	}
	transcript, code, err := lineageSavedGuard(r, b, saved)
	if err != nil {
		return proof, err
	}
	if err = checkLineageSavedSupervisor(report.Runtime, r, b, consumer, saved, transcript.PID, report.SupervisorCommand, report.SupervisorWatch, lineageEnvironmentReaderFailed); err != nil {
		return proof, err
	}
	observation, binding, err := ReadLineageObservation(report.Observation, r.Contract)
	if err != nil {
		return proof, err
	}
	if saved.Observation != report.Observation || report.NativePID != transcript.PID || observation.PID != transcript.PID || !reflect.DeepEqual(report.Binding, binding) || !reflect.DeepEqual(report.Code, code) || binding.Architecture != "sm_120a" || binding.Output.SHA256 != lineageOutputSHA {
		return proof, fmt.Errorf("numeric lineage replay/raw binding differs")
	}
	proof.Runtime, proof.Context = r, observation.Context
	proof.Inputs = maps.Clone(r.Custody)
	for _, f := range []File{anchor, report.Runtime, report.SourceLoad, report.SupervisorCommand, report.SupervisorWatch, report.Observation, observation.InputCopy, binding.Input, binding.Output, report.ConsumerPlan, report.ConsumerController, report.ProducerPlan, report.ProducerController, report.NativeBuild, report.NativeExecutable} {
		if err = addPin(proof.Inputs, f); err != nil {
			return proof, err
		}
	}
	var replayConsumer LineagePlan
	if err = readJSON(report.ConsumerPlan, &replayConsumer); err != nil {
		return proof, err
	}
	if err = verifyLineagePlan(replayConsumer, false, false); err != nil {
		return proof, err
	}
	maps.Copy(proof.Inputs, replayConsumer.SnapshotPins)
	return proof, checkPins(proof.Inputs)
}

func numericEnvironment(p supgpu.RuntimePolicy) []string {
	return []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "LD_AUDIT=" + p.Audit.Path, "LD_LIBRARY_PATH=" + strings.Join(p.Directories, ":"), "LD_BIND_NOW=1"}
}

func equalEnvironment(a, b []string) bool {
	x, y := slices.Clone(a), slices.Clone(b)
	slices.Sort(x)
	slices.Sort(y)
	return slices.Equal(x, y)
}

func compareNumericEnvironments(load, numeric []string) error {
	if len(load) != 8 || len(numeric) != 8 {
		return fmt.Errorf("numeric producer environment extent differs")
	}
	values := func(env []string) (map[string]string, error) {
		out := map[string]string{}
		for _, entry := range env {
			k, v, ok := strings.Cut(entry, "=")
			if !ok || k == "" {
				return nil, fmt.Errorf("numeric environment syntax")
			}
			if _, ok = out[k]; ok {
				return nil, fmt.Errorf("numeric environment duplicate")
			}
			out[k] = v
		}
		return out, nil
	}
	a, err := values(load)
	if err != nil {
		return err
	}
	b, err := values(numeric)
	if err != nil {
		return err
	}
	for k, v := range a {
		other, ok := b[k]
		if !ok || k != "LD_LIBRARY_PATH" && other != v {
			return fmt.Errorf("undeclared cross-producer environment difference: %s", k)
		}
	}
	for _, env := range []map[string]string{a, b} {
		for k := range env {
			if strings.HasPrefix(k, "CUDA_") || strings.HasPrefix(k, "NVIDIA_") {
				return fmt.Errorf("numeric CUDA/JIT control environment differs")
			}
		}
	}
	return nil
}

func sharedNumericProvider(name string) bool {
	return name == "libcuda.so.1" || strings.HasPrefix(name, "libcupti.so") || strings.HasPrefix(name, "libnvidia-")
}
func compareNumericProviders(a, b supgpu.RuntimePolicy) error {
	selected := func(p supgpu.RuntimePolicy) (map[string]supgpu.Pin, error) {
		m := map[string]supgpu.Pin{}
		for _, c := range p.Allowed {
			if sharedNumericProvider(c.SONAME) {
				if _, ok := m[c.SONAME]; ok {
					return nil, fmt.Errorf("ambiguous numeric shared provider")
				}
				m[c.SONAME] = c.Object
			}
		}
		return m, nil
	}
	x, err := selected(a)
	if err != nil {
		return err
	}
	y, err := selected(b)
	if err != nil {
		return err
	}
	if len(x) < 2 || !maps.Equal(x, y) {
		return fmt.Errorf("numeric shared driver/CUPTI/JIT providers differ")
	}
	cuda, cupti := false, false
	for name := range x {
		cuda = cuda || name == "libcuda.so.1"
		cupti = cupti || strings.HasPrefix(name, "libcupti.so")
	}
	if !cuda || !cupti {
		return fmt.Errorf("numeric shared driver/CUPTI providers absent")
	}
	return nil
}

func verifyNumericConditions(r RuntimePlan, current bool) error {
	return verifyNumericConditionsScoped(r, current, nil)
}

func verifyNumericConditionsScoped(r RuntimePlan, current bool, proofs *numericProofs) error {
	proof, err := proofs.historical(r.NumericLineage)
	if err != nil {
		return err
	}
	if r.LoaderCache != proof.Runtime.LoaderCache {
		return fmt.Errorf("numeric loader-cache identity differs")
	}
	if err = compareNumericProviders(proof.Runtime.Policy, r.Policy); err != nil {
		return err
	}
	oldEnv, err := lineageExecEnvironment(proof.Runtime.Contract.Environment)
	if err != nil {
		return err
	}
	if !equalEnvironment(oldEnv, proof.Context.Environment) {
		return fmt.Errorf("numeric lineage real environment differs")
	}
	if err = compareNumericEnvironments(oldEnv, numericEnvironment(r.Policy)); err != nil {
		return err
	}
	if r.KernelVersion.SHA256 != proof.Runtime.KernelVersion.SHA256 {
		return fmt.Errorf("numeric kernel-driver identity differs")
	}
	if err = r.KernelVersion.verify(); err != nil {
		return err
	}
	if current {
		raw, e := os.ReadFile("/proc/driver/nvidia/version")
		if e != nil {
			return e
		}
		if digest(raw) != r.KernelVersion.SHA256 {
			return fmt.Errorf("numeric live kernel-driver changed")
		}
	}
	return nil
}

func readCaptureImages(r RuntimePlan, dir string, original ImagePolicy) (ImageReport, error) {
	return readCaptureImagesScoped(r, dir, original, nil)
}

func readCaptureImagesScoped(r RuntimePlan, dir string, original ImagePolicy, proofs *numericProofs) (ImageReport, error) {
	if r.Schema != "nanopore-decoder-runtime-plan/v4" {
		return ReadImageReport(dir, original)
	}
	proof, err := proofs.historical(r.NumericLineage)
	if err != nil {
		return ImageReport{}, err
	}
	if !reflect.DeepEqual(original, proof.Runtime.Contract.Policy) {
		return ImageReport{}, fmt.Errorf("numeric original image policy differs from lineage")
	}
	image := DeviceImage{Architecture: "sm_120a/reproduced-finalized", Image: proof.Report.Binding.Output, Symbols: map[string]uint32{}}
	f, err := elf.Open(image.Image.Path)
	if err != nil {
		return ImageReport{}, err
	}
	symbols, err := f.Symbols()
	valid := f.Class == elf.ELFCLASS64 && f.Data == elf.ELFDATA2LSB && f.Machine == elf.EM_CUDA
	f.Close()
	if err != nil || !valid {
		return ImageReport{}, fmt.Errorf("numeric delivered image ELF differs")
	}
	for i, s := range symbols {
		for _, name := range decoderSymbols {
			if s.Name == name && elf.ST_TYPE(s.Info) == elf.STT_FUNC && s.Section != elf.SHN_UNDEF {
				if image.Symbols[name] != 0 {
					return ImageReport{}, fmt.Errorf("numeric delivered symbol ambiguous")
				}
				image.Symbols[name] = uint32(i + 1)
			}
		}
	}
	if !reflect.DeepEqual(image.Symbols, proof.Runtime.Contract.Input.Symbols) {
		return ImageReport{}, fmt.Errorf("numeric delivered symbol indices differ")
	}
	return ReadImageReport(dir, ImagePolicy{Schema: "nanopore-decoder-finalized-image/v1", Member: original.Member, Images: []DeviceImage{image}})
}

func nativePreparationSources(schema string) (map[string]string, error) {
	names, err := nativeSourceNames(schema)
	if err != nil {
		return nil, err
	}
	inputs := map[string]string{}
	for _, name := range names {
		if err = filePin(inputs, filepath.Join("verification/decodergpu/native", name)); err != nil {
			return nil, err
		}
	}
	return inputs, nil
}
func requireNativePreparationSources(inputs map[string]string, schema string) error {
	expected, err := nativePreparationSources(schema)
	if err != nil {
		return err
	}
	for path, sha := range expected {
		if inputs[path] != sha {
			return fmt.Errorf("required numeric preparation source pin omitted or changed: %s", path)
		}
	}
	return nil
}
