package decodergpu

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// RuntimePlan is post-link pure metadata. It is not a selected-image receipt,
// and ELF resolution here never executes a loader or invokes a device API.
type RuntimePlan struct {
	Schema                                    string
	Build, Plan, Controller                   File
	ConsumerPreparation, Consumer             File
	Seeds                                     []nativeproof.Pin
	SearchDirs                                []string
	ELF                                       nativeproof.ELFClosure
	LoaderCache                               File
	Inputs                                    map[string]string
	ImageGate                                 string
	NumericLineage, KernelVersion             File `json:",omitempty"`
	Limits                                    []string
	AuditProof, Guard, PhaseContract, Watcher File
	WatcherInterpreter                        File
	Policy                                    supgpu.RuntimePolicy
	Custody                                   map[string]string
	CaptureDirectory, WorkingDirectory        string
}

const passiveImageGate = "required-cupti-context-module-function-image-binding/v2"
const legacyPassiveImageGate = "held-passive-cupti-module-bytes-before-first-kernel/v1"

func runtimeSeeds(binary File, recipe NativeRecipe) ([]nativeproof.Pin, []string, error) {
	dirs := []string{filepath.Dir(recipe.CUDART.Path), "/usr/lib", "/lib", "/opt/cuda/lib"}
	paths := []string{binary.Path, recipe.CUDART.Path}
	prefixes := []string{"libcudart", "libcuda", "libnvidia-gpucomp", "libnvidia-nvvm", "libnvidia-ptxjitcompiler", "libnvidia-allocator", "libnvrtc", "libnvJitLink"}
	if recipe.CUPTI.Path != "" {
		paths = append(paths, recipe.CUPTI.Path)
		dirs = append(dirs, filepath.Dir(recipe.CUPTI.Path))
		prefixes = append(prefixes, "libcupti")
	}
	// Explicit dynamic families cover CUDA's lazy driver/JIT loads; Torch and
	// cuDNN are absent from this standalone decoder recipe.
	for _, dir := range dirs {
		for _, prefix := range prefixes {
			matches, err := filepath.Glob(filepath.Join(dir, prefix+"*.so*"))
			if err != nil {
				return nil, nil, err
			}
			for _, path := range matches {
				f, err := os.Open(path)
				if err != nil {
					return nil, nil, err
				}
				var b [4]byte
				n, _ := f.Read(b[:])
				f.Close()
				if n == 4 && string(b[:]) == "\x7fELF" {
					paths = append(paths, path)
				}
			}
		}
	}
	slices.Sort(paths)
	seeds := []nativeproof.Pin{}
	seen := map[string]bool{}
	for _, path := range paths {
		p, err := artifactPin(path)
		if err != nil {
			return nil, nil, err
		}
		if !seen[p.Path] {
			seen[p.Path] = true
			seeds = append(seeds, nativeproof.Pin{Path: p.Path, SHA256: p.SHA256})
		}
	}
	return seeds, dirs, nil
}

func PlanRuntime(build File, out string) (r RuntimePlan, err error) {
	return planRuntime(build, File{}, out)
}

func PlanRuntimeConsumer(build, consumer File, out string) (r RuntimePlan, err error) {
	return planRuntime(build, consumer, out)
}

func planRuntime(build, consumer File, out string) (r RuntimePlan, err error) {
	r, err = makeRuntime(build, consumer)
	if err != nil {
		return r, err
	}
	dir, e := runner.NewAttempt(out)
	if e != nil {
		return r, e
	}
	return r, writeJSON(filepath.Join(dir, "runtime.json"), r)
}

func makeRuntime(build, consumer File) (r RuntimePlan, err error) {
	return makeRuntimeScoped(build, consumer, nil)
}

func makeRuntimeScoped(build, consumer File, proofs *numericProofs) (r RuntimePlan, err error) {
	build, err = normalizeAnchor(build)
	if err != nil {
		return r, err
	}
	var a ProducerConsumer
	if consumer.Path != "" {
		a, err = checkProducerConsumerScoped(build, consumer, proofs)
		if err != nil {
			return r, err
		}
	}
	b, p, err := readBuiltPhaseScoped(build, consumer.Path == "", proofs)
	if err != nil {
		return r, err
	}
	r = RuntimePlan{Schema: "nanopore-decoder-runtime-plan/v1", Build: build, Plan: b.Plan, Controller: p.Controller, Inputs: map[string]string{}, ImageGate: passiveImageGate, Limits: []string{"Prospective explicit ELF search; loader cache is pinned, not emulated.", "Actual process mappings must match before first stage and after every stage.", "Device SM alone cannot establish selected SM120/SM120a/PTX; passive cubin observation remains held."}}
	if consumer.Path != "" {
		r.Schema = "nanopore-decoder-runtime-plan/v2"
		r.ConsumerPreparation, r.Consumer = a.ConsumerPreparation, a.ConsumerController
	}
	r.Seeds, r.SearchDirs, err = runtimeSeeds(b.Output, p.Recipe)
	if err != nil {
		return r, err
	}
	r.ELF, err = nativeproof.CloseELF(r.Seeds, r.SearchDirs)
	if err != nil {
		return r, err
	}
	if err = checkHostELF(r.ELF); err != nil {
		return r, err
	}
	for _, p := range r.ELF.Inputs {
		if err = addPin(r.Inputs, File{p.Path, p.SHA256}); err != nil {
			return r, err
		}
	}
	r.LoaderCache, err = artifactPin("/etc/ld.so.cache")
	if err != nil {
		return r, err
	}
	if err = addPin(r.Inputs, r.LoaderCache); err != nil {
		return r, err
	}
	return r, nil
}

func verifyRuntimePlan(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan) error {
	if (r.Schema != "nanopore-decoder-runtime-plan/v1" && r.Schema != "nanopore-decoder-runtime-plan/v2") || r.Plan != b.Plan || r.Controller != p.Controller || (r.ImageGate != passiveImageGate && r.ImageGate != legacyPassiveImageGate) {
		return fmt.Errorf("runtime/producer policy binding changed")
	}
	seeds, dirs, err := runtimeSeeds(b.Output, p.Recipe)
	if err != nil {
		return err
	}
	closure, err := nativeproof.CloseELF(seeds, dirs)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(seeds, r.Seeds) || !slices.Equal(dirs, r.SearchDirs) || !reflect.DeepEqual(closure, r.ELF) {
		return fmt.Errorf("prospective CUDA runtime family changed")
	}
	cache, err := artifactPin("/etc/ld.so.cache")
	if err != nil {
		return err
	}
	if cache != r.LoaderCache {
		return fmt.Errorf("runtime loader cache changed")
	}
	required := map[string]string{cache.Path: cache.SHA256}
	for _, f := range closure.Inputs {
		required[f.Path] = f.SHA256
	}
	if !maps.Equal(required, r.Inputs) {
		return fmt.Errorf("prospective runtime input omitted or added")
	}
	return checkPins(r.Inputs)
}

// CheckRuntimeMaps is also used by capture admission. It checks complete
// file-backed mappings, never infers a closure from what happened.
func CheckRuntimeMaps(body string, allowed map[string]string) error {
	seen := 0
	paths := map[string]bool{}
	for _, line := range strings.Split(body, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := strings.Join(fields[5:], " ")
		if !filepath.IsAbs(path) {
			continue
		}
		if strings.HasSuffix(path, " (deleted)") {
			return fmt.Errorf("deleted file-backed mapping is outside runtime closure")
		}
		if paths[path] {
			continue
		}
		paths[path] = true
		p, err := artifactPin(path)
		if err != nil {
			return err
		}
		if allowed[p.Path] != p.SHA256 {
			return fmt.Errorf("unknown or changed executable runtime mapping: %s", path)
		}
		seen++
	}
	if seen == 0 {
		return fmt.Errorf("runtime mapping observation is empty")
	}
	return nil
}

type CaptureOptions struct {
	Build, Runtime, Consumer File
	Out                      string
	Admitted                 bool
}

func RunCapture(o CaptureOptions) error {
	_, err := Capture(o)
	return err
}
