package supgpu

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

type ELFNode struct {
	Object      Pin      `json:"object"`
	Needed      []string `json:"needed"`
	RunPath     []string `json:"runpath"`
	RPath       []string `json:"rpath"`
	Interpreter string   `json:"interpreter"`
}
type RuntimePlan struct {
	Schema      string        `json:"schema"`
	Build       Pin           `json:"build"`
	Prepared    Pin           `json:"prepared"`
	Controller  Pin           `json:"controller"`
	Consumer    Pin           `json:"consumer"`
	Family      string        `json:"dlopen_family"`
	Seeds       []Pin         `json:"seeds"`
	Nodes       []ELFNode     `json:"elf_nodes"`
	Inputs      []Pin         `json:"inputs"`
	LoaderCache Pin           `json:"loader_cache"`
	SearchDirs  []string      `json:"search_directories"`
	Policy      RuntimePolicy `json:"policy"`
}

type RuntimeAnchors struct {
	Build      Pin `json:"build"`
	Prepared   Pin `json:"prepared"`
	Controller Pin `json:"controller"`
	Consumer   Pin `json:"consumer"`
}

const runtimePlanSchema = "nanopore-supgpu-runtime-plan/v3"

func canonicalPin(path string) (Pin, error) {
	actual, err := filepath.EvalSymlinks(path)
	if err != nil {
		return Pin{}, err
	}
	return pin(actual)
}

func isELF(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	prefix := make([]byte, 4)
	n, err := f.Read(prefix)
	if err != nil && n == 0 {
		return false, err
	}
	return n == 4 && string(prefix) == "\x7fELF", nil
}

// PlanRuntime is a separate host-only, post-build step. It reads ELF metadata;
// it never executes ldd, a loader, a native executable or a device query.
func PlanRuntime(buildPath, buildSHA, out, consumerPath, consumerSHA string) (RuntimePlan, error) {
	r := RuntimePlan{Schema: runtimePlanSchema, Family: "fixed-sup-cuda13-dlopen/v1", Seeds: []Pin{}, Nodes: []ELFNode{}, Inputs: []Pin{}, SearchDirs: []string{}}
	a, b, p, err := runtimeConsumerBuild(buildPath, buildSHA, consumerPath, consumerSHA)
	if err != nil {
		return r, err
	}
	r.Build, r.Prepared, r.Controller, r.Consumer = a.Build, a.Prepared, a.Controller, a.Consumer
	r.Seeds, r.SearchDirs, err = runtimeSeeds(b, p)
	if err != nil {
		return r, err
	}
	r.Nodes, r.Inputs, err = elfClosure(r.Seeds, r.SearchDirs)
	if err != nil {
		return r, err
	}
	r.LoaderCache, err = canonicalPin("/etc/ld.so.cache")
	if err != nil {
		return r, err
	}
	r.Inputs = append(r.Inputs, r.LoaderCache)
	r.Policy, err = makeRuntimePolicy(r, b, p)
	if err != nil {
		return r, err
	}
	if err = writeJSON(out, r); err != nil {
		return r, err
	}
	return r, nil
}

// CheckRuntimeConsumer validates an accepted compiler producer and a separately
// frozen current Go consumer without constructing a runtime plan or any NN TU.
func CheckRuntimeConsumer(buildPath, buildSHA, consumerPath, consumerSHA string) (RuntimeAnchors, error) {
	a, _, _, err := runtimeConsumerBuild(buildPath, buildSHA, consumerPath, consumerSHA)
	return a, err
}

func runtimeConsumerBuild(buildPath, buildSHA, consumerPath, consumerSHA string) (a RuntimeAnchors, b Build, p BuildPlan, err error) {
	a.Consumer, err = pin(consumerPath)
	if err != nil {
		return
	}
	if a.Consumer.SHA256 != consumerSHA {
		err = fmt.Errorf("runtime consumer manifest changed")
		return
	}
	if _, err = readController(a.Consumer, true); err != nil {
		return
	}
	if err = readPinnedJSON(buildPath, buildSHA, &b); err != nil {
		return
	}
	if err = verifyBuild(b, b.PreparedSHA256); err != nil {
		return
	}
	if err = readPinnedJSON(b.Plan.Path, b.Plan.SHA256, &p); err != nil {
		return
	}
	// All old inputs, copies, binary and compiler/link closure still validate.
	// Only the old live Go sources/executing-binary requirement is historical.
	if err = verifyPlan(p, false); err != nil {
		return
	}
	a.Build, err = pin(buildPath)
	a.Prepared, a.Controller = p.Prepared, p.Controller
	return
}

func runtimeSeeds(b Build, p BuildPlan) ([]Pin, []string, error) {
	paths := map[string]bool{b.Reference.Path: true, b.Observer.Path: true, b.Audit.Path: true}
	dirs := []string{}
	seenDir := map[string]bool{}
	addDir := func(dir string) {
		if dir != "" && !seenDir[dir] {
			seenDir[dir] = true
			dirs = append(dirs, dir)
		}
	}
	for _, binding := range p.LinkBindings {
		if strings.Contains(filepath.Base(binding.Input.Path), ".so") {
			binary, err := isELF(binding.Input.Path)
			if err != nil {
				return nil, nil, err
			}
			if binary {
				paths[binding.Input.Path] = true
			}
			addDir(filepath.Dir(binding.Input.Path))
		}
	}
	var prepared Prepared
	if err := readPinnedJSON(p.Prepared.Path, p.Prepared.SHA256, &prepared); err != nil {
		return nil, nil, err
	}
	if err := verifyPin("", prepared.Profile); err != nil {
		return nil, nil, err
	}
	raw, err := os.ReadFile(prepared.Profile.Path)
	if err != nil {
		return nil, nil, err
	}
	var profile struct {
		Runtime map[string]string `json:"runtime_sha256"`
	}
	if err = json.Unmarshal(raw, &profile); err != nil {
		return nil, nil, err
	}
	profilePaths := []string{}
	for path := range profile.Runtime {
		profilePaths = append(profilePaths, path)
	}
	sort.Strings(profilePaths)
	cache, err := canonicalPin("/etc/ld.so.cache")
	if err != nil {
		return nil, nil, err
	}
	for _, path := range profilePaths {
		sha := profile.Runtime[path]
		if !strings.Contains(filepath.Base(path), ".so") || strings.Contains(filepath.Base(path), "libnanopore_tensor") {
			continue
		}
		v, code, err := profileRuntimeSeed(path, sha, cache)
		if err != nil {
			return nil, nil, err
		}
		if !code {
			continue
		}
		paths[v.Path] = true
		addDir(filepath.Dir(v.Path))
	}
	addDir("/usr/lib")
	addDir("/lib")
	addDir("/opt/cuda/lib")
	// cuDNN engines, NVRTC/JIT and CUDA compute drivers can load DSOs lazily.
	// Freeze the selected distribution families before any model process.
	for _, dir := range append([]string{}, dirs...) {
		for _, prefix := range []string{"libcudnn", "libcusparseLt", "libcublas", "libcudart", "libcufft", "libcusolver", "libcusparse", "libnvJitLink", "libnvrtc", "libcuda", "libnvidia-gpucomp", "libnvidia-nvvm", "libnvidia-ptxjitcompiler", "libnvidia-allocator"} {
			matches, err := filepath.Glob(filepath.Join(dir, prefix+"*.so*"))
			if err != nil {
				return nil, nil, err
			}
			for _, path := range matches {
				info, err := os.Stat(path)
				if err != nil {
					return nil, nil, err
				}
				if info.Mode().IsRegular() {
					binary, err := isELF(path)
					if err != nil {
						return nil, nil, err
					}
					if binary {
						paths[path] = true
					}
				}
			}
		}
	}
	ordered := []string{}
	for path := range paths {
		ordered = append(ordered, path)
	}
	sort.Strings(ordered)
	seeds := []Pin{}
	seen := map[string]bool{}
	for _, path := range ordered {
		v, err := canonicalPin(path)
		if err != nil {
			return nil, nil, err
		}
		if !seen[v.Path] {
			seen[v.Path] = true
			seeds = append(seeds, v)
		}
	}
	return seeds, dirs, nil
}

func profileRuntimeSeed(path, sha string, cache Pin) (Pin, bool, error) {
	v, err := canonicalPin(path)
	if err != nil {
		return v, false, err
	}
	if v.SHA256 != sha {
		return v, false, fmt.Errorf("retained P01 runtime input changed: %s", path)
	}
	if v.Path == cache.Path {
		if v != cache {
			return v, false, fmt.Errorf("loader cache changed while preparing runtime seeds")
		}
		return v, false, nil
	}
	code, err := isELF(v.Path)
	if err != nil {
		return v, false, err
	}
	if !code {
		return v, false, fmt.Errorf("retained runtime code input is not ELF: %s", path)
	}
	return v, true, nil
}

func elfClosure(seeds []Pin, search []string) ([]ELFNode, []Pin, error) {
	nodes := []ELFNode{}
	inputs := []Pin{}
	seen := map[string]bool{}
	queue := append([]Pin{}, seeds...)
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		if seen[p.Path] {
			continue
		}
		seen[p.Path] = true
		if err := verifyPin("", p); err != nil {
			return nil, nil, err
		}
		f, err := elf.Open(p.Path)
		if err != nil {
			return nil, nil, fmt.Errorf("runtime seed is not ELF %s: %w", p.Path, err)
		}
		if f.Class != elf.ELFCLASS64 || f.Machine != elf.EM_X86_64 {
			f.Close()
			return nil, nil, fmt.Errorf("runtime ELF architecture is outside the qualified target")
		}
		needed, err := f.ImportedLibraries()
		if err != nil {
			f.Close()
			return nil, nil, err
		}
		runpath, err := f.DynString(elf.DT_RUNPATH)
		if err != nil {
			f.Close()
			return nil, nil, err
		}
		rpath, err := f.DynString(elf.DT_RPATH)
		if err != nil {
			f.Close()
			return nil, nil, err
		}
		interpreter := ""
		if section := f.Section(".interp"); section != nil {
			raw, err := section.Data()
			if err != nil {
				f.Close()
				return nil, nil, err
			}
			interpreter = strings.TrimRight(string(raw), "\x00")
		}
		f.Close()
		if needed == nil {
			needed = []string{}
		}
		if runpath == nil {
			runpath = []string{}
		}
		if rpath == nil {
			rpath = []string{}
		}
		nodes = append(nodes, ELFNode{p, needed, runpath, rpath, interpreter})
		inputs = append(inputs, p)
		dirs := []string{filepath.Dir(p.Path)}
		for _, value := range append(append([]string{}, rpath...), runpath...) {
			for _, dir := range strings.Split(value, ":") {
				dir = strings.ReplaceAll(strings.ReplaceAll(dir, "${ORIGIN}", filepath.Dir(p.Path)), "$ORIGIN", filepath.Dir(p.Path))
				if strings.Contains(dir, "$") {
					return nil, nil, fmt.Errorf("unsupported runtime search expansion %s", dir)
				}
				if dir == "" {
					dir = "/home/jake/Projects/nanopore_basecaller_go"
				}
				dirs = append(dirs, dir)
			}
		}
		dirs = append(dirs, search...)
		for _, name := range needed {
			found := ""
			if strings.ContainsRune(name, '/') {
				return nil, nil, fmt.Errorf("unexpected pathname DT_NEEDED")
			}
			for _, dir := range dirs {
				candidate := filepath.Join(dir, name)
				info, err := os.Stat(candidate)
				if err == nil && info.Mode().IsRegular() {
					found = candidate
					break
				}
			}
			if found == "" {
				return nil, nil, fmt.Errorf("unresolved prospective DT_NEEDED %s from %s", name, p.Path)
			}
			v, err := canonicalPin(found)
			if err != nil {
				return nil, nil, err
			}
			queue = append(queue, v)
		}
		if interpreter != "" {
			v, err := canonicalPin(interpreter)
			if err != nil {
				return nil, nil, err
			}
			queue = append(queue, v)
		}
	}
	return nodes, inputs, nil
}

func readRuntimePlan(p Pin, b Build, build Pin, current bool) (RuntimePlan, error) {
	var r RuntimePlan
	if err := verifyPin("", p); err != nil {
		return r, err
	}
	if err := readPinnedJSON(p.Path, p.SHA256, &r); err != nil {
		return r, err
	}
	var plan BuildPlan
	if err := readPinnedJSON(b.Plan.Path, b.Plan.SHA256, &plan); err != nil {
		return r, err
	}
	if err := verifyPlan(plan, false); err != nil {
		return r, err
	}
	if err := verifyRuntimeAnchors(r, b, plan, build, current); err != nil {
		return r, err
	}
	seeds, dirs, err := runtimeSeeds(b, plan)
	if err != nil {
		return r, err
	}
	if !reflect.DeepEqual(seeds, r.Seeds) || !reflect.DeepEqual(dirs, r.SearchDirs) {
		return r, fmt.Errorf("prospective dlopen/runtime seed set changed")
	}
	nodes, inputs, err := elfClosure(seeds, dirs)
	if err != nil {
		return r, err
	}
	cache, err := canonicalPin("/etc/ld.so.cache")
	if err != nil {
		return r, err
	}
	inputs = append(inputs, cache)
	if cache != r.LoaderCache || !reflect.DeepEqual(nodes, r.Nodes) || !reflect.DeepEqual(inputs, r.Inputs) {
		return r, fmt.Errorf("prospective ELF runtime closure changed")
	}
	if err = verifyRuntimePolicy(r, b, plan); err != nil {
		return r, err
	}
	return r, nil
}

func verifyRuntimeAnchors(r RuntimePlan, b Build, plan BuildPlan, build Pin, current bool) error {
	if r.Schema != runtimePlanSchema || r.Family != "fixed-sup-cuda13-dlopen/v1" || r.Build != build || r.Prepared != plan.Prepared || b.PreparedSHA256 != plan.Prepared.SHA256 || r.Controller != plan.Controller {
		return fmt.Errorf("runtime plan identity changed")
	}
	if _, err := readController(r.Controller, false); err != nil {
		return err
	}
	_, err := readController(r.Consumer, current)
	return err
}

func requireRuntimeMappings(actual, frozen []Pin) error {
	allowed := map[string]Pin{}
	for _, p := range frozen {
		allowed[p.Path] = p
	}
	for _, p := range actual {
		canonical, err := canonicalPin(p.Path)
		if err != nil {
			return err
		}
		if allowed[canonical.Path] != canonical || canonical.SHA256 != p.SHA256 || canonical.Bytes != p.Bytes {
			return fmt.Errorf("runtime object absent from pre-model closure: %s", p.Path)
		}
	}
	return nil
}
