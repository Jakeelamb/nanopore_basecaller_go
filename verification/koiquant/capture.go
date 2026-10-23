package koiquant

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type CaptureOptions struct {
	Root, Prepared, Basic, Build, Out, ReuseBinary string
	Ordinal                                        int
	Admitted                                       bool
	Invocation                                     []string
}

type qualifiedBuild struct {
	ABI        int               `json:"abi_version"`
	Status     string            `json:"status"`
	Family     string            `json:"cuda_family"`
	Torch      string            `json:"torch_root"`
	Archive    string            `json:"archive"`
	ArchiveSHA string            `json:"archive_sha256"`
	Output     string            `json:"output"`
	OutputSHA  string            `json:"output_sha256"`
	Inputs     map[string]string `json:"input_sha256"`
	Runtime    map[string]string `json:"runtime_sha256"`
}

func qualification(basicPath, buildPath string) (qualifiedBuild, error) {
	var build qualifiedBuild
	if err := checkHash(basicPath, BasicSHA256); err != nil {
		return build, err
	}
	if err := checkHash(buildPath, BuildSHA256); err != nil {
		return build, err
	}
	data, err := os.ReadFile(buildPath)
	if err != nil {
		return build, err
	}
	if err = json.Unmarshal(data, &build); err != nil {
		return build, err
	}
	var basic struct {
		ABI      int                     `json:"abi_version"`
		Status   string                  `json:"status"`
		Inputs   map[string]string       `json:"input_sha256"`
		Commands []runner.CommandReceipt `json:"commands"`
	}
	data, err = os.ReadFile(basicPath)
	if err != nil {
		return build, err
	}
	if err = json.Unmarshal(data, &basic); err != nil {
		return build, err
	}
	if build.ABI != 4 || build.Status != "built" || build.Family != "13.0" || build.OutputSHA != BackendSHA256 || basic.ABI != 4 || basic.Status != "source_differential_passed" || basic.Inputs[build.Output] != BackendSHA256 {
		return build, fmt.Errorf("missing qualified ABI4 CUDA13 evidence")
	}
	matched := false
	for path, sha := range basic.Inputs {
		if filepath.Base(path) == "manifest.json" && sha == BuildSHA256 {
			matched = true
		}
	}
	if !matched {
		return build, fmt.Errorf("basic runtime does not pin selected build")
	}
	if err = checkHash(build.Output, BackendSHA256); err != nil {
		return build, err
	}
	return build, nil
}

func effectiveEnvironment() (map[string]EnvironmentValue, []string) {
	values := map[string]EnvironmentValue{}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "LD_LIBRARY_PATH", "LD_PRELOAD", "DORADO_MATMUL_FP16_MODE"} {
		value, present := os.LookupEnv(key)
		values[key] = EnvironmentValue{present, value}
	}
	overrides := []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}
	for _, kv := range overrides {
		key, value, _ := strings.Cut(kv, "=")
		values[key] = EnvironmentValue{true, value}
	}
	return values, overrides
}

func pinFiles(paths []string) (map[string]string, error) {
	out := map[string]string{}
	for _, path := range paths {
		if _, exists := out[path]; exists {
			continue
		}
		pin, err := filePin(path)
		if err != nil {
			return nil, err
		}
		out[path] = pin.SHA256
	}
	return out, nil
}

func mappedPaths(raw []byte) ([]string, error) {
	paths := []string{}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		if strings.HasPrefix(fields[5], "/dev/") {
			continue
		}
		if len(fields) != 6 {
			return nil, fmt.Errorf("unsupported or deleted mapped path: %s", line)
		}
		path := fields[5]
		if !seen[path] {
			paths = append(paths, path)
			seen[path] = true
		}
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("missing actual loaded mappings")
	}
	return paths, nil
}

func lddPaths(raw []byte) ([]string, error) {
	if strings.Contains(string(raw), "not found") {
		return nil, fmt.Errorf("unresolved runtime library")
	}
	paths := regexp.MustCompile(`(?m)(?:=>\s+|^\s*)(/[^\s]+)`).FindAllSubmatch(raw, -1)
	out := []string{}
	for _, path := range paths {
		out = append(out, string(path[1]))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no linked libraries observed")
	}
	return out, nil
}

func pinTree(root string) (map[string]string, error) {
	paths := []string{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("nonregular raw output %s", path)
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return pinFiles(paths)
}

func copyFile(from, to string) error {
	b, err := os.ReadFile(from)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(to), 0755); err != nil {
		return err
	}
	return os.WriteFile(to, b, 0644)
}

// Capture is the only native entry point. Admission and the existing lock precede every external command.
func Capture(o CaptureOptions) (m RawCapture, err error) {
	if !o.Admitted {
		return m, fmt.Errorf("runtime admission required; zero external commands launched")
	}
	for _, path := range []*string{&o.Prepared, &o.Basic, &o.Build} {
		*path, err = filepath.Abs(*path)
		if err != nil {
			return m, err
		}
	}
	if o.ReuseBinary != "" {
		o.ReuseBinary, err = filepath.Abs(o.ReuseBinary)
		if err != nil {
			return m, err
		}
	}
	build, err := qualification(o.Basic, o.Build)
	if err != nil {
		return m, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return m, err
	}
	defer lock.Close()
	o.Root, err = filepath.Abs(o.Root)
	if err != nil {
		return m, err
	}
	o.Out, err = filepath.Abs(o.Out)
	if err != nil {
		return m, err
	}
	if err = admitOutput(o.Root, o.Out); err != nil {
		return m, err
	}
	prepared, err := CheckPrepared(o.Root, o.Prepared)
	if err != nil {
		return m, err
	}
	if o.Ordinal < 0 || prepared.Device == "cpu" && o.Ordinal != 0 {
		return m, fmt.Errorf("invalid ordinal")
	}
	if err = os.Mkdir(o.Out, 0755); err != nil {
		return m, err
	}
	m = RawCapture{Schema: CaptureSchema, Status: "capture_failed", Started: clock(), Device: prepared.Device, Ordinal: o.Ordinal, Prepared: prepared, Torch: build.Torch, CUDA: "/opt/cuda", Invocation: o.Invocation,
		ProducerSources: map[string]string{}, Dependencies: map[string]string{}, ArchiveMembers: map[string]string{}, Linked: map[string]string{}, MappedBefore: map[string]string{}, MappedAfter: map[string]string{}, Outputs: map[string]string{}, Commands: []runner.CommandReceipt{}}
	defer func() {
		m.Finished = clock()
		if err != nil {
			m.Error = err.Error()
		}
		if outputs, pinErr := pinTree(filepath.Join(o.Out, "observations")); pinErr == nil {
			m.Outputs = outputs
		}
		writeErr := writeJSON(filepath.Join(o.Out, "raw-manifest.json"), m)
		if err == nil {
			err = writeErr
		}
	}()
	for path, target := range map[string]*Pin{o.Prepared: &m.Preparation, o.Basic: &m.Basic, o.Build: &m.Build, build.Archive: &m.Archive} {
		*target, err = filePin(path)
		if err != nil {
			return m, err
		}
	}
	if m.Archive.SHA256 != build.ArchiveSHA {
		return m, fmt.Errorf("qualified Torch archive pin changed")
	}
	for path, want := range build.Runtime {
		if err = checkHash(path, want); err != nil {
			return m, err
		}
	}
	for path, want := range prepared.Sources {
		target := filepath.Join(o.Out, "producer-source", path)
		if err = copyFile(filepath.Join(o.Root, path), target); err != nil {
			return m, err
		}
		if err = checkHash(target, want); err != nil {
			return m, err
		}
		m.ProducerSources[path] = want
	}
	for name, want := range prepared.Generated {
		if err = copyFile(filepath.Join(filepath.Dir(o.Prepared), name), filepath.Join(o.Out, name)); err != nil {
			return m, err
		}
		if err = checkHash(filepath.Join(o.Out, name), want); err != nil {
			return m, err
		}
	}
	for _, fixture := range prepared.Fixtures {
		if err = copyFile(fixture.Input.Path, filepath.Join(o.Out, "inputs", fixture.Name+"-u16le.bin")); err != nil {
			return m, err
		}
	}
	var env []string
	m.Environment, env = effectiveEnvironment()
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		r, e := runner.Execute(o.Out, label, env, args...)
		m.Commands = append(m.Commands, r)
		return r, e
	}
	bookend := func(label string) error {
		r, e := execute(label+"-gpu", "nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits")
		if e != nil {
			return e
		}
		data, e := os.ReadFile(r.Stdout)
		if e != nil {
			return e
		}
		identity := strings.TrimSpace(string(data))
		if identity == "" || !strings.Contains(identity, "GPU-") {
			return fmt.Errorf("missing GPU UUID and driver bookend")
		}
		if m.GPUIdentity != "" && identity != m.GPUIdentity {
			return fmt.Errorf("GPU identity changed")
		}
		m.GPUIdentity = identity
		r, e = execute(label+"-apps", "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits")
		if e != nil {
			return e
		}
		data, e = os.ReadFile(r.Stdout)
		if e != nil {
			return e
		}
		if strings.TrimSpace(string(data)) != "" {
			return fmt.Errorf("foreign compute process present")
		}
		return nil
	}
	if err = bookend("before"); err != nil {
		return m, err
	}
	if o.ReuseBinary != "" {
		var prior RawCapture
		if err = readClosed(o.ReuseBinary, &prior); err != nil {
			return m, err
		}
		if err = validateRaw(o.ReuseBinary, prior); err != nil {
			return m, err
		}
		if prior.Device != "cpu" || prior.Build.SHA256 != m.Build.SHA256 || !reflect.DeepEqual(prior.Prepared.Generated, prepared.Generated) || !reflect.DeepEqual(prior.Prepared.Sources, prepared.Sources) {
			return m, fmt.Errorf("reused binary source/build mismatch")
		}
		m.Binary, m.Compiler = prior.Binary, prior.Compiler
		m.BinaryOrigin, err = filePin(o.ReuseBinary)
		if err != nil {
			return m, err
		}
		m.Dependencies, m.ArchiveMembers = prior.Dependencies, prior.ArchiveMembers
	} else {
		compiler, e := exec.LookPath("g++")
		if e != nil {
			return m, e
		}
		compiler, e = filepath.EvalSymlinks(compiler)
		if e != nil {
			return m, e
		}
		m.Compiler, err = filePin(compiler)
		if err != nil {
			return m, err
		}
		if build.Inputs[compiler] != m.Compiler.SHA256 {
			return m, fmt.Errorf("compiler differs from qualified build")
		}
		if _, err = execute("compiler-version", compiler, "--version"); err != nil {
			return m, err
		}
		toolPaths := []string{compiler}
		for _, tool := range []string{"cc1plus", "as", "collect2", "ld"} {
			r, e := execute("compiler-"+tool, compiler, "-print-prog-name="+tool)
			if e != nil {
				return m, e
			}
			b, e := os.ReadFile(r.Stdout)
			if e != nil {
				return m, e
			}
			path, e := exec.LookPath(strings.TrimSpace(string(b)))
			if e != nil {
				return m, e
			}
			path, e = filepath.EvalSymlinks(path)
			if e != nil {
				return m, e
			}
			p, e := filePin(path)
			if e != nil {
				return m, e
			}
			m.Dependencies[p.Path] = p.SHA256
			toolPaths = append(toolPaths, p.Path)
		}
		for i, path := range toolPaths {
			r, e := execute(fmt.Sprintf("compiler-linkage-%d", i), "ldd", path)
			if e != nil {
				return m, e
			}
			b, e := os.ReadFile(r.Stdout)
			if e != nil {
				return m, e
			}
			paths, e := lddPaths(b)
			if e != nil {
				return m, e
			}
			paths = append(paths, path)
			for _, p := range paths {
				if _, ok := m.Dependencies[p]; ok {
					continue
				}
				pin, e := filePin(p)
				if e != nil {
					return m, e
				}
				m.Dependencies[p] = pin.SHA256
			}
		}
		binary := filepath.Join(o.Out, "oracle")
		lib := filepath.Join(build.Torch, "lib")
		args := []string{compiler, "-std=c++17", "-O1", "-g0", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-I" + filepath.Join(build.Torch, "include"), "-I" + filepath.Join(build.Torch, "include/torch/csrc/api/include"), "-I" + filepath.Join(m.CUDA, "include"), "-MD", "-MF", filepath.Join(o.Out, "oracle.d"), filepath.Join(o.Out, "oracle.cpp"), "-o", binary,
			"-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10", "-Wl,--no-as-needed", "-ltorch_cuda", "-lc10_cuda", "-Wl,--as-needed", "-L" + filepath.Join(m.CUDA, "lib64"), "-Wl,-rpath," + filepath.Join(m.CUDA, "lib64"), "-lcudart"}
		if _, err = execute("compile", args...); err != nil {
			return m, err
		}
		m.Binary, err = filePin(binary)
		if err != nil {
			return m, err
		}
		depfile := filepath.Join(o.Out, "oracle.d")
		data, e := os.ReadFile(depfile)
		if e != nil {
			return m, e
		}
		text := strings.ReplaceAll(string(data), "\\\n", "")
		_, text, ok := strings.Cut(text, ":")
		if !ok || strings.Contains(text, "\\") {
			return m, fmt.Errorf("unsupported depfile")
		}
		paths := []string{depfile}
		for _, p := range strings.Fields(text) {
			path, e := filepath.Abs(p)
			if e != nil {
				return m, e
			}
			paths = append(paths, path)
		}
		pins, e := pinFiles(paths)
		if e != nil {
			return m, e
		}
		for path, h := range pins {
			m.Dependencies[path] = h
		}
	}
	r, err := execute("linkage", "ldd", m.Binary.Path)
	if err != nil {
		return m, err
	}
	data, err := os.ReadFile(r.Stdout)
	if err != nil {
		return m, err
	}
	paths, err := lddPaths(data)
	if err != nil {
		return m, err
	}
	m.Linked, err = pinFiles(paths)
	if err != nil {
		return m, err
	}
	if err = archiveClosure(build, m.Dependencies, m.Linked, m.ArchiveMembers); err != nil {
		return m, err
	}
	r, nativeErr := execute("native", m.Binary.Path, o.Out, m.Device, strconv.Itoa(m.Ordinal))
	if err = validateReceipt(r, true); err != nil {
		return m, err
	}
	for label, target := range map[string]*map[string]string{"before": &m.MappedBefore, "after": &m.MappedAfter} {
		data, err = os.ReadFile(filepath.Join(o.Out, "observations", label+"-maps.txt"))
		if err != nil {
			return m, err
		}
		paths, err = mappedPaths(data)
		if err != nil {
			return m, err
		}
		*target, err = pinFiles(paths)
		if err != nil {
			return m, err
		}
		if err = archiveClosure(build, *target, nil, m.ArchiveMembers); err != nil {
			return m, err
		}
	}
	if err = bookend("after"); err != nil {
		return m, err
	}
	if err = checkSources(o.Root, prepared.Sources); err != nil {
		return m, err
	}
	if nativeErr != nil {
		m.Status = "native_observed_errors"
	} else {
		m.Status = "native_completed"
	}
	return m, nil
}

func archiveClosure(build qualifiedBuild, first, second, members map[string]string) error {
	z, err := zip.OpenReader(build.Archive)
	if err != nil {
		return err
	}
	defer z.Close()
	seen := map[string]bool{}
	for _, pins := range []map[string]string{first, second} {
		for path, want := range pins {
			if !strings.HasPrefix(path, build.Torch+"/") || seen[path] {
				continue
			}
			seen[path] = true
			if archivedPathMatches(build.Torch, path, want, members) == nil {
				continue
			}
			matched, e := runner.ArchiveMember(z, build.Torch, path, want)
			if e != nil {
				return e
			}
			for name, h := range matched {
				members[name] = h
			}
		}
	}
	return nil
}

func archivedPathMatches(root, path, want string, members map[string]string) error {
	for depth := 0; depth < 32; depth++ {
		if !strings.HasPrefix(path, root+"/") {
			return fmt.Errorf("archive member escapes Torch")
		}
		name := "libtorch/" + strings.TrimPrefix(path, root+"/")
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink == 0 {
			if !info.Mode().IsRegular() || members[name] != want {
				return fmt.Errorf("missing archive target pin %s", name)
			}
			return nil
		}
		target, err := os.Readlink(path)
		if err != nil {
			return err
		}
		if filepath.IsAbs(target) || members[name] != digest([]byte(target)) {
			return fmt.Errorf("missing archive symlink pin %s", name)
		}
		path = filepath.Clean(filepath.Join(filepath.Dir(path), target))
	}
	return fmt.Errorf("archive symlink depth exceeds limit")
}
