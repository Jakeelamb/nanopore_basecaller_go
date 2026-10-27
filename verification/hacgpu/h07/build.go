package h07

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type OracleBuildOptions struct {
	ControllerBuild, ControllerBuildSHA          string
	BuildDir, DoradoSource, RuntimeManifest, Out string
	Admitted                                     bool
}
type OracleBuildReceipt struct {
	ControllerBuild, ControllerBuildSHA                              string
	Controller, ControllerSHA                                        string
	Schema, Status, Error, Started, Finished, Attempt                string
	Compiler, Output, OutputSHA, RuntimeManifest, RuntimeManifestSHA string
	Toolchain                                                        hacgpu.OracleBuildToolchain
	Inputs, Runtime, Objects                                         map[string]string
	Dependencies                                                     map[string][]string
	Commands                                                         []runner.CommandReceipt
	Compile, Link                                                    [][]string
}

// ParseDependencies reads GCC -M/-MD output, including escaped whitespace and
// line continuations. Paths are resolved in the compiler's working directory.
func ParseDependencies(body, dir string) ([]string, error) {
	body = strings.ReplaceAll(body, "\\\r\n", "")
	body = strings.ReplaceAll(body, "\\\n", "")
	colon := -1
	escaped := false
	for i, c := range body {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == ':' {
			colon = i
			break
		}
	}
	if colon < 0 {
		return nil, fmt.Errorf("dependency target is absent")
	}
	var token strings.Builder
	out := []string{}
	escaped = false
	flush := func() error {
		if token.Len() == 0 {
			return nil
		}
		name := token.String()
		token.Reset()
		if !filepath.IsAbs(name) {
			name = filepath.Join(dir, name)
		}
		actual, err := filepath.EvalSymlinks(name)
		if err != nil {
			return err
		}
		out = append(out, actual)
		return nil
	}
	for _, c := range body[colon+1:] {
		if escaped {
			token.WriteRune(c)
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			if err := flush(); err != nil {
				return nil, err
			}
		} else {
			token.WriteRune(c)
		}
	}
	if escaped {
		return nil, fmt.Errorf("unterminated dependency escape")
	}
	if err := flush(); err != nil {
		return nil, err
	}
	slices.Sort(out)
	out = slices.Compact(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("empty header dependency closure")
	}
	return out, nil
}
func pinFile(pins map[string]string, path string) error {
	actual, err := filepath.EvalSymlinks(path)
	if err != nil {
		return err
	}
	actual, err = filepath.Abs(actual)
	if err != nil {
		return err
	}
	info, err := os.Stat(actual)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("closure input is not regular: %s", actual)
	}
	h, err := runner.SHA(actual)
	if err != nil {
		return err
	}
	if prior, ok := pins[actual]; ok && prior != h {
		return fmt.Errorf("closure input changed: %s", actual)
	}
	pins[actual] = h
	return nil
}
func checkPins(pins map[string]string) error {
	for path, pin := range pins {
		if err := requirePin(path, pin); err != nil {
			return err
		}
	}
	return nil
}

// BuildOracle compiles the actual pinned projection and WorkingMemory source
// separately. No producer is invoked. A missing implicit link input fails the
// proof rather than being added after the link.
func BuildOracle(o OracleBuildOptions) (r OracleBuildReceipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("genuine C++ compiler execution requires separate admission")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = OracleBuildReceipt{Schema: "nanopore-hac-output-oracle-build/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Inputs: map[string]string{}, Runtime: map[string]string{}, Objects: map[string]string{}, Dependencies: map[string][]string{}}
	r.Attempt, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(r.Attempt, "build.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	if o.BuildDir, err = filepath.Abs(o.BuildDir); err != nil {
		return r, err
	}
	if o.DoradoSource, err = filepath.Abs(o.DoradoSource); err != nil {
		return r, err
	}
	r.Toolchain, err = hacgpu.OracleToolchainForSource(o.BuildDir, "nn/FactorisedLinearLayer.cpp")
	if err != nil {
		return r, err
	}
	r.Controller, err = os.Executable()
	if err != nil {
		return r, err
	}
	r.ControllerSHA, err = runner.SHA(r.Controller)
	if err != nil {
		return r, err
	}
	r.ControllerBuild = o.ControllerBuild
	r.ControllerBuildSHA = o.ControllerBuildSHA
	custody, e := ReadControllerBuild(o.ControllerBuild, o.ControllerBuildSHA, r.Controller, r.ControllerSHA)
	if e != nil {
		return r, e
	}
	if err = pinFile(r.Inputs, r.Controller); err != nil {
		return r, err
	}
	if err = pinFile(r.Inputs, o.ControllerBuild); err != nil {
		return r, err
	}
	for path, pin := range custody.SnapshotSHA256 {
		if err = requirePin(path, pin); err != nil {
			return r, err
		}
		r.Inputs[path] = pin
	}
	for path, pin := range custody.ToolSHA256 {
		if err = requirePin(path, pin); err != nil {
			return r, err
		}
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
	}
	r.Compiler, err = exec.LookPath(r.Toolchain.Compiler)
	if err != nil {
		return r, err
	}
	if r.Compiler, err = filepath.Abs(r.Compiler); err != nil {
		return r, err
	}
	for _, p := range []string{r.Compiler, "/usr/bin/env", filepath.Join(o.BuildDir, "compile_commands.json"), filepath.Join(o.BuildDir, "build.ninja")} {
		if err = pinFile(r.Inputs, p); err != nil {
			return r, err
		}
	}
	r.RuntimeManifest = o.RuntimeManifest
	r.RuntimeManifestSHA, err = runner.SHA(o.RuntimeManifest)
	if err != nil {
		return r, err
	}
	raw, err := os.ReadFile(o.RuntimeManifest)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(raw, &r.Runtime); err != nil {
		return r, err
	}
	if len(r.Runtime) < 10 {
		return r, fmt.Errorf("explicit complete prospective runtime map required")
	}
	for path, pin := range r.Runtime {
		if _, err = readPinned(path, pin); err != nil {
			return r, err
		}
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
	}
	if err = pinFile(r.Inputs, o.RuntimeManifest); err != nil {
		return r, err
	}
	execute := func(label string, args []string) (runner.CommandReceipt, error) {
		c, e := runner.Execute(r.Attempt, label, []string{"LD_PRELOAD=", "LD_LIBRARY_PATH="}, append([]string{"/usr/bin/env", "--chdir=" + o.BuildDir}, args...)...)
		r.Commands = append(r.Commands, c)
		return c, e
	}
	query := func(label, flag string) (string, error) {
		c, e := execute(label, []string{r.Compiler, flag})
		if e != nil {
			return "", e
		}
		b, e := os.ReadFile(c.Stdout)
		if e != nil {
			return "", e
		}
		name := strings.TrimSpace(string(b))
		if name == "" || strings.ContainsAny(name, "\r\n") {
			return "", fmt.Errorf("invalid compiler resource response")
		}
		if !filepath.IsAbs(name) {
			if strings.ContainsRune(name, '/') {
				name = filepath.Join(o.BuildDir, name)
			} else {
				name, e = exec.LookPath(name)
			}
		}
		return name, e
	}
	toolFiles := []string{r.Compiler}
	for _, tool := range []string{"cc1plus", "as", "ld", "collect2"} {
		path, e := query("tool-"+tool, "-print-prog-name="+tool)
		if e != nil {
			return r, e
		}
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
		toolFiles = append(toolFiles, path)
	}
	ldd, e := exec.LookPath("ldd")
	if e != nil {
		return r, e
	}
	for _, path := range []string{ldd, "/bin/bash"} {
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
	}
	for i, tool := range toolFiles {
		c, e := execute(fmt.Sprintf("tool-runtime-%d", i), []string{ldd, tool})
		if e != nil {
			return r, e
		}
		body, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if strings.Contains(string(body), "not found") {
			return r, fmt.Errorf("compiler tool runtime is unresolved")
		}
		for _, word := range strings.Fields(string(body)) {
			if filepath.IsAbs(word) {
				if err = pinFile(r.Inputs, word); err != nil {
					return r, err
				}
			}
		}
	}
	for _, name := range []string{"Scrt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "libstdc++.so", "libc.so", "libc_nonshared.a", "libm.so"} {
		path, e := query("resource-"+strings.ReplaceAll(name, ".", "-"), "-print-file-name="+name)
		if e != nil {
			return r, e
		}
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
	}
	for _, arg := range r.Toolchain.Libraries {
		if strings.HasPrefix(arg, "-") {
			if strings.HasPrefix(arg, "-l") {
				name := strings.TrimPrefix(arg, "-l")
				file := "lib" + name + ".so"
				if strings.HasPrefix(name, ":") {
					file = name[1:]
				}
				path, e := query(fmt.Sprintf("link-lib-%d", len(r.Commands)), "-print-file-name="+file)
				if e != nil {
					if strings.HasPrefix(name, ":") {
						return r, e
					}
					path, e = query(fmt.Sprintf("link-lib-static-%d", len(r.Commands)), "-print-file-name=lib"+name+".a")
					if e != nil {
						return r, e
					}
				}
				if err = pinFile(r.Inputs, path); err != nil {
					return r, err
				}
			}
			continue
		}
		path := arg
		if !filepath.IsAbs(path) {
			path = filepath.Join(o.BuildDir, path)
		}
		if err = pinFile(r.Inputs, path); err != nil {
			return r, err
		}
	}
	resolveScript := func(input, script string) (string, error) {
		if filepath.IsAbs(input) {
			return input, nil
		}
		if !strings.HasPrefix(input, "-l") {
			candidate := filepath.Join(filepath.Dir(script), input)
			if info, e := os.Stat(candidate); e == nil && info.Mode().IsRegular() {
				return candidate, nil
			}
		}
		file := input
		if strings.HasPrefix(input, "-l") {
			name := input[2:]
			file = "lib" + name + ".so"
			if strings.HasPrefix(name, ":") {
				file = name[1:]
			}
		}
		path, e := query(fmt.Sprintf("script-input-%d", len(r.Commands)), "-print-file-name="+file)
		if e != nil && strings.HasPrefix(input, "-l") && !strings.HasPrefix(input, "-l:") {
			return query(fmt.Sprintf("script-static-%d", len(r.Commands)), "-print-file-name=lib"+input[2:]+".a")
		}
		return path, e
	}
	if err = freezeLinkerScripts(r.Inputs, resolveScript); err != nil {
		return r, err
	}
	oracle, err := filepath.Abs("verification/hacgpu/native/output_cache_oracle.cpp")
	if err != nil {
		return r, err
	}
	sources := []string{oracle, filepath.Join(o.DoradoSource, "dorado/nn/FactorisedLinearLayer.cpp"), filepath.Join(o.DoradoSource, "dorado/nn/WorkingMemory.cpp")}
	for path, want := range map[string]string{sources[1]: "2f41d8688ad3aab27c20fa257320f66e8f32eb9c29348163b7b297833a636288", sources[2]: "4b512195a11e418ce35f8a60cf050dc499cec1b3c5504fa6ce8e21d6986e2b0d"} {
		if _, err = readPinned(path, want); err != nil {
			return r, err
		}
	}
	base := append(slices.Clone(r.Toolchain.Defines), r.Toolchain.Includes...)
	base = append(base, "-O2", "-DNDEBUG", "-std=c++23", "-fPIC")
	for i, source := range sources {
		dep := filepath.Join(r.Attempt, fmt.Sprintf("pre-%d.d", i))
		args := append(slices.Clone(base), "-M", "-MT", "oracle", "-MF", dep, source)
		if _, err = execute(fmt.Sprintf("deps-%d", i), append([]string{r.Compiler}, args...)); err != nil {
			return r, err
		}
		body, e := os.ReadFile(dep)
		if e != nil {
			return r, e
		}
		deps, e := ParseDependencies(string(body), o.BuildDir)
		if e != nil {
			return r, e
		}
		r.Dependencies[source] = deps
		for _, path := range deps {
			if err = pinFile(r.Inputs, path); err != nil {
				return r, err
			}
		}
	}
	freeze, err := json.MarshalIndent(r.Inputs, "", "  ")
	if err != nil {
		return r, err
	}
	if err = os.WriteFile(filepath.Join(r.Attempt, "before-compile-inputs.json"), append(freeze, '\n'), 0644); err != nil {
		return r, err
	}
	objects := []string{}
	for i, source := range sources {
		if err = checkPins(r.Inputs); err != nil {
			return r, err
		}
		object := filepath.Join(r.Attempt, fmt.Sprintf("source-%d.o", i))
		dep := filepath.Join(r.Attempt, fmt.Sprintf("actual-%d.d", i))
		args := append(slices.Clone(base), "-MD", "-MF", dep, "-c", source, "-o", object)
		r.Compile = append(r.Compile, append([]string{r.Compiler}, args...))
		if _, err = execute(fmt.Sprintf("compile-%d", i), append([]string{r.Compiler}, args...)); err != nil {
			return r, err
		}
		body, e := os.ReadFile(dep)
		if e != nil {
			return r, e
		}
		actual, e := ParseDependencies(string(body), o.BuildDir)
		if e != nil {
			return r, e
		}
		if !slices.Equal(actual, r.Dependencies[source]) {
			return r, fmt.Errorf("actual compiler dependency set changed for %s", source)
		}
		if err = checkPins(r.Inputs); err != nil {
			return r, err
		}
		r.Objects[object], err = runner.SHA(object)
		if err != nil {
			return r, err
		}
		objects = append(objects, object)
	}
	r.Output = filepath.Join(r.Attempt, "output-cache-oracle")
	link := append(slices.Clone(objects), r.Toolchain.Libraries...)
	link = append(link, "-Wl,-t", "-Wl,-Map="+filepath.Join(r.Attempt, "link.map"), "-o", r.Output)
	r.Link = append(r.Link, append([]string{r.Compiler}, link...))
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	command, err := execute("link", append([]string{r.Compiler}, link...))
	if err != nil {
		return r, err
	}
	trace, err := os.ReadFile(command.Stdout)
	if err != nil {
		return r, err
	}
	if err = VerifyLinkTrace(string(trace), o.BuildDir, r.Inputs, r.Objects); err != nil {
		return r, err
	}
	linkMap, err := os.ReadFile(filepath.Join(r.Attempt, "link.map"))
	if err != nil {
		return r, err
	}
	if strings.Contains(string(linkMap), "libdorado_nn.a(FactorisedLinearLayer.cpp.o)") || strings.Contains(string(linkMap), "libdorado_nn.a(WorkingMemory.cpp.o)") {
		return r, fmt.Errorf("link selected an ambiguous archived projection or WorkingMemory implementation")
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Objects); err != nil {
		return r, err
	}
	r.OutputSHA, err = runner.SHA(r.Output)
	if err != nil {
		return r, err
	}
	r.Status = "closed_build_passed"
	return r, nil
}

func VerifyLinkTrace(trace, dir string, inputs, objects map[string]string) error {
	seen := 0
	for _, line := range strings.Split(trace, "\n") {
		path := strings.TrimSpace(line)
		if path == "" {
			continue
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, path)
		}
		info, e := os.Stat(path)
		if e != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("unrecognized linker trace input: %s", line)
		}
		actual, e := filepath.EvalSymlinks(path)
		if e != nil {
			return e
		}
		pin, ok := inputs[actual]
		if !ok {
			pin, ok = objects[actual]
		}
		if !ok {
			return fmt.Errorf("link used input absent from before-compile closure: %s", actual)
		}
		if e = requirePin(actual, pin); e != nil {
			return e
		}
		seen++
	}
	if seen == 0 {
		return fmt.Errorf("empty linker input trace")
	}
	return nil
}
