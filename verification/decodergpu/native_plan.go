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

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const koiSHA = "17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f"
const compilerManifestSHA = "edf1675775340279d7fe2aef08b51d38673549e99d54c3f9b2c67e56fa7f0497"

type SelectedTool struct {
	Name       string
	Executable File
}
type CompilerDomain struct {
	Name string
	Root File
	ELF  nativeproof.ELFClosure
}
type CompilerClosure struct {
	Tools       []SelectedTool
	Plugins     []File
	LoaderCache File
	SearchDirs  []string
	Domains     []CompilerDomain
	Inputs      map[string]string
}
type ResourceQuery struct {
	Flag     string
	Expected File
}
type NativeRecipe struct {
	Compiler                           File
	Koi, CUDART, KoiHeader, CUDAHeader File
	CUPTI                              File
	Flags, Libraries, SearchDirs       []string
	Resources                          []ResourceQuery
	Inputs                             map[string]string
}

func nativeSourceNames(schema string) ([]string, error) {
	switch schema {
	case "nanopore-decoder-closed-build/v1":
		return []string{"oracle.cpp", "sha256.inc"}, nil
	case "nanopore-decoder-closed-build/v3":
		return []string{"oracle.cpp", "sha256.inc", "runtime_gate.inc", "image_observer.inc", "numeric_identity.inc"}, nil
	case "nanopore-decoder-closed-build/v2":
		return []string{"oracle.cpp", "sha256.inc", "runtime_gate.inc", "image_observer.inc"}, nil
	default:
		return nil, fmt.Errorf("unsupported decoder native source schema")
	}
}

func observedRecipe(t hacgpu.OracleBuildToolchain, buildDir string) (r NativeRecipe, err error) {
	r, err = minimalRecipe(t, buildDir)
	if err != nil {
		return r, err
	}
	r.CUPTI, err = artifactPin("/opt/cuda/lib64/libcupti.so")
	if err != nil {
		return r, err
	}
	for _, name := range []string{"cupti.h", "cupti_activity.h", "cupti_callbacks.h", "cupti_driver_cbid.h"} {
		if err = filePin(r.Inputs, filepath.Join("/opt/cuda/include", name)); err != nil {
			return r, err
		}
	}
	closure, err := nativeproof.CloseELF([]nativeproof.Pin{{Path: r.CUPTI.Path, SHA256: r.CUPTI.SHA256}}, []string{filepath.Dir(r.CUPTI.Path), "/usr/lib", "/lib"})
	if err != nil {
		return r, err
	}
	for _, pin := range closure.Inputs {
		if err = addPin(r.Inputs, File{pin.Path, pin.SHA256}); err != nil {
			return r, err
		}
	}
	r.Flags = append(r.Flags, "-isystem", "/opt/cuda/include")
	r.Libraries = append(r.Libraries, r.CUPTI.Path, "-Wl,-rpath,"+filepath.Dir(r.CUPTI.Path))
	return r, nil
}

type ClosedBuildPlan struct {
	Schema                       string
	Preparation                  File
	Controller                   File
	Directory, BuildDir, Source  string
	SourceBindings, SnapshotPins map[string]string
	Recipe                       NativeRecipe
	Compiler                     CompilerClosure
	Inputs                       map[string]string
}

// D01 has one standalone TU. Stock flags/archives remain provenance, but no
// Torch API or object is called and no unrelated stock library enters this link.
func minimalRecipe(t hacgpu.OracleBuildToolchain, buildDir string) (r NativeRecipe, err error) {
	r.Inputs = map[string]string{}
	r.Compiler, err = artifactPin(t.Compiler)
	if err != nil {
		return r, err
	}
	dirs := []string{}
	for i := 0; i < len(t.Includes); i++ {
		a := t.Includes[i]
		dir := ""
		if (a == "-I" || a == "-isystem") && i+1 < len(t.Includes) {
			i++
			dir = t.Includes[i]
		} else if strings.HasPrefix(a, "-I") {
			dir = a[2:]
		}
		if dir != "" {
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(buildDir, dir)
			}
			dirs = append(dirs, dir)
		}
	}
	findHeader := func(name string) (File, error) {
		for _, d := range dirs {
			if info, e := os.Stat(filepath.Join(d, name)); e == nil && info.Mode().IsRegular() {
				return artifactPin(filepath.Join(d, name))
			}
		}
		return File{}, fmt.Errorf("required standalone header absent: %s", name)
	}
	r.KoiHeader, err = findHeader("koi.h")
	if err != nil {
		return r, err
	}
	if r.KoiHeader.SHA256 != "8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88" {
		return r, fmt.Errorf("selected Koi header changed")
	}
	r.CUDAHeader, err = findHeader("cuda_runtime_api.h")
	if err != nil {
		return r, err
	}
	files, _, err := LinkReferences(t.Libraries)
	if err != nil {
		return r, err
	}
	for _, name := range files {
		if !filepath.IsAbs(name) {
			name = filepath.Join(buildDir, name)
		}
		base := filepath.Base(name)
		if base == "libkoi.a" {
			if r.Koi.Path != "" {
				return r, fmt.Errorf("ambiguous Koi archive")
			}
			r.Koi, err = artifactPin(name)
		}
		if base == "libcudart.so" {
			if r.CUDART.Path != "" {
				return r, fmt.Errorf("ambiguous CUDA runtime")
			}
			r.CUDART, err = artifactPin(name)
		}
		if err != nil {
			return r, err
		}
	}
	if r.Koi.SHA256 != koiSHA || r.CUDART.Path == "" {
		return r, fmt.Errorf("fixed original Koi and dynamic CUDA runtime required")
	}
	r.Flags = []string{"-O2", "-DNDEBUG", "-std=c++23", "-fPIC", "-fno-lto", "-I" + filepath.Dir(r.KoiHeader.Path), "-isystem", filepath.Dir(r.CUDAHeader.Path)}
	r.Libraries = []string{"-pie", "-fno-lto", r.Koi.Path, r.CUDART.Path, "-Wl,-rpath," + filepath.Dir(r.CUDART.Path)}
	usr := filepath.Dir(filepath.Dir(r.Compiler.Path))
	matches, e := filepath.Glob(filepath.Join(usr, "lib/gcc/*/*/crtbeginS.o"))
	if e != nil || len(matches) != 1 {
		return r, fmt.Errorf("qualified GCC resource directory ambiguous")
	}
	r.SearchDirs = []string{filepath.Dir(matches[0]), filepath.Join(usr, "lib"), "/usr/lib"}
	for _, name := range []string{"Scrt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "libstdc++.so", "libc.so", "libc_nonshared.a", "libm.so"} {
		p, e := resolveFile(name, r.SearchDirs)
		if e != nil {
			return r, e
		}
		r.Resources = append(r.Resources, ResourceQuery{"-print-file-name=" + name, p})
		if err = addPin(r.Inputs, p); err != nil {
			return r, err
		}
	}
	for _, p := range []File{r.Compiler, r.Koi, r.CUDART, r.KoiHeader, r.CUDAHeader} {
		if err = addPin(r.Inputs, p); err != nil {
			return r, err
		}
	}
	resolveScript := func(name, from string) (string, error) {
		if filepath.IsAbs(name) {
			return name, nil
		}
		if !strings.HasPrefix(name, "-l") {
			if info, e := os.Stat(filepath.Join(filepath.Dir(from), name)); e == nil && info.Mode().IsRegular() {
				return filepath.Join(filepath.Dir(from), name), nil
			}
		}
		p, e := resolveLibrary(name, r.SearchDirs)
		return p.Path, e
	}
	if err = ExpandLinkerScripts(r.Inputs, resolveScript); err != nil {
		return r, err
	}
	return r, nil
}
func resolveLibrary(name string, dirs []string) (File, error) {
	names := []string{name}
	if strings.HasPrefix(name, "-l:") {
		names = []string{name[3:]}
	} else if strings.HasPrefix(name, "-l") {
		names = []string{"lib" + name[2:] + ".so", "lib" + name[2:] + ".a"}
	}
	for _, dir := range dirs {
		for _, file := range names {
			p := filepath.Join(dir, file)
			if info, e := os.Stat(p); e == nil && info.Mode().IsRegular() {
				return artifactPin(p)
			}
		}
	}
	return File{}, fmt.Errorf("prospective linker-script library unresolved: %s", name)
}
func resolveFile(name string, dirs []string) (File, error) {
	if filepath.IsAbs(name) {
		return artifactPin(name)
	}
	for _, d := range dirs {
		p := filepath.Join(d, name)
		if info, e := os.Stat(p); e == nil && info.Mode().IsRegular() {
			return artifactPin(p)
		}
	}
	return File{}, fmt.Errorf("prospective resource unresolved: %s", name)
}

// All selected tool/plugin ELF dependencies are frozen by pure metadata reads
// before the first compiler child. No ldd, loader, compiler or probe is invoked.
func compilerClosure(driver File) (c CompilerClosure, err error) {
	c.Inputs = map[string]string{}
	c.SearchDirs = []string{"/usr/lib", "/lib"}
	usr := filepath.Dir(filepath.Dir(driver.Path))
	matches, e := filepath.Glob(filepath.Join(usr, "lib/gcc/*/*/cc1plus"))
	if e != nil || len(matches) != 1 {
		return c, fmt.Errorf("qualified cc1plus directory ambiguous")
	}
	gcc := filepath.Dir(matches[0])
	for _, t := range []struct{ name, path string }{{"driver", driver.Path}, {"cc1plus", matches[0]}, {"collect2", filepath.Join(gcc, "collect2")}, {"as", "/usr/bin/as"}, {"ld", "/usr/bin/ld"}, {"nm", "/usr/bin/nm"}, {"lto-wrapper", filepath.Join(gcc, "lto-wrapper")}, {"lto1", filepath.Join(gcc, "lto1")}, {"cc1", filepath.Join(gcc, "cc1")}, {"gcc-driver", filepath.Join(usr, "bin/gcc-15")}} {
		p, e := artifactPin(t.path)
		if e != nil {
			return c, e
		}
		c.Tools = append(c.Tools, SelectedTool{t.name, p})
	}
	plugin, e := artifactPin(filepath.Join(gcc, "liblto_plugin.so"))
	if e != nil {
		return c, e
	}
	c.Plugins = []File{plugin}
	bfd, e := filepath.Glob("/usr/lib/bfd-plugins/*.so*")
	if e != nil {
		return c, e
	}
	for _, path := range bfd {
		p, e := artifactPin(path)
		if e != nil {
			return c, e
		}
		if !slices.Contains(c.Plugins, p) {
			c.Plugins = append(c.Plugins, p)
		}
	}
	// This is a compiler process family, not one loader namespace: GCC's
	// selected plugin and a system nm/BFD plugin may have the same SONAME.
	// Preserve every absolute root and its own bindings; ambiguity inside any
	// one prospective loader domain still rejects in CloseELF.
	roots := []SelectedTool{}
	roots = append(roots, c.Tools...)
	for _, p := range c.Plugins {
		roots = append(roots, SelectedTool{"plugin:" + p.Path, p})
	}
	for _, root := range roots {
		closure, e := nativeproof.CloseELF([]nativeproof.Pin{{Path: root.Executable.Path, SHA256: root.Executable.SHA256}}, c.SearchDirs)
		if e != nil {
			return c, e
		}
		if e = checkHostELF(closure); e != nil {
			return c, e
		}
		c.Domains = append(c.Domains, CompilerDomain{root.Name, root.Executable, closure})
		for _, p := range closure.Inputs {
			if err = addPin(c.Inputs, File{p.Path, p.SHA256}); err != nil {
				return c, err
			}
		}
	}
	c.LoaderCache, err = artifactPin("/etc/ld.so.cache")
	if err != nil {
		return c, err
	}
	if err = addPin(c.Inputs, c.LoaderCache); err != nil {
		return c, err
	}
	return c, nil
}
func checkHostELF(c nativeproof.ELFClosure) error {
	for _, node := range c.Nodes {
		if node.Class != elf.ELFCLASS64 || node.Data != elf.ELFDATA2LSB || node.Machine != elf.EM_X86_64 {
			return fmt.Errorf("D01 native object is outside little-endian x86_64: %s", node.Pin.Path)
		}
	}
	return nil
}
func compilerAssets(buildDir string) (map[string]string, error) {
	pins := map[string]string{}
	path := filepath.Join(filepath.Dir(buildDir), "evidence/gcc15-files.sha256")
	p, err := artifactPin(path)
	if err != nil {
		return nil, err
	}
	if p.SHA256 != compilerManifestSHA {
		return nil, fmt.Errorf("qualified compiler manifest changed")
	}
	if err = addPin(pins, p); err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	root := filepath.Join(filepath.Dir(buildDir), "toolchain-gcc15")
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if len(line) < 67 || line[64:66] != "  " || !strings.HasPrefix(line[66:], root+"/") {
			return nil, fmt.Errorf("invalid qualified compiler asset")
		}
		if err = addPin(pins, File{line[66:], line[:64]}); err != nil {
			return nil, err
		}
	}
	return pins, nil
}

func PlanClosedBuild(preparation File, buildDir, out string) (p ClosedBuildPlan, err error) {
	preparation, err = normalizeAnchor(preparation)
	if err != nil {
		return p, err
	}
	r, err := ReadPreparation(preparation, true)
	if err != nil {
		return p, err
	}
	buildDir, err = filepath.Abs(buildDir)
	if err != nil {
		return p, err
	}
	p = ClosedBuildPlan{Schema: "nanopore-decoder-closed-build/v2", Preparation: preparation, Controller: File{r.Go.Controller, r.Go.ControllerSHA}, BuildDir: buildDir, Inputs: map[string]string{}}
	if r.Schema == "nanopore-decoder-fixed-hac-preparation/v3" {
		p.Schema = "nanopore-decoder-closed-build/v3"
	}
	t, err := hacgpu.OracleToolchainForSource(buildDir, "basecall/decode/CUDADecoder.cpp")
	if err != nil {
		return p, err
	}
	p.Recipe, err = observedRecipe(t, buildDir)
	if err == nil && numericBuild(p) {
		p.Recipe.Flags = append(p.Recipe.Flags, "-DD01_NUMERIC_IDENTITY=1")
	}
	if err != nil {
		return p, err
	}
	p.Compiler, err = compilerClosure(p.Recipe.Compiler)
	if err != nil {
		return p, err
	}
	p.Inputs, err = compilerAssets(buildDir)
	if err != nil {
		return p, err
	}
	for _, inputs := range []map[string]string{p.Recipe.Inputs, p.Compiler.Inputs} {
		maps.Copy(p.Inputs, inputs)
	}
	for _, path := range []string{filepath.Join(buildDir, "compile_commands.json"), filepath.Join(buildDir, "build.ninja")} {
		if err = filePin(p.Inputs, path); err != nil {
			return p, err
		}
	}
	p.Directory, err = runner.NewAttempt(out)
	if err != nil {
		return p, err
	}
	sources := map[string]string{}
	names, _ := nativeSourceNames(p.Schema)
	for _, name := range names {
		if err = filePin(sources, filepath.Join("verification/decodergpu/native", name)); err != nil {
			return p, err
		}
	}
	p.SourceBindings, p.SnapshotPins, err = snapshotPins(sources, filepath.Join(p.Directory, "native-inputs"))
	if err != nil {
		return p, err
	}
	// Preserve include spelling in the immutable source directory.
	for original, copy := range p.SourceBindings {
		name := filepath.Join(p.Directory, "native-inputs", filepath.Base(original))
		body, e := os.ReadFile(copy)
		if e != nil {
			return p, e
		}
		if e = os.WriteFile(name, body, 0400); e != nil {
			return p, e
		}
		if err = filePin(p.SnapshotPins, name); err != nil {
			return p, err
		}
		if filepath.Base(original) == "oracle.cpp" {
			p.Source = name
		}
	}
	maps.Copy(p.Inputs, sources)
	maps.Copy(p.Inputs, p.SnapshotPins)
	if err = checkPins(p.Inputs); err != nil {
		return p, err
	}
	if err = writeJSON(filepath.Join(p.Directory, "plan.json"), p); err != nil {
		return p, err
	}
	return p, nil
}

func verifyClosedPlan(p ClosedBuildPlan) error {
	return verifyClosedPlanMode(p, true)
}

func verifyClosedPlanMode(p ClosedBuildPlan, current bool) error {
	return verifyClosedPlanModeScoped(p, current, nil)
}

func verifyClosedPlanModeScoped(p ClosedBuildPlan, current bool, proofs *numericProofs) error {
	names, err := nativeSourceNames(p.Schema)
	if err != nil || p.Source == "" {
		return fmt.Errorf("invalid finite D01 build plan")
	}
	r, err := readPreparation(p.Preparation, current, proofs)
	if err != nil {
		return err
	}
	if (p.Schema == "nanopore-decoder-closed-build/v3") != (r.Schema == "nanopore-decoder-fixed-hac-preparation/v3") {
		return fmt.Errorf("numeric producer/preparation contract differs")
	}
	if p.Controller != (File{r.Go.Controller, r.Go.ControllerSHA}) {
		return fmt.Errorf("plan/controller binding changed")
	}
	t, err := hacgpu.OracleToolchainForSource(p.BuildDir, "basecall/decode/CUDADecoder.cpp")
	if err != nil {
		return err
	}
	recipe, err := minimalRecipe(t, p.BuildDir)
	if p.Schema == "nanopore-decoder-closed-build/v2" || p.Schema == "nanopore-decoder-closed-build/v3" {
		recipe, err = observedRecipe(t, p.BuildDir)
		if err == nil && numericBuild(p) {
			recipe.Flags = append(recipe.Flags, "-DD01_NUMERIC_IDENTITY=1")
		}
	}
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(recipe, p.Recipe) {
		return fmt.Errorf("standalone D01 compiler/link recipe changed")
	}
	assets, err := compilerAssets(p.BuildDir)
	if err != nil {
		return err
	}
	c, err := compilerClosure(p.Recipe.Compiler)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(c, p.Compiler) {
		return fmt.Errorf("compiler selected tool/plugin/ELF closure changed")
	}
	for _, inputs := range []map[string]string{assets, p.Compiler.Inputs, p.Recipe.Inputs, p.SnapshotPins} {
		for path, sha := range inputs {
			if p.Inputs[path] != sha {
				return fmt.Errorf("required before-query input omitted: %s", path)
			}
		}
	}
	if p.Source != filepath.Join(p.Directory, "native-inputs/oracle.cpp") || len(p.SourceBindings) != len(names) {
		return fmt.Errorf("immutable native source selection changed")
	}
	if err = verifyNativeSourceMembership(p.SourceBindings, r.Go, names, current); err != nil {
		return err
	}
	expectedInputs := maps.Clone(assets)
	maps.Copy(expectedInputs, p.Compiler.Inputs)
	maps.Copy(expectedInputs, p.Recipe.Inputs)
	maps.Copy(expectedInputs, p.SnapshotPins)
	for _, path := range []string{filepath.Join(p.BuildDir, "compile_commands.json"), filepath.Join(p.BuildDir, "build.ninja")} {
		if err = filePin(expectedInputs, path); err != nil {
			return err
		}
	}
	for original, copy := range p.SourceBindings {
		want, ok := r.Inputs[original]
		if !ok || p.Inputs[original] != want || p.SnapshotPins[copy] != want || p.SnapshotPins[filepath.Join(p.Directory, "native-inputs", filepath.Base(original))] != want {
			return fmt.Errorf("native snapshot differs from preparation: %s", original)
		}
		expectedInputs[original] = want
	}
	if !maps.Equal(expectedInputs, p.Inputs) {
		return fmt.Errorf("native input set differs from complete reconstructed recipe")
	}
	if err = checkProducerInputs(r.Inputs, p, current); err != nil {
		return err
	}
	return checkProducerInputs(p.Inputs, p, current)
}
func historicalProducerRoot(g GoClosure) (string, error) {
	if len(g.Commands) != 3 {
		return "", fmt.Errorf("historical Go environment receipt absent")
	}
	command := g.Commands[1]
	var environment map[string]string
	if err := readJSON(File{command.Stdout, command.StdoutSHA}, &environment); err != nil {
		return "", err
	}
	module := environment["GOMOD"]
	actual, bound := g.Selection[module]
	if !filepath.IsAbs(module) || filepath.Clean(module) != module || filepath.Base(module) != "go.mod" || !bound || !filepath.IsAbs(actual) || filepath.Clean(actual) != actual || filepath.Base(actual) != "go.mod" || g.Pins[actual] == "" {
		return "", fmt.Errorf("historical canonical Go module selection/content pin absent")
	}
	return filepath.Dir(actual), nil
}

func verifyNativeSourceMembership(bindings map[string]string, g GoClosure, names []string, current bool) error {
	producerRoot := ""
	if !current {
		var err error
		producerRoot, err = historicalProducerRoot(g)
		if err != nil {
			return err
		}
	}
	for _, name := range names {
		original := filepath.Join(producerRoot, "verification/decodergpu/native", name)
		if current {
			var err error
			original, err = canonical(original)
			if err != nil {
				return err
			}
		}
		if _, ok := bindings[original]; !ok {
			return fmt.Errorf("required native source binding omitted: %s", name)
		}
	}
	return nil
}

func compilerClosuresEqual(a, b CompilerClosure) bool { return reflect.DeepEqual(a, b) }

// CheckClosedBuildPlan exercises the exact pre-query proof without acquiring a
// native slot or invoking any command. Rehashed omission controls use this path.
func CheckClosedBuildPlan(anchor File) error {
	anchor, err := normalizeAnchor(anchor)
	if err != nil {
		return err
	}
	var p ClosedBuildPlan
	if err = readJSON(anchor, &p); err != nil {
		return err
	}
	return verifyClosedPlan(p)
}
