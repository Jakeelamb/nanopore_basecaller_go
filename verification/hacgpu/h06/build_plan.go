package h06

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

type BuildUnit struct {
	Name   string
	Source nativeproof.Pin
	Flags  []string
}
type BuildTarget struct {
	Name  string
	Units []string
}
type BuildPlan struct {
	Schema, BuildDir                                     string
	Prepared, CompilerContract, LinkContract, Controller nativeproof.Pin
	Compiler                                             compilerClosure
	Driver                                               nativeproof.Pin
	Libraries                                            []string
	Queries                                              []linkQuery
	Units                                                []BuildUnit
	Targets                                              []BuildTarget
	Inputs                                               map[string]string
}
type linkQuery struct {
	Argv     []string `json:"argv"`
	Expected string   `json:"expected"`
}
type selectedTool struct {
	Name       string
	Executable nativeproof.Pin
}
type compilerDomain struct {
	Name string
	Root nativeproof.Pin
	ELF  nativeproof.ELFClosure
}
type compilerClosure struct {
	Tools       []selectedTool
	Plugins     []nativeproof.Pin
	LoaderCache nativeproof.Pin
	SearchDirs  []string
	Domains     []compilerDomain
	Inputs      map[string]string
}
type compilerPlanContract struct {
	Schema, BuildDir string
	Compiler         compilerClosure
	Recipe           struct{ Compiler nativeproof.Pin }
}
type linkContract struct {
	Schema   string            `json:"schema"`
	Compiler nativeproof.Pin   `json:"compiler"`
	Original []string          `json:"original_libraries"`
	Inputs   []nativeproof.Pin `json:"inputs"`
	Bindings []struct {
		Input nativeproof.Pin `json:"input"`
	} `json:"link_bindings"`
	Queries  []linkQuery `json:"queries"`
	Commands []struct {
		Directory string   `json:"directory"`
		Argv      []string `json:"argv"`
		LinkMap   string   `json:"link_map"`
	} `json:"commands"`
}
type BuildPlanOptions struct {
	Prepared, CompilerContract, LinkContract, Controller nativeproof.Pin
	BuildDir                                             string
}

func buildPin(path string) (nativeproof.Pin, error) {
	actual, err := filepath.Abs(path)
	if err != nil {
		return nativeproof.Pin{}, err
	}
	actual, err = filepath.EvalSymlinks(actual)
	if err != nil {
		return nativeproof.Pin{}, err
	}
	info, err := os.Stat(actual)
	if err != nil {
		return nativeproof.Pin{}, err
	}
	if !info.Mode().IsRegular() {
		return nativeproof.Pin{}, fmt.Errorf("regular build input required: %s", actual)
	}
	return artifactPin(actual)
}
func checkBuildPin(p nativeproof.Pin) error {
	got, err := buildPin(p.Path)
	if err != nil {
		return err
	}
	if p.SHA256 == "" || got.SHA256 != p.SHA256 {
		return fmt.Errorf("H06 pinned bytes changed: %s", p.Path)
	}
	return nil
}
func readBuildJSON(p nativeproof.Pin, value any) error {
	if err := checkBuildPin(p); err != nil {
		return err
	}
	body, err := os.ReadFile(p.Path)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, value)
}
func addBuildPin(pins map[string]string, p nativeproof.Pin) error {
	if err := checkBuildPin(p); err != nil {
		return err
	}
	actual, err := filepath.EvalSymlinks(p.Path)
	if err != nil {
		return err
	}
	if old, ok := pins[actual]; ok && old != p.SHA256 {
		return fmt.Errorf("conflicting H06 build input: %s", actual)
	}
	pins[actual] = p.SHA256
	return nil
}
func checkBuildPins(pins map[string]string) error {
	for _, path := range slices.Sorted(maps.Keys(pins)) {
		if err := checkBuildPin(nativeproof.Pin{Path: path, SHA256: pins[path]}); err != nil {
			return err
		}
	}
	return nil
}

// This consumes an independently pinned selected-tool contract. Each loader
// domain is rederived by the shared metadata reader before native execution.
func checkCompilerContract(c compilerClosure, inputs map[string]string) error {
	if len(c.Tools) != 10 || len(c.Plugins) < 1 || len(c.Domains) != len(c.Tools)+len(c.Plugins) || c.LoaderCache.Path != "/etc/ld.so.cache" || !slices.Equal(c.SearchDirs, []string{"/usr/lib", "/lib"}) {
		return fmt.Errorf("incomplete selected H06 compiler contract")
	}
	if err := checkCompilerDomains(c, inputs); err != nil {
		return err
	}
	plugins := []nativeproof.Pin{{Path: c.Plugins[0].Path, SHA256: c.Plugins[0].SHA256}}
	paths, err := filepath.Glob("/usr/lib/bfd-plugins/*.so*")
	if err != nil {
		return err
	}
	for _, path := range paths {
		pin, e := buildPin(path)
		if e != nil {
			return e
		}
		if !slices.Contains(plugins, pin) {
			plugins = append(plugins, pin)
		}
	}
	got := []nativeproof.Pin{}
	for _, pin := range c.Plugins {
		got = append(got, nativeproof.Pin{Path: pin.Path, SHA256: pin.SHA256})
	}
	sortPins := func(p []nativeproof.Pin) {
		slices.SortFunc(p, func(a, b nativeproof.Pin) int { return strings.Compare(a.Path, b.Path) })
	}
	sortPins(plugins)
	sortPins(got)
	if !slices.Equal(plugins, got) {
		return fmt.Errorf("selected BFD plugin discovery family changed")
	}
	return nil
}

func checkCompilerDomains(c compilerClosure, inputs map[string]string) error {
	roots := map[string]nativeproof.Pin{}
	for _, t := range c.Tools {
		roots[t.Name] = nativeproof.Pin{Path: t.Executable.Path, SHA256: t.Executable.SHA256}
	}
	for _, name := range []string{"driver", "cc1plus", "collect2", "as", "ld", "nm", "lto-wrapper", "lto1", "cc1", "gcc-driver"} {
		if roots[name].Path == "" {
			return fmt.Errorf("selected compiler tool omitted: %s", name)
		}
	}
	for _, p := range c.Plugins {
		roots["plugin:"+p.Path] = nativeproof.Pin{Path: p.Path, SHA256: p.SHA256}
	}
	if len(roots) != len(c.Domains) {
		return fmt.Errorf("duplicate compiler tool/plugin domain")
	}
	union := map[string]string{}
	for _, d := range c.Domains {
		root, ok := roots[d.Name]
		if !ok || root != (nativeproof.Pin{Path: d.Root.Path, SHA256: d.Root.SHA256}) {
			return fmt.Errorf("compiler domain root changed")
		}
		delete(roots, d.Name)
		actual, err := nativeproof.CloseELF([]nativeproof.Pin{root}, c.SearchDirs)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(actual, d.ELF) {
			return fmt.Errorf("selected compiler ELF domain changed: %s", d.Name)
		}
		for _, node := range actual.Nodes {
			if node.Class != elf.ELFCLASS64 || node.Data != elf.ELFDATA2LSB || node.Machine != elf.EM_X86_64 {
				return fmt.Errorf("compiler is outside x86_64")
			}
		}
		for _, pin := range actual.Inputs {
			union[pin.Path] = pin.SHA256
		}
	}
	if err := addBuildPin(union, nativeproof.Pin{Path: c.LoaderCache.Path, SHA256: c.LoaderCache.SHA256}); err != nil {
		return err
	}
	if !maps.Equal(union, c.Inputs) {
		return fmt.Errorf("compiler input union differs from selected domains and cache")
	}
	for path, sha := range union {
		if inputs[path] != sha {
			return fmt.Errorf("compiler input omitted before query: %s", path)
		}
	}
	return nil
}

// PlanBuild reads only pinned contract data, source and ELF metadata. It never
// invokes a compiler, preprocessor, linker, loader, or model.
func PlanBuild(o BuildPlanOptions) (p BuildPlan, err error) {
	p = BuildPlan{Schema: "nanopore-hac-half-build-plan/v1", Prepared: o.Prepared, CompilerContract: o.CompilerContract, LinkContract: o.LinkContract, Controller: o.Controller, Inputs: map[string]string{}}
	p.BuildDir, err = filepath.Abs(o.BuildDir)
	if err != nil {
		return p, err
	}
	var prepared Prepared
	if err = readBuildJSON(o.Prepared, &prepared); err != nil {
		return p, err
	}
	if prepared.Schema != "nanopore-hac-half-prepared/v2" || !prepared.Generated.Reversible || prepared.Generated.SourceSHA != FLSTMSourceSHA {
		return p, fmt.Errorf("fixed prepared H06 source required")
	}
	original, err := os.ReadFile(prepared.Generated.Source)
	if err != nil {
		return p, err
	}
	observed, insertions, err := observedSource(original)
	if err != nil {
		return p, err
	}
	if digest(original) != FLSTMSourceSHA || digest(observed) != prepared.Generated.ObservedSHA || !reflect.DeepEqual(insertions, prepared.Generated.Insertions) {
		return p, fmt.Errorf("prepared observer is not the reversible source insertion")
	}
	var compiler compilerPlanContract
	if err = readBuildJSON(o.CompilerContract, &compiler); err != nil {
		return p, err
	}
	if compiler.Schema != "nanopore-decoder-closed-build/v1" || compiler.BuildDir != p.BuildDir {
		return p, fmt.Errorf("accepted compiler contract/build directory differs")
	}
	p.Compiler = compiler.Compiler
	p.Driver = nativeproof.Pin{Path: compiler.Recipe.Compiler.Path, SHA256: compiler.Recipe.Compiler.SHA256}
	driverMatches := false
	for _, tool := range p.Compiler.Tools {
		if tool.Name == "driver" && tool.Executable == p.Driver {
			driverMatches = true
		}
	}
	if !driverMatches {
		return p, fmt.Errorf("recipe driver is outside selected compiler domains")
	}
	var link linkContract
	if err = readBuildJSON(o.LinkContract, &link); err != nil {
		return p, err
	}
	if link.Schema != "nanopore-supgpu-build-plan/v1" || link.Compiler != p.Driver {
		return p, fmt.Errorf("closed Torch link contract/compiler differs")
	}
	t, err := hacgpu.OracleToolchainForSource(p.BuildDir, "nn/FLSTMStack.cpp")
	if err != nil {
		return p, err
	}
	if t.Compiler != p.Driver.Path || !slices.Equal(t.Libraries, link.Original) {
		return p, fmt.Errorf("stock compiler/link selection changed")
	}
	for path, sha := range p.Compiler.Inputs {
		p.Inputs[path] = sha
	}
	for _, binding := range link.Bindings {
		if err = addBuildPin(p.Inputs, binding.Input); err != nil {
			return p, err
		}
	}
	linkInputs := map[string]string{}
	for _, pin := range link.Inputs {
		linkInputs[pin.Path] = pin.SHA256
	}
	for _, q := range link.Queries {
		if len(q.Argv) != 2 || q.Argv[0] != p.Driver.Path || !(strings.HasPrefix(q.Argv[1], "-print-file-name=") || strings.HasPrefix(q.Argv[1], "-print-prog-name=")) {
			return p, fmt.Errorf("closed link query is not a driver resource query")
		}
	}
	p.Queries, err = bindSelectedQueries(link.Queries, linkInputs, p.Inputs)
	if err != nil {
		return p, err
	}
	for _, command := range link.Commands {
		if command.LinkMap == "" {
			continue
		}
		start := len(command.Argv) - 4 - len(link.Original)
		if command.Directory != p.BuildDir || start < 2 || command.Argv[0] != p.Driver.Path || command.Argv[len(command.Argv)-4] != "-Wl,-t" || command.Argv[len(command.Argv)-2] != "-o" {
			return p, fmt.Errorf("closed link template differs")
		}
		libs := command.Argv[start : len(command.Argv)-4]
		if p.Libraries != nil && !slices.Equal(p.Libraries, libs) {
			return p, fmt.Errorf("closed link targets use different libraries")
		}
		p.Libraries = slices.Clone(libs)
	}
	if len(p.Libraries) == 0 {
		return p, fmt.Errorf("closed Torch link template absent")
	}
	files, e := closedLinkFiles(p.Libraries)
	if e != nil {
		return p, e
	}
	for _, path := range files {
		actual, e := buildPin(path)
		if e != nil {
			return p, e
		}
		if p.Inputs[actual.Path] != actual.SHA256 {
			return p, fmt.Errorf("link file absent from closed bindings: %s", path)
		}
	}
	for _, pin := range []nativeproof.Pin{o.Prepared, o.CompilerContract, o.LinkContract, o.Controller, prepared.Main, prepared.Helper, {Path: prepared.Generated.Source, SHA256: prepared.Generated.SourceSHA}, {Path: prepared.Generated.Observed, SHA256: prepared.Generated.ObservedSHA}, {Path: prepared.Generated.Header, SHA256: prepared.Generated.HeaderSHA}} {
		if err = addBuildPin(p.Inputs, pin); err != nil {
			return p, err
		}
	}
	controller, e := ReadControllerBuild(o.Controller, true)
	if e != nil {
		return p, e
	}
	for path, sha := range controller.Inputs {
		if err = addBuildPin(p.Inputs, nativeproof.Pin{Path: path, SHA256: sha}); err != nil {
			return p, err
		}
	}
	for path, copy := range controller.Snapshots {
		if err = addBuildPin(p.Inputs, nativeproof.Pin{Path: copy, SHA256: controller.Inputs[path]}); err != nil {
			return p, err
		}
	}
	if err = addBuildPin(p.Inputs, controller.Binary); err != nil {
		return p, err
	}
	for _, path := range []string{filepath.Join(p.BuildDir, "compile_commands.json"), filepath.Join(p.BuildDir, "build.ninja")} {
		pin, e := buildPin(path)
		if e != nil {
			return p, e
		}
		if err = addBuildPin(p.Inputs, pin); err != nil {
			return p, err
		}
	}
	base := append(slices.Clone(t.Defines), t.Includes...)
	base = append(base, "-O2", "-DNDEBUG", "-std=c++23", "-fPIC", "-I"+filepath.Dir(prepared.Generated.Header))
	p.Units = []BuildUnit{{Name: "reference-main", Source: prepared.Main, Flags: append(slices.Clone(base), "-DH06_OBSERVED=0")}, {Name: "observed-main", Source: prepared.Main, Flags: append(slices.Clone(base), "-DH06_OBSERVED=1")}, {Name: "observed-flstm", Source: nativeproof.Pin{Path: prepared.Generated.Observed, SHA256: prepared.Generated.ObservedSHA}, Flags: append(slices.Clone(base), "-iquote", filepath.Dir(prepared.Generated.Source))}}
	p.Targets = []BuildTarget{{Name: "reference", Units: []string{"reference-main"}}, {Name: "observed", Units: []string{"observed-main", "observed-flstm"}}}
	if err = checkCompilerContract(p.Compiler, p.Inputs); err != nil {
		return p, err
	}
	return p, checkBuildPins(p.Inputs)
}

func addClosedQueryInput(lookup string, accepted, inputs map[string]string) error {
	actual, err := buildPin(lookup)
	if err != nil {
		return err
	}
	// S00 preserves linker lookup spelling, including symlinks. Authenticate
	// that accepted lookup's bytes before storing H06's canonical interval key.
	want := accepted[lookup]
	if want == "" {
		want = inputs[actual.Path]
	}
	if want == "" || want != actual.SHA256 {
		return fmt.Errorf("compiler resource differs from accepted link input: %s", lookup)
	}
	return addBuildPin(inputs, nativeproof.Pin{Path: lookup, SHA256: want})
}

func bindSelectedQueries(queries []linkQuery, accepted, inputs map[string]string) ([]linkQuery, error) {
	var selected []linkQuery
	for _, q := range queries {
		if !filepath.IsAbs(q.Expected) {
			if strings.HasPrefix(q.Argv[1], "-print-prog-name=") {
				return nil, fmt.Errorf("selected compiler program is unresolved")
			}
			continue
		}
		known := accepted[q.Expected] != ""
		if !known {
			actual, err := filepath.EvalSymlinks(q.Expected)
			if err == nil {
				known = inputs[actual] != ""
			}
		}
		// The source plan also probes unselected .a/.so alternatives. H06 links
		// its resolved absolute recipe and does not need those filename probes.
		if !known {
			continue
		}
		if err := addClosedQueryInput(q.Expected, accepted, inputs); err != nil {
			return nil, err
		}
		selected = append(selected, q)
	}
	return selected, nil
}

// The external contract already resolved its library search. This accepts only
// those absolute files and the fixed source linker controls, not a new search.
func closedLinkFiles(args []string) ([]string, error) {
	var files []string
	for _, arg := range args {
		if filepath.IsAbs(arg) {
			files = append(files, arg)
			continue
		}
		if !strings.HasPrefix(arg, "-Wl,") {
			return nil, fmt.Errorf("unresolved closed linker argument: %s", arg)
		}
		parts := strings.Split(arg[4:], ",")
		for i := 0; i < len(parts); i++ {
			part := parts[i]
			switch part {
			case "-rpath":
				i++
				if i == len(parts) {
					return nil, fmt.Errorf("linker rpath value absent")
				}
			case "--as-needed", "--no-as-needed":
			default:
				if !filepath.IsAbs(part) {
					return nil, fmt.Errorf("unresolved wrapped linker input: %s", part)
				}
				files = append(files, part)
			}
		}
	}
	return files, nil
}

// CheckBuildPlan reconstructs the complete recipe from independent anchors.
func CheckBuildPlan(p BuildPlan) error {
	expected, err := PlanBuild(BuildPlanOptions{Prepared: p.Prepared, CompilerContract: p.CompilerContract, LinkContract: p.LinkContract, Controller: p.Controller, BuildDir: p.BuildDir})
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expected, p) {
		return fmt.Errorf("H06 build plan differs from pinned source/compiler/link contracts")
	}
	return nil
}
