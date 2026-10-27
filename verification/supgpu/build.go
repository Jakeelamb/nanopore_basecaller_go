package supgpu

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

type BuildCommand struct {
	Directory string   `json:"directory"`
	Argv      []string `json:"argv"`
	Depfile   string   `json:"depfile"`
	LinkMap   string   `json:"link_map"`
}
type BuildPlan struct {
	Schema            string          `json:"schema"`
	Prepared          Pin             `json:"prepared"`
	Generated         Pin             `json:"generated"`
	Compiler          Pin             `json:"compiler"`
	Inputs            []Pin           `json:"inputs"`
	Commands          []BuildCommand  `json:"commands"`
	Reference         string          `json:"reference"`
	Observer          string          `json:"observer"`
	Audit             string          `json:"audit"`
	Snapshots         []Snapshot      `json:"snapshots"`
	Controller        Pin             `json:"controller"`
	OriginalLibraries []string        `json:"original_libraries"`
	LinkBindings      []LinkBinding   `json:"link_bindings"`
	Queries           []ToolQuery     `json:"queries"`
	Preprocess        []BuildCommand  `json:"preprocess"`
	CompilerRuntime   CompilerRuntime `json:"compiler_runtime"`
}

const nativeCompileCount = 6
const nativeCommandCount = 9
const nativeLinkCount = 3

type Snapshot struct {
	Original string `json:"original"`
	Copy     Pin    `json:"copy"`
}

// PlanBuild reads CMake's actual compile/link instructions without executing a
// compiler, preprocessor, library or device query. Separate generated objects
// satisfy the observed graph's symbols before the original static archives.
func PlanBuild(preparedPath, preparedSHA, generatedDir, buildDir, out, controllerPath, controllerSHA string) (BuildPlan, error) {
	p := BuildPlan{Schema: "nanopore-supgpu-build-plan/v2", Inputs: []Pin{}, Commands: []BuildCommand{}, Snapshots: []Snapshot{}}
	if _, err := ReadPrepared(preparedPath, preparedSHA); err != nil {
		return p, err
	}
	var err error
	p.Controller, err = pin(controllerPath)
	if err != nil {
		return p, err
	}
	if p.Controller.SHA256 != controllerSHA {
		return p, fmt.Errorf("controller manifest changed")
	}
	controller, err := readController(p.Controller, true)
	if err != nil {
		return p, err
	}
	p.Prepared, err = pin(preparedPath)
	if err != nil {
		return p, err
	}
	p.Generated, err = pin(filepath.Join(generatedDir, "generated.json"))
	if err != nil {
		return p, err
	}
	var generated Generated
	if err = readPinnedJSON(p.Generated.Path, p.Generated.SHA256, &generated); err != nil {
		return p, err
	}
	if generated.PreparedSHA256 != preparedSHA || len(generated.Files) != 7 || len(generated.Insertions) != 16 {
		return p, fmt.Errorf("incomplete observer source generation")
	}
	for _, v := range generated.Files {
		if err = verifyPin("", v); err != nil {
			return p, err
		}
		p.Inputs = append(p.Inputs, v)
	}
	for _, path := range []string{filepath.Join(buildDir, "compile_commands.json"), filepath.Join(buildDir, "build.ninja")} {
		v, e := pin(path)
		if e != nil {
			return p, e
		}
		p.Inputs = append(p.Inputs, v)
	}
	toolchain, err := hacgpu.OracleToolchain(buildDir)
	if err != nil {
		return p, err
	}
	absOut, err := filepath.Abs(out)
	if err != nil {
		return p, err
	}
	p.Reference = filepath.Join(absOut, "supgpu-reference")
	p.Observer = filepath.Join(absOut, "supgpu-observer")
	p.Audit = filepath.Join(absOut, "supgpu-loader-audit.so")
	compiler := toolchain.Compiler
	observedObjects := []string{}
	referenceObject := ""
	for _, unit := range []struct{ suffix, source, object, macro string }{
		{"basecall/model/CRFModel.cpp", "graph_oracle.cpp", "supgpu-reference-main.o", "0"},
		{"basecall/model/CRFModel.cpp", "graph_oracle.cpp", "supgpu-observed-main.o", "1"},
		{"nn/TxModules.cpp", "TxModules.cpp", "supgpu-observed-TxModules.o", "1"},
		{"nn/ConvStack.cpp", "ConvStack.cpp", "supgpu-observed-ConvStack.o", "1"},
		{"basecall/model/TxModel.cpp", "TxModel.cpp", "supgpu-observed-TxModel.o", "1"},
	} {
		flags, e := hacgpu.OracleToolchainForSource(buildDir, unit.suffix)
		if e != nil {
			return p, e
		}
		if compiler != flags.Compiler {
			return p, fmt.Errorf("mixed stock compilers")
		}
		object := filepath.Join(absOut, unit.object)
		depfile := object + ".d"
		c := BuildCommand{Directory: buildDir, Argv: []string{compiler}, Depfile: depfile, LinkMap: ""}
		c.Argv = append(c.Argv, flags.Defines...)
		c.Argv = append(c.Argv, flags.Includes...)
		c.Argv = append(c.Argv, "-O3", "-DNDEBUG", "-std=c++23", "-fPIC")
		origin, e := stockSource(buildDir, unit.suffix)
		if e != nil {
			return p, e
		}
		p.Inputs = append(p.Inputs, origin)
		c.Argv = append(c.Argv, "-iquote", filepath.Dir(origin.Path))
		include, err := filepath.Abs(generatedDir)
		if err != nil {
			return p, err
		}
		c.Argv = append(c.Argv, "-I"+include, "-DSUPGPU_OBSERVER="+unit.macro, "-MD", "-MF", depfile, "-c", filepath.Join(include, unit.source), "-o", object)
		headers, e := quotedHeaderInputs(c)
		if e != nil {
			return p, e
		}
		p.Inputs = append(p.Inputs, headers...)
		p.Commands = append(p.Commands, c)
		preDep := object + ".before.d"
		preArgs := append([]string{}, c.Argv[:len(c.Argv)-7]...)
		preArgs = append(preArgs, "-M", "-MT", object, "-MF", preDep, filepath.Join(include, unit.source))
		p.Preprocess = append(p.Preprocess, BuildCommand{Directory: buildDir, Argv: preArgs, Depfile: preDep, LinkMap: ""})
		if unit.macro == "0" {
			referenceObject = object
		} else {
			observedObjects = append(observedObjects, object)
		}
	}
	auditObject := filepath.Join(absOut, "supgpu-loader-audit.o")
	auditSource, err := filepath.Abs(filepath.Join(generatedDir, "loader_audit.c"))
	if err != nil {
		return p, err
	}
	auditDep := auditObject + ".d"
	auditFlags := []string{compiler, "-x", "c", "-std=gnu11", "-O2", "-fPIC", "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables"}
	args := append(append([]string{}, auditFlags...), "-MD", "-MF", auditDep, "-c", auditSource, "-o", auditObject)
	p.Commands = append(p.Commands, BuildCommand{buildDir, args, auditDep, ""})
	preDep := auditObject + ".before.d"
	args = append(append([]string{}, auditFlags...), "-M", "-MT", auditObject, "-MF", preDep, auditSource)
	p.Preprocess = append(p.Preprocess, BuildCommand{buildDir, args, preDep, ""})
	p.Compiler, err = pin(compiler)
	if err != nil {
		return p, err
	}
	compilerAssets, e := compilerAssetPins(buildDir)
	if e != nil {
		return p, e
	}
	p.Inputs = append(p.Inputs, compilerAssets...)
	p.CompilerRuntime, err = planCompilerRuntime(compiler)
	if err != nil {
		return p, err
	}
	p.Inputs = append(p.Inputs, p.CompilerRuntime.Inputs...)
	p.OriginalLibraries = toolchain.Libraries
	libs, bindings, queries, e := linkClosure(toolchain.Libraries, buildDir, compiler)
	if e != nil {
		return p, e
	}
	p.LinkBindings = bindings
	p.Queries = append(compilerQueries(p.CompilerRuntime), queries...)
	for _, b := range bindings {
		p.Inputs = append(p.Inputs, b.Input)
	}
	for i, objects := range [][]string{{referenceObject}, observedObjects} {
		binary := p.Reference
		if i == 1 {
			binary = p.Observer
		}
		linkmap := binary + ".map"
		argv := append([]string{compiler, "-O3", "-DNDEBUG"}, objects...)
		argv = append(argv, libs...)
		argv = append(argv, "-Wl,-t", "-Wl,-Map,"+linkmap, "-o", binary)
		p.Commands = append(p.Commands, BuildCommand{buildDir, argv, "", linkmap})
	}
	p.Commands = append(p.Commands, auditLinkCommand(buildDir, compiler, auditObject, p.Audit))
	if err = os.Mkdir(out, 0755); err != nil {
		return p, err
	}
	p.Snapshots = controller.Snapshots
	p.Inputs = append(p.Inputs, p.Controller, controller.Binary, controller.Before, controller.Dependencies, controller.Environment)
	for _, s := range controller.Snapshots {
		p.Inputs = append(p.Inputs, s.Copy)
	}
	unique := []Pin{}
	seen := map[string]bool{}
	for _, v := range p.Inputs {
		if !seen[v.Path] {
			seen[v.Path] = true
			unique = append(unique, v)
		}
	}
	p.Inputs = unique
	if err = verifyCompilerRuntime(p); err != nil {
		return p, err
	}
	if err = verifyFrozenLinkBindings(p); err != nil {
		return p, err
	}
	if err = writeJSON(filepath.Join(out, "plan.json"), p); err != nil {
		return p, err
	}
	return p, nil
}

func compilerAssetPins(buildDir string) ([]Pin, error) {
	manifest := filepath.Join(filepath.Dir(filepath.Clean(buildDir)), "evidence", "gcc15-files.sha256")
	root, err := filepath.Abs(filepath.Join(filepath.Dir(filepath.Clean(buildDir)), "toolchain-gcc15"))
	if err != nil {
		return nil, err
	}
	m, err := pin(manifest)
	if err != nil {
		return nil, err
	}
	if m.SHA256 != "edf1675775340279d7fe2aef08b51d38673549e99d54c3f9b2c67e56fa7f0497" {
		return nil, fmt.Errorf("unrecognized qualified GCC asset manifest")
	}
	out := []Pin{m}
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if len(line) < 67 || line[64:66] != "  " || !digestRE.MatchString(line[:64]) {
			return nil, fmt.Errorf("invalid GCC asset manifest")
		}
		path := line[66:]
		if !strings.HasPrefix(path, root+string(filepath.Separator)) {
			return nil, fmt.Errorf("GCC asset escapes qualified toolchain")
		}
		v, err := pin(path)
		if err != nil {
			return nil, err
		}
		if v.SHA256 != line[:64] {
			return nil, fmt.Errorf("qualified GCC asset changed: %s", path)
		}
		out = append(out, v)
	}
	for _, path := range []string{"/usr/bin/as", "/usr/bin/ld", "/etc/ld.so.cache"} {
		v, err := pin(path)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// CheckPlan is a host-only current-controller/input/header/link guard. Native
// query and preprocessing commands are observed here as data, never executed.
func CheckPlan(path, sha string) (BuildPlan, error) {
	var p BuildPlan
	if err := readPinnedJSON(path, sha, &p); err != nil {
		return p, err
	}
	if err := verifyPlan(p, true); err != nil {
		return p, err
	}
	return p, nil
}

func verifyPlan(p BuildPlan, current bool) error {
	if p.Schema != "nanopore-supgpu-build-plan/v2" || len(p.Commands) != nativeCommandCount || len(p.Preprocess) != nativeCompileCount || len(p.Queries) == 0 {
		return fmt.Errorf("incomplete prospective native plan")
	}
	c, err := readController(p.Controller, current)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(c.Snapshots, p.Snapshots) {
		return fmt.Errorf("Go dependency snapshots differ from controller")
	}
	inputs := map[string]Pin{}
	for _, v := range append(append([]Pin{}, p.Inputs...), p.Compiler, p.Generated, p.Prepared) {
		if err = verifyPin("", v); err != nil {
			return err
		}
		inputs[v.Path] = v
	}
	for i := 0; i < nativeCompileCount; i++ {
		headers, err := quotedHeaderInputs(p.Commands[i])
		if err != nil {
			return err
		}
		for _, v := range headers {
			if inputs[v.Path] != v {
				return fmt.Errorf("quoted header absent from frozen inputs")
			}
		}
	}
	if err = verifyCompilerRuntime(p); err != nil {
		return err
	}
	return verifyFrozenLinkBindings(p)
}

func auditLinkCommand(dir, compiler, object, binary string) BuildCommand {
	linkmap := binary + ".map"
	return BuildCommand{dir, []string{compiler, "-shared", "-nostdlib", "-nodefaultlibs", "-nostartfiles", object, "-Wl,-soname,supgpu-loader-audit.so", "-Wl,-t", "-Wl,-Map," + linkmap, "-o", binary}, "", linkmap}
}

// DepfileInputs parses the compiler's complete -MD dependency observation.
// A malformed or incomplete depfile cannot become a successful build receipt.
func DepfileInputs(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	body := strings.ReplaceAll(string(raw), "\\\n", "")
	_, deps, ok := strings.Cut(body, ":")
	if !ok {
		return nil, fmt.Errorf("depfile lacks target")
	}
	words := []string{}
	var word strings.Builder
	escaped := false
	s := bufio.NewReader(strings.NewReader(deps))
	for {
		r, _, e := s.ReadRune()
		if e != nil {
			break
		}
		if escaped {
			word.WriteRune(r)
			escaped = false
			continue
		}
		if r == '\\' {
			escaped = true
			continue
		}
		if r == ' ' || r == '\n' || r == '\t' {
			if word.Len() > 0 {
				words = append(words, word.String())
				word.Reset()
			}
		} else {
			word.WriteRune(r)
		}
	}
	if escaped {
		return nil, fmt.Errorf("unterminated depfile escape")
	}
	if word.Len() > 0 {
		words = append(words, word.String())
	}
	if len(words) == 0 {
		return nil, fmt.Errorf("empty depfile dependencies")
	}
	return words, nil
}
