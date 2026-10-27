package hacgpu

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// BuildManifest records exactly what produced the oracle binary.
type BuildManifest struct {
	Schema      string            `json:"schema"`
	Status      string            `json:"status"`
	ABI         int               `json:"abi_version"`
	Started     string            `json:"started_utc"`
	Finished    string            `json:"finished_utc"`
	BuildDir    string            `json:"build_dir"`
	DoradoSourc string            `json:"dorado_source"`
	Compiler    string            `json:"compiler"`
	Output      string            `json:"output"`
	OutputSHA   string            `json:"output_sha256"`
	Inputs      map[string]string `json:"input_sha256"`
	Compile     []string          `json:"compile_argv"`
	Link        []string          `json:"link_argv"`
}

// BuildOptions names the qualified GCC15 build tree and the pinned source.
type BuildOptions struct {
	BuildDir     string
	DoradoSource string
	Out          string
	// Source is the oracle translation unit, relative to the repository root.
	// An empty value selects the H01 stage oracle. Each card that needs its own
	// observation publishes its own pinned source instead of editing a file whose
	// SHA an earlier capture manifest already pins.
	Source string
}

// OracleBuildToolchain describes the retained CMake compiler and link inputs.
// Paths and arguments retain their build-directory interpretation.
type OracleBuildToolchain struct {
	Compiler                     string
	Includes, Defines, Libraries []string
}

// OracleToolchain reads build metadata without compiling or loading a library.
func OracleToolchain(buildDir string) (OracleBuildToolchain, error) {
	return OracleToolchainForSource(buildDir, "basecall/model/CRFModel.cpp")
}

// OracleToolchainForSource selects the stock flags for one source-file suffix.
func OracleToolchainForSource(buildDir, sourceSuffix string) (OracleBuildToolchain, error) {
	compiler, includes, defines, err := compilerInvocationForSource(buildDir, sourceSuffix)
	if err != nil {
		return OracleBuildToolchain{}, err
	}
	libraries, err := linkLibraries(buildDir)
	if err != nil {
		return OracleBuildToolchain{}, err
	}
	return OracleBuildToolchain{Compiler: compiler, Includes: includes, Defines: defines, Libraries: libraries}, nil
}

// compileEntry mirrors one record of compile_commands.json.
type compileEntry struct {
	Directory string `json:"directory"`
	Command   string `json:"command"`
	File      string `json:"file"`
}

// BuildOracle compiles verification/hacgpu/native/oracle.cpp against the pinned
// static libraries and links it with the exact library list CMake used for the
// stock `dorado` executable, minus the stock main object.
func BuildOracle(o BuildOptions) (BuildManifest, error) {
	m := BuildManifest{
		Schema:  "nanopore-hacgpu-oracle-build/v1",
		ABI:     4,
		Status:  "failed",
		Started: time.Now().UTC().Format(time.RFC3339Nano),
		Inputs:  map[string]string{},
	}
	abs := func(p string) (string, error) { return filepath.Abs(p) }
	var err error
	if o.BuildDir, err = abs(o.BuildDir); err != nil {
		return m, err
	}
	if o.DoradoSource, err = abs(o.DoradoSource); err != nil {
		return m, err
	}
	if o.Out, err = abs(o.Out); err != nil {
		return m, err
	}
	m.BuildDir, m.DoradoSourc = o.BuildDir, o.DoradoSource

	compiler, includes, defines, err := compilerInvocation(o.BuildDir)
	if err != nil {
		return m, err
	}
	m.Compiler = compiler
	libs, err := linkLibraries(o.BuildDir)
	if err != nil {
		return m, err
	}

	source, err := abs(filepath.Join("verification", "hacgpu", "native", "oracle.cpp"))
	if err != nil {
		return m, err
	}
	if o.Source != "" {
		if source, err = abs(o.Source); err != nil {
			return m, err
		}
	}
	if err = os.MkdirAll(o.Out, 0o755); err != nil {
		return m, err
	}
	object := filepath.Join(o.Out, "oracle.o")
	m.Output = filepath.Join(o.Out, "hacgpu-oracle")

	compileArgs := append([]string{}, defines...)
	compileArgs = append(compileArgs, includes...)
	compileArgs = append(compileArgs, "-O2", "-DNDEBUG", "-std=c++23", "-fPIC", "-c", source, "-o", object)
	m.Compile = append([]string{compiler}, compileArgs...)
	if err = run(m.Compile, o.Out, "compile"); err != nil {
		return m, err
	}

	linkArgs := append([]string{"-O3", "-DNDEBUG", object}, libs...)
	linkArgs = append(linkArgs, "-o", m.Output)
	m.Link = append([]string{compiler}, linkArgs...)
	if err = run(m.Link, o.BuildDir, "link"); err != nil {
		return m, err
	}

	if m.OutputSHA, err = fileSHA(m.Output); err != nil {
		return m, err
	}
	m.Inputs[source], err = fileSHA(source)
	if err != nil {
		return m, err
	}
	for _, path := range []string{
		filepath.Join(o.BuildDir, "compile_commands.json"),
		filepath.Join(o.BuildDir, "build.ninja"),
	} {
		m.Inputs[path], err = fileSHA(path)
		if err != nil {
			return m, err
		}
	}
	for _, path := range []string{
		filepath.Join(o.BuildDir, "dorado", "nn", "libdorado_nn.a"),
		filepath.Join(o.BuildDir, "dorado", "basecall", "libdorado_basecall.a"),
		filepath.Join(o.BuildDir, "dorado", "config", "libdorado_config.a"),
		filepath.Join(o.BuildDir, "dorado", "torch_utils", "libdorado_torch_utils.a"),
	} {
		m.Inputs[path], err = fileSHA(path)
		if err != nil {
			return m, err
		}
	}
	if err = PinStageSources(o.DoradoSource, m.Inputs); err != nil {
		return m, err
	}

	m.Status = "built"
	m.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return m, err
	}
	if err = os.WriteFile(filepath.Join(o.Out, "manifest.json"), append(raw, '\n'), 0o644); err != nil {
		return m, err
	}
	return m, nil
}

func run(argv []string, dir, label string) error {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %v\n%s", label, err, tail(string(out), 40))
	}
	return nil
}

func tail(s string, lines int) string {
	parts := strings.Split(strings.TrimSpace(s), "\n")
	if len(parts) > lines {
		parts = parts[len(parts)-lines:]
	}
	return strings.Join(parts, "\n")
}

// compilerInvocation extracts the C++ compiler and the include/define flags the
// qualified build used for a stock basecall translation unit.
func compilerInvocation(buildDir string) (string, []string, []string, error) {
	return compilerInvocationForSource(buildDir, "basecall/model/CRFModel.cpp")
}

func compilerInvocationForSource(buildDir, sourceSuffix string) (string, []string, []string, error) {
	if sourceSuffix == "" {
		return "", nil, nil, fmt.Errorf("source-file suffix is required")
	}
	raw, err := os.ReadFile(filepath.Join(buildDir, "compile_commands.json"))
	if err != nil {
		return "", nil, nil, err
	}
	var entries []compileEntry
	if err = json.Unmarshal(raw, &entries); err != nil {
		return "", nil, nil, err
	}
	var chosen *compileEntry
	for i := range entries {
		if strings.HasSuffix(entries[i].File, sourceSuffix) {
			if chosen != nil {
				return "", nil, nil, fmt.Errorf("compile_commands.json has multiple %s entries", sourceSuffix)
			}
			chosen = &entries[i]
		}
	}
	if chosen == nil {
		return "", nil, nil, fmt.Errorf("compile_commands.json has no %s entry", sourceSuffix)
	}
	fields, err := splitCommand(chosen.Command)
	if err != nil {
		return "", nil, nil, err
	}
	if len(fields) == 0 || fields[0] == "" {
		return "", nil, nil, fmt.Errorf("empty compile command")
	}
	var includes, defines []string
	for i := 0; i < len(fields); i++ {
		switch f := fields[i]; {
		case f == "-isystem":
			if i+1 < len(fields) {
				includes = append(includes, "-isystem", fields[i+1])
				i++
			}
		case strings.HasPrefix(f, "-isystem"):
			includes = append(includes, f)
		case strings.HasPrefix(f, "-I"), strings.HasPrefix(f, "-D"):
			if strings.HasPrefix(f, "-D") {
				defines = append(defines, f)
			} else {
				includes = append(includes, f)
			}
		}
	}
	return fields[0], includes, defines, nil
}

// linkLibraries extracts the LINK_LIBRARIES list CMake used for bin/dorado.
func linkLibraries(buildDir string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(buildDir, "build.ninja"))
	if err != nil {
		return nil, err
	}
	text := string(raw)
	start := strings.Index(text, "build bin/dorado: CXX_EXECUTABLE_LINKER__dorado_Release")
	if start < 0 {
		return nil, fmt.Errorf("build.ninja has no bin/dorado link rule")
	}
	const key = "  LINK_LIBRARIES = "
	from := strings.Index(text[start:], key)
	if from < 0 {
		return nil, fmt.Errorf("bin/dorado link rule has no LINK_LIBRARIES")
	}
	from += start + len(key)
	end := strings.Index(text[from:], "\n  LINK_PATH")
	if end < 0 {
		return nil, fmt.Errorf("bin/dorado LINK_LIBRARIES is unterminated")
	}
	body := strings.ReplaceAll(text[from:from+end], "$\n", " ")
	body = strings.ReplaceAll(body, "\"", "")
	fields := strings.Fields(body)
	if len(fields) == 0 {
		return nil, fmt.Errorf("empty link library list")
	}
	return fields, nil
}

// splitCommand splits a shell command line, preserving quoted -D values.
// It tokenizes literal arguments without expanding variables or running a shell.
func splitCommand(line string) ([]string, error) {
	var out []string
	var token strings.Builder
	var quote byte
	started := false
	for i := 0; i < len(line); i++ {
		c := line[i]
		if quote == '\'' {
			if c == '\'' {
				quote = 0
			} else {
				token.WriteByte(c)
			}
			continue
		}
		if c == '\\' {
			if i+1 == len(line) {
				return nil, fmt.Errorf("trailing escape in compile command")
			}
			next := line[i+1]
			if quote == '"' && next != '"' && next != '$' && next != '`' && next != '\\' && next != '\n' {
				token.WriteByte(c)
				continue
			}
			i++
			if next != '\n' {
				token.WriteByte(next)
				started = true
			}
			continue
		}
		if quote == '"' {
			if c == '"' {
				quote = 0
			} else {
				token.WriteByte(c)
			}
			continue
		}
		switch c {
		case '\'', '"':
			quote, started = c, true
		case ' ', '\t', '\n':
			if started {
				out = append(out, token.String())
				token.Reset()
				started = false
			}
		default:
			token.WriteByte(c)
			started = true
		}
	}
	if quote != 0 {
		return nil, fmt.Errorf("unterminated quote in compile command")
	}
	if started {
		out = append(out, token.String())
	}
	return out, nil
}
