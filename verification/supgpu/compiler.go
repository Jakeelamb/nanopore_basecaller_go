package supgpu

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

type CompilerTool struct {
	Name       string `json:"name"`
	Executable Pin    `json:"executable"`
}

// CompilerRuntime precedes even -print-prog-name and -M. The post-build model
// RuntimePlan authenticates a different process family and cannot replace it.
type CompilerRuntime struct {
	Schema      string         `json:"schema"`
	Environment []string       `json:"environment"`
	Tools       []CompilerTool `json:"tools"`
	Plugins     []Pin          `json:"plugins"`
	Nodes       []ELFNode      `json:"elf_nodes"`
	Inputs      []Pin          `json:"inputs"`
	LoaderCache Pin            `json:"loader_cache"`
	SearchDirs  []string       `json:"search_directories"`
}

func planCompilerRuntime(compiler string) (CompilerRuntime, error) {
	usr := filepath.Dir(filepath.Dir(compiler))
	matches, err := filepath.Glob(filepath.Join(usr, "lib", "gcc", "*", "*", "cc1plus"))
	if err != nil || len(matches) != 1 {
		return CompilerRuntime{}, fmt.Errorf("qualified GCC program directory is ambiguous: %v", err)
	}
	gcc := filepath.Dir(matches[0])
	tools := []CompilerTool{}
	for _, t := range []struct{ name, path string }{
		{"driver", compiler}, {"cc1plus", matches[0]},
		{"collect2", filepath.Join(gcc, "collect2")},
		{"as", "/usr/bin/as"}, {"ld", "/usr/bin/ld"}, {"nm", "/usr/bin/nm"},
		{"lto-wrapper", filepath.Join(gcc, "lto-wrapper")},
		{"lto1", filepath.Join(gcc, "lto1")},
		{"cc1", filepath.Join(gcc, "cc1")},
		{"gcc-driver", filepath.Join(usr, "bin", "gcc-15")},
	} {
		p, err := canonicalPin(t.path)
		if err != nil {
			return CompilerRuntime{}, err
		}
		tools = append(tools, CompilerTool{t.name, p})
	}
	plugin, err := canonicalPin(filepath.Join(gcc, "liblto_plugin.so"))
	if err != nil {
		return CompilerRuntime{}, err
	}
	plugins := []Pin{plugin}
	seen := map[string]bool{plugin.Path: true}
	// BFD may discover plugins independently of the driver's explicit LTO
	// plugin, notably when nm inspects archive members. Freeze that finite
	// directory family before launch as well as recursively closing its DSOs.
	bfd, err := filepath.Glob("/usr/lib/bfd-plugins/*.so*")
	if err != nil {
		return CompilerRuntime{}, err
	}
	for _, path := range bfd {
		p, err := canonicalPin(path)
		if err != nil {
			return CompilerRuntime{}, err
		}
		if !seen[p.Path] {
			seen[p.Path] = true
			plugins = append(plugins, p)
		}
	}
	cache, err := canonicalPin("/etc/ld.so.cache")
	if err != nil {
		return CompilerRuntime{}, err
	}
	return compilerRuntimeForTools(tools, plugins, []string{"/usr/lib", "/lib"}, cache)
}

func compilerRuntimeForTools(tools []CompilerTool, plugins []Pin, dirs []string, cache Pin) (CompilerRuntime, error) {
	c := CompilerRuntime{Schema: "nanopore-supgpu-compiler-runtime/v1", Environment: nativeEnvironment(), Tools: tools, Plugins: plugins, LoaderCache: cache, SearchDirs: dirs}
	seeds := append([]Pin{}, plugins...)
	for _, tool := range tools {
		seeds = append(seeds, tool.Executable)
	}
	var err error
	c.Nodes, c.Inputs, err = elfClosure(seeds, dirs)
	if err != nil {
		return c, err
	}
	if err = verifyPin("", cache); err != nil {
		return c, err
	}
	c.Inputs = append(c.Inputs, cache)
	return c, nil
}

func compilerQueries(c CompilerRuntime) []ToolQuery {
	queries := []ToolQuery{}
	for _, tool := range c.Tools {
		if tool.Name == "driver" || tool.Name == "gcc-driver" {
			continue
		}
		queries = append(queries, ToolQuery{[]string{c.Tools[0].Executable.Path, "-print-prog-name=" + tool.Name}, tool.Executable.Path})
	}
	for i, plugin := range c.Plugins {
		if i != 0 {
			break
		}
		// Only the first plugin is selected by GCC's own file lookup; the
		// remaining BFD plugins come from the separately frozen directory family.
		queries = append(queries, ToolQuery{[]string{c.Tools[0].Executable.Path, "-print-file-name=" + filepath.Base(plugin.Path)}, plugin.Path})
	}
	return queries
}

func requireCompilerInputs(c CompilerRuntime, inputs []Pin) error {
	allowed := map[string]Pin{}
	for _, p := range inputs {
		allowed[p.Path] = p
	}
	for _, p := range c.Inputs {
		if allowed[p.Path] != p {
			return fmt.Errorf("compiler runtime input absent from frozen plan: %s", p.Path)
		}
		if err := verifyPin("", p); err != nil {
			return err
		}
	}
	return nil
}

// Rebuild from selected paths, rather than trusting a self-consistent edited
// node/input list. This also checks symlink resolution and DT_NEEDED changes.
func verifyCompilerRuntime(p BuildPlan) error {
	c, err := planCompilerRuntime(p.Compiler.Path)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(c, p.CompilerRuntime) {
		return fmt.Errorf("prospective compiler ELF runtime closure changed")
	}
	if err = requireCompilerInputs(c, p.Inputs); err != nil {
		return err
	}
	for _, command := range append(append([]BuildCommand{}, p.Preprocess...), p.Commands...) {
		if len(command.Argv) == 0 || command.Argv[0] != p.Compiler.Path {
			return fmt.Errorf("compiler command selects an unfrozen driver")
		}
		for _, arg := range command.Argv[1:] {
			for _, prefix := range []string{"-B", "-wrapper", "-specs", "--specs", "-fplugin", "-fuse-ld", "--sysroot", "-isysroot", "-flto", "-fuse-linker-plugin"} {
				if strings.HasPrefix(arg, prefix) {
					return fmt.Errorf("compiler tool selection override is outside the frozen recipe: %s", arg)
				}
			}
		}
	}
	return nil
}

func resolveCompilerProgram(raw, dir string) (string, error) {
	if filepath.IsAbs(raw) {
		return filepath.EvalSymlinks(raw)
	}
	if strings.ContainsRune(raw, '/') {
		return filepath.EvalSymlinks(filepath.Join(dir, raw))
	}
	// This is the exact PATH supplied to all native commands, independent of the
	// controller's inherited environment. The query itself executes no subtool.
	for _, root := range []string{"/usr/bin", "/bin"} {
		candidate := filepath.Join(root, raw)
		info, err := os.Stat(candidate)
		if err == nil && info.Mode().IsRegular() && info.Mode().Perm()&0111 != 0 {
			return filepath.EvalSymlinks(candidate)
		}
	}
	return "", fmt.Errorf("compiler selected an unresolved program %q", raw)
}
