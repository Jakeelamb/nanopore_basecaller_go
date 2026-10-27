package closedbuild

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild/recipe"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const ArchiveSHA = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"
const PthreadArchiveSHA = "f0a17a43c74d2fe5474fa2fd29c8f14799e777d7d75a2cc4d11c20a6e7b161c5"

type Query struct {
	Argv     []string `json:"argv"`
	Expected string   `json:"expected"`
}
type linkContract struct {
	Schema   string                        `json:"schema"`
	Compiler struct{ Path, SHA256 string } `json:"compiler"`
	Bindings []struct {
		Argument   string                        `json:"argument"`
		Input      struct{ Path, SHA256 string } `json:"input"`
		FromScript string                        `json:"from_script"`
	} `json:"link_bindings"`
	Queries []Query `json:"queries"`
}
type AssetContract struct {
	Torch      string            `json:"torch_root"`
	Archive    string            `json:"archive"`
	ArchiveSHA string            `json:"archive_sha256"`
	Source     string            `json:"dorado_source"`
	Runtime    map[string]string `json:"runtime_sha256"`
}
type PlanOptions struct {
	Recipe, CompilerContract, LinkContract, AssetContract, Controller, Driver Pin
	PthreadArchive                                                            Pin
	Profiles                                                                  []string
}

// The maintained -pthread flag adds this driver-selected link input. It is
// declared before any query; a later query authenticates the actual selection.
func closePthreadArchive(p Pin, driver string, inputs map[string]string) (Query, error) {
	if e := verify(p); e != nil {
		return Query{}, e
	}
	q, e := PinFile(p.Path)
	if e != nil {
		return Query{}, e
	}
	if filepath.Base(q.Path) != "libpthread.a" || q.SHA256 != PthreadArchiveSHA {
		return Query{}, fmt.Errorf("exact pinned empty pthread archive required")
	}
	body, e := os.ReadFile(q.Path)
	if e != nil {
		return Query{}, e
	}
	if string(body) != "!<arch>\n" {
		return Query{}, fmt.Errorf("pthread archive bytes differ")
	}
	if e = add(inputs, q); e != nil {
		return Query{}, e
	}
	return Query{Argv: []string{driver, "-print-file-name=libpthread.a"}, Expected: q.Path}, nil
}

type Unit struct {
	Recipe    recipe.Unit
	Source    Pin
	Libraries []string
}
type Plan struct {
	Schema, Directory string
	Options           PlanOptions
	Recipe            recipe.Recipe
	Compiler          Compiler
	Units             []Unit
	Queries           []Query
	Inputs            map[string]string
	PathBindings      map[string]string
	Runtime           []nativeproof.ELFMetadata
	ArchiveMembers    map[string]string
	ToolkitVersion    int
	RuntimeHeld       bool
}

func pathBindings(paths []string, cwd string) (map[string]string, error) {
	m := map[string]string{}
	for _, path := range paths {
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		target, e := filepath.EvalSymlinks(path)
		if e != nil {
			return nil, e
		}
		m[path] = target
	}
	return m, nil
}
func checkPathBindings(bindings map[string]string) error {
	for path, want := range bindings {
		actual, e := filepath.EvalSymlinks(path)
		if e != nil || actual != want {
			return fmt.Errorf("selected path binding changed: %s: %v", path, e)
		}
	}
	return nil
}

func prospectiveRuntime(paths map[string]string) ([]nativeproof.ELFMetadata, error) {
	byName := map[string]Pin{}
	nodes := []nativeproof.ELFMetadata{}
	for _, path := range slices.Sorted(maps.Keys(paths)) {
		m, e := nativeproof.InspectELF(path)
		if e != nil {
			return nil, e
		}
		if m.Pin.SHA256 != paths[path] {
			return nil, fmt.Errorf("prospective runtime changed: %s", path)
		}
		nodes = append(nodes, m)
		for _, name := range []string{m.SONAME, filepath.Base(path)} {
			if name == "" {
				continue
			}
			if p, ok := byName[name]; ok && p != m.Pin {
				return nil, fmt.Errorf("ambiguous selected runtime SONAME: %s", name)
			}
			byName[name] = m.Pin
		}
	}
	for _, m := range nodes {
		for _, name := range m.Needed {
			if byName[name].Path == "" {
				return nil, fmt.Errorf("selected runtime lacks %s required by %s", name, m.Pin.Path)
			}
		}
		if m.Interpreter != "" {
			p, e := PinFile(m.Interpreter)
			if e != nil {
				return nil, e
			}
			if paths[p.Path] != p.SHA256 {
				return nil, fmt.Errorf("selected runtime lacks ELF interpreter")
			}
		}
	}
	return nodes, nil
}
func archiveMembers(torch, archive string, inputs map[string]string) (map[string]string, error) {
	z, e := zip.OpenReader(archive)
	if e != nil {
		return nil, e
	}
	defer z.Close()
	matched := map[string]string{}
	for _, path := range slices.Sorted(maps.Keys(inputs)) {
		if strings.HasPrefix(path, torch+string(filepath.Separator)) {
			m, e := runner.ArchiveMember(z, torch, path, inputs[path])
			if e != nil {
				return nil, e
			}
			maps.Copy(matched, m)
		}
	}
	return matched, nil
}
func PlanBuild(o PlanOptions) (p Plan, err error) {
	if len(o.Profiles) == 0 {
		return p, fmt.Errorf("explicit compiler profiles required")
	}
	seen := map[string]bool{}
	for _, s := range o.Profiles {
		if seen[s] || !slices.Contains([]string{"backend", "fixtures", "failure-test"}, s) {
			return p, fmt.Errorf("unsupported or duplicate compiler profile %q", s)
		}
		seen[s] = true
	}
	if o.PthreadArchive.Path == "" || o.PthreadArchive.SHA256 == "" {
		return p, fmt.Errorf("explicit pthread archive pin required before compiler work")
	}
	p = Plan{Schema: "nanopore-tensor-closed-plan/v1", Options: o, Inputs: map[string]string{}, RuntimeHeld: true}
	p.Directory, err = os.Getwd()
	if err != nil {
		return p, err
	}
	for _, q := range []Pin{o.Recipe, o.CompilerContract, o.LinkContract, o.AssetContract, o.Controller, o.Driver, o.PthreadArchive} {
		if err = add(p.Inputs, q); err != nil {
			return p, err
		}
	}
	if err = read(o.Recipe, &p.Recipe); err != nil {
		return p, err
	}
	expected, e := recipe.New(p.Recipe.Options)
	if e != nil || !equal(p.Recipe, expected) {
		return p, fmt.Errorf("maintained recipe differs: %v", e)
	}
	r := p.Recipe.Options
	if r.Family != "13.0" {
		return p, fmt.Errorf("this closed backend profile requires CUDA family13.0")
	}
	for _, path := range []string{r.Compiler, r.Torch, r.Archive, r.Dorado, r.CUDA, r.Koi, r.Weight} {
		if path != "" && !filepath.IsAbs(path) {
			return p, fmt.Errorf("absolute selected paths required")
		}
	}
	paths := []string{r.Compiler, r.Torch, r.Dorado, r.CUDA, r.Koi, o.PthreadArchive.Path}
	for _, flag := range append(slices.Clone(p.Recipe.Common), p.Recipe.Link...) {
		if strings.HasPrefix(flag, "-I") || strings.HasPrefix(flag, "-L") {
			paths = append(paths, flag[2:])
		}
	}
	p.PathBindings, err = pathBindings(paths, p.Directory)
	if err != nil {
		return p, err
	}
	var compiler CompilerContract
	if err = read(o.CompilerContract, &compiler); err != nil {
		return p, err
	}
	if compiler.Schema != "nanopore-decoder-closed-build/v1" {
		return p, fmt.Errorf("unsupported selected compiler contract")
	}
	p.Compiler = compiler.Compiler
	if err = p.Compiler.Check(); err != nil {
		return p, err
	}
	driver, e := PinFile(r.Compiler)
	if e != nil || driver != compiler.Recipe.Compiler {
		return p, fmt.Errorf("recipe did not select qualified compiler: %v", e)
	}
	pthreadQuery, e := closePthreadArchive(o.PthreadArchive, driver.Path, p.Inputs)
	if e != nil {
		return p, e
	}
	maps.Copy(p.Inputs, p.Compiler.Inputs)
	var lc linkContract
	if err = read(o.LinkContract, &lc); err != nil {
		return p, err
	}
	if lc.Schema != "nanopore-supgpu-build-plan/v1" || lc.Compiler.Path != driver.Path || lc.Compiler.SHA256 != driver.SHA256 {
		return p, fmt.Errorf("closed link contract differs")
	}
	allowed := map[string]string{}
	for _, b := range lc.Bindings {
		q := Pin{Path: b.Input.Path, SHA256: b.Input.SHA256}
		if err = add(allowed, q); err != nil {
			return p, err
		}
		if strings.HasPrefix(b.Argument, "implicit:") || b.FromScript != "" {
			if err = add(p.Inputs, q); err != nil {
				return p, err
			}
		}
	}
	var assets AssetContract
	if err = read(o.AssetContract, &assets); err != nil {
		return p, err
	}
	if assets.Torch != r.Torch || assets.Archive != r.Archive || assets.ArchiveSHA != ArchiveSHA || assets.Source != r.Dorado {
		return p, fmt.Errorf("selected distribution asset contract differs")
	}
	archive, e := PinFile(r.Archive)
	if e != nil || archive.SHA256 != ArchiveSHA {
		return p, fmt.Errorf("qualified archive differs: %v", e)
	}
	if err = add(p.Inputs, archive); err != nil {
		return p, err
	}
	if strings.Contains(o.Driver.Path, "/stubs/") {
		return p, fmt.Errorf("CUDA driver stub is not a runtime driver")
	}
	driverMeta, e := nativeproof.InspectELF(o.Driver.Path)
	if e != nil || driverMeta.Pin.SHA256 != o.Driver.SHA256 || driverMeta.SONAME != "libcuda.so.1" {
		return p, fmt.Errorf("actual CUDA driver pin/SONAME differs: %v", e)
	}
	driverClosure, e := nativeproof.CloseELF([]Pin{driverMeta.Pin}, []string{"/usr/lib", "/lib"})
	if e != nil {
		return p, e
	}
	runtime := map[string]string{}
	for path, want := range assets.Runtime {
		if err = add(runtime, Pin{Path: path, SHA256: want}); err != nil {
			return p, err
		}
	}
	for _, q := range driverClosure.Inputs {
		if err = add(runtime, q); err != nil {
			return p, err
		}
	}
	p.Runtime, e = prospectiveRuntime(runtime)
	if e != nil {
		return p, e
	}
	maps.Copy(p.Inputs, runtime)
	resolve := func(args []string) ([]string, error) {
		out := []string{}
		for _, s := range args {
			if strings.HasPrefix(s, "-L") {
				continue
			}
			if strings.HasPrefix(s, "-Wl,-rpath,") {
				out = append(out, s)
				continue
			}
			path := s
			if strings.HasPrefix(s, "-l") {
				name := strings.TrimPrefix(s, "-l")
				switch name {
				case "torch_cpu", "c10", "torch_cuda", "c10_cuda":
					path = filepath.Join(r.Torch, "lib/lib"+name+".so")
				case "cudart", "cublas":
					path = filepath.Join(r.CUDA, "lib64/lib"+name+".so")
				case "cuda":
					path = driverMeta.Pin.Path
				default:
					return nil, fmt.Errorf("unsupported maintained link search %q", s)
				}
			}
			if !filepath.IsAbs(path) {
				return nil, fmt.Errorf("unsupported maintained link argument %q", s)
			}
			q, e := PinFile(path)
			if e != nil {
				return nil, e
			}
			if q != driverMeta.Pin && allowed[q.Path] != q.SHA256 {
				return nil, fmt.Errorf("selected library lacks independently closed binding: %s", path)
			}
			if e = add(p.Inputs, q); e != nil {
				return nil, e
			}
			out = append(out, q.Path)
		}
		return out, nil
	}
	for _, name := range o.Profiles {
		u, e := p.Recipe.Unit(name, "")
		if e != nil {
			return p, e
		}
		source, e := PinFile(filepath.Join(p.Directory, u.Source))
		if e != nil {
			return p, e
		}
		libs, e := resolve(u.Link)
		if e != nil {
			return p, e
		}
		p.Units = append(p.Units, Unit{u, source, libs})
	}
	for _, root := range []string{"internal/tensor", "verification/tensor"} {
		err = filepath.WalkDir(root, func(path string, d os.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.IsDir() {
				return nil
			}
			ext := filepath.Ext(path)
			if !slices.Contains([]string{".h", ".cpp", ".inc"}, ext) && path != "verification/tensor/source-pins.json" {
				return nil
			}
			q, e := PinFile(path)
			if e != nil {
				return e
			}
			return add(p.Inputs, q)
		})
		if err != nil {
			return p, err
		}
	}
	sourceBody, e := os.ReadFile("verification/tensor/source-pins.json")
	if e != nil {
		return p, e
	}
	var sourcePins struct {
		Commit string            `json:"commit"`
		Files  map[string]string `json:"files"`
	}
	if e = json.Unmarshal(sourceBody, &sourcePins); e != nil {
		return p, e
	}
	if sourcePins.Commit != "8b8fc5d36a9c0baab262a743cb175b5878e38ca6" || len(sourcePins.Files) == 0 {
		return p, fmt.Errorf("pinned Dorado source contract differs")
	}
	for rel, want := range sourcePins.Files {
		if filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, "../") {
			return p, fmt.Errorf("invalid stock source member")
		}
		if e = add(p.Inputs, Pin{Path: filepath.Join(r.Dorado, rel), SHA256: want}); e != nil {
			return p, e
		}
	}
	for _, path := range []string{filepath.Join(r.Torch, "build-version"), filepath.Join(r.Koi, "include/koi.h"), filepath.Join(r.CUDA, "include/cuda.h")} {
		q, e := PinFile(path)
		if e != nil {
			return p, e
		}
		if err = add(p.Inputs, q); err != nil {
			return p, err
		}
	}
	header, e := os.ReadFile(filepath.Join(r.CUDA, "include/cuda.h"))
	if e != nil {
		return p, e
	}
	match := regexp.MustCompile(`(?m)^#define CUDA_VERSION\s+13040\s*$`).Find(header)
	if match == nil {
		return p, fmt.Errorf("selected toolkit headers must be CUDA13.4 (13040)")
	}
	p.ToolkitVersion = 13040
	q, e := PinFile(r.Weight)
	if e != nil {
		return p, e
	}
	if filepath.Base(r.Weight) != "0.conv.weight.tensor" || q.SHA256 != "8c6b47fe5414d9c7e785b80d4d265b4b4290fd5487b26436eaffbe956d5c3e28" {
		return p, fmt.Errorf("closed backend plan requires genuine pinned HAC0 convolution weight for the later CPU fixture gate")
	}
	if err = add(p.Inputs, q); err != nil {
		return p, err
	}
	for _, q := range lc.Queries {
		if len(q.Argv) != 2 || q.Argv[0] != driver.Path || !filepath.IsAbs(q.Expected) {
			continue
		}
		selected, e := PinFile(q.Expected)
		if e != nil {
			continue
		}
		if p.Inputs[selected.Path] == selected.SHA256 {
			p.Queries = append(p.Queries, q)
		}
	}
	if len(p.Queries) == 0 {
		return p, fmt.Errorf("no selected compiler resource queries")
	}
	p.Queries = append(p.Queries, pthreadQuery)
	c, e := ReadController(o.Controller, false)
	if e != nil {
		return p, e
	}
	maps.Copy(p.Inputs, c.Inputs)
	for _, q := range c.Snapshots {
		if err = add(p.Inputs, q); err != nil {
			return p, err
		}
	}
	for _, q := range c.Binaries {
		if err = add(p.Inputs, q); err != nil {
			return p, err
		}
	}
	for _, command := range c.Commands {
		for path, want := range map[string]string{command.Stdout: command.StdoutSHA, command.Stderr: command.StderrSHA} {
			if err = add(p.Inputs, Pin{Path: path, SHA256: want}); err != nil {
				return p, err
			}
		}
	}
	p.ArchiveMembers, err = archiveMembers(r.Torch, r.Archive, p.Inputs)
	return p, err
}
func CheckPlan(p Plan) error {
	actual, e := PlanBuild(p.Options)
	if e != nil {
		return e
	}
	if !equal(actual, p) {
		return fmt.Errorf("reviewed compiler plan differs from current maintained recipe or closure")
	}
	return nil
}
