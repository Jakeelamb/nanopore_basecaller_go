// Package cgobuild prepares the fixed H08 CGO1 compiler boundary. Preparing a
// plan executes no Go, Cgo, compiler, linker or produced target.
package cgobuild

import (
	"encoding/json"
	"fmt"
	"go/build"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type Pin = nativeproof.Pin

const Module = "github.com/Jakeelamb/nanopore_basecaller_go"
const Target = Module + "/verification/hacgpu/h08/cmd"

type Options struct {
	Role                       BuildRole
	Root, GoRoot, WorkRoot     string
	CompilerPlan, BackendProof Pin
}
type Package struct {
	ImportPath, Directory                      string
	Go, Cgo, C, Headers, Assembly, Syso, Embed []string
	CPPFlags, CFlags, CXXFlags, LDFlags        []string
	Imports                                    []string
	TestGo, TestImports, XTestGo, XTestImports []string
}
type Resource struct {
	Name  string
	Input Pin
	Query []string
}
type Query struct {
	Name     string
	Argv     []string
	Expected Pin
}
type Profile struct {
	Name, Source, SuccessfulExit, InputsBeforeChild, Evidence string
	Variants                                                  []string
}
type Plan struct {
	Role             BuildRole
	CPU13            *CPU13Profile
	Schema, Status   string
	Options          Options
	CC, CXX          Pin
	Compiler         closedbuild.Compiler
	CompilerCCDomain nativeproof.ELFClosure
	Backend          closedbuild.ArtifactOptions
	Packages         []Package
	Inputs, GoTools  map[string]string
	Resources        []Resource
	Queries          []Query
	Profiles         []Profile
	Environment      map[string]string
	BuildArgv        []string
	Generated        map[string][]string
	Boundaries       []string
}

func explicitEmbed(dir, pattern string) (Pin, error) {
	if !filepath.IsLocal(pattern) || filepath.Clean(pattern) != pattern || strings.ContainsAny(pattern, "*?[]\\:") {
		return Pin{}, fmt.Errorf("unadmitted embed shape %s", pattern)
	}
	root, e := filepath.EvalSymlinks(dir)
	if e != nil {
		return Pin{}, e
	}
	q, e := closedPin(filepath.Join(dir, pattern))
	if e != nil || !under(root, q.Path) {
		return Pin{}, fmt.Errorf("embed is not a regular file within its package: %s", pattern)
	}
	return q, nil
}

func pinned(path string, inputs map[string]string) (Pin, error) {
	p, e := closedbuild.PinFile(path)
	if e != nil {
		return p, e
	}
	if old, ok := inputs[p.Path]; ok && old != p.SHA256 {
		return p, fmt.Errorf("conflicting input pin")
	}
	inputs[p.Path] = p.SHA256
	return p, nil
}
func read(p Pin, v any) error {
	a, e := closedbuild.PinFile(p.Path)
	if e != nil {
		return e
	}
	if a != p || len(p.SHA256) != 64 {
		return fmt.Errorf("contract identity differs: %s", p.Path)
	}
	b, e := os.ReadFile(p.Path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func toolPin(c closedbuild.Compiler, name string) Pin {
	for _, t := range c.Tools {
		if t.Name == name {
			return t.Executable
		}
	}
	return Pin{}
}

// SelectSources uses Go 1.27's build constraints without invoking cmd/go or CC.
// This fixed module has no third-party modules or directory/glob embeddings.
// The actual admitted CGO1 build must independently match its package list.
func SelectSources(root, goRoot string) ([]Package, map[string]string, error) {
	return selectSources(root, goRoot, H08Command)
}
func selectSources(root, goRoot string, role BuildRole) ([]Package, map[string]string, error) {
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.Compiler, ctx.CgoEnabled, ctx.GOROOT, ctx.GOPATH = "linux", "amd64", "gc", true, goRoot, ""
	inputs := map[string]string{}
	if _, e := pinned(filepath.Join(root, "go.mod"), inputs); e != nil {
		return nil, nil, e
	}
	queue := []string{Target}
	if role == CPU13Tests {
		queue = []string{TensorPackage, Module + "/verification/hacgpu/h08", "os", "reflect", "testing", "testing/internal/testdeps"}
	}
	seen := map[string]bool{}
	packages := []Package{}
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		if seen[name] || name == "C" || name == "unsafe" {
			continue
		}
		seen[name] = true
		var dir string
		if strings.HasPrefix(name, Module+"/") {
			dir = filepath.Join(root, strings.TrimPrefix(name, Module+"/"))
		} else {
			if strings.Contains(strings.Split(name, "/")[0], ".") || filepath.IsAbs(name) || strings.Contains(name, "..") {
				return nil, nil, fmt.Errorf("unadmitted module import %s", name)
			}
			dir = filepath.Join(goRoot, "src", name)
		}
		// IgnoreVendor also disables go/build's module-mode cmd/go fallback;
		// source selection must remain filesystem-only.
		p, e := ctx.ImportDir(dir, build.IgnoreVendor)
		if e != nil {
			return nil, nil, fmt.Errorf("static selection %s: %w", name, e)
		}
		if len(p.CXXFiles)+len(p.FFiles)+len(p.SwigFiles)+len(p.SwigCXXFiles) != 0 || len(p.CgoPkgConfig) != 0 {
			return nil, nil, fmt.Errorf("unadmitted CXX/Fortran/SWIG/pkg-config input: %s", name)
		}
		if len(p.CgoFiles) > 0 && name != Module+"/internal/tensor" && name != "runtime/cgo" {
			return nil, nil, fmt.Errorf("unadmitted Cgo package %s", name)
		}
		x := Package{ImportPath: name, Directory: dir, Go: p.GoFiles, Cgo: p.CgoFiles, C: p.CFiles, Headers: p.HFiles, Assembly: p.SFiles, Syso: p.SysoFiles, CPPFlags: p.CgoCPPFLAGS, CFlags: p.CgoCFLAGS, CXXFlags: p.CgoCXXFLAGS, LDFlags: p.CgoLDFLAGS, Imports: p.Imports}
		if role == CPU13Tests && name == TensorPackage {
			if len(p.TestEmbedPatterns)+len(p.XTestEmbedPatterns) != 0 {
				return nil, nil, fmt.Errorf("CPU13 test embeddings require a separate source contract")
			}
			x.TestGo, x.TestImports = p.TestGoFiles, p.TestImports
			x.XTestGo, x.XTestImports = p.XTestGoFiles, p.XTestImports
			queue = append(queue, p.TestImports...)
			queue = append(queue, p.XTestImports...)
		}
		for _, pattern := range p.EmbedPatterns {
			q, e := explicitEmbed(dir, pattern)
			if e != nil {
				return nil, nil, e
			}
			inputs[q.Path] = q.SHA256
			x.Embed = append(x.Embed, pattern)
		}
		for _, files := range [][]string{x.Go, x.Cgo, x.C, x.Headers, x.Assembly, x.Syso, x.TestGo, x.XTestGo} {
			for _, file := range files {
				if filepath.Base(file) != file {
					return nil, nil, fmt.Errorf("nonlocal source member")
				}
				if _, e = pinned(filepath.Join(dir, file), inputs); e != nil {
					return nil, nil, e
				}
			}
		}
		packages = append(packages, x)
		queue = append(queue, p.Imports...)
		if len(p.CgoFiles) > 0 {
			queue = append(queue, "runtime/cgo")
		}
	}
	slices.SortFunc(packages, func(a, b Package) int { return strings.Compare(a.ImportPath, b.ImportPath) })
	return packages, inputs, nil
}

func RequiredGenerated(p Package) []string {
	if len(p.Cgo) == 0 {
		return nil
	}
	files := []string{"_cgo_gotypes.go", "_cgo_export.c", "_cgo_export.h", "_cgo_main.c"}
	for _, file := range p.Cgo {
		base := strings.TrimSuffix(file, ".go")
		files = append(files, base+".cgo1.go", base+".cgo2.c")
	}
	slices.Sort(files)
	return files
}

var resourceNames = []string{"cc1", "collect2", "as", "ld", "nm", "lto-wrapper", "lto1", "liblto_plugin.so", "crt1.o", "Scrt1.o", "crti.o", "crtn.o", "crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "libc.so", "libc.so.6", "libc_nonshared.a", "libdl.a", "libpthread.a", "ld-linux-x86-64.so.2"}

// ValidateSpecification is pure: omission/path controls fail before any ELF or
// file inspection, and therefore before a future compiler child can exist.
func ValidateSpecification(p Plan) error {
	if p.CC != toolPin(p.Compiler, "gcc-driver") || p.CXX != toolPin(p.Compiler, "driver") || filepath.Base(p.CC.Path) != "gcc-15" || filepath.Base(p.CXX.Path) != "g++-15" {
		return fmt.Errorf("explicit qualified GCC15 CC/CXX binding required")
	}
	if len(p.Compiler.Plugins) == 0 {
		return fmt.Errorf("compiler LTO/BFD plugins omitted")
	}
	seen := map[string]bool{}
	for _, r := range p.Resources {
		if seen[r.Name] || r.Input.Path == "" || len(r.Input.SHA256) != 64 {
			return fmt.Errorf("invalid/duplicate C link resource")
		}
		seen[r.Name] = true
		if p.Inputs[r.Input.Path] != r.Input.SHA256 {
			return fmt.Errorf("C link resource not declared before queries")
		}
	}
	if len(seen) != len(resourceNames) {
		return fmt.Errorf("finite C link resource family incomplete")
	}
	for _, name := range resourceNames {
		if !seen[name] {
			return fmt.Errorf("C link resource omitted: %s", name)
		}
	}
	plugin := false
	for _, q := range p.Compiler.Plugins {
		if filepath.Base(q.Path) == "liblto_plugin.so" {
			plugin = true
		}
	}
	if !plugin {
		return fmt.Errorf("compiler LTO plugin omitted")
	}
	return nil
}

func Prepare(o Options) (p Plan, err error) {
	o.Role, err = normalizedRole(o.Role)
	if err != nil {
		return p, err
	}
	for _, dir := range []string{o.Root, o.GoRoot, o.WorkRoot} {
		if !filepath.IsAbs(dir) {
			return p, fmt.Errorf("absolute root/work paths required")
		}
	}
	if o.WorkRoot == o.Root || strings.HasPrefix(o.Root, o.WorkRoot+"/") {
		return p, fmt.Errorf("dedicated fresh work root required")
	}
	p = Plan{Role: o.Role, Schema: "nanopore-h08-cgo-build-plan/v2", Status: "prospective_no_native_queries", Options: o, Inputs: map[string]string{}, GoTools: map[string]string{}, Generated: map[string][]string{}}
	var native closedbuild.Plan
	if err = read(o.CompilerPlan, &native); err != nil {
		return p, err
	}
	if native.Schema != "nanopore-tensor-closed-plan/v1" {
		return p, fmt.Errorf("accepted compiler plan schema required")
	}
	p.Compiler = native.Compiler
	p.CC = toolPin(p.Compiler, "gcc-driver")
	p.CXX = toolPin(p.Compiler, "driver")
	for _, d := range p.Compiler.Domains {
		if d.Name == "gcc-driver" && d.Root == p.CC {
			p.CompilerCCDomain = d.ELF
		}
	}
	if len(p.CompilerCCDomain.Nodes) == 0 {
		return p, fmt.Errorf("selected CC own ELF domain omitted")
	}
	if err = p.Compiler.Check(); err != nil {
		return p, err
	}
	maps.Copy(p.Inputs, p.Compiler.Inputs)
	var proof struct {
		Schema  string
		Backend closedbuild.BackendArtifact
	}
	if err = read(o.BackendProof, &proof); err != nil {
		return p, err
	}
	if proof.Schema != "nanopore-h08-backend-proof/v1" || proof.Backend.Schema != "nanopore-tensor-backend-artifact/v1" || proof.Backend.Plan != o.CompilerPlan || !proof.Backend.RuntimeHeld {
		return p, fmt.Errorf("authentic typed backend proof required")
	}
	p.Backend = closedbuild.ArtifactOptions{Build: proof.Backend.Build, Plan: proof.Backend.Plan, Producer: proof.Backend.Producer, Output: proof.Backend.Output}
	for _, q := range []Pin{o.CompilerPlan, o.BackendProof, p.Backend.Build, p.Backend.Producer, p.Backend.Output} {
		actual, e := pinned(q.Path, p.Inputs)
		if e != nil || actual != q {
			return p, fmt.Errorf("backend/compiler anchor changed: %s: %v", q.Path, e)
		}
	}
	p.Packages, p.Inputs, err = selectInto(o, p.Inputs)
	if err != nil {
		return p, err
	}
	for _, q := range p.Packages {
		if len(q.Cgo) > 0 {
			p.Generated[q.ImportPath] = RequiredGenerated(q)
		}
	}
	if len(p.Generated) != 2 {
		return p, fmt.Errorf("fixed H08 requires exactly tensor and runtime/cgo")
	}
	tools := []string{filepath.Join(o.GoRoot, "bin/go")}
	entries, e := os.ReadDir(filepath.Join(o.GoRoot, "pkg/tool/linux_amd64"))
	if e != nil {
		return p, e
	}
	for _, f := range entries {
		if !f.IsDir() {
			tools = append(tools, filepath.Join(o.GoRoot, "pkg/tool/linux_amd64", f.Name()))
		}
	}
	for _, path := range tools {
		m, e := nativeproof.InspectELF(path)
		if e != nil || m.Interpreter != "" || len(m.Needed) != 0 {
			return p, fmt.Errorf("static Go tool required: %s: %v", path, e)
		}
		p.GoTools[m.Pin.Path] = m.Pin.SHA256
		p.Inputs[m.Pin.Path] = m.Pin.SHA256
	}
	for _, path := range []string{filepath.Join(o.GoRoot, "VERSION"), filepath.Join(o.GoRoot, "go.env")} {
		if _, err = pinned(path, p.Inputs); err != nil {
			return p, err
		}
	}
	v, e := os.ReadFile(filepath.Join(o.GoRoot, "VERSION"))
	if e != nil || !strings.HasPrefix(string(v), "go1.27.3\n") {
		return p, fmt.Errorf("pinned Go1.27.3 required")
	}
	// Pin generator and driver source as well as their executables. These lines
	// define the finite request profiles, expected failing probes and outputs.
	for _, rel := range []string{"src/go/build/build.go", "src/cmd/cgo/main.go", "src/cmd/cgo/gcc.go", "src/cmd/cgo/util.go", "src/cmd/cgo/out.go", "src/cmd/go/internal/work/exec.go", "src/cmd/go/internal/work/buildid.go", "src/cmd/link/internal/ld/lib.go"} {
		if _, err = pinned(filepath.Join(o.GoRoot, rel), p.Inputs); err != nil {
			return p, err
		}
	}
	if o.Role == CPU13Tests {
		for _, rel := range []string{"src/cmd/go/internal/load/test.go", "src/cmd/go/internal/load/pkg.go", "src/cmd/go/internal/test/test.go", "src/cmd/go/internal/list/list.go"} {
			if _, err = pinned(filepath.Join(o.GoRoot, rel), p.Inputs); err != nil {
				return p, err
			}
		}
		p.CPU13, err = cpu13Profile(p)
		if err != nil {
			return p, err
		}
	}
	err = filepath.WalkDir(filepath.Join(o.GoRoot, "pkg/include"), func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() {
			return nil
		}
		_, e = pinned(path, p.Inputs)
		return e
	})
	if err != nil {
		return p, err
	}
	lib := filepath.Dir(toolPin(p.Compiler, "cc1").Path)
	for _, name := range resourceNames {
		var path, query string
		switch name {
		case "cc1", "collect2", "as", "ld", "nm", "lto-wrapper", "lto1":
			path = toolPin(p.Compiler, name).Path
			query = "-print-prog-name=" + name
		case "liblto_plugin.so", "crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1":
			path = filepath.Join(lib, name)
			query = "-print-file-name=" + name
		default:
			path = filepath.Join("/usr/lib", name)
			if name != "libc.so.6" && name != "ld-linux-x86-64.so.2" {
				query = "-print-file-name=" + name
			}
		}
		q, e := pinned(path, p.Inputs)
		if e != nil {
			return p, e
		}
		if old, ok := native.Inputs[q.Path]; ok && old != q.SHA256 {
			return p, fmt.Errorf("accepted002 native resource changed: %s", name)
		}
		if (name == "libpthread.a" || name == "libdl.a") && q.SHA256 != closedbuild.PthreadArchiveSHA {
			return p, fmt.Errorf("exact empty merged glibc archive required: %s", name)
		}
		r := Resource{Name: name, Input: q}
		if query != "" {
			r.Query = []string{p.CC.Path, query}
		}
		p.Resources = append(p.Resources, r)
		if len(r.Query) > 0 {
			p.Queries = append(p.Queries, Query{Name: name, Argv: r.Query, Expected: r.Input})
		}
	}
	if err = ValidateSpecification(p); err != nil {
		return p, err
	}
	p.Profiles = profiles(o.GoRoot)
	p.Queries = append(p.Queries, Query{Name: "cc-version", Argv: []string{p.CC.Path, "--version"}}, Query{Name: "cc-tool-id", Argv: []string{p.CC.Path, "-###", "-x", "c", "-c", "-"}})
	p.Environment = map[string]string{"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "GOROOT": o.GoRoot, "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOSUMDB": "off", "GOENV": "off", "GOWORK": "off", "GOFLAGS": "", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "CGO_ENABLED": "1", "CGO_CFLAGS": "-O2 -g", "CGO_CPPFLAGS": "", "CGO_CXXFLAGS": "-O2 -g", "CGO_LDFLAGS": "-O2 -g", "GOMAXPROCS": "1", "GOMEMLIMIT": "3GiB", "GOCACHE": filepath.Join(o.WorkRoot, "cache"), "GOTMPDIR": filepath.Join(o.WorkRoot, "go-work"), "TMPDIR": filepath.Join(o.WorkRoot, "cgo-tmp"), "GOPATH": filepath.Join(o.WorkRoot, "gopath"), "GOEXPERIMENT": "", "GOFIPS140": "off"}
	p.Environment["CC"], p.Environment["CXX"] = "<SEALED_CURRENT_INTERPOSER_CC>", "<SEALED_CURRENT_INTERPOSER_CXX>"
	p.BuildArgv = plannedBuildArgs(p)
	p.Boundaries = []string{"Plan-only: interposer executable/argv must be sealed before any admitted native phase; placeholders are not executable argv.", "Cgo translation is native compiler work, including expected nonzero GCC classification and unsupported-flag probes.", "Generated C temp bytes must be snapshotted before each -M and actual compiler child; final Cgo outputs before Go resumes compilation, dynimport Go before use.", "Every native TU uses same-option -M before actual -MD; intentional error probes retain -MD equality when a depfile exists, otherwise an explicit source-grounded unavailable record.", "Implicit CRT/scripts/archive/runtime/plugin/cache inputs are pinned before queries; actual query answers and link trace/map must match those pins.", "No backend compilation or load: the accepted typed producer is history. Current compiler consumes only Cgo shims and C runtime resources.", "Go/Cgo build, linked output execution, CPU13, runtime guard, driver/PTX/GPU and model phases remain separately held; Consume cannot be armed by this plan."}
	p.Boundaries = append(p.Boundaries, "A rejected interposer request fails the entire build even if cmd/go or cgo tolerates its nonzero exit as a feature probe; no silent cache/feature fallback can qualify it.")
	if err = validateTarget(p); err != nil {
		return p, err
	}
	return p, nil
}

func selectInto(o Options, inputs map[string]string) ([]Package, map[string]string, error) {
	pkgs, selected, e := selectSources(o.Root, o.GoRoot, o.Role)
	if e != nil {
		return nil, nil, e
	}
	maps.Copy(inputs, selected)
	return pkgs, inputs, nil
}

func profiles(goRoot string) []Profile {
	return []Profile{
		{"resources", filepath.Join(goRoot, "src/cmd/go/internal/work/buildid.go"), "0", "Complete selected compiler/plugin/interpreter/cache + all resource pins before first query", "Exact argv/raw stdout/stderr and canonical selected-resource equality", []string{"planned -print-prog-name/-print-file-name", "--version", "-### -x c -c -"}},
		{"flag-probes", filepath.Join(goRoot, "src/cmd/go/internal/work/exec.go"), "0 or compiler nonzero, interpreted only by Go1.27's pinned diagnostic predicate", "Empty stdin or pinned Go linker trivial.c; all compile/link resources already declared", "Expected failure is retained, never promoted to all-exit-zero", []string{"-fno-caret-diagnostics", "-Qunused-arguments", "-Wl,--no-gc-sections", "-fdebug-prefix-map=a=b", "-ffile-prefix-map=a=b", "-gno-record-gcc-switches", "-frandom-seed=1", "-no-pie", "-nopie", "Go linker build-id/export-dynamic/response-file probes"}},
		{"translation", filepath.Join(goRoot, "src/cmd/cgo/gcc.go"), "0 for defines/DWARF; diagnostic classification intentionally may fail", "Pinned source package/CGO flags and cgo-gcc-input-*.c before native call; no arbitrary external path", "-M then actual -MD when available, exact generated bytes/stdout/stderr/exit", []string{"gccDefines -E -dM", "gccErrors -O0 with optional -fdiagnostics-color=never", "gccDebug -gdwarf-2 -fno-lto"}},
		{"compile-and-link", filepath.Join(goRoot, "src/cmd/go/internal/work/exec.go"), "0; dynimport failure is not accepted for this fixed build", "All generated .c/.h/.go sets and original C/S/ABI sources; object pins before link", "-M=-MD; object identity, exact argv; closed trace/map; dynimport source and final ELF identity", []string{"runtime/cgo C/S and generated C", "tensor loader.c and generated C", "Cgo dynimport executable link (unexecuted)", "Go external linker probes and final executable link (unexecuted)"}},
	}
}
