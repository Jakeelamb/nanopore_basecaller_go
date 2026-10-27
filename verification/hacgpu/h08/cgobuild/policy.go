package cgobuild

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// Lease identifies one actual pinned Go-tool invocation. It is written before
// that child starts and inherited only by its descendants, never inferred from
// a global current-package flag.
type Lease struct {
	Schema, Kind, Package, Directory, ObjectDir string
	Tool                                        Pin
	Args                                        []string
	Inputs                                      map[string]string
}

type ccRequest struct {
	Role, Directory, Output, OutputTarget string
	Args                                  []string
	Source                                Pin
	Inputs                                map[string]string
	Bindings                              map[string]string
	Expected                              []int
	Link                                  bool
	Support                               *compilerSupportOrigin
}

type compilerSupportOrigin struct {
	Package, ObjectDir, Trial string
	Producer, Types           Pin
}

var supportFlags = []string{"-fno-caret-diagnostics", "-Qunused-arguments", "-Wl,--no-gc-sections", "-fdebug-prefix-map=a=b", "-ffile-prefix-map=a=b", "-gno-record-gcc-switches", "-frandom-seed=1", "-no-pie", "-nopie"}
var linkProbeFlags = []string{"-Wl,--build-id=0x1234567890abcdef", "-Wl,--export-dynamic-symbol=main", "-Qunused-arguments", "-no-pie", "-nopie"}

func packageByName(p Plan, name string) (Package, error) {
	for _, q := range p.Packages {
		if q.ImportPath == name && len(q.Cgo) > 0 {
			return q, nil
		}
	}
	return Package{}, fmt.Errorf("request is outside actual Cgo packages")
}
func exactArgs(a, b []string) bool { return slices.Equal(a, b) }
func absArg(cwd, path string) string {
	if !filepath.IsAbs(path) {
		return filepath.Join(cwd, path)
	}
	return filepath.Clean(path)
}
func privateOutput(p Plan, cwd, path string) bool {
	path = absArg(cwd, path)
	if !under(p.Options.WorkRoot, path) {
		return false
	}
	dir, e := filepath.EvalSymlinks(filepath.Dir(path))
	if e != nil || dir != p.Options.WorkRoot && !under(p.Options.WorkRoot, dir) {
		return false
	}
	// A safe parent does not make an existing final symlink safe. GCC and Go
	// tools open this path themselves, so admit only absent or regular outputs.
	info, e := os.Lstat(path)
	return os.IsNotExist(e) || e == nil && info.Mode().IsRegular()
}

func privateInput(p Plan, path string) (Pin, error) {
	q, e := closedPin(path)
	if e != nil || !under(p.Options.WorkRoot, path) || !under(p.Options.WorkRoot, q.Path) {
		return Pin{}, fmt.Errorf("derived input outside private work: %s", path)
	}
	return q, nil
}

// classifyCC admits only the source-traced Linux/amd64 H08 request families.
// It resolves every input before any compiler (including -M) can start.
func classifyCC(p Plan, cwd string, args []string, stdin []byte, lease *Lease, derived []GeneratedSet) (r ccRequest, err error) {
	return classifyRequest(p, cwd, args, stdin, lease, derived, nil)
}

// A reader resolves historical transient inputs through the exact before-child
// snapshots, rather than requiring cmd/cgo's already removed file to reappear.
func classifyRequest(p Plan, cwd string, args []string, stdin []byte, lease *Lease, derived []GeneratedSet, recorded *operation) (r ccRequest, err error) {
	r = ccRequest{Directory: cwd, Args: slices.Clone(args), Inputs: map[string]string{}, Bindings: map[string]string{}, Expected: []int{0}}
	resolve := func(path string) (Pin, error) {
		var q Pin
		var e error
		if recorded == nil {
			q, e = closedPin(path)
		} else {
			canonical := recorded.Request.Bindings[path]
			q = Pin{Path: canonical, SHA256: recorded.Inputs[canonical]}
			snap, ok := recorded.Snapshots[canonical]
			if !ok || snap.SHA256 != q.SHA256 || q.Path == "" {
				return q, fmt.Errorf("request input snapshot omitted")
			}
			e = verifyPin(snap)
		}
		if e == nil {
			r.Bindings[path] = q.Path
		}
		return q, e
	}
	private := func(path string) (Pin, error) {
		q, e := resolve(path)
		if e != nil || !under(p.Options.WorkRoot, path) || !under(p.Options.WorkRoot, q.Path) {
			return Pin{}, fmt.Errorf("derived input outside private work: %s", path)
		}
		return q, nil
	}
	output := func(path string) bool {
		if recorded == nil {
			return privateOutput(p, cwd, path)
		}
		return under(p.Options.WorkRoot, absArg(cwd, path)) && under(p.Options.WorkRoot, recorded.Request.OutputTarget)
	}
	defer func() {
		if err == nil && r.Output != "" {
			if !output(r.Output) {
				err = fmt.Errorf("unsafe private compiler output")
				return
			}
			if recorded == nil {
				dir, _ := filepath.EvalSymlinks(filepath.Dir(r.Output))
				r.OutputTarget = filepath.Join(dir, filepath.Base(r.Output))
			} else {
				r.OutputTarget = recorded.Request.OutputTarget
			}
		}
	}()
	for _, q := range p.Queries {
		if exactArgs(args, q.Argv[1:]) {
			r.Role = "metadata"
			if len(stdin) != 0 {
				return r, fmt.Errorf("metadata query has unexpected stdin")
			}
			return r, nil
		}
	}
	for _, flag := range supportFlags {
		want := []string{flag, "-O2", "-g"}
		if !strings.HasPrefix(flag, "-Wl,") {
			want = append(want, "-c")
		}
		want = append(want, "-x", "c", "-", "-o", "/dev/null")
		if exactArgs(args, want) {
			if len(stdin) != 0 {
				return r, fmt.Errorf("Go support probe stdin must be empty")
			}
			r.Role = "go-flag-probe"
			r.Expected = []int{0, 1}
			r.Link = strings.HasPrefix(flag, "-Wl,")
			return r, nil
		}
	}
	if len(args) > 0 && args[0] == "-I" && slices.Contains(args, "-x") && slices.Contains(args, "-") {
		origin, e := prefixedCompilerSupport(p, cwd, args, stdin, derived)
		if e != nil {
			return r, e
		}
		for _, expected := range []Pin{origin.Producer, origin.Types} {
			q, e := resolve(expected.Path)
			if e != nil || q != expected {
				return r, fmt.Errorf("support probe Cgo context input differs: %v", e)
			}
			r.Inputs[q.Path] = q.SHA256
		}
		r.Role, r.Expected, r.Support = "go-flag-probe", []int{0, 1}, &origin
		return r, nil
	}
	if len(stdin) != 0 {
		return r, fmt.Errorf("unadmitted compiler stdin")
	}
	if len(args) == 0 {
		return r, fmt.Errorf("empty compiler request")
	}
	var source string
	var files []string
	flags := map[string]bool{}
	includes := []string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-o" || a == "-I":
			if i+1 == len(args) {
				return r, fmt.Errorf("missing compiler option value")
			}
			i++
			if a == "-o" {
				if r.Output != "" {
					return r, fmt.Errorf("duplicate compiler output")
				}
				r.Output = absArg(cwd, args[i])
			} else {
				includes = append(includes, absArg(cwd, args[i]))
			}
		case strings.HasPrefix(a, "-o") && len(a) > 2:
			if r.Output != "" {
				return r, fmt.Errorf("duplicate compiler output")
			}
			r.Output = absArg(cwd, a[2:])
		case strings.HasPrefix(a, "-I") && len(a) > 2:
			includes = append(includes, absArg(cwd, a[2:]))
		case slices.Contains([]string{"-E", "-dM", "-w", "-Wno-error", "-gdwarf-2", "-c", "-m64", "-fno-lto", "-O0", "-O2", "-g", "-Wall", "-Werror", "-fno-stack-protector", "-Wdeclaration-after-statement", "-std=gnu90", "-fdiagnostics-color=never", "-fPIC", "-pthread", "-fno-caret-diagnostics", "-fmessage-length=0", "-gno-record-gcc-switches", "-Wl,--no-gc-sections", "-no-pie", "-nopie", "-rdynamic", "-Wl,-z,now", "-Wl,-z,nocopyreloc"}, a):
			flags[a] = true
		case strings.HasPrefix(a, "-frandom-seed="):
			if !regexp.MustCompile(`^-frandom-seed=[A-Za-z0-9_-]{20}$`).MatchString(a) {
				return r, fmt.Errorf("invalid Go action seed")
			}
		case strings.HasPrefix(a, "-ffile-prefix-map=") || strings.HasPrefix(a, "-fdebug-prefix-map="):
			value := strings.SplitN(a, "=", 3)
			if len(value) != 3 {
				return r, fmt.Errorf("invalid Go prefix map")
			}
			from, to := value[1], value[2]
			if from != p.Options.Root && from != p.Options.GoRoot && !under(p.Options.WorkRoot, from) && from != p.Options.WorkRoot {
				return r, fmt.Errorf("unknown Go prefix-map input")
			}
			if to != "/_/GOROOT" && to != "/_/"+Module && to != "/tmp/go-build" {
				return r, fmt.Errorf("unknown Go prefix-map output")
			}
		case slices.Contains(linkProbeFlags, a):
			flags[a] = true
		case strings.HasPrefix(a, "-Wl,--build-id="):
			if !regexp.MustCompile(`^-Wl,--build-id=(0x[0-9a-f]+|none)$`).MatchString(a) {
				return r, fmt.Errorf("unadmitted link build-id")
			}
			flags[a] = true
		case strings.HasPrefix(a, "-Wl,--export-dynamic-symbol="):
			name := strings.TrimPrefix(a, "-Wl,--export-dynamic-symbol=")
			if !regexp.MustCompile(`^(main|_cgo_[A-Za-z0-9_]+|crosscall2)$`).MatchString(name) {
				return r, fmt.Errorf("unadmitted dynamic export")
			}
			flags[a] = true
		case a == "-ldl" || a == "-lpthread":
			flags[a] = true
		case strings.HasPrefix(a, "-") || strings.HasPrefix(a, "@"):
			return r, fmt.Errorf("unknown compiler/linker argument %s", a)
		default:
			path := absArg(cwd, a)
			files = append(files, path)
			if filepath.Ext(path) == ".c" || filepath.Ext(path) == ".S" {
				if source != "" {
					return r, fmt.Errorf("multiple native sources")
				}
				source = path
			}
		}
	}
	if source != "" && flags["-c"] || source != "" && flags["-E"] {
		if len(files) != 1 {
			return r, fmt.Errorf("compile has extra inputs")
		}
		if r.Output != "" && !output(r.Output) {
			return r, fmt.Errorf("compiler output outside private work")
		}
		if lease != nil && lease.Kind == "cgo" && strings.HasPrefix(filepath.Base(source), "cgo-gcc-input-") {
			pkg, e := packageByName(p, lease.Package)
			if e != nil {
				return r, e
			}
			if cwd != pkg.Directory || !under(filepath.Join(p.Options.WorkRoot, "cgo-tmp"), source) {
				return r, fmt.Errorf("Cgo temporary source outside active package")
			}
			for _, inc := range includes {
				if inc != pkg.Directory && inc != lease.ObjectDir {
					return r, fmt.Errorf("unknown translation include directory")
				}
			}
			if flags["-E"] && flags["-dM"] {
				r.Role = "cgo-defines"
			} else if flags["-w"] && flags["-Wno-error"] && flags["-gdwarf-2"] && flags["-fno-lto"] {
				r.Role = "cgo-dwarf"
				if flags["-O0"] {
					r.Role = "cgo-diagnostics"
					r.Expected = []int{0, 1}
				}
			} else {
				return r, fmt.Errorf("unknown Cgo translation profile")
			}
		} else {
			pkg, obj, e := compileContext(p, source, derived, resolve, recorded != nil)
			if e != nil {
				return r, e
			}
			if !flags["-O2"] || flags["-O0"] || !flags["-g"] || !flags["-fPIC"] || !flags["-m64"] || !flags["-pthread"] {
				return r, fmt.Errorf("native TU arithmetic/geometry flags differ")
			}
			for _, inc := range includes {
				if inc != pkg.Directory && inc != obj {
					return r, fmt.Errorf("unknown native include directory")
				}
			}
			r.Role = "native-tu"
		}
		q, e := resolve(source)
		if strings.HasPrefix(filepath.Base(source), "cgo-gcc-input-") {
			q, e = private(source)
		}
		if e != nil {
			return r, e
		}
		r.Source = q
		r.Inputs[q.Path] = q.SHA256
		return r, nil
	}
	// Go's external linker creates one literal probe under its own active lease.
	if source != "" && lease != nil && lease.Kind == "link" && filepath.Base(source) == "trivial.c" {
		if len(files) != 1 || !under(p.Options.WorkRoot, source) || !output(r.Output) {
			return r, fmt.Errorf("invalid Go linker probe paths")
		}
		bodyPath := source
		if recorded != nil {
			q, e := private(source)
			if e != nil {
				return r, e
			}
			bodyPath = recorded.Snapshots[q.Path].Path
		}
		body, e := os.ReadFile(bodyPath)
		if e != nil || string(body) != "int main() { return 0; }" {
			return r, fmt.Errorf("Go linker probe source differs")
		}
		known := false
		for _, flag := range linkProbeFlags {
			known = known || flags[flag]
		}
		if !known {
			return r, fmt.Errorf("unknown Go linker probe")
		}
		r.Role = "go-link-probe"
		r.Expected = []int{0, 1}
		r.Link = true
	} else if source == "" && len(files) > 0 && output(r.Output) {
		if lease != nil && lease.Kind == "link" {
			r.Role = "go-link"
		} else {
			ok := false
			for _, g := range derived {
				for _, q := range g.Files {
					if filepath.Dir(q.Path) == filepath.Dir(r.Output) && filepath.Base(r.Output) == "_cgo_.o" {
						ok = true
					}
				}
			}
			if !ok {
				return r, fmt.Errorf("link outside admitted Cgo object directory")
			}
			r.Role = "cgo-dynlink"
		}
		r.Link = true
	} else {
		return r, fmt.Errorf("unknown native request profile")
	}
	for _, file := range files {
		if !under(p.Options.WorkRoot, file) {
			return r, fmt.Errorf("link input outside private Go work")
		}
		q, e := private(file)
		if e != nil {
			return r, e
		}
		r.Inputs[q.Path] = q.SHA256
		if file == source {
			r.Source = q
		}
	}
	for _, inc := range includes {
		if !under(p.Options.WorkRoot, inc) {
			found := false
			for _, pkg := range p.Packages {
				if len(pkg.Cgo) > 0 && inc == pkg.Directory {
					found = true
				}
			}
			if !found {
				return r, fmt.Errorf("unknown link include path")
			}
		}
	}
	return r, nil
}

// ccompile (Go1.27 exec.go2327,2330,2375) passes the complete compilerCmd
// prefix to these three trials. compilerCmd's other flag trials use bare CC.
func prefixedCompilerSupport(p Plan, cwd string, args []string, stdin []byte, derived []GeneratedSet) (compilerSupportOrigin, error) {
	var found *compilerSupportOrigin
	if len(stdin) != 0 || len(args) < 9 {
		return compilerSupportOrigin{}, fmt.Errorf("invalid prefixed support probe stdin/shape")
	}
	tail := args[len(args)-9:]
	trial := tail[0]
	if !slices.Contains([]string{"-fdebug-prefix-map=a=b", "-ffile-prefix-map=a=b", "-frandom-seed=1"}, trial) || !slices.Equal(tail[1:], []string{"-O2", "-g", "-c", "-x", "c", "-", "-o", "/dev/null"}) {
		return compilerSupportOrigin{}, fmt.Errorf("unknown prefixed support trial/stdin/output")
	}
	prefix := args[:len(args)-9]
	for _, g := range derived {
		if g.Stage != "translation" || g.Package != "runtime/cgo" && g.Package != TensorPackage {
			continue
		}
		pkg, e := packageByName(p, g.Package)
		if e != nil || len(pkg.Cgo) == 0 {
			continue
		}
		types, ok := g.Files["_cgo_gotypes.go"]
		obj := filepath.Dir(types.Path)
		if !ok || !under(filepath.Join(p.Options.WorkRoot, "go-work"), obj) || cwd != filepath.Dir(obj) || !compilerCmdPrefix(prefix, pkg.Directory, obj) {
			continue
		}
		if found != nil {
			return compilerSupportOrigin{}, fmt.Errorf("ambiguous support probe Cgo context")
		}
		origin := compilerSupportOrigin{g.Package, obj, trial, g.Producer, types}
		found = &origin
	}
	if found == nil {
		return compilerSupportOrigin{}, fmt.Errorf("support probe prefix is not from captured Cgo context")
	}
	return *found, nil
}

func compilerCmdPrefix(args []string, sourceDir, objDir string) bool {
	base := []string{"-I", sourceDir, "-fPIC", "-m64", "-pthread"}
	if len(args) < len(base) || !slices.Equal(args[:len(base)], base) {
		return false
	}
	i := len(base)
	optional := func(flag string) {
		if i < len(args) && args[i] == flag {
			i++
		}
	}
	// Exact Linux/amd64 compilerCmd order; each optional is a source-defined
	// support result. No user flag, alternate include or arbitrary map is added.
	optional("-fno-caret-diagnostics")
	optional("-Qunused-arguments")
	optional("-Wl,--no-gc-sections")
	if i == len(args) || args[i] != "-fmessage-length=0" {
		return false
	}
	i++
	if i < len(args) && (args[i] == "-ffile-prefix-map="+objDir+"=/tmp/go-build" || args[i] == "-fdebug-prefix-map="+objDir+"=/tmp/go-build") {
		i++
	}
	optional("-gno-record-gcc-switches")
	return i == len(args)
}

func compileContext(p Plan, source string, derived []GeneratedSet, resolve func(string) (Pin, error), historical bool) (Package, string, error) {
	for _, pkg := range p.Packages {
		if len(pkg.Cgo) == 0 {
			continue
		}
		for _, file := range append(slices.Clone(pkg.C), pkg.Assembly...) {
			if source == filepath.Join(pkg.Directory, file) {
				q, e := resolve(source)
				if e != nil || p.Inputs[q.Path] != q.SHA256 {
					return Package{}, "", fmt.Errorf("native source changed")
				}
				for _, g := range derived {
					if g.Package == pkg.ImportPath && g.Stage == "translation" {
						if !historical {
							e = CheckGeneratedLive(g)
						}
						if e != nil {
							return Package{}, "", e
						}
						for _, q := range g.Files {
							return pkg, filepath.Dir(q.Path), nil
						}
					}
				}
				return Package{}, "", fmt.Errorf("Cgo generation must precede original C/S compilation")
			}
		}
	}
	for _, g := range derived {
		if !historical {
			if e := CheckGeneratedLive(g); e != nil {
				return Package{}, "", e
			}
		}
		for _, q := range g.Files {
			if source == q.Path {
				pkg, e := packageByName(p, g.Package)
				return pkg, filepath.Dir(source), e
			}
		}
	}
	return Package{}, "", fmt.Errorf("native source is not from actual Cgo roots")
}
