package cgobuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"text/template"
	"unicode"
	"unicode/utf8"
)

type BuildRole string

const (
	H08Command      BuildRole = "h08-command"
	CPU13Tests      BuildRole = "cpu13-tensor-tests"
	SUPGraphCommand BuildRole = "sup-graph-command"
	TensorPackage             = Module + "/internal/tensor"
	testSuffix                = " [" + TensorPackage + ".test]"
	OverlayName               = "h08_phase_main_test.go"
)

type CPU13Profile struct {
	Template                                Pin
	OverlayTarget, OverlayPath, OverlayJSON string
	TestCopies                              []string
	MainSource, MainSHA256                  string
}

func digest(body []byte) string { q := sha256.Sum256(body); return hex.EncodeToString(q[:]) }
func normalizedRole(role BuildRole) (BuildRole, error) {
	if role == "" {
		role = H08Command
	}
	if role != H08Command && role != CPU13Tests && role != SUPGraphCommand {
		return "", fmt.Errorf("unadmitted fixed build role %q", role)
	}
	return role, nil
}
func commandTarget(role BuildRole) (importPath, basename string, err error) {
	switch role {
	case H08Command:
		return Target, "h08", nil
	case SUPGraphCommand:
		return Module + "/verification/supgpu/graph/cmd", "sup-graph", nil
	default:
		return "", "", fmt.Errorf("unadmitted command role %q", role)
	}
}
func commandArgument(role BuildRole) string {
	path, _, e := commandTarget(role)
	if e != nil {
		return ""
	}
	return "./" + strings.TrimPrefix(path, Module+"/")
}
func outputPath(p Plan) string {
	_, name, e := commandTarget(p.Role)
	if p.Role == CPU13Tests {
		name = "tensor.test"
	} else if e != nil {
		return ""
	}
	return filepath.Join(p.Options.WorkRoot, name)
}
func operationInputs(p Plan, s Session) map[string]string {
	inputs := maps.Clone(p.Inputs)
	if s.Phase == "build" && p.Role == CPU13Tests {
		inputs[p.CPU13.OverlayPath] = digest([]byte(p.CPU13.OverlayJSON))
	}
	return inputs
}
func validateMainInputs(p Plan, name string, args []string, inputs map[string]string) error {
	if p.Role != CPU13Tests {
		return nil
	}
	if name == TensorPackage+".test" {
		count := 0
		for _, a := range args {
			if !strings.HasSuffix(a, ".go") {
				continue
			}
			if filepath.Base(a) != "_testmain.go" || !under(p.Options.WorkRoot, a) || inputs[a] != p.CPU13.MainSHA256 {
				return fmt.Errorf("generated testmain differs before compilation")
			}
			count++
		}
		if count != 1 {
			return fmt.Errorf("exact generated testmain input omitted")
		}
	}
	if name == TensorPackage+"_test"+testSuffix {
		if !slices.Contains(args, p.CPU13.Template.Path) || inputs[p.CPU13.Template.Path] != p.CPU13.Template.SHA256 {
			return fmt.Errorf("external TestMain template input omitted/rebound")
		}
	}
	return nil
}
func listArgs(p Plan) []string {
	args := []string{filepath.Join(p.Options.GoRoot, "bin/go"), "list", "-deps", "-json"}
	if p.Role == CPU13Tests {
		return append(args, "-test", "-overlay="+p.CPU13.OverlayPath, "./internal/tensor")
	}
	return append(args, commandArgument(p.Role))
}
func plannedBuildArgs(p Plan) []string {
	args := []string{filepath.Join(p.Options.GoRoot, "bin/go"), "build"}
	if p.Role == CPU13Tests {
		args[1] = "test"
		args = append(args, "-c", "-vet=off")
	}
	args = append(args, "-a", "-x", "-work", "-p=1", "-pgo=off", "-trimpath", "-buildvcs=false", "-toolexec=<SEALED_CURRENT_INTERPOSER_TOOL>", "-ldflags=-linkmode=external -compressdwarf=false -buildid= -extld=<SEALED_CURRENT_INTERPOSER_CC>")
	if p.Role == CPU13Tests {
		return append(args, "-overlay="+p.CPU13.OverlayPath, "-o", outputPath(p), "./internal/tensor")
	}
	return append(args, "-o", outputPath(p), commandArgument(p.Role))
}
func sourcePackage(p Plan, name string) (Package, bool) {
	for _, q := range p.Packages {
		if q.ImportPath == name {
			return q, true
		}
	}
	return Package{}, false
}
func testDescription(p Plan, desc string) (base string, variant bool, ok bool) {
	if _, ok = sourcePackage(p, desc); ok {
		return desc, false, true
	}
	if p.Role != CPU13Tests || p.CPU13 == nil {
		return "", false, false
	}
	if desc == TensorPackage+".test" || desc == TensorPackage+"_test"+testSuffix {
		return desc, true, true
	}
	if strings.HasSuffix(desc, testSuffix) {
		base = strings.TrimSuffix(desc, testSuffix)
		return base, true, slices.Contains(p.CPU13.TestCopies, base)
	}
	return "", false, false
}
func cpu13Profile(p Plan) (*CPU13Profile, error) {
	q, e := closedPin(filepath.Join(p.Options.Root, "verification/hacgpu/h08/cpu13/testmain.go.txt"))
	if e != nil {
		return nil, e
	}
	if p.Inputs[q.Path] != q.SHA256 {
		return nil, fmt.Errorf("TestMain template absent from current source closure")
	}
	profile := &CPU13Profile{Template: q, OverlayTarget: filepath.Join(p.Options.Root, "internal/tensor", OverlayName), OverlayPath: filepath.Join(p.Options.WorkRoot, "cpu13-overlay.json")}
	if _, e = os.Lstat(profile.OverlayTarget); !os.IsNotExist(e) {
		return nil, fmt.Errorf("CPU13 overlay is add-only; target already exists or is inaccessible")
	}
	parent, e := filepath.EvalSymlinks(filepath.Dir(profile.OverlayTarget))
	if e != nil || parent != filepath.Join(p.Options.Root, "internal/tensor") {
		return nil, fmt.Errorf("CPU13 overlay target parent differs")
	}
	body, e := json.Marshal(struct{ Replace map[string]string }{map[string]string{profile.OverlayTarget: profile.Template.Path}})
	if e != nil {
		return nil, e
	}
	profile.OverlayJSON = string(body)
	copies := map[string]bool{TensorPackage: true}
	for changed := true; changed; {
		changed = false
		for _, pkg := range p.Packages {
			if copies[pkg.ImportPath] {
				continue
			}
			for _, name := range pkg.Imports {
				if copies[name] {
					copies[pkg.ImportPath], changed = true, true
					break
				}
			}
		}
	}
	for name := range copies {
		profile.TestCopies = append(profile.TestCopies, name)
	}
	slices.Sort(profile.TestCopies)
	profile.MainSource, e = expectedTestMain(p, profile.Template)
	if e != nil {
		return nil, e
	}
	profile.MainSHA256 = digest([]byte(profile.MainSource))
	return profile, nil
}

type mainFunction struct {
	Package, Name, Output string
	Unordered             bool
}
type mainData struct {
	Tests, Benchmarks, FuzzTargets, Examples     []mainFunction
	TestMain                                     *mainFunction
	Package                                      struct{ ImportPath string }
	ImportTest, NeedTest, ImportXtest, NeedXtest bool
	Cover                                        any
}

func (mainData) ModulePath() string { return Module }
func (mainData) ImportPath() string { return TensorPackage }

func expectedTestMain(p Plan, overlay Pin) (string, error) {
	pkg, ok := sourcePackage(p, TensorPackage)
	if !ok {
		return "", fmt.Errorf("maintained tensor source package omitted")
	}
	d := mainData{ImportTest: true, NeedTest: true, ImportXtest: true, NeedXtest: true, TestMain: &mainFunction{Package: "_xtest", Name: "TestMain"}}
	d.Package.ImportPath = TensorPackage
	for _, group := range []struct {
		Package string
		Files   []string
	}{{"_test", pkg.TestGo}, {"_xtest", pkg.XTestGo}} {
		for _, name := range group.Files {
			f, e := parser.ParseFile(token.NewFileSet(), filepath.Join(pkg.Directory, name), nil, 0)
			if e != nil {
				return "", e
			}
			for _, decl := range f.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Recv != nil {
					continue
				}
				n := fn.Name.Name
				for _, prefix := range []string{"Example", "Benchmark", "Fuzz", "TestMain"} {
					if strings.HasPrefix(n, prefix) {
						return "", fmt.Errorf("unsupported tensor test entry %s", n)
					}
				}
				if !strings.HasPrefix(n, "Test") {
					continue
				}
				r, _ := utf8.DecodeRuneInString(strings.TrimPrefix(n, "Test"))
				if unicode.IsLower(r) {
					continue
				}
				if fn.Type.Params.NumFields() != 1 || fn.Type.Results.NumFields() != 0 || fn.Type.TypeParams.NumFields() != 0 {
					return "", fmt.Errorf("invalid maintained test signature")
				}
				ptr, ok := fn.Type.Params.List[0].Type.(*ast.StarExpr)
				if !ok {
					return "", fmt.Errorf("invalid maintained test parameter")
				}
				valid := false
				if x, ok := ptr.X.(*ast.Ident); ok {
					valid = x.Name == "T"
				}
				if x, ok := ptr.X.(*ast.SelectorExpr); ok {
					valid = x.Sel.Name == "T"
				}
				if !valid {
					return "", fmt.Errorf("invalid maintained test parameter")
				}
				d.Tests = append(d.Tests, mainFunction{Package: group.Package, Name: n})
			}
		}
	}
	if len(d.Tests) == 0 {
		return "", fmt.Errorf("maintained test functions omitted")
	}
	f, e := parser.ParseFile(token.NewFileSet(), overlay.Path, nil, 0)
	if e != nil {
		return "", e
	}
	if f.Name.Name != "tensor_test" {
		return "", fmt.Errorf("CPU13 TestMain is not external tensor_test")
	}
	var funcs []string
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcs = append(funcs, fn.Name.Name)
		}
	}
	if !slices.Equal(funcs, []string{"TestMain"}) {
		return "", fmt.Errorf("CPU13 overlay must only wrap TestMain")
	}
	gen := filepath.Join(p.Options.GoRoot, "src/cmd/go/internal/load/test.go")
	f, e = parser.ParseFile(token.NewFileSet(), gen, nil, 0)
	if e != nil {
		return "", e
	}
	var text string
	ast.Inspect(f, func(n ast.Node) bool {
		v, ok := n.(*ast.ValueSpec)
		if !ok || len(v.Names) != 1 || v.Names[0].Name != "testmainTmpl" || len(v.Values) != 1 {
			return true
		}
		call, ok := v.Values[0].(*ast.CallExpr)
		if !ok || len(call.Args) != 2 {
			return false
		}
		lit, ok := call.Args[1].(*ast.BasicLit)
		if ok {
			text, _ = strconv.Unquote(lit.Value)
		}
		return false
	})
	if text == "" || p.Inputs[gen] == "" {
		return "", fmt.Errorf("pinned Go testmain generator template omitted")
	}
	t, e := template.New("main").Parse(text)
	if e != nil {
		return "", e
	}
	var out bytes.Buffer
	if e = t.Execute(&out, d); e != nil {
		return "", e
	}
	return out.String(), nil
}

func validateTarget(p Plan) error {
	role, e := normalizedRole(p.Role)
	if p.Schema != "nanopore-h08-cgo-build-plan/v2" || e != nil || role != p.Role || p.Options.Role != p.Role {
		return fmt.Errorf("explicit plan role differs")
	}
	if role != CPU13Tests {
		if _, _, e := commandTarget(role); e != nil {
			return e
		}
		if p.CPU13 != nil {
			return fmt.Errorf("ordinary command cannot carry a test overlay")
		}
	} else {
		c := p.CPU13
		if c == nil || c.OverlayTarget != filepath.Join(p.Options.Root, "internal/tensor", OverlayName) || c.OverlayPath != filepath.Join(p.Options.WorkRoot, "cpu13-overlay.json") || c.Template.Path != filepath.Join(p.Options.Root, "verification/hacgpu/h08/cpu13/testmain.go.txt") || p.Inputs[c.Template.Path] != c.Template.SHA256 || c.MainSource == "" || digest([]byte(c.MainSource)) != c.MainSHA256 {
			return fmt.Errorf("fixed CPU13 source/overlay/generator binding differs")
		}
		body, _ := json.Marshal(struct{ Replace map[string]string }{map[string]string{c.OverlayTarget: c.Template.Path}})
		if c.OverlayJSON != string(body) {
			return fmt.Errorf("CPU13 overlay mapping differs")
		}
	}
	if !slices.Equal(p.BuildArgv, plannedBuildArgs(p)) {
		return fmt.Errorf("fixed target build argv differs")
	}
	return nil
}
