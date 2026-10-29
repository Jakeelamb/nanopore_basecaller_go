package supgpu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type GoController struct {
	Schema         string                  `json:"schema"`
	Binary         Pin                     `json:"binary"`
	Before         Pin                     `json:"before_build_inputs"`
	Dependencies   Pin                     `json:"go_dependencies"`
	Environment    Pin                     `json:"go_environment"`
	Snapshots      []Snapshot              `json:"snapshots"`
	Commands       []runner.CommandReceipt `json:"commands"`
	ProjectGoFiles int                     `json:"project_go_files"`
	ProjectEmbeds  int                     `json:"project_embeds"`
	GoVersion      string                  `json:"go_version"`
}

type goEnvironment struct{ GOROOT, GOTOOLDIR, GOVERSION, GOOS, GOARCH, GOAMD64, CGO_ENABLED, GOFLAGS, GOWORK, GOENV string }
type goPackage struct {
	Dir, Root, ImportPath string
	Module                *struct {
		Path, Dir, GoMod string
		Main             bool
	}
	Standard                                                 bool
	GoFiles, CgoFiles, SFiles, HFiles, SysoFiles, EmbedFiles []string
	Error                                                    *struct{ Err string }
}

func hostGoEnvironment() []string {
	return []string{"GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOFLAGS=", "GOWORK=off", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1"}
}

func selectedGoInputs(deps, envRaw []byte, goExecutable string) ([]string, int, int, goEnvironment, error) {
	root, err := os.Getwd()
	if err != nil {
		return nil, 0, 0, goEnvironment{}, err
	}
	return selectedGoInputsAtRoot(deps, envRaw, goExecutable, root)
}

func selectedGoInputsAtRoot(deps, envRaw []byte, goExecutable, root string) ([]string, int, int, goEnvironment, error) {
	var env goEnvironment
	if err := json.Unmarshal(envRaw, &env); err != nil {
		return nil, 0, 0, env, err
	}
	if env.CGO_ENABLED != "0" || env.GOOS != "linux" || env.GOARCH != "amd64" || env.GOAMD64 != "v1" || env.GOFLAGS != "" || env.GOWORK != "off" || env.GOENV != "" || !filepath.IsAbs(env.GOROOT) || !filepath.IsAbs(env.GOTOOLDIR) {
		return nil, 0, 0, env, fmt.Errorf("unsupported Go build environment")
	}
	seen := map[string]bool{}
	projectGo, projectEmbeds := 0, 0
	decoder := json.NewDecoder(bytes.NewReader(deps))
	packages := 0
	for {
		var pkg goPackage
		err := decoder.Decode(&pkg)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, 0, env, err
		}
		packages++
		if packages > 10000 || pkg.Error != nil || !filepath.IsAbs(pkg.Dir) {
			return nil, 0, 0, env, fmt.Errorf("invalid actual Go dependency package")
		}
		if !pkg.Standard {
			projectGo += len(pkg.GoFiles) + len(pkg.CgoFiles)
			projectEmbeds += len(pkg.EmbedFiles)
		}
		for _, names := range [][]string{pkg.GoFiles, pkg.CgoFiles, pkg.SFiles, pkg.HFiles, pkg.SysoFiles, pkg.EmbedFiles} {
			for _, name := range names {
				path := filepath.Join(pkg.Dir, name)
				if !filepath.IsAbs(path) {
					return nil, 0, 0, env, fmt.Errorf("relative Go input")
				}
				seen[path] = true
			}
		}
	}
	if packages == 0 || projectGo == 0 || (projectEmbeds != 2 && projectEmbeds != 3) {
		return nil, 0, 0, env, fmt.Errorf("empty/incomplete Go dependency closure")
	}
	for _, dir := range []string{env.GOTOOLDIR, filepath.Join(env.GOROOT, "pkg", "include")} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, 0, 0, env, err
		}
		for _, entry := range entries {
			if entry.IsDir() {
				return nil, 0, 0, env, fmt.Errorf("unexpected nested Go tool/include directory")
			}
			seen[filepath.Join(dir, entry.Name())] = true
		}
	}
	goBin := filepath.Join(env.GOROOT, "bin", "go")
	seen[goBin] = true
	if !filepath.IsAbs(goExecutable) {
		return nil, 0, 0, env, fmt.Errorf("actual Go executable must be absolute")
	}
	seen[goExecutable] = true
	for _, path := range []string{"go.mod", "verification/supgpu/host-controls.sh"} {
		seen[filepath.Join(root, path)] = true
	}
	paths := make([]string, 0, len(seen))
	for path := range seen {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths, projectGo, projectEmbeds, env, nil
}

func loadHostCommand(dir, label string) (runner.CommandReceipt, error) {
	c := runner.CommandReceipt{Environment: hostGoEnvironment(), Stdout: filepath.Join(dir, label+".stdout"), Stderr: filepath.Join(dir, label+".stderr")}
	args, err := os.ReadFile(filepath.Join(dir, label+".argv.nul"))
	if err != nil {
		return c, err
	}
	if len(args) == 0 || args[len(args)-1] != 0 {
		return c, fmt.Errorf("missing actual argv")
	}
	c.Args = strings.Split(string(args[:len(args)-1]), "\x00")
	read := func(suffix string) (string, error) {
		raw, err := os.ReadFile(filepath.Join(dir, label+suffix))
		return strings.TrimSpace(string(raw)), err
	}
	if c.Started, err = read(".started"); err != nil {
		return c, err
	}
	if c.Finished, err = read(".finished"); err != nil {
		return c, err
	}
	exit, err := read(".exit")
	if err != nil {
		return c, err
	}
	if c.Exit, err = strconv.Atoi(exit); err != nil {
		return c, err
	}
	for _, v := range []struct {
		path string
		sha  *string
	}{{c.Stdout, &c.StdoutSHA}, {c.Stderr, &c.StderrSHA}} {
		p, err := pin(v.path)
		if err != nil {
			return c, err
		}
		*v.sha = p.SHA256
	}
	return c, verifyCommand(c)
}

// FreezeController authenticates the before-build source/tool manifest against
// actual go-list selection, then retains the executable that is running it.
// It runs no compiler or child. The host script records the Go build separately.
func FreezeController(dir string) (GoController, error) {
	c := GoController{Schema: "nanopore-supgpu-go-controller/v1", Snapshots: []Snapshot{}, Commands: []runner.CommandReceipt{}}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return c, err
	}
	dir = abs
	for _, label := range []string{"go-environment", "go-dependencies", "host-build"} {
		cmd, err := loadHostCommand(dir, label)
		if err != nil {
			return c, err
		}
		c.Commands = append(c.Commands, cmd)
	}
	goExecutable := c.Commands[0].Args[0]
	if !reflect.DeepEqual(c.Commands[0].Args, []string{goExecutable, "env", "-json", "GOROOT", "GOTOOLDIR", "GOVERSION", "GOOS", "GOARCH", "GOAMD64", "CGO_ENABLED", "GOFLAGS", "GOWORK", "GOENV"}) || !reflect.DeepEqual(c.Commands[1].Args, []string{goExecutable, "list", "-deps", "-json", "./verification/supgpu/cmd"}) {
		return c, fmt.Errorf("Go metadata command changed")
	}
	deps, err := os.ReadFile(c.Commands[1].Stdout)
	if err != nil {
		return c, err
	}
	envRaw, err := os.ReadFile(c.Commands[0].Stdout)
	if err != nil {
		return c, err
	}
	paths, ng, ne, env, err := selectedGoInputs(deps, envRaw, goExecutable)
	if err != nil {
		return c, err
	}
	c.ProjectGoFiles, c.ProjectEmbeds, c.GoVersion = ng, ne, env.GOVERSION
	beforePath := filepath.Join(dir, "before-go-build.sha256")
	before, err := os.ReadFile(beforePath)
	if err != nil {
		return c, err
	}
	expected, err := parseSHAList(string(before))
	if err != nil {
		return c, err
	}
	if len(expected) != len(paths) {
		return c, fmt.Errorf("before-build Go source/tool set differs from actual selection")
	}
	exe, err := os.Executable()
	if err != nil {
		return c, err
	}
	exe, err = filepath.Abs(exe)
	if err != nil {
		return c, err
	}
	if !reflect.DeepEqual(c.Commands[2].Args, []string{goExecutable, "build", "-buildvcs=false", "-o", exe, "./verification/supgpu/cmd"}) {
		return c, fmt.Errorf("actual controller differs from Go build output")
	}
	info, ok := debug.ReadBuildInfo()
	if !ok || info.GoVersion != env.GOVERSION {
		return c, fmt.Errorf("executing controller Go toolchain differs")
	}
	c.Binary, err = pin(exe)
	if err != nil {
		return c, err
	}
	c.Before, err = pin(beforePath)
	if err != nil {
		return c, err
	}
	c.Dependencies, err = pin(c.Commands[1].Stdout)
	if err != nil {
		return c, err
	}
	c.Environment, err = pin(c.Commands[0].Stdout)
	if err != nil {
		return c, err
	}
	if err = os.Mkdir(filepath.Join(dir, "go-inputs"), 0755); err != nil {
		return c, err
	}
	for _, path := range paths {
		v, err := pin(path)
		if err != nil {
			return c, err
		}
		if expected[path] != v.SHA256 {
			return c, fmt.Errorf("Go source/tool changed during build: %s", path)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return c, err
		}
		copyPath := filepath.Join(dir, "go-inputs", v.SHA256+filepath.Ext(path))
		if err = os.WriteFile(copyPath, raw, 0644); err != nil {
			return c, err
		}
		copy, err := pin(copyPath)
		if err != nil {
			return c, err
		}
		c.Snapshots = append(c.Snapshots, Snapshot{path, copy})
	}
	if err = writeJSON(filepath.Join(dir, "controller.json"), c); err != nil {
		return c, err
	}
	return c, nil
}

func parseSHAList(body string) (map[string]string, error) {
	out := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(body), "\n") {
		if len(line) < 67 || line[64:66] != "  " || !digestRE.MatchString(line[:64]) || !filepath.IsAbs(line[66:]) {
			return nil, fmt.Errorf("invalid before-build SHA manifest")
		}
		path := line[66:]
		if _, ok := out[path]; ok {
			return nil, fmt.Errorf("duplicate before-build input")
		}
		out[path] = line[:64]
	}
	return out, nil
}

func readController(p Pin, current bool) (GoController, error) {
	var c GoController
	if err := verifyPin("", p); err != nil {
		return c, err
	}
	if err := readPinnedJSON(p.Path, p.SHA256, &c); err != nil {
		return c, err
	}
	if c.Schema != "nanopore-supgpu-go-controller/v1" || len(c.Commands) != 3 {
		return c, fmt.Errorf("invalid Go controller")
	}
	for _, v := range []Pin{c.Binary, c.Before, c.Dependencies, c.Environment} {
		if err := verifyPin("", v); err != nil {
			return c, err
		}
	}
	for _, cmd := range c.Commands {
		if err := verifyCommand(cmd); err != nil {
			return c, err
		}
		if !reflect.DeepEqual(cmd.Environment, hostGoEnvironment()) {
			return c, fmt.Errorf("Go controller environment changed")
		}
	}
	if err := verifyGoCommandRoles(c); err != nil {
		return c, err
	}
	deps, err := os.ReadFile(c.Dependencies.Path)
	if err != nil {
		return c, err
	}
	envRaw, err := os.ReadFile(c.Environment.Path)
	if err != nil {
		return c, err
	}
	var root string
	if current {
		root, err = os.Getwd()
	} else {
		root, err = controllerSourceRoot(deps, "github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu/cmd")
	}
	if err != nil {
		return c, err
	}
	paths, ng, ne, env, err := selectedGoInputsAtRoot(deps, envRaw, c.Commands[0].Args[0], root)
	if err != nil {
		return c, err
	}
	if ng != c.ProjectGoFiles || ne != c.ProjectEmbeds || env.GOVERSION != c.GoVersion || len(paths) != len(c.Snapshots) {
		return c, fmt.Errorf("Go controller dependency membership changed")
	}
	before, err := os.ReadFile(c.Before.Path)
	if err != nil {
		return c, err
	}
	expected, err := parseSHAList(string(before))
	if err != nil {
		return c, err
	}
	if len(expected) != len(paths) {
		return c, fmt.Errorf("Go controller input closure differs")
	}
	for i, s := range c.Snapshots {
		if s.Original != paths[i] || expected[s.Original] != s.Copy.SHA256 {
			return c, fmt.Errorf("Go source/tool binding changed")
		}
		if err = verifyPin("", s.Copy); err != nil {
			return c, err
		}
		if current {
			if err = verifyCurrentSnapshot(s); err != nil {
				return c, err
			}
		}
	}
	if current {
		if err = verifyControllerBinary(c.Binary); err != nil {
			return c, err
		}
	}
	return c, nil
}

func verifyGoCommandRoles(c GoController) error {
	if len(c.Commands) != 3 || len(c.Commands[0].Args) == 0 {
		return fmt.Errorf("incomplete Go command roles")
	}
	goexe := c.Commands[0].Args[0]
	expected := [][]string{{goexe, "env", "-json", "GOROOT", "GOTOOLDIR", "GOVERSION", "GOOS", "GOARCH", "GOAMD64", "CGO_ENABLED", "GOFLAGS", "GOWORK", "GOENV"}, {goexe, "list", "-deps", "-json", "./verification/supgpu/cmd"}, {goexe, "build", "-buildvcs=false", "-o", c.Binary.Path, "./verification/supgpu/cmd"}}
	for i, args := range expected {
		if !reflect.DeepEqual(c.Commands[i].Args, args) {
			return fmt.Errorf("Go producer command role changed")
		}
	}
	if c.Commands[0].Stdout != c.Environment.Path || c.Commands[0].StdoutSHA != c.Environment.SHA256 || c.Commands[1].Stdout != c.Dependencies.Path || c.Commands[1].StdoutSHA != c.Dependencies.SHA256 {
		return fmt.Errorf("Go metadata stdout binding changed")
	}
	return nil
}

func verifyCurrentSnapshot(s Snapshot) error {
	actual, err := pin(s.Original)
	if err != nil {
		return err
	}
	if actual.SHA256 != s.Copy.SHA256 || actual.Bytes != s.Copy.Bytes {
		return fmt.Errorf("Go source/tool drift before native phase: %s", s.Original)
	}
	return nil
}
func verifyControllerBinary(expected Pin) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	actual, err := pin(exe)
	if err != nil {
		return err
	}
	if actual != expected {
		return fmt.Errorf("executing controller binary differs from frozen Go build")
	}
	return nil
}
