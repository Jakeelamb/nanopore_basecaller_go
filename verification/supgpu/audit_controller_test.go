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
	"strings"
)

// Metadata is private to the test controller; the frozen production reader's
// goPackage wire selection remains unchanged.
func auditSourceRoot(deps []byte) (string, error) {
	const module = "github.com/Jakeelamb/nanopore_basecaller_go"
	const target = module + "/verification/supgpu"
	decoder := json.NewDecoder(bytes.NewReader(deps))
	root, targets, packages := "", 0, 0
	for {
		var p struct {
			Dir, Root, ImportPath string
			Module                *struct {
				Path, Dir, GoMod string
				Main             bool
			}
		}
		err := decoder.Decode(&p)
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		packages++
		if packages > 10000 {
			return "", fmt.Errorf("too many test dependency packages")
		}
		name := strings.Split(p.ImportPath, " [")[0]
		project := name == module || strings.HasPrefix(name, module+"/")
		if !project && (p.Module == nil || !p.Module.Main) {
			continue
		}
		if !project || p.Module == nil || p.Module.Path != module || !p.Module.Main || !filepath.IsAbs(p.Root) || filepath.Clean(p.Root) != p.Root || p.Module.Dir != p.Root || p.Module.GoMod != filepath.Join(p.Root, "go.mod") {
			return "", fmt.Errorf("invalid audit source module/root evidence")
		}
		rel := strings.TrimPrefix(name, module+"/")
		if name == module {
			rel = "."
		}
		if name == target+".test" {
			rel = "verification/supgpu"
		}
		if p.Dir != filepath.Join(p.Root, rel) || (root != "" && root != p.Root) {
			return "", fmt.Errorf("ambiguous audit source root")
		}
		root = p.Root
		if p.ImportPath == target {
			targets++
		}
	}
	if root == "" || targets != 1 {
		return "", fmt.Errorf("missing/ambiguous audit source package root")
	}
	return root, nil
}

func auditControllerSourceRoot(c GoController) (string, error) {
	if err := verifyPin("", c.Dependencies); err != nil {
		return "", err
	}
	raw, err := os.ReadFile(c.Dependencies.Path)
	if err != nil {
		return "", err
	}
	return auditSourceRoot(raw)
}

// The test controller is separate from the frozen model consumer. Actual
// go-list -test output includes the generated test main as an absolute GoFile.
func auditTestInputs(deps, envRaw []byte, goexe string) ([]string, error) {
	root, err := auditSourceRoot(deps)
	if err != nil {
		return nil, err
	}
	var env goEnvironment
	if err := json.Unmarshal(envRaw, &env); err != nil {
		return nil, err
	}
	if env.CGO_ENABLED != "0" || env.GOOS != "linux" || env.GOARCH != "amd64" || env.GOAMD64 != "v1" || env.GOFLAGS != "" || env.GOWORK != "off" || env.GOENV != "" || !filepath.IsAbs(env.GOROOT) || !filepath.IsAbs(env.GOTOOLDIR) {
		return nil, fmt.Errorf("unsupported test-controller environment")
	}
	seen := map[string]bool{}
	decoder := json.NewDecoder(bytes.NewReader(deps))
	packages := 0
	for {
		var p goPackage
		err := decoder.Decode(&p)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		packages++
		if p.Error != nil || !filepath.IsAbs(p.Dir) || packages > 10000 {
			return nil, fmt.Errorf("invalid test dependency selection")
		}
		for _, names := range [][]string{p.GoFiles, p.CgoFiles, p.SFiles, p.HFiles, p.SysoFiles, p.EmbedFiles} {
			for _, name := range names {
				path := name
				if !filepath.IsAbs(path) {
					path = filepath.Join(p.Dir, name)
				}
				seen[path] = true
			}
		}
	}
	if packages == 0 {
		return nil, fmt.Errorf("empty test dependency selection")
	}
	for _, dir := range []string{env.GOTOOLDIR, filepath.Join(env.GOROOT, "pkg", "include")} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.IsDir() {
				return nil, fmt.Errorf("nested Go tool directory")
			}
			seen[filepath.Join(dir, e.Name())] = true
		}
	}
	if !filepath.IsAbs(goexe) {
		return nil, fmt.Errorf("actual Go executable must be absolute")
	}
	seen[goexe], seen[filepath.Join(env.GOROOT, "bin", "go")] = true, true
	for _, path := range append([]string{"go.mod", "verification/supgpu/audit-controls.sh"}, auditControlSources...) {
		seen[filepath.Join(root, path)] = true
	}
	paths := []string{}
	for path := range seen {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths, nil
}

func auditControllerMetadata(c GoController) ([]string, error) {
	if c.Schema != "nanopore-supgpu-audit-controller/v1" || len(c.Commands) != 3 || len(c.Commands[0].Args) == 0 {
		return nil, fmt.Errorf("invalid audit controller")
	}
	goexe := c.Commands[0].Args[0]
	expected := [][]string{{goexe, "env", "-json", "GOROOT", "GOTOOLDIR", "GOVERSION", "GOOS", "GOARCH", "GOAMD64", "CGO_ENABLED", "GOFLAGS", "GOWORK", "GOENV"}, {goexe, "list", "-deps", "-test", "-json", "./verification/supgpu"}, {goexe, "test", "-c", "-buildvcs=false", "-o", c.Binary.Path, "./verification/supgpu"}}
	for i, cmd := range c.Commands {
		if !reflect.DeepEqual(cmd.Args, expected[i]) || !reflect.DeepEqual(cmd.Environment, hostGoEnvironment()) {
			return nil, fmt.Errorf("audit controller command role changed")
		}
		if err := verifyCommand(cmd); err != nil {
			return nil, err
		}
	}
	if c.Commands[0].Stdout != c.Environment.Path || c.Commands[0].StdoutSHA != c.Environment.SHA256 || c.Commands[1].Stdout != c.Dependencies.Path || c.Commands[1].StdoutSHA != c.Dependencies.SHA256 {
		return nil, fmt.Errorf("audit metadata stdout binding changed")
	}
	deps, err := os.ReadFile(c.Dependencies.Path)
	if err != nil {
		return nil, err
	}
	env, err := os.ReadFile(c.Environment.Path)
	if err != nil {
		return nil, err
	}
	var observed goEnvironment
	if err = json.Unmarshal(env, &observed); err != nil || observed.GOVERSION != c.GoVersion {
		return nil, fmt.Errorf("audit Go toolchain metadata differs")
	}
	return auditTestInputs(deps, env, goexe)
}

func freezeAuditController(dir string) (GoController, error) {
	c := GoController{Schema: "nanopore-supgpu-audit-controller/v1", Snapshots: []Snapshot{}}
	var err error
	dir, err = filepath.Abs(dir)
	if err != nil {
		return c, err
	}
	exe, err := os.Executable()
	if err != nil {
		return c, err
	}
	c.Binary, err = pin(exe)
	if err != nil {
		return c, err
	}
	for _, label := range []string{"go-environment", "go-dependencies", "host-build"} {
		cmd, e := loadHostCommand(dir, label)
		if e != nil {
			return c, e
		}
		c.Commands = append(c.Commands, cmd)
	}
	c.Before, err = pin(filepath.Join(dir, "before-go-build.sha256"))
	if err != nil {
		return c, err
	}
	c.Environment, err = pin(c.Commands[0].Stdout)
	if err != nil {
		return c, err
	}
	c.Dependencies, err = pin(c.Commands[1].Stdout)
	if err != nil {
		return c, err
	}
	var env goEnvironment
	raw, err := os.ReadFile(c.Environment.Path)
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(raw, &env); err != nil {
		return c, err
	}
	info, ok := debug.ReadBuildInfo()
	if !ok || info.GoVersion != env.GOVERSION {
		return c, fmt.Errorf("audit executable Go toolchain differs")
	}
	c.GoVersion = env.GOVERSION
	paths, err := auditControllerMetadata(c)
	if err != nil {
		return c, err
	}
	raw, err = os.ReadFile(c.Before.Path)
	if err != nil {
		return c, err
	}
	before, err := parseSHAList(string(raw))
	if err != nil {
		return c, err
	}
	if len(before) != len(paths) {
		return c, fmt.Errorf("test-controller before-build selection differs")
	}
	if err = os.Mkdir(filepath.Join(dir, "go-inputs"), 0755); err != nil {
		return c, err
	}
	for _, path := range paths {
		p, e := pin(path)
		if e != nil {
			return c, e
		}
		if before[path] != p.SHA256 {
			return c, fmt.Errorf("test input changed during build: %s", path)
		}
		raw, e := os.ReadFile(path)
		if e != nil {
			return c, e
		}
		dst := filepath.Join(dir, "go-inputs", p.SHA256+filepath.Ext(path))
		if e = os.WriteFile(dst, raw, 0644); e != nil {
			return c, e
		}
		cp, e := pin(dst)
		if e != nil {
			return c, e
		}
		c.Snapshots = append(c.Snapshots, Snapshot{path, cp})
	}
	if err = writeJSON(filepath.Join(dir, "controller.json"), c); err != nil {
		return c, err
	}
	return c, nil
}

func readAuditController(p Pin, current bool) (GoController, error) {
	var c GoController
	if err := readPinnedJSON(p.Path, p.SHA256, &c); err != nil {
		return c, err
	}
	for _, v := range []Pin{c.Binary, c.Before, c.Environment, c.Dependencies} {
		if err := verifyPin("", v); err != nil {
			return c, err
		}
	}
	paths, err := auditControllerMetadata(c)
	if err != nil {
		return c, err
	}
	raw, err := os.ReadFile(c.Before.Path)
	if err != nil {
		return c, err
	}
	before, err := parseSHAList(string(raw))
	if err != nil {
		return c, err
	}
	if len(paths) != len(c.Snapshots) || len(before) != len(paths) {
		return c, fmt.Errorf("audit controller source membership changed")
	}
	for i, s := range c.Snapshots {
		if s.Original != paths[i] || before[s.Original] != s.Copy.SHA256 {
			return c, fmt.Errorf("audit controller source binding changed")
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
		root, e := auditControllerSourceRoot(c)
		if e != nil {
			return c, e
		}
		cwd, e := os.Getwd()
		if e != nil {
			return c, e
		}
		if cwd != root {
			return c, fmt.Errorf("current audit source root differs from working directory")
		}
		if err = verifyControllerBinary(c.Binary); err != nil {
			return c, err
		}
	}
	return c, nil
}
