package closedbuild

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Controller struct {
	Schema, Status, Error, Started, Finished, GoRoot string
	Binaries                                         map[string]Pin
	Tools, Inputs                                    map[string]string
	Snapshots                                        map[string]Pin
	Lists                                            []Pin
	Commands                                         []runner.CommandReceipt
	Limits                                           map[string]string
}
type ControllerOptions struct {
	Out, GoRoot string
	Admitted    bool
}
type goPackage struct {
	Dir                                                                        string
	GoFiles, CgoFiles, CFiles, CXXFiles, HFiles, SFiles, SysoFiles, EmbedFiles []string
	Module                                                                     *struct{ GoMod string }
	Error                                                                      *struct{ Err string }
}

func selectedGo(body []byte) ([]string, error) {
	d := json.NewDecoder(bytes.NewReader(body))
	set := map[string]bool{}
	for {
		var p goPackage
		e := d.Decode(&p)
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, e
		}
		if p.Error != nil {
			return nil, fmt.Errorf("Go dependency selection: %s", p.Error.Err)
		}
		if p.Dir == "" || !filepath.IsAbs(p.Dir) || len(p.CgoFiles)+len(p.CFiles)+len(p.CXXFiles) != 0 {
			return nil, fmt.Errorf("CGO0 source selection required")
		}
		for _, files := range [][]string{p.GoFiles, p.HFiles, p.SFiles, p.SysoFiles, p.EmbedFiles} {
			for _, s := range files {
				if filepath.IsAbs(s) || s == ".." || strings.HasPrefix(s, "../") {
					return nil, fmt.Errorf("invalid Go source member")
				}
				set[filepath.Join(p.Dir, s)] = true
			}
		}
		if p.Module != nil && p.Module.GoMod != "" {
			set[p.Module.GoMod] = true
		}
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("empty Go source selection")
	}
	return slices.Sorted(maps.Keys(set)), nil
}
func staticTools(goRoot string) (map[string]string, error) {
	m := map[string]string{}
	paths := []string{filepath.Join(goRoot, "bin/go")}
	entries, e := os.ReadDir(filepath.Join(goRoot, "pkg/tool/linux_amd64"))
	if e != nil {
		return nil, e
	}
	for _, d := range entries {
		if !d.IsDir() {
			paths = append(paths, filepath.Join(goRoot, "pkg/tool/linux_amd64", d.Name()))
		}
	}
	for _, p := range paths {
		meta, e := nativeproof.InspectELF(p)
		if e != nil {
			return nil, e
		}
		if meta.Interpreter != "" || len(meta.Needed) != 0 {
			return nil, fmt.Errorf("Go-only tool must be static: %s", p)
		}
		if e = add(m, meta.Pin); e != nil {
			return nil, e
		}
	}
	for _, path := range []string{filepath.Join(goRoot, "go.env"), filepath.Join(goRoot, "VERSION")} {
		p, e := PinFile(path)
		if e != nil {
			return nil, e
		}
		if e = add(m, p); e != nil {
			return nil, e
		}
	}
	e = filepath.WalkDir(filepath.Join(goRoot, "pkg/include"), func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() {
			return nil
		}
		p, e := PinFile(path)
		if e != nil {
			return e
		}
		return add(m, p)
	})
	return m, e
}
func FreezeControllers(o ControllerOptions) (c Controller, err error) {
	if !o.Admitted {
		return c, fmt.Errorf("%w: explicit Go-only controller admission required", runner.ErrAdmissionBlocked)
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return c, err
	}
	defer lock.Close()
	limits, err := Limits()
	if err != nil {
		return c, err
	}
	root, err := os.Getwd()
	if err != nil {
		return c, err
	}
	goRoot, err := filepath.EvalSymlinks(o.GoRoot)
	if err != nil {
		return c, err
	}
	tools, err := staticTools(goRoot)
	if err != nil {
		return c, err
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return c, err
	}
	c = Controller{Schema: "nanopore-tensor-closed-controller/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), GoRoot: goRoot, Tools: tools, Inputs: maps.Clone(tools), Binaries: map[string]Pin{}, Snapshots: map[string]Pin{}, Limits: limits}
	defer func() {
		c.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			c.Error = err.Error()
		}
		err = errors.Join(err, Write(filepath.Join(out, "controller.json"), c))
	}()
	cache := filepath.Join(out, "cache")
	temp := filepath.Join(out, "tmp")
	if err = os.Mkdir(cache, 0700); err != nil {
		return c, err
	}
	if err = os.Mkdir(temp, 0700); err != nil {
		return c, err
	}
	env := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "GOROOT=" + goRoot, "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off", "GOFLAGS=", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "CGO_ENABLED=0", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "GOCACHE=" + cache, "GOTMPDIR=" + temp, "GOPATH=" + filepath.Join(out, "gopath")}
	goBin := filepath.Join(goRoot, "bin/go")
	run := func(label string, args ...string) (runner.CommandReceipt, error) {
		if e := checkPins(c.Inputs); e != nil {
			return runner.CommandReceipt{}, e
		}
		r, e := command(out, label, root, append([]string{goBin}, args...), env)
		c.Commands = append(c.Commands, r)
		return r, errors.Join(e, checkPins(c.Inputs))
	}
	r, e := run("go-env", "env", "-json", "GOROOT", "GOTOOLDIR", "CGO_ENABLED", "GOOS", "GOARCH")
	if e != nil {
		return c, e
	}
	raw, e := os.ReadFile(r.Stdout)
	if e != nil {
		return c, e
	}
	var ge map[string]string
	if e = json.Unmarshal(raw, &ge); e != nil {
		return c, e
	}
	if ge["GOROOT"] != goRoot || ge["GOTOOLDIR"] != filepath.Join(goRoot, "pkg/tool/linux_amd64") || ge["CGO_ENABLED"] != "0" || ge["GOOS"] != "linux" || ge["GOARCH"] != "amd64" {
		return c, fmt.Errorf("Go selection differs")
	}
	listArgs := []string{"list", "-deps", "-json", "./verification/tensor/closedbuild/cmd", "./verification/tensor/build"}
	r, e = run("go-list-before", listArgs...)
	if e != nil {
		return c, e
	}
	raw, e = os.ReadFile(r.Stdout)
	if e != nil {
		return c, e
	}
	selected, e := selectedGo(raw)
	if e != nil {
		return c, e
	}
	lp, e := PinFile(r.Stdout)
	if e != nil {
		return c, e
	}
	c.Lists = append(c.Lists, lp)
	for _, path := range selected {
		p, e := PinFile(path)
		if e != nil {
			return c, e
		}
		if e = add(c.Inputs, p); e != nil {
			return c, e
		}
	}
	if err = os.Mkdir(filepath.Join(out, "sources"), 0700); err != nil {
		return c, err
	}
	for _, path := range slices.Sorted(maps.Keys(c.Inputs)) {
		body, e := os.ReadFile(path)
		if e != nil {
			return c, e
		}
		dest := filepath.Join(out, "sources", c.Inputs[path])
		if _, e = os.Stat(dest); os.IsNotExist(e) {
			if e = os.WriteFile(dest, body, 0400); e != nil {
				return c, e
			}
		}
		p, e := PinFile(dest)
		if e != nil || p.SHA256 != c.Inputs[path] {
			return c, fmt.Errorf("controller snapshot differs: %s: %v", path, e)
		}
		c.Snapshots[path] = p
	}
	for _, target := range []struct{ Name, Package string }{{"closed", "./verification/tensor/closedbuild/cmd"}, {"maintained", "./verification/tensor/build"}} {
		binary := filepath.Join(out, target.Name)
		if _, e = run("go-build-"+target.Name, "build", "-a", "-x", "-work", "-p=2", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", binary, target.Package); e != nil {
			return c, e
		}
		p, e := PinFile(binary)
		if e != nil {
			return c, e
		}
		c.Binaries[target.Name] = p
	}
	r, e = run("go-list-after", listArgs...)
	if e != nil {
		return c, e
	}
	raw, e = os.ReadFile(r.Stdout)
	if e != nil {
		return c, e
	}
	after, e := selectedGo(raw)
	if e != nil {
		return c, e
	}
	if !slices.Equal(selected, after) {
		return c, fmt.Errorf("Go membership changed during rebuild")
	}
	lp, e = PinFile(r.Stdout)
	if e != nil {
		return c, e
	}
	c.Lists = append(c.Lists, lp)
	if e = checkPins(c.Inputs); e != nil {
		return c, e
	}
	c.Status = "built"
	return c, nil
}
func ReadController(p Pin, current bool) (c Controller, err error) {
	if err = read(p, &c); err != nil {
		return c, err
	}
	if c.Schema != "nanopore-tensor-closed-controller/v1" || c.Status != "built" || c.Error != "" || len(c.Lists) != 2 || len(c.Commands) != 5 || len(c.Binaries) != 2 || len(c.Snapshots) != len(c.Inputs) {
		return c, fmt.Errorf("incomplete controller custody")
	}
	tools, e := staticTools(c.GoRoot)
	if e != nil || !equal(tools, c.Tools) {
		return c, fmt.Errorf("controller Go tools changed: %v", e)
	}
	inputs := maps.Clone(tools)
	var original []string
	for i, lp := range c.Lists {
		if e = verify(lp); e != nil {
			return c, e
		}
		b, e := os.ReadFile(lp.Path)
		if e != nil {
			return c, e
		}
		s, e := selectedGo(b)
		if e != nil {
			return c, e
		}
		if i == 0 {
			original = s
		} else if !slices.Equal(original, s) {
			return c, fmt.Errorf("controller source selection differs")
		}
	}
	for _, path := range original {
		p, e := PinFile(path)
		if e != nil {
			return c, e
		}
		if e = add(inputs, p); e != nil {
			return c, e
		}
	}
	if !equal(inputs, c.Inputs) {
		return c, fmt.Errorf("controller input selection changed")
	}
	for path, p := range c.Snapshots {
		if e = verify(p); e != nil {
			return c, e
		}
		if p.SHA256 != inputs[path] {
			return c, fmt.Errorf("snapshot mapping changed")
		}
	}
	for _, r := range c.Commands {
		if r.Exit != 0 || r.Started == "" || r.Finished < r.Started {
			return c, fmt.Errorf("failed controller command")
		}
		for path, want := range map[string]string{r.Stdout: r.StdoutSHA, r.Stderr: r.StderrSHA} {
			if e = verify(Pin{Path: path, SHA256: want}); e != nil {
				return c, e
			}
		}
	}
	for _, p := range c.Binaries {
		if e = verify(p); e != nil {
			return c, e
		}
		m, e := nativeproof.InspectELF(p.Path)
		if e != nil || m.Interpreter != "" || len(m.Needed) != 0 {
			return c, fmt.Errorf("controller is not static: %v", e)
		}
	}
	if current {
		exe, e := os.Executable()
		if e != nil {
			return c, e
		}
		actual, e := PinFile(exe)
		if e != nil || actual != c.Binaries["closed"] {
			return c, fmt.Errorf("executing controller differs from sealed binary")
		}
	}
	return c, checkPins(c.Inputs)
}
