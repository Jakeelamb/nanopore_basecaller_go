package decodergpu

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const lineageGoPackage = "./verification/decodergpu/lineagecmd"
const lineageGoExperiment = "nodwarf5"

var lineageGoEnvironment = []string{"GOENV=off", "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOFLAGS=", "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "GOEXPERIMENT=" + lineageGoExperiment, "GODEBUG="}

func lineageGoSelection(dependencies, environment []byte, goBin string, recorded map[string]string) (map[string]string, error) {
	var env map[string]string
	if err := json.Unmarshal(environment, &env); err != nil {
		return nil, err
	}
	if env["CGO_ENABLED"] != "0" || env["GOOS"] != "linux" || env["GOARCH"] != "amd64" || env["GOAMD64"] != "v1" || env["GOFLAGS"] != "" || env["GOWORK"] != "off" || env["GOENV"] != "" || env["GOEXPERIMENT"] != lineageGoExperiment || !filepath.IsAbs(env["GOMOD"]) || filepath.Base(env["GOMOD"]) != "go.mod" {
		return nil, fmt.Errorf("lineage actual Go environment/module selection differs")
	}
	selected := map[string]string{}
	add := func(path string) error {
		if recorded != nil {
			actual, ok := recorded[path]
			if !ok {
				return fmt.Errorf("lineage historical Go selection omitted: %s", path)
			}
			selected[path] = actual
			return nil
		}
		actual, err := canonical(path)
		if err == nil {
			selected[path] = actual
		}
		return err
	}
	for _, path := range []string{goBin, env["GOMOD"], filepath.Join(env["GOTOOLDIR"], "compile"), filepath.Join(env["GOTOOLDIR"], "asm"), filepath.Join(env["GOTOOLDIR"], "link")} {
		if err := add(path); err != nil {
			return nil, err
		}
	}
	sum := filepath.Join(filepath.Dir(env["GOMOD"]), "go.sum")
	if _, err := os.Stat(sum); err == nil {
		if err = add(sum); err != nil {
			return nil, err
		}
	}
	entries, err := os.ReadDir(filepath.Join(env["GOROOT"], "pkg/include"))
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			if err = add(filepath.Join(env["GOROOT"], "pkg/include", entry.Name())); err != nil {
				return nil, err
			}
		}
	}
	dec := json.NewDecoder(strings.NewReader(string(dependencies)))
	count := 0
	root := false
	for {
		var p goPackage
		err := dec.Decode(&p)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if p.Error != nil || len(p.DepsErrors) > 0 || !filepath.IsAbs(p.Dir) {
			return nil, fmt.Errorf("lineage selected Go package error")
		}
		count++
		root = root || strings.HasSuffix(p.ImportPath, "/verification/decodergpu/lineagecmd")
		for _, list := range [][]string{p.GoFiles, p.CgoFiles, p.CFiles, p.CXXFiles, p.HFiles, p.SFiles, p.SysoFiles, p.EmbedFiles} {
			for _, name := range list {
				if err = add(filepath.Join(p.Dir, name)); err != nil {
					return nil, err
				}
			}
		}
	}
	if count == 0 || !root {
		return nil, fmt.Errorf("lineage actual controller dependency selection absent")
	}
	return selected, nil
}

func freezeLineageGo(out string) (g GoClosure, err error) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		return g, err
	}
	run := func(label string, args ...string) (runner.CommandReceipt, error) {
		c, e := runner.Execute(out, label, lineageGoEnvironment, append([]string{goBin}, args...)...)
		g.Commands = append(g.Commands, c)
		return c, e
	}
	deps, err := run("go-dependencies", "list", "-deps", "-json", "-pgo=off", lineageGoPackage)
	if err != nil {
		return g, err
	}
	env, err := run("go-environment", "env", "-json")
	if err != nil {
		return g, err
	}
	depRaw, err := os.ReadFile(deps.Stdout)
	if err != nil {
		return g, err
	}
	envRaw, err := os.ReadFile(env.Stdout)
	if err != nil {
		return g, err
	}
	g.Selection, err = lineageGoSelection(depRaw, envRaw, goBin, nil)
	if err != nil {
		return g, err
	}
	g.Pins = map[string]string{}
	for _, actual := range g.Selection {
		if err = filePin(g.Pins, actual); err != nil {
			return g, err
		}
	}
	var environment map[string]string
	json.Unmarshal(envRaw, &environment)
	for _, path := range []string{goBin, filepath.Join(environment["GOTOOLDIR"], "compile"), filepath.Join(environment["GOTOOLDIR"], "asm"), filepath.Join(environment["GOTOOLDIR"], "link")} {
		f, e := elf.Open(path)
		if e != nil {
			return g, e
		}
		dynamic := false
		for _, p := range f.Progs {
			dynamic = dynamic || p.Type == elf.PT_INTERP || p.Type == elf.PT_DYNAMIC
		}
		f.Close()
		if dynamic {
			return g, fmt.Errorf("lineage Go tools require an explicit dynamic runtime closure")
		}
	}
	g.Snapshots, g.SnapshotPins, err = snapshotPins(g.Pins, filepath.Join(out, "go-inputs"))
	if err != nil {
		return g, err
	}
	g.Controller = filepath.Join(out, "lineage-controller")
	if _, err = run("go-controller-build", "build", "-x", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", g.Controller, lineageGoPackage); err != nil {
		return g, err
	}
	g.ControllerSHA, err = runner.SHA(g.Controller)
	if err != nil {
		return g, err
	}
	return g, checkLineageGo(g, true, false)
}

func checkLineageGo(g GoClosure, current, executing bool) error {
	if len(g.Commands) != 3 || len(g.Commands[0].Args) == 0 {
		return fmt.Errorf("lineage Go command custody absent")
	}
	goBin := g.Commands[0].Args[0]
	want := [][]string{{goBin, "list", "-deps", "-json", "-pgo=off", lineageGoPackage}, {goBin, "env", "-json"}, {goBin, "build", "-x", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", g.Controller, lineageGoPackage}}
	for i, c := range g.Commands {
		if !slices.Equal(c.Args, want[i]) || !slices.Equal(c.Environment, lineageGoEnvironment) {
			return fmt.Errorf("lineage actual Go command/environment roles differ")
		}
		if err := commandEvidence(c); err != nil {
			return err
		}
	}
	deps, err := os.ReadFile(g.Commands[0].Stdout)
	if err != nil {
		return err
	}
	env, err := os.ReadFile(g.Commands[1].Stdout)
	if err != nil {
		return err
	}
	recorded := g.Selection
	if current {
		recorded = nil
	}
	selected, e := lineageGoSelection(deps, env, goBin, recorded)
	if e != nil {
		return e
	}
	if !maps.Equal(selected, g.Selection) {
		return fmt.Errorf("lineage selected Go path membership changed")
	}
	expected := map[string]string{}
	for _, path := range selected {
		want, ok := g.Pins[path]
		if !ok {
			return fmt.Errorf("lineage Go selected input omitted")
		}
		expected[path] = want
		if current {
			if err = (File{path, want}).verify(); err != nil {
				return err
			}
		}
	}
	if !maps.Equal(expected, g.Pins) {
		return fmt.Errorf("lineage Go dependency/tool/module pin membership changed")
	}
	for path, want := range g.Pins {
		copy, ok := g.Snapshots[path]
		if !ok || g.SnapshotPins[copy] != want {
			return fmt.Errorf("lineage immutable Go source binding omitted")
		}
	}
	if len(g.Pins) == 0 || len(g.Snapshots) != len(g.Pins) {
		return fmt.Errorf("lineage Go snapshot membership differs")
	}
	if err = checkPins(g.SnapshotPins); err != nil {
		return err
	}
	if err = (File{g.Controller, g.ControllerSHA}).verify(); err != nil {
		return err
	}
	if executing {
		path, e := os.Executable()
		if e != nil {
			return e
		}
		got, e := artifactPin(path)
		if e != nil {
			return e
		}
		if got != (File{g.Controller, g.ControllerSHA}) {
			return fmt.Errorf("running lineage controller differs from retained binary")
		}
	}
	return nil
}
