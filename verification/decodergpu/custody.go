package decodergpu

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type File struct{ Path, SHA256 string }

func normalizeAnchor(p File) (File, error) {
	got, err := artifactPin(p.Path)
	if err != nil {
		return File{}, err
	}
	if p.SHA256 == "" || got.SHA256 != p.SHA256 {
		return File{}, fmt.Errorf("external anchor digest differs: %s", p.Path)
	}
	return got, nil
}

func artifactPin(path string) (File, error) {
	pins := map[string]string{}
	if err := filePin(pins, path); err != nil {
		return File{}, err
	}
	for path, sha := range pins {
		return File{path, sha}, nil
	}
	return File{}, fmt.Errorf("empty pin")
}
func (p File) verify() error {
	actual, err := artifactPin(p.Path)
	if err != nil {
		return err
	}
	if actual != p {
		return fmt.Errorf("pinned path/bytes changed: %s", p.Path)
	}
	return nil
}
func readJSON(p File, out any) error {
	raw, err := readPinnedMetadata(p, maxPinnedMetadataBytes)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

func readPinnedMetadata(p File, limit int64) ([]byte, error) {
	actual, err := canonical(p.Path)
	if err != nil {
		return nil, err
	}
	if actual != p.Path || limit < 0 {
		return nil, fmt.Errorf("pinned path/bytes changed: %s", p.Path)
	}
	f, err := os.Open(actual)
	if err != nil {
		return nil, err
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, limit+1))
	closeErr := f.Close()
	if readErr != nil {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("pinned metadata exceeds bounded input: %s", p.Path)
	}
	after, err := canonical(p.Path)
	if err != nil {
		return nil, err
	}
	if after != p.Path || digest(raw) != p.SHA256 {
		return nil, fmt.Errorf("pinned path/bytes changed: %s", p.Path)
	}
	return raw, nil
}

func writeJSON(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(append(raw, '\n'))
	closed := f.Close()
	if err != nil {
		return err
	}
	return closed
}

func snapshotPins(inputs map[string]string, dir string) (map[string]string, map[string]string, error) {
	if err := os.Mkdir(dir, 0700); err != nil {
		return nil, nil, err
	}
	bindings, copies := map[string]string{}, map[string]string{}
	paths := slices.Sorted(maps.Keys(inputs))
	for _, original := range paths {
		p := File{original, inputs[original]}
		if err := p.verify(); err != nil {
			return nil, nil, err
		}
		copy := filepath.Join(dir, p.SHA256+filepath.Ext(original))
		if _, exists := copies[copy]; !exists {
			raw, err := os.ReadFile(original)
			if err != nil {
				return nil, nil, err
			}
			if digest(raw) != p.SHA256 {
				return nil, nil, fmt.Errorf("source changed while snapshotting: %s", original)
			}
			if err = os.WriteFile(copy, raw, 0400); err != nil {
				return nil, nil, err
			}
			copies[copy] = p.SHA256
		}
		bindings[original] = copy
	}
	return bindings, copies, checkPins(copies)
}

// ReadPreparation distinguishes immutable historical source custody from the
// current-input/executable checks required before any native phase.
func ReadPreparation(p File, current bool) (r Preparation, err error) {
	return readPreparation(p, current, nil)
}

func readPreparation(p File, current bool, proofs *numericProofs) (r Preparation, err error) {
	if err = readJSON(p, &r); err != nil {
		return r, err
	}
	if (r.Schema != "nanopore-decoder-fixed-hac-preparation/v1" && r.Schema != "nanopore-decoder-fixed-hac-preparation/v2" && r.Schema != "nanopore-decoder-fixed-hac-preparation/v3") || r.Status != "host_prepared_native_held" || r.Error != "" || r.Contract.Validate() != nil || len(r.Go.Pins) == 0 || len(r.Go.Snapshots) != len(r.Go.Pins) || len(r.Go.SnapshotPins) == 0 || len(r.Go.Commands) != 3 {
		return r, fmt.Errorf("incomplete D01 preparation/controller custody")
	}
	if current && ((r.Schema != "nanopore-decoder-fixed-hac-preparation/v2" && r.Schema != "nanopore-decoder-fixed-hac-preparation/v3") || len(r.Images.Images) != 2) {
		return r, fmt.Errorf("current consumer requires observed v2 preparation; v1 producer is historical")
	}
	if r.Schema == "nanopore-decoder-fixed-hac-preparation/v2" || r.Schema == "nanopore-decoder-fixed-hac-preparation/v3" {
		if len(r.Images.Images) != 2 {
			return r, fmt.Errorf("prospective decoder images absent")
		}
		expected, e := PlanDeviceImages(filepath.Dir(r.Images.Member.Path), r.Images.Images[1].Image.Path)
		if e != nil || !reflect.DeepEqual(expected, r.Images) {
			return r, fmt.Errorf("prospective decoder member/images changed: %v", e)
		}
		for _, image := range r.Images.Images {
			if r.Inputs[image.Image.Path] != image.Image.SHA256 {
				return r, fmt.Errorf("decoder image input omitted")
			}
		}
	}
	if r.Schema == "nanopore-decoder-fixed-hac-preparation/v3" {
		if current {
			if e := requireNativePreparationSources(r.Inputs, "nanopore-decoder-closed-build/v3"); e != nil {
				return r, e
			}
		}
		proof, e := proofs.historical(r.NumericLineage)
		if e != nil {
			return r, e
		}
		for path, sha := range proof.Inputs {
			if r.Inputs[path] != sha {
				return r, fmt.Errorf("numeric lineage custody omitted: %s", path)
			}
		}
	} else if r.NumericLineage != (File{}) {
		return r, fmt.Errorf("legacy preparation has numeric policy")
	}
	for original, want := range r.Go.Pins {
		copy, ok := r.Go.Snapshots[original]
		if !ok || r.Go.SnapshotPins[copy] != want {
			return r, fmt.Errorf("Go snapshot binding missing: %s", original)
		}
	}
	if err = checkPins(r.Go.SnapshotPins); err != nil {
		return r, err
	}
	if err = (File{r.Go.Controller, r.Go.ControllerSHA}).verify(); err != nil {
		return r, err
	}
	for _, command := range r.Go.Commands {
		if command.Exit != 0 || command.Started == "" || command.Finished == "" {
			return r, fmt.Errorf("unsuccessful Go build custody")
		}
		if err = (File{command.Stdout, command.StdoutSHA}).verify(); err != nil {
			return r, err
		}
		if err = (File{command.Stderr, command.StderrSHA}).verify(); err != nil {
			return r, err
		}
	}
	if err = verifyGoSelection(r.Go, current); err != nil {
		return r, err
	}
	if current {
		if err = checkPins(r.Inputs); err != nil {
			return r, err
		}
		if err = checkPins(r.Go.Pins); err != nil {
			return r, err
		}
		exe, e := os.Executable()
		if e != nil {
			return r, e
		}
		got, e := artifactPin(exe)
		if e != nil {
			return r, e
		}
		if got != (File{r.Go.Controller, r.Go.ControllerSHA}) {
			return r, fmt.Errorf("executing D01 controller differs from retained build")
		}
	}
	return r, nil
}

func verifyGoSelection(g GoClosure, current bool) error {
	if len(g.Commands) != 3 || len(g.Commands[0].Args) == 0 {
		return fmt.Errorf("actual Go metadata/build roles absent")
	}
	goBin := g.Commands[0].Args[0]
	want := [][]string{{goBin, "list", "-deps", "-json", "-pgo=off", "./verification/decodergpu/cmd"}, {goBin, "env", "-json"}, {goBin, "build", "-x", "-pgo=off", "-trimpath", "-buildvcs=true", "-o", g.Controller, "./verification/decodergpu/cmd"}}
	for i, c := range g.Commands {
		if !slices.Equal(c.Args, want[i]) {
			return fmt.Errorf("actual Go command role changed")
		}
	}
	envRaw, err := os.ReadFile(g.Commands[1].Stdout)
	if err != nil {
		return err
	}
	var env map[string]string
	if err = json.Unmarshal(envRaw, &env); err != nil {
		return err
	}
	if env["CGO_ENABLED"] != "0" || env["GOOS"] != "linux" || env["GOARCH"] != "amd64" || env["GOAMD64"] != "v1" || env["GOFLAGS"] != "" || env["GOWORK"] != "off" || env["GOENV"] != "" {
		return fmt.Errorf("Go build environment outside fixed CGO0 recipe")
	}
	require := func(path string) error {
		actual, ok := g.Selection[path]
		if !ok {
			return fmt.Errorf("actual Go input binding omitted: %s", path)
		}
		if _, ok := g.Pins[actual]; !ok {
			return fmt.Errorf("actual Go selected input omitted: %s", actual)
		}
		if current {
			got, e := canonical(path)
			if e != nil {
				return e
			}
			if got != actual {
				return fmt.Errorf("Go selected path changed: %s", path)
			}
		}
		return nil
	}
	for _, path := range []string{goBin, filepath.Join(env["GOTOOLDIR"], "compile"), filepath.Join(env["GOTOOLDIR"], "asm"), filepath.Join(env["GOTOOLDIR"], "link")} {
		if err = require(path); err != nil {
			return err
		}
	}
	if !filepath.IsAbs(env["GOMOD"]) || env["GOMOD"] == "/dev/null" {
		return fmt.Errorf("actual Go module selection absent")
	}
	if err = require(env["GOMOD"]); err != nil {
		return err
	}
	body, err := os.ReadFile(g.Commands[0].Stdout)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	count := 0
	for {
		var pkg goPackage
		e := decoder.Decode(&pkg)
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		if !filepath.IsAbs(pkg.Dir) || pkg.Error != nil || len(pkg.DepsErrors) > 0 {
			return fmt.Errorf("invalid actual Go dependency metadata")
		}
		count++
		for _, list := range [][]string{pkg.GoFiles, pkg.CgoFiles, pkg.CFiles, pkg.CXXFiles, pkg.HFiles, pkg.SFiles, pkg.SysoFiles, pkg.EmbedFiles} {
			for _, name := range list {
				if err = require(filepath.Join(pkg.Dir, name)); err != nil {
					return err
				}
			}
		}
	}
	if count == 0 {
		return fmt.Errorf("empty actual Go dependency selection")
	}
	return nil
}

func addPin(inputs map[string]string, p File) error {
	if err := p.verify(); err != nil {
		return err
	}
	if old, ok := inputs[p.Path]; ok && old != p.SHA256 {
		return fmt.Errorf("conflicting pin: %s", p.Path)
	}
	inputs[p.Path] = p.SHA256
	return nil
}

func commandEvidence(c runner.CommandReceipt) error {
	if c.Exit != 0 || c.Started == "" || c.Finished == "" {
		return fmt.Errorf("command did not complete successfully")
	}
	if err := (File{c.Stdout, c.StdoutSHA}).verify(); err != nil {
		return err
	}
	return (File{c.Stderr, c.StderrSHA}).verify()
}
