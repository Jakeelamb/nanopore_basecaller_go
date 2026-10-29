// Command preparecmd constructs or preflights CPU13 metadata without execution.
package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type admissionPreflight struct {
	Schema                                       string
	Admission, ParentBuild, ParentBuildSelection h08.Pin
	ControllerBuild, ControllerBuildSelection    h08.Pin
}

type fakePreflight struct {
	Schema                          string
	Proof, Target, Build, Selection h08.Pin
}

func pinShape(p h08.Pin) error {
	b, err := hex.DecodeString(p.SHA256)
	if !filepath.IsAbs(p.Path) || filepath.Clean(p.Path) != p.Path || err != nil || len(b) != 32 {
		return fmt.Errorf("absolute clean path and full independent SHA256 required: %s", p.Path)
	}
	return nil
}

func fixtureBoundary(o h08.FixtureOptions) error {
	for _, p := range []h08.Pin{o.SourceManifest, o.Archive, o.Backend, o.Weight} {
		if err := pinShape(p); err != nil {
			return err
		}
	}
	for path, hash := range o.Runtime {
		if err := pinShape(h08.Pin{Path: path, SHA256: hash}); err != nil {
			return err
		}
	}
	if o.Runtime == nil {
		return fmt.Errorf("explicit runtime provider map required")
	}
	switch filepath.Base(o.Backend.Path) {
	case "manifest.json", "runtime.json", "view.json":
		return fmt.Errorf("backend symlink collides with generated fixture metadata")
	}
	if !filepath.IsAbs(o.Directory) || filepath.Clean(o.Directory) != o.Directory {
		return fmt.Errorf("absolute fresh fixture directory required")
	}
	if _, err := os.Lstat(o.Directory); !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("fixture output must be absent: %v", err)
	}
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(filepath.Join(root, "artifacts"), o.Directory)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, "../") || filepath.IsAbs(rel) {
		return fmt.Errorf("fixture directory must be under working checkout artifacts")
	}
	parent := filepath.Dir(o.Directory)
	canonical, err := filepath.EvalSymlinks(parent)
	if err != nil || canonical != parent {
		return fmt.Errorf("fixture parent must exist without symlink aliases: %v", err)
	}
	return nil
}

func run(args []string, out, diagnostics io.Writer) error {
	f := flag.NewFlagSet("cpu13-prepare", flag.ContinueOnError)
	f.SetOutput(diagnostics)
	op := f.String("operation", "", "fixture, admission, cpu13-preflight, or fake-preflight")
	path := f.String("request", "", "absolute typed JSON request path")
	sha := f.String("request-sha", "", "independent request SHA256")
	if err := f.Parse(args); err != nil {
		return err
	}
	if f.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	p := h08.Pin{Path: *path, SHA256: *sha}
	if err := pinShape(p); err != nil {
		return err
	}
	var result any
	switch *op {
	case "fixture":
		var o h08.FixtureOptions
		if err := h08.ReadPreparationJSON(p, &o); err != nil {
			return err
		}
		if err := fixtureBoundary(o); err != nil {
			return err
		}
		v, err := h08.PrepareFixtureView(o)
		if err != nil {
			return err
		}
		pin, err := closedbuild.PinFile(filepath.Join(v.Directory, "view.json"))
		if err != nil {
			return err
		}
		result = struct {
			View        h08.FixtureView
			Pin         h08.Pin
			RuntimeHeld bool
		}{v, pin, true}
	case "admission":
		r, err := h08.PrepareCPU13Admission(p)
		if err != nil {
			return err
		}
		result = r
	case "cpu13-preflight":
		var o admissionPreflight
		if err := h08.ReadPreparationJSON(p, &o); err != nil {
			return err
		}
		if o.Schema != "nanopore-h08-cpu13-preflight/v1" {
			return fmt.Errorf("unsupported CPU13 preflight schema")
		}
		for _, pin := range []h08.Pin{o.Admission, o.ParentBuild, o.ParentBuildSelection, o.ControllerBuild, o.ControllerBuildSelection} {
			if err := pinShape(pin); err != nil {
				return err
			}
		}
		a, q, err := h08.ReadCPU13Admission(o.Admission)
		if err != nil {
			return err
		}
		if a.ParentBuild != o.ParentBuild || a.ParentBuildSelection != o.ParentBuildSelection || q.ControllerBuild != o.ControllerBuild || q.ControllerBuildSelection != o.ControllerBuildSelection {
			return fmt.Errorf("CPU13 preflight selected build pair differs")
		}
		result = struct {
			Admission   h08.Pin
			RuntimeHeld bool
		}{o.Admission, true}
	case "fake-preflight":
		var o fakePreflight
		if err := h08.ReadPreparationJSON(p, &o); err != nil {
			return err
		}
		if o.Schema != "nanopore-h08-fake-preflight/v1" {
			return fmt.Errorf("unsupported fake preflight schema")
		}
		for _, pin := range []h08.Pin{o.Proof, o.Target, o.Build, o.Selection} {
			if err := pinShape(pin); err != nil {
				return err
			}
		}
		b, err := cgobuild.ReadCPU13Artifact(o.Build, o.Selection)
		if err != nil {
			return err
		}
		if b.Output() != o.Target {
			return fmt.Errorf("fake preflight selected target differs")
		}
		if _, err = h08.ReadFakeLoaderProof(o.Proof, o.Target, o.Build, o.Selection); err != nil {
			return err
		}
		result = struct {
			Proof       h08.Pin
			RuntimeHeld bool
		}{o.Proof, true}
	default:
		return fmt.Errorf("unsupported preparation operation: %s", *op)
	}
	return json.NewEncoder(out).Encode(result)
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
