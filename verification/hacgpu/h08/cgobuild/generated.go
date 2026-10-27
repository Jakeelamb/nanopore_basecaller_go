package cgobuild

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type GeneratedSet struct {
	Package, Stage   string
	Producer         Pin
	Files, Snapshots map[string]Pin
	Bindings         map[string]string
}

// CheckGenerated is a structural pre-child guard, not a post-build audit.
// Translation and dynimport are separate Go cgo tool invocations.
func CheckGenerated(p Package, stage string, files map[string]Pin) error {
	if p.ImportPath != "runtime/cgo" && p.ImportPath != Module+"/internal/tensor" {
		return fmt.Errorf("unadmitted generated package")
	}
	want := RequiredGenerated(p)
	switch stage {
	case "translation":
	case "dynimport":
		want = []string{"_cgo_import.go"}
	default:
		return fmt.Errorf("unadmitted generated stage")
	}
	if len(files) != len(want) {
		return fmt.Errorf("generated source/header set incomplete")
	}
	for _, name := range want {
		q, ok := files[name]
		if !ok || len(q.SHA256) != 64 || filepath.Base(q.Path) != name {
			return fmt.Errorf("generated file omitted or rebound: %s", name)
		}
	}
	return nil
}

func under(root, path string) bool {
	rel, e := filepath.Rel(root, path)
	return e == nil && rel != "." && rel != ".." && !filepath.IsAbs(rel) && len(rel) > 0 && rel[:1] != "."
}

// CaptureGenerated snapshots actual outputs at the cgo-tool return boundary,
// before Go compilation resumes. It never invokes a producer or native tool.
func CaptureGenerated(p Package, stage, workRoot, objectDir, snapshotDir string, producer Pin) (g GeneratedSet, err error) {
	actual, e := closedbuild.PinFile(producer.Path)
	if e != nil || actual != producer {
		return g, fmt.Errorf("cgo producer receipt required: %v", e)
	}
	root, e := filepath.EvalSymlinks(workRoot)
	if e != nil {
		return g, e
	}
	dir, e := filepath.EvalSymlinks(objectDir)
	if e != nil || !under(root, dir) {
		return g, fmt.Errorf("generated object directory outside private work root")
	}
	names := RequiredGenerated(p)
	if stage == "dynimport" {
		names = []string{"_cgo_import.go"}
	}
	g = GeneratedSet{Package: p.ImportPath, Stage: stage, Producer: producer, Files: map[string]Pin{}, Snapshots: map[string]Pin{}, Bindings: map[string]string{}}
	for _, name := range names {
		q, e := closedbuild.PinFile(filepath.Join(dir, name))
		if e != nil {
			return g, e
		}
		if !under(root, q.Path) || filepath.Base(q.Path) != name {
			return g, fmt.Errorf("generated source escaped work root")
		}
		g.Files[name] = q
		g.Bindings[filepath.Join(dir, name)] = q.Path
	}
	if err = CheckGenerated(p, stage, g.Files); err != nil {
		return g, err
	}
	if err = os.MkdirAll(snapshotDir, 0700); err != nil {
		return g, err
	}
	for _, name := range slices.Sorted(maps.Keys(g.Files)) {
		q := g.Files[name]
		body, e := os.ReadFile(q.Path)
		if e != nil {
			return g, e
		}
		path := filepath.Join(snapshotDir, q.SHA256)
		f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0400)
		if os.IsExist(e) {
			existing, x := closedbuild.PinFile(path)
			if x != nil || existing.SHA256 != q.SHA256 {
				return g, fmt.Errorf("snapshot identity differs")
			}
		} else if e != nil {
			return g, e
		} else {
			_, e = f.Write(body)
			closeErr := f.Close()
			if e != nil {
				return g, e
			}
			if closeErr != nil {
				return g, closeErr
			}
		}
		s, e := closedbuild.PinFile(path)
		if e != nil || s.SHA256 != q.SHA256 {
			return g, fmt.Errorf("generated snapshot differs: %v", e)
		}
		g.Snapshots[name] = s
	}
	return g, CheckGeneratedLive(g)
}

func CheckGeneratedLive(g GeneratedSet) error {
	if len(g.Files) == 0 || len(g.Files) != len(g.Snapshots) || len(g.Bindings) != len(g.Files) {
		return fmt.Errorf("generated snapshots incomplete")
	}
	for path, want := range g.Bindings {
		actual, e := filepath.EvalSymlinks(path)
		if e != nil || actual != want {
			return fmt.Errorf("generated source path binding changed: %s", path)
		}
	}
	for name, q := range g.Files {
		live, e := closedbuild.PinFile(q.Path)
		if e != nil || live != q {
			return fmt.Errorf("generated input changed before native child: %s", name)
		}
		s := g.Snapshots[name]
		snapshot, e := closedbuild.PinFile(s.Path)
		if e != nil || snapshot != s || s.SHA256 != q.SHA256 {
			return fmt.Errorf("generated snapshot changed: %s", name)
		}
	}
	return nil
}
