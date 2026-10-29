package supgpu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// controllerSourceRoot derives the producer root only from the pinned go-list
// selection. Both the production controller and audit controller use this rule.
func controllerSourceRoot(deps []byte, target string) (string, error) {
	const module = "github.com/Jakeelamb/nanopore_basecaller_go"
	decoder := json.NewDecoder(bytes.NewReader(deps))
	root, targets, packages := "", 0, 0
	for {
		var p goPackage
		err := decoder.Decode(&p)
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		packages++
		if packages > 10000 || p.Error != nil {
			return "", fmt.Errorf("invalid Go dependency selection")
		}
		name := strings.Split(p.ImportPath, " [")[0]
		project := name == module || strings.HasPrefix(name, module+"/")
		if !project && (p.Module == nil || !p.Module.Main) {
			// This port has no external modules. Unknown non-standard source
			// must not be mistaken for a standard-library dependency.
			if !p.Standard {
				return "", fmt.Errorf("unknown Go source module")
			}
			continue
		}
		if !project || p.Standard || p.Module == nil || p.Module.Path != module || !p.Module.Main || !filepath.IsAbs(p.Root) || filepath.Clean(p.Root) != p.Root || p.Module.Dir != p.Root || p.Module.GoMod != filepath.Join(p.Root, "go.mod") {
			return "", fmt.Errorf("invalid Go source module/root evidence")
		}
		rel := strings.TrimPrefix(name, module+"/")
		if name == module {
			rel = "."
		}
		if name == target+".test" {
			rel = strings.TrimPrefix(target, module+"/")
		}
		if p.Dir != filepath.Join(p.Root, rel) || (root != "" && root != p.Root) {
			return "", fmt.Errorf("ambiguous Go source root")
		}
		root = p.Root
		if p.ImportPath == target {
			targets++
		}
	}
	if root == "" || targets != 1 {
		return "", fmt.Errorf("missing/ambiguous Go source package root")
	}
	return root, nil
}
