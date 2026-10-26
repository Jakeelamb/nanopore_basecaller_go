package supgpu

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func stockSource(buildDir, suffix string) (Pin, error) {
	raw, err := os.ReadFile(filepath.Join(buildDir, "compile_commands.json"))
	if err != nil {
		return Pin{}, err
	}
	var entries []struct {
		File string `json:"file"`
	}
	if err = json.Unmarshal(raw, &entries); err != nil {
		return Pin{}, err
	}
	found := []string{}
	for _, e := range entries {
		if strings.HasSuffix(e.File, "/"+suffix) {
			found = append(found, e.File)
		}
	}
	if len(found) != 1 {
		return Pin{}, fmt.Errorf("stock source selector %s is missing/ambiguous", suffix)
	}
	return pin(found[0])
}

var quotedInclude = regexp.MustCompile(`(?m)^\s*#\s*include\s*"([^"]+)"`)

// quotedHeaderInputs checks actual copied-source quote lookup without invoking
// a preprocessor. Full transitive dependencies are frozen by admitted -M later.
func quotedHeaderInputs(c BuildCommand) ([]Pin, error) {
	if len(c.Argv) < 7 {
		return nil, fmt.Errorf("short compile command")
	}
	source := c.Argv[len(c.Argv)-3]
	dirs := []string{filepath.Dir(source)}
	after := []string{}
	for i := 1; i < len(c.Argv); i++ {
		arg := c.Argv[i]
		switch {
		case arg == "-iquote":
			if i+1 >= len(c.Argv) {
				return nil, fmt.Errorf("missing quote path")
			}
			dirs = append(dirs, c.Argv[i+1])
			i++
		case arg == "-I" || arg == "-isystem":
			if i+1 >= len(c.Argv) {
				return nil, fmt.Errorf("missing include path")
			}
			after = append(after, c.Argv[i+1])
			i++
		case strings.HasPrefix(arg, "-I"):
			after = append(after, arg[2:])
		}
	}
	dirs = append(dirs, after...)
	raw, err := os.ReadFile(source)
	if err != nil {
		return nil, err
	}
	out := []Pin{}
	for _, m := range quotedInclude.FindAllSubmatch(raw, -1) {
		name := string(m[1])
		found := false
		for _, dir := range dirs {
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(c.Directory, dir)
			}
			path := filepath.Join(dir, name)
			info, err := os.Stat(path)
			if err != nil {
				if os.IsNotExist(err) {
					continue
				}
				return nil, err
			}
			if !info.Mode().IsRegular() {
				return nil, fmt.Errorf("quoted header is nonregular")
			}
			v, err := pin(path)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
			found = true
			break
		}
		if !found {
			return nil, fmt.Errorf("planned quoted include %q is unresolved for %s", name, source)
		}
	}
	return out, nil
}
