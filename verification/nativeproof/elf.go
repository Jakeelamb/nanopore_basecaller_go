// Package nativeproof inspects explicitly selected files without executing a
// tool, target, dynamic loader, or device API. Callers select tools, plugins,
// dlopen families, environment policy, and any loader-cache observations.
package nativeproof

import (
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type Pin struct{ Path, SHA256 string }

type ELFMetadata struct {
	Pin                    Pin
	Class                  elf.Class
	Data                   elf.Data
	Machine                elf.Machine
	Type                   elf.Type
	Interpreter, SONAME    string
	Needed, RPath, RunPath []string
}

type ELFBinding struct {
	From, Kind, Name string
	Target           Pin
	SearchDirs       []string
}

type ELFClosure struct {
	Nodes    []ELFMetadata
	Inputs   []Pin
	Bindings []ELFBinding
}

func digest(f *os.File) (string, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func canonical(path string) (string, error) {
	if path == "" || strings.ContainsRune(path, 0) {
		return "", fmt.Errorf("empty or invalid file path")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(path)
}

// InspectELF returns metadata and the canonical regular-file pin for ET_EXEC or
// ET_DYN. It makes no claim about the host loader's actual selected libraries.
func InspectELF(path string) (ELFMetadata, error) { return inspectELF(path, "") }

func inspectELF(path, expected string) (m ELFMetadata, err error) {
	path, err = canonical(path)
	if err != nil {
		return m, err
	}
	f, err := os.Open(path)
	if err != nil {
		return m, err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !before.Mode().IsRegular() {
		return m, fmt.Errorf("ELF input must be a regular file: %s: %v", path, err)
	}
	pin, err := digest(f)
	if err != nil {
		return m, err
	}
	// Seed identity is checked before the ELF parser reads any metadata.
	if expected != "" && pin != expected {
		return m, fmt.Errorf("ELF seed SHA256 mismatch: %s", path)
	}
	e, err := elf.NewFile(f)
	if err != nil {
		return m, fmt.Errorf("unsupported ELF input %s: %w", path, err)
	}
	if e.Type != elf.ET_EXEC && e.Type != elf.ET_DYN {
		return m, fmt.Errorf("unsupported ELF type %s: %s", e.Type, path)
	}
	if e.Machine == elf.EM_NONE {
		return m, fmt.Errorf("ELF machine is unspecified: %s", path)
	}
	m = ELFMetadata{Pin: Pin{path, pin}, Class: e.Class, Data: e.Data, Machine: e.Machine, Type: e.Type}
	for _, p := range e.Progs {
		if p.Type != elf.PT_INTERP {
			continue
		}
		body, readErr := io.ReadAll(io.LimitReader(p.Open(), 4097))
		if readErr != nil || m.Interpreter != "" || len(body) < 2 || len(body) > 4096 || body[len(body)-1] != 0 || strings.ContainsRune(string(body[:len(body)-1]), 0) {
			return m, fmt.Errorf("invalid or repeated ELF interpreter: %s", path)
		}
		m.Interpreter = string(body[:len(body)-1])
		if !filepath.IsAbs(m.Interpreter) || strings.Contains(m.Interpreter, "$") {
			return m, fmt.Errorf("unsupported ELF interpreter path: %s", m.Interpreter)
		}
	}
	stringsByTag, err := runtimeDynamicStrings(e, f, uint64(before.Size()))
	if err != nil {
		return m, fmt.Errorf("invalid ELF dynamic metadata %s: %w", path, err)
	}
	m.Needed, m.RPath, m.RunPath = stringsByTag[elf.DT_NEEDED], stringsByTag[elf.DT_RPATH], stringsByTag[elf.DT_RUNPATH]
	names := stringsByTag[elf.DT_SONAME]
	if len(names) == 1 {
		m.SONAME = names[0]
		if !libraryName(m.SONAME) {
			return m, fmt.Errorf("unsupported ELF SONAME: %q", m.SONAME)
		}
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return m, fmt.Errorf("ELF input changed during inspection: %s", path)
	}
	second, err := digest(f)
	if err != nil || second != pin {
		return m, fmt.Errorf("ELF bytes changed during inspection: %s: %v", path, err)
	}
	return m, nil
}

func libraryName(name string) bool {
	return name != "" && name != "." && name != ".." && !strings.ContainsAny(name, "/$\x00")
}

func directories(dirs []string) ([]string, error) {
	out := []string{}
	for _, dir := range dirs {
		if !filepath.IsAbs(dir) || strings.ContainsRune(dir, 0) || strings.Contains(dir, "$") {
			return nil, fmt.Errorf("unsupported ELF search directory: %q", dir)
		}
		actual, err := canonical(dir)
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(actual)
		if err != nil || !info.IsDir() {
			return nil, fmt.Errorf("ELF search path is not a directory: %s", dir)
		}
		out = append(out, actual)
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

func origin(value, parent string) (string, error) {
	var out strings.Builder
	for len(value) != 0 {
		at := strings.IndexByte(value, '$')
		if at < 0 {
			out.WriteString(value)
			break
		}
		out.WriteString(value[:at])
		value = value[at:]
		n := 0
		if strings.HasPrefix(value, "${ORIGIN}") {
			n = len("${ORIGIN}")
		} else if strings.HasPrefix(value, "$ORIGIN") {
			n = len("$ORIGIN")
			if len(value) > n {
				c := value[n]
				if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' {
					n = 0
				}
			}
		}
		if n == 0 {
			return "", fmt.Errorf("unsupported ELF path token: %q", value)
		}
		out.WriteString(filepath.Dir(parent))
		value = value[n:]
	}
	value = out.String()
	if strings.Contains(value, "$") || !filepath.IsAbs(value) || strings.ContainsRune(value, 0) {
		return "", fmt.Errorf("unsupported ELF path or token: %q", value)
	}
	return value, nil
}

func embeddedDirs(values []string, parent string) ([]string, error) {
	out := []string{}
	if len(values) > 1 {
		return nil, fmt.Errorf("repeated ELF RPATH/RUNPATH: %s", parent)
	}
	for _, value := range values {
		for _, part := range strings.Split(value, ":") {
			path, err := origin(part, parent)
			if err != nil {
				return nil, err
			}
			out = append(out, path)
		}
	}
	return directories(out)
}

func resolve(name, parent string, dirs []string) (string, error) {
	if strings.ContainsAny(name, "/$") {
		path, err := origin(name, parent)
		if err != nil {
			return "", err
		}
		return canonical(path)
	}
	if !libraryName(name) {
		return "", fmt.Errorf("invalid ELF dependency name: %q", name)
	}
	matches := []string{}
	for _, dir := range dirs {
		path := filepath.Join(dir, name)
		_, err := os.Lstat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return "", err
		}
		path, err = canonical(path)
		if err != nil {
			return "", err
		}
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return "", fmt.Errorf("ELF dependency is not a regular file: %s", path)
		}
		matches = append(matches, path)
	}
	slices.Sort(matches)
	matches = slices.Compact(matches)
	if len(matches) != 1 {
		return "", fmt.Errorf("ELF dependency %q from %s has %d distinct bindings: %v", name, parent, len(matches), matches)
	}
	return matches[0], nil
}

// CloseELF freezes the recursively resolved metadata closure. Explicit search
// directories and embedded paths are conservative candidates: two distinct
// matches reject instead of emulating precedence, a loader cache, or hwcaps.
// RUNPATH applies directly; RPATH also propagates to descendants. Cycles and
// inconsistent bindings reject. Callers must verify actual runtime mappings.
func CloseELF(seeds []Pin, searchDirs []string) (out ELFClosure, err error) {
	if len(seeds) == 0 {
		return out, fmt.Errorf("at least one pinned ELF seed is required")
	}
	dirs, err := directories(searchDirs)
	if err != nil {
		return out, err
	}
	nodes := map[string]ELFMetadata{}
	for _, seed := range seeds {
		decoded, e := hex.DecodeString(seed.SHA256)
		if e != nil || len(decoded) != sha256.Size || strings.ToLower(seed.SHA256) != seed.SHA256 {
			return out, fmt.Errorf("valid lowercase seed SHA256 required: %s", seed.Path)
		}
		m, e := inspectELF(seed.Path, seed.SHA256)
		if e != nil {
			return out, e
		}
		if before, ok := nodes[m.Pin.Path]; ok && before.Pin != m.Pin {
			return out, fmt.Errorf("conflicting ELF seed identity: %s", m.Pin.Path)
		}
		nodes[m.Pin.Path] = m
	}
	bindings := map[string]ELFBinding{}
	names := map[string]Pin{}
	active, done := map[string]bool{}, map[string]bool{}
	var visit func(string, []string) error
	visit = func(path string, inherited []string) error {
		if active[path] {
			return fmt.Errorf("ELF dependency cycle at %s", path)
		}
		context := path + "\x00" + strings.Join(inherited, "\x00")
		if done[context] {
			return nil
		}
		active[path] = true
		defer delete(active, path)
		m := nodes[path]
		if m.SONAME != "" {
			if before, ok := names[m.SONAME]; ok && before != m.Pin {
				return fmt.Errorf("ambiguous global ELF SONAME %q", m.SONAME)
			}
			names[m.SONAME] = m.Pin
		}
		rpath, e := embeddedDirs(m.RPath, path)
		if e != nil {
			return e
		}
		runpath, e := embeddedDirs(m.RunPath, path)
		if e != nil {
			return e
		}
		childInherited := slices.Clone(inherited)
		local := runpath
		if len(m.RunPath) == 0 {
			local = rpath
			childInherited = append(childInherited, rpath...)
		}
		childInherited, e = directories(childInherited)
		if e != nil {
			return e
		}
		effective := append(slices.Clone(dirs), inherited...)
		effective = append(effective, local...)
		effective, e = directories(effective)
		if e != nil {
			return e
		}
		add := func(kind, name string, candidates, next []string) error {
			target, e := resolve(name, path, candidates)
			if e != nil {
				return e
			}
			dep, ok := nodes[target]
			if !ok {
				dep, e = InspectELF(target)
				if e != nil {
					return e
				}
				nodes[target] = dep
			}
			if dep.Class != m.Class || dep.Data != m.Data || dep.Machine != m.Machine {
				return fmt.Errorf("incompatible ELF dependency ABI: %s -> %s", path, target)
			}
			key := path + "\x00" + kind + "\x00" + name
			if before, ok := bindings[key]; ok {
				if before.Target != dep.Pin {
					return fmt.Errorf("context-dependent ELF binding: %s", name)
				}
				candidates, e = directories(append(slices.Clone(candidates), before.SearchDirs...))
				if e != nil {
					return e
				}
			}
			bindings[key] = ELFBinding{From: path, Kind: kind, Name: name, Target: dep.Pin, SearchDirs: slices.Clone(candidates)}
			if kind == "needed" && libraryName(name) {
				if before, ok := names[name]; ok && before != dep.Pin {
					return fmt.Errorf("ambiguous global ELF dependency %q", name)
				}
				names[name] = dep.Pin
			}
			return visit(target, next)
		}
		if m.Interpreter != "" {
			if e = add("interpreter", m.Interpreter, nil, nil); e != nil {
				return e
			}
		}
		needed := slices.Clone(m.Needed)
		slices.Sort(needed)
		for _, name := range slices.Compact(needed) {
			if e = add("needed", name, effective, childInherited); e != nil {
				return e
			}
		}
		done[context] = true
		return nil
	}
	paths := []string{}
	for path := range nodes {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	for _, path := range paths {
		if err = visit(path, nil); err != nil {
			return ELFClosure{}, err
		}
	}
	for _, node := range nodes {
		out.Nodes = append(out.Nodes, node)
		out.Inputs = append(out.Inputs, node.Pin)
	}
	slices.SortFunc(out.Nodes, func(a, b ELFMetadata) int { return strings.Compare(a.Pin.Path, b.Pin.Path) })
	slices.SortFunc(out.Inputs, func(a, b Pin) int { return strings.Compare(a.Path, b.Path) })
	for _, binding := range bindings {
		out.Bindings = append(out.Bindings, binding)
	}
	slices.SortFunc(out.Bindings, func(a, b ELFBinding) int {
		return strings.Compare(a.From+"\x00"+a.Kind+"\x00"+a.Name, b.From+"\x00"+b.Kind+"\x00"+b.Name)
	})
	return out, nil
}
