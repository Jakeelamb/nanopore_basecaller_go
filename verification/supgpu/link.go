package supgpu

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

type ToolQuery struct {
	Argv     []string `json:"argv"`
	Expected string   `json:"expected"`
}
type LinkBinding struct {
	Argument   string `json:"argument"`
	Input      Pin    `json:"input"`
	FromScript string `json:"from_script"`
}

// linkClosure resolves this qualified compiler's finite link recipe without
// executing it. Admitted compiler queries must confirm these choices before -M
// or compilation, and actual -t/map inputs must stay inside the frozen set.
func linkClosure(libs []string, buildDir, compiler string) ([]string, []LinkBinding, []ToolQuery, error) {
	usr := filepath.Dir(filepath.Dir(compiler))
	gccDirs, err := filepath.Glob(filepath.Join(usr, "lib", "gcc", "*", "*", "libstdc++fs.a"))
	if err != nil {
		return nil, nil, nil, err
	}
	if len(gccDirs) != 1 {
		return nil, nil, nil, fmt.Errorf("qualified GCC library directory is ambiguous")
	}
	dirs := []string{}
	for i, arg := range libs {
		if arg == "-L" {
			if i+1 >= len(libs) {
				return nil, nil, nil, fmt.Errorf("missing -L directory")
			}
			dirs = append(dirs, libs[i+1])
		} else if strings.HasPrefix(arg, "-L") {
			dirs = append(dirs, arg[2:])
		}
	}
	dirs = append(dirs, filepath.Dir(gccDirs[0]), filepath.Join(usr, "lib"), "/usr/lib")
	for i, dir := range dirs {
		if !filepath.IsAbs(dir) {
			dirs[i] = filepath.Join(buildDir, dir)
		}
	}
	bindings := []LinkBinding{}
	queries := []ToolQuery{}
	querySeen := map[string]bool{}
	queue := []string{}
	seen := map[string]bool{}
	addQuery := func(name, expected string) {
		key := name + "\x00" + expected
		if !querySeen[key] {
			querySeen[key] = true
			queries = append(queries, ToolQuery{[]string{compiler, "-print-file-name=" + name}, expected})
		}
	}
	add := func(arg, path, from string) (string, error) {
		if !filepath.IsAbs(path) {
			path = filepath.Join(buildDir, path)
		}
		p, err := pin(path)
		if err != nil {
			return "", err
		}
		bindings = append(bindings, LinkBinding{arg, p, from})
		if !seen[p.Path] {
			seen[p.Path] = true
			queue = append(queue, p.Path)
		}
		return p.Path, nil
	}
	resolveFile := func(name string) (string, bool) {
		for _, dir := range dirs {
			path := filepath.Join(dir, name)
			info, err := os.Stat(path)
			if err == nil && info.Mode().IsRegular() {
				return path, true
			}
		}
		return "", false
	}
	resolveLib := func(arg, from string) (string, error) {
		name := strings.TrimPrefix(arg, "-l")
		names := []string{"lib" + name + ".so", "lib" + name + ".a"}
		if strings.HasPrefix(name, ":") {
			names = []string{name[1:]}
		}
		selected := ""
		for _, dir := range dirs {
			for _, file := range names {
				candidate := filepath.Join(dir, file)
				info, e := os.Stat(candidate)
				if e == nil && info.Mode().IsRegular() {
					selected = candidate
					break
				}
			}
			if selected != "" {
				break
			}
		}
		if selected == "" {
			return "", fmt.Errorf("unresolved qualified linker lookup %s", arg)
		}
		for _, file := range names {
			path, ok := resolveFile(file)
			if !ok {
				addQuery(file, file)
				continue
			}
			addQuery(file, path)
		}
		return add(arg, selected, from)
	}
	resolved := []string{}
	for i := 0; i < len(libs); i++ {
		arg := libs[i]
		switch {
		case arg == "-L":
			if i+1 >= len(libs) {
				return nil, nil, nil, fmt.Errorf("missing -L")
			}
			resolved = append(resolved, arg, libs[i+1])
			i++
		case strings.HasPrefix(arg, "-l"):
			path, e := resolveLib(arg, "")
			if e != nil {
				return nil, nil, nil, e
			}
			resolved = append(resolved, path)
		case strings.HasPrefix(arg, "-Wl,"):
			parts := strings.Split(arg[4:], ",")
			for j := 0; j < len(parts); j++ {
				part := parts[j]
				if part == "-rpath" || part == "--rpath" || part == "-Map" || part == "--version-script" {
					if j+1 >= len(parts) {
						return nil, nil, nil, fmt.Errorf("incomplete wrapped linker option")
					}
					if part == "--version-script" {
						if _, e := add(part, parts[j+1], ""); e != nil {
							return nil, nil, nil, e
						}
					}
					j++
					continue
				}
				if strings.HasPrefix(part, "-l") {
					path, e := resolveLib(part, "")
					if e != nil {
						return nil, nil, nil, e
					}
					parts[j] = path
				} else if !strings.HasPrefix(part, "-") {
					path, e := add(part, part, "")
					if e != nil {
						return nil, nil, nil, e
					}
					parts[j] = path
				}
			}
			resolved = append(resolved, "-Wl,"+strings.Join(parts, ","))
		case strings.HasPrefix(arg, "-"):
			resolved = append(resolved, arg)
		default:
			path, e := add(arg, arg, "")
			if e != nil {
				return nil, nil, nil, e
			}
			resolved = append(resolved, path)
		}
	}
	for _, name := range []string{"Scrt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "libstdc++.so", "libc.so", "libc_nonshared.a", "libm.so"} {
		path, ok := resolveFile(name)
		if !ok {
			return nil, nil, nil, fmt.Errorf("unresolved implicit compiler link resource %s", name)
		}
		addQuery(name, path)
		if _, err = add("implicit:"+name, path, ""); err != nil {
			return nil, nil, nil, err
		}
	}
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]
		f, e := os.Open(path)
		if e != nil {
			return nil, nil, nil, e
		}
		prefix := make([]byte, 8)
		n, _ := f.Read(prefix)
		f.Close()
		if strings.HasPrefix(string(prefix[:n]), "\x7fELF") || strings.HasPrefix(string(prefix[:n]), "!<arch>\n") {
			continue
		}
		info, e := os.Stat(path)
		if e != nil {
			return nil, nil, nil, e
		}
		if info.Size() > 1<<20 {
			return nil, nil, nil, fmt.Errorf("nonbinary link input exceeds script bound")
		}
		body, e := os.ReadFile(path)
		if e != nil {
			return nil, nil, nil, e
		}
		inputs, e := linkerScriptInputs(string(body))
		if e != nil {
			return nil, nil, nil, e
		}
		if len(inputs) == 0 {
			return nil, nil, nil, fmt.Errorf("unrecognized nonbinary linker input %s", path)
		}
		for _, input := range inputs {
			if strings.HasPrefix(input, "-l") {
				if _, e = resolveLib(input, path); e != nil {
					return nil, nil, nil, e
				}
				continue
			}
			candidate := input
			if !filepath.IsAbs(input) {
				candidate = filepath.Join(filepath.Dir(path), input)
				if info, e := os.Stat(candidate); e != nil || !info.Mode().IsRegular() {
					var ok bool
					candidate, ok = resolveFile(input)
					if !ok {
						return nil, nil, nil, fmt.Errorf("unresolved script input %s", input)
					}
				}
			}
			if _, e = add(input, candidate, path); e != nil {
				return nil, nil, nil, e
			}
		}
	}
	return resolved, bindings, queries, nil
}

var linkerComments = regexp.MustCompile(`(?s)/\*.*?\*/`)

// This bounded parser follows the existing H07 GROUP/INPUT/AS_NEEDED contract.
func linkerScriptInputs(body string) ([]string, error) {
	body = linkerComments.ReplaceAllString(body, "")
	var token strings.Builder
	tokens := []string{}
	quoted, escaped := false, false
	flush := func() {
		if token.Len() > 0 {
			tokens = append(tokens, token.String())
			token.Reset()
		}
	}
	for _, c := range body {
		if escaped {
			token.WriteRune(c)
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == '"' {
			quoted = !quoted
			continue
		}
		if quoted {
			token.WriteRune(c)
			continue
		}
		if unicode.IsSpace(c) || c == '(' || c == ')' || c == ',' {
			flush()
			if c == '(' || c == ')' {
				tokens = append(tokens, string(c))
			}
		} else {
			token.WriteRune(c)
		}
	}
	if quoted || escaped {
		return nil, fmt.Errorf("unterminated linker token")
	}
	flush()
	stack := []bool{}
	active := false
	out := []string{}
	for i, t := range tokens {
		switch t {
		case "(":
			function := ""
			if i > 0 {
				function = tokens[i-1]
			}
			stack = append(stack, active)
			active = active || function == "GROUP" || function == "INPUT" || function == "AS_NEEDED"
		case ")":
			if len(stack) == 0 {
				return nil, fmt.Errorf("unbalanced linker script")
			}
			active = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		default:
			if t == "SEARCH_DIR" {
				return nil, fmt.Errorf("linker SEARCH_DIR requires an explicit planned search path")
			}
			if i+1 < len(tokens) && tokens[i+1] == "(" {
				continue
			}
			if t == "INCLUDE" {
				if i+1 >= len(tokens) {
					return nil, fmt.Errorf("missing linker INCLUDE file")
				}
				out = append(out, tokens[i+1])
				continue
			}
			if i > 0 && tokens[i-1] == "INCLUDE" {
				continue
			}
			if active {
				out = append(out, t)
			}
		}
	}
	if len(stack) != 0 {
		return nil, fmt.Errorf("unbalanced linker script")
	}
	return out, nil
}

func verifyLinkInputs(body, dir string, inputs, objects []Pin, linkMap bool) error {
	allowed := map[string]Pin{}
	for _, p := range append(append([]Pin{}, inputs...), objects...) {
		actual, err := filepath.EvalSymlinks(p.Path)
		if err != nil {
			return err
		}
		allowed[actual] = p
	}
	seen := 0
	for _, line := range strings.Split(body, "\n") {
		path := strings.TrimSpace(line)
		if linkMap {
			if !strings.HasPrefix(path, "LOAD ") {
				continue
			}
			path = strings.TrimSpace(strings.TrimPrefix(path, "LOAD "))
		}
		if path == "" {
			continue
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, path)
		}
		actual, err := filepath.EvalSymlinks(path)
		if err != nil {
			return fmt.Errorf("unrecognized linker input %s", path)
		}
		p, ok := allowed[actual]
		if !ok {
			return fmt.Errorf("unfrozen linker input %s", actual)
		}
		if err = verifyPin("", p); err != nil {
			return err
		}
		seen++
	}
	if seen == 0 {
		return fmt.Errorf("empty actual linker input observation")
	}
	return nil
}

func sortedDependencies(path, dir string) ([]string, error) {
	deps, err := DepfileInputs(path)
	if err != nil {
		return nil, err
	}
	set := map[string]bool{}
	for _, p := range deps {
		if !filepath.IsAbs(p) {
			p = filepath.Join(dir, p)
		}
		set[filepath.Clean(p)] = true
	}
	out := []string{}
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out, nil
}

func sameQueryResult(raw string, q ToolQuery, dir string) error {
	actual := strings.TrimSpace(raw)
	expected := q.Expected
	if filepath.IsAbs(expected) {
		var a string
		var err error
		if len(q.Argv) > 1 && strings.HasPrefix(q.Argv[1], "-print-prog-name=") {
			a, err = resolveCompilerProgram(actual, dir)
		} else {
			if !filepath.IsAbs(actual) {
				actual = filepath.Join(dir, actual)
			}
			a, err = filepath.EvalSymlinks(actual)
		}
		if err != nil {
			return err
		}
		b, err := filepath.EvalSymlinks(expected)
		if err != nil {
			return err
		}
		if a != b {
			return fmt.Errorf("compiler lookup differs from prospective binding")
		}
	} else if actual != expected {
		return fmt.Errorf("previously unresolved library lookup changed")
	}
	return nil
}

func verifyFrozenLinkBindings(p BuildPlan) error {
	libs, bindings, queries, err := linkClosure(p.OriginalLibraries, p.Commands[0].Directory, p.Compiler.Path)
	if err != nil {
		return err
	}
	queries = append(compilerQueries(p.CompilerRuntime), queries...)
	if !reflect.DeepEqual(bindings, p.LinkBindings) || !reflect.DeepEqual(queries, p.Queries) {
		return fmt.Errorf("prospective link closure changed")
	}
	for i := nativeCompileCount; i < nativeCompileCount+2; i++ {
		cmd := p.Commands[i]
		start := 4
		if i == nativeCompileCount+1 {
			start = 7
		}
		if len(cmd.Argv) < start+4 || !reflect.DeepEqual(cmd.Argv[start:len(cmd.Argv)-4], libs) {
			return fmt.Errorf("resolved link argv differs")
		}
	}
	if !reflect.DeepEqual(p.Commands[8], auditLinkCommand(p.Commands[0].Directory, p.Compiler.Path, p.Commands[5].Argv[len(p.Commands[5].Argv)-1], p.Audit)) {
		return fmt.Errorf("loader audit link recipe changed")
	}
	inputs := map[string]Pin{}
	for _, v := range p.Inputs {
		inputs[v.Path] = v
	}
	for _, b := range bindings {
		if inputs[b.Input.Path] != b.Input {
			return fmt.Errorf("link input absent from frozen plan: %s", b.Input.Path)
		}
	}
	return nil
}
