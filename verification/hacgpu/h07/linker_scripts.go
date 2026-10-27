package h07

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"unicode"
)

var linkerComments = regexp.MustCompile(`(?s)/\*.*?\*/`)

// LinkerScriptInputs returns the file references in GNU INPUT/GROUP clauses,
// including nested AS_NEEDED. OUTPUT_FORMAT arguments are not file references.
func LinkerScriptInputs(body string) ([]string, error) {
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
		return nil, fmt.Errorf("unterminated linker script token")
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
			if i+1 < len(tokens) && tokens[i+1] == "(" {
				continue
			}
			if t == "INCLUDE" {
				if i+1 >= len(tokens) {
					return nil, fmt.Errorf("linker INCLUDE requires a file")
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

func freezeLinkerScripts(pins map[string]string, resolve func(string, string) (string, error)) error {
	queue := make([]string, 0, len(pins))
	for path := range pins {
		queue = append(queue, path)
	}
	slices.Sort(queue)
	seen := map[string]bool{}
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]
		if seen[path] {
			continue
		}
		seen[path] = true
		base := filepath.Base(path)
		if !strings.HasPrefix(base, "lib") && !strings.HasPrefix(base, "ld-") {
			continue
		}
		if !strings.Contains(base, ".so") && !strings.HasSuffix(base, ".a") && !strings.HasSuffix(base, ".ld") {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		prefix := make([]byte, 8)
		n, e := f.Read(prefix)
		f.Close()
		if e != nil && e != io.EOF {
			return e
		}
		prefix = prefix[:n]
		if strings.HasPrefix(string(prefix), "\x7fELF") || strings.HasPrefix(string(prefix), "!<arch>\n") {
			continue
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if len(body) > 1<<20 {
			return fmt.Errorf("linker script exceeds bounded text size: %s", path)
		}
		inputs, err := LinkerScriptInputs(string(body))
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		for _, input := range inputs {
			actual, e := resolve(input, path)
			if e != nil {
				return e
			}
			if e = pinFile(pins, actual); e != nil {
				return e
			}
			actual, e = filepath.EvalSymlinks(actual)
			if e != nil {
				return e
			}
			actual, e = filepath.Abs(actual)
			if e != nil {
				return e
			}
			queue = append(queue, actual)
		}
	}
	return nil
}
