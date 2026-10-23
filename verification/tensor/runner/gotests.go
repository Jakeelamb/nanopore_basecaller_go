package runner

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

func RequireGoTests(path, pkg string, required []string) error {
	return requireGoTests(path, pkg, required, false)
}
func RequireGoTestRoots(path, pkg string, required []string) error {
	return requireGoTests(path, pkg, required, true)
}
func requireGoTests(path, pkg string, required []string, allowSubtests bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	passed := map[string]int{}
	for {
		var event struct{ Action, Package, Test string }
		if err := dec.Decode(&event); err == io.EOF {
			break
		} else if err != nil {
			return fmt.Errorf("invalid go test JSON evidence: %w", err)
		}
		if event.Package != pkg {
			continue
		}
		if event.Action == "skip" || event.Action == "fail" {
			return fmt.Errorf("required native gate had %s event for %s", event.Action, event.Test)
		}
		if event.Action == "pass" && event.Test != "" {
			passed[event.Test]++
		}
	}
	for _, test := range required {
		if passed[test] != 1 {
			return fmt.Errorf("required native test %s executed %d times; expected exactly one successful execution", test, passed[test])
		}
	}
	if allowSubtests {
		for name := range passed {
			allowed := false
			for _, root := range required {
				if name == root || strings.HasPrefix(name, root+"/") {
					allowed = true
					break
				}
			}
			if !allowed {
				return fmt.Errorf("unexpected native test %s", name)
			}
		}
		return nil
	}
	if len(passed) != len(required) {
		return fmt.Errorf("native test coverage differs: got %d passed tests expected %d", len(passed), len(required))
	}
	return nil
}
