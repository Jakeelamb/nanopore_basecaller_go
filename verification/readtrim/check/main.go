// Command check retains MUX trim acceptance receipts in a fresh ignored attempt.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
func run() error {
	native := false
	if len(os.Args) > 1 {
		if len(os.Args) != 2 || os.Args[1] != "--native" {
			return fmt.Errorf("usage: verify.sh [--native]")
		}
		native = true
	}
	if native && (os.Getenv("DORADO_SOURCE") == "" || os.Getenv("READTRIM_CAPTURE_MANIFESTS") == "") {
		return fmt.Errorf("--native requires DORADO_SOURCE and READTRIM_CAPTURE_MANIFESTS; no inference is run")
	}
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(root, "internal/readtrim/mux.go")); err != nil {
		return fmt.Errorf("run from repository root: %w", err)
	}
	artifactRoot := os.Getenv("READTRIM_ARTIFACT_ROOT")
	if artifactRoot == "" {
		artifactRoot = filepath.Join(root, "artifacts/readtrim")
	}
	artifactRoot, err = filepath.Abs(artifactRoot)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(artifactRoot, 0700); err != nil {
		return err
	}
	attempt, err := os.MkdirTemp(artifactRoot, "checks-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		return err
	}
	fmt.Println(attempt)
	pins := map[string]string{}
	for _, dir := range []string{"internal/readtrim", "verification/readtrim"} {
		err := filepath.WalkDir(dir, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if d.IsDir() {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			pins[path] = fmt.Sprintf("%x", sha256.Sum256(data))
			return nil
		})
		if err != nil {
			return err
		}
	}
	data, err := os.ReadFile("docs/readtrim.md")
	if err != nil {
		return err
	}
	pins["docs/readtrim.md"] = fmt.Sprintf("%x", sha256.Sum256(data))
	head, err := exec.Command("git", "rev-parse", "HEAD").CombinedOutput()
	if err != nil {
		return err
	}
	toolVersion, err := exec.Command("go", "version").CombinedOutput()
	if err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(attempt, "inputs.json"), map[string]any{"source_pins": pins, "git_head": strings.TrimSpace(string(head)), "go_version": strings.TrimSpace(string(toolVersion)), "native_requested": native}); err != nil {
		return err
	}
	type check struct {
		name      string
		args, env []string
	}
	checks := []check{
		{"ordinary", []string{"test", "-count=1", "-v", "./internal/readtrim", "./verification/readtrim/..."}, nil},
		{"race", []string{"test", "-race", "-count=1", "./internal/readtrim", "./verification/readtrim/..."}, []string{"CGO_ENABLED=1"}},
		{"no-cgo", []string{"test", "-count=1", "./internal/readtrim", "./verification/readtrim/..."}, []string{"CGO_ENABLED=0"}},
		{"vet", []string{"vet", "./internal/readtrim", "./verification/readtrim/..."}, nil},
		{"mutations", []string{"test", "-count=1", "-v", "./verification/readtrim", "-run", "TestBehaviorMutations"}, []string{"READTRIM_MUTATIONS=1"}},
	}
	if native {
		checks = append(checks, check{"native", []string{"test", "-count=1", "-v", "./internal/readtrim", "-run", "TestPinnedNativeMux"}, []string{"DORADO_SOURCE=" + os.Getenv("DORADO_SOURCE"), "READTRIM_CAPTURE_MANIFESTS=" + os.Getenv("READTRIM_CAPTURE_MANIFESTS")}})
	}
	receipts := []map[string]any{}
	failed := false
	for _, c := range checks {
		cmd := exec.Command("go", c.args...)
		cmd.Env = append(os.Environ(), "DORADO_SOURCE=", "READTRIM_CAPTURE_MANIFESTS=", "READTRIM_MUTATIONS=", "READTRIM_ARTIFACT_ROOT="+attempt)
		cmd.Env = append(cmd.Env, c.env...)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		started := time.Now().UTC()
		runErr := cmd.Run()
		finished := time.Now().UTC()
		code := -1
		if cmd.ProcessState != nil {
			code = cmd.ProcessState.ExitCode()
		}
		for path, content := range map[string][]byte{c.name + ".stdout": stdout.Bytes(), c.name + ".stderr": stderr.Bytes()} {
			if err := os.WriteFile(filepath.Join(attempt, path), content, 0600); err != nil {
				return err
			}
		}
		receipt := map[string]any{"name": c.name, "command": cmd.Args, "environment_overrides": append([]string{"DORADO_SOURCE=", "READTRIM_CAPTURE_MANIFESTS=", "READTRIM_MUTATIONS=", "READTRIM_ARTIFACT_ROOT=" + attempt}, c.env...), "started": started, "finished": finished, "exit_code": code, "stdout_sha256": fmt.Sprintf("%x", sha256.Sum256(stdout.Bytes())), "stderr_sha256": fmt.Sprintf("%x", sha256.Sum256(stderr.Bytes()))}
		receipts = append(receipts, receipt)
		if err := writeJSON(filepath.Join(attempt, c.name+".receipt.json"), receipt); err != nil {
			return err
		}
		fmt.Printf("%s exit=%d\n", c.name, code)
		if runErr != nil {
			failed = true
			fmt.Print(stdout.String(), stderr.String())
		}
	}
	if err := writeJSON(filepath.Join(attempt, "summary.json"), map[string]any{"checks": receipts, "passed": !failed, "native_qualified": native && !failed}); err != nil {
		return err
	}
	if failed {
		return fmt.Errorf("acceptance checks failed; retained at %s", attempt)
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
