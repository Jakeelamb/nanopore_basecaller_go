// Command check runs and retains separate CPU CRF acceptance gates.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type command struct {
	Args     []string `json:"command"`
	Env      []string `json:"environment_overrides"`
	Started  string   `json:"started_utc"`
	Finished string   `json:"finished_utc"`
	Exit     int      `json:"exit_code"`
	Log      string   `json:"log"`
	LogSHA   string   `json:"log_sha256"`
}
type receipt struct {
	Head, Status, Error string
	Commands            []command
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (ret error) {
	backend := flag.String("backend", "", "qualified ABI2 CPU tensor library")
	source := flag.String("source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout")
	out := flag.String("out", filepath.Join("artifacts", "crf", time.Now().UTC().Format("check-20060102T150405.000000000Z")), "new receipt directory")
	flag.Parse()
	if *backend == "" || *source == "" {
		return fmt.Errorf("-backend and -source required; the source-linked native gate cannot skip")
	}
	for _, p := range []*string{backend, source, out} {
		v, err := filepath.Abs(*p)
		if err != nil {
			return err
		}
		*p = v
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return err
	}
	m := receipt{Status: "failed"}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		data, err := json.MarshalIndent(m, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(*out, "commands.json"), append(data, '\n'), 0644)
		}
		if ret == nil {
			ret = err
		}
	}()
	head, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return err
	}
	m.Head = strings.TrimSpace(string(head))
	execute := func(label string, env []string, args ...string) error {
		r := command{Args: args, Env: env, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Log: filepath.Join(*out, label+".log")}
		cmd := exec.Command(args[0], args[1:]...)
		all := os.Environ()
		for _, kv := range env {
			key, _, _ := strings.Cut(kv, "=")
			filtered := all[:0]
			for _, prior := range all {
				if !strings.HasPrefix(prior, key+"=") {
					filtered = append(filtered, prior)
				}
			}
			all = append(filtered, kv)
		}
		cmd.Env = all
		data, err := cmd.CombinedOutput()
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if cmd.ProcessState != nil {
			r.Exit = cmd.ProcessState.ExitCode()
		}
		sum := sha256.Sum256(data)
		r.LogSHA = hex.EncodeToString(sum[:])
		if writeErr := os.WriteFile(r.Log, data, 0644); writeErr != nil {
			return writeErr
		}
		m.Commands = append(m.Commands, r)
		if err != nil {
			return fmt.Errorf("%s failed: %v; see %s", label, err, r.Log)
		}
		fmt.Printf("%s passed\n", label)
		return nil
	}
	clear := []string{"NANOPORE_TENSOR_BACKEND=", "DORADO_SOURCE=", "CRF_ARTIFACT_DIR="}
	if err := execute("go-version", nil, "go", "version"); err != nil {
		return err
	}
	if err := execute("ordinary", clear, "go", "test", "-count=1", "-v", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	noCGO := append(append([]string{}, clear...), "CGO_ENABLED=0")
	if err := execute("nocgo", noCGO, "go", "test", "-count=1", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	env := []string{"CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND=" + *backend, "DORADO_SOURCE=" + *source, "CRF_ARTIFACT_DIR=" + filepath.Join(*out, "native-attempts"), "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"}
	if err := execute("native", env, "go", "test", "-count=1", "-v", "./internal/crf"); err != nil {
		return err
	}
	if err := execute("race", env, "go", "test", "-count=1", "-race", "-run", "TestNativeValidationAndCancellation|TestCanceledBeforeNative", "./internal/crf"); err != nil {
		return err
	}
	if err := execute("vet", nil, "go", "vet", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	m.Status = "passed"
	fmt.Println(filepath.Join(*out, "commands.json"))
	return nil
}
