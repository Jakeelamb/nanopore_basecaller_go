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

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	crforacle "github.com/Jakeelamb/nanopore_basecaller_go/verification/crf"
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
	Mode                string
	Pins                map[string]string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (ret error) {
	mode := flag.String("mode", "replay", "replay or native-produce; replay never builds or runs an oracle")
	retained := flag.String("retained", "", "accepted check-002 native attempt directory for replay")
	preflight := flag.Bool("preflight-only", false, "verify retained pins and backend loading, then stop")
	backend := flag.String("backend", "", "qualified ABI2 CPU tensor library")
	source := flag.String("source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout")
	out := flag.String("out", filepath.Join("artifacts", "crf", time.Now().UTC().Format("check-20060102T150405.000000000Z")), "new receipt directory")
	flag.Parse()
	if *mode != "replay" && *mode != "native-produce" {
		return fmt.Errorf("invalid -mode %q", *mode)
	}
	if *mode == "replay" && *retained == "" {
		return fmt.Errorf("-retained required for producer-free replay")
	}
	if *preflight && *mode != "replay" {
		return fmt.Errorf("-preflight-only requires replay mode")
	}
	if *backend == "" || *source == "" {
		return fmt.Errorf("-backend and -source required; the CRF gate cannot skip")
	}
	paths := []*string{backend, source, out}
	if *retained != "" {
		paths = append(paths, retained)
	}
	for _, p := range paths {
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
	m := receipt{Status: "failed", Mode: *mode}
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
	if *mode == "replay" {
		r := command{Args: []string{"in-process", "crforacle.Replay", *source, *backend, *retained}, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: 1, Log: filepath.Join(*out, "preflight.log")}
		_, pins, preflightErr := crforacle.Replay(*source, *backend, *retained)
		if preflightErr == nil {
			m.Pins = pins
			var e *tensor.Engine
			e, preflightErr = tensor.Open(*backend)
			if preflightErr == nil {
				preflightErr = e.Close()
			}
		}
		data := []byte("retained pins and backend load passed; no native producer invoked\n")
		if preflightErr != nil {
			data = []byte(preflightErr.Error() + "\n")
		} else {
			r.Exit = 0
		}
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		sum := sha256.Sum256(data)
		r.LogSHA = hex.EncodeToString(sum[:])
		if err := os.WriteFile(r.Log, data, 0644); err != nil {
			return err
		}
		m.Commands = append(m.Commands, r)
		if preflightErr != nil {
			return preflightErr
		}
		fmt.Println("retained pins and backend load passed")
		if *preflight {
			m.Status = "passed"
			return nil
		}
	}
	clear := []string{"NANOPORE_TENSOR_BACKEND=", "DORADO_SOURCE=", "CRF_ARTIFACT_DIR=", "CRF_ORACLE_MODE=", "CRF_RETAINED_ATTEMPT="}
	if err := execute("go-version", nil, "go", "version"); err != nil {
		return err
	}
	if err := execute("ordinary", clear, "go", "test", "-p=2", "-timeout=5m", "-count=1", "-v", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	noCGO := append(append([]string{}, clear...), "CGO_ENABLED=0")
	if err := execute("nocgo", noCGO, "go", "test", "-p=2", "-timeout=5m", "-count=1", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	env := []string{"CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND=" + *backend, "DORADO_SOURCE=" + *source, "CRF_ARTIFACT_DIR=" + filepath.Join(*out, "native-attempts"), "CRF_ORACLE_MODE=" + *mode, "CRF_RETAINED_ATTEMPT=" + *retained, "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"}
	if err := execute(*mode, env, "go", "test", "-p=2", "-timeout=5m", "-count=1", "-v", "./internal/crf", "./verification/crf"); err != nil {
		return err
	}
	if err := execute("race", env, "go", "test", "-p=2", "-timeout=5m", "-count=1", "-race", "-run", "TestNativeValidationAndCancellation|TestNativeBatch|Test.*CanceledBeforeNative", "./internal/crf"); err != nil {
		return err
	}
	if err := execute("vet", nil, "go", "vet", "./internal/crf", "./verification/crf/..."); err != nil {
		return err
	}
	m.Status = "passed"
	fmt.Println(filepath.Join(*out, "commands.json"))
	return nil
}
