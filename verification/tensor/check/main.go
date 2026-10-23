// Command check records the tensor package's ordinary, unsupported, and native gates.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type commandReceipt = runner.CommandReceipt

type evidence struct {
	Head     string            `json:"head"`
	Source   map[string]string `json:"source_sha256"`
	Inputs   map[string]string `json:"input_sha256"`
	Commands []commandReceipt  `json:"commands"`
	Status   string            `json:"status"`
	Error    string            `json:"error,omitempty"`
}

func hash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}
func run() (ret error) {
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	pure := flag.Bool("pure-only", false, "ordinary, no-cgo, race and vet checks only; no native library load")
	backend := flag.String("backend", "", "compiled tensor library")
	weight := flag.String("weight", "", "optional real Dorado weight, captured by builder -weight")
	oldBackend := flag.String("old-backend", "", "retained ABI 3 library for actual version rejection")
	out := flag.String("out", filepath.Join("artifacts", "tensor", time.Now().UTC().Format("20060102T150405.000000000Z")+"-checks"), "new receipt directory")
	flag.Parse()
	if *backend == "" && !*pure {
		return fmt.Errorf("-backend is required; this gate never skips native execution")
	}
	for _, p := range []*string{backend, out} {
		v, err := filepath.Abs(*p)
		if err != nil {
			return err
		}
		*p = v
	}
	if *weight != "" {
		v, err := filepath.Abs(*weight)
		if err != nil {
			return err
		}
		*weight = v
	}
	if *oldBackend != "" {
		v, err := filepath.Abs(*oldBackend)
		if err != nil {
			return err
		}
		*oldBackend = v
	}
	attempt, err := runner.NewAttempt(*out)
	if err != nil {
		return err
	}
	*out = attempt
	m := evidence{Source: map[string]string{}, Inputs: map[string]string{}, Status: "failed"}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		data, err := json.MarshalIndent(m, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(*out, "manifest.json"), append(data, '\n'), 0644)
		}
		if ret == nil && err != nil {
			ret = err
		}
	}()
	if !*pure {
		lock, err := runner.Acquire(runner.HostLock)
		if err != nil {
			m.Status = "admission_blocked"
			return err
		}
		defer lock.Close()
	}
	for _, root := range []string{"internal/tensor", "verification/tensor"} {
		if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			h, err := hash(path)
			if err != nil {
				return err
			}
			m.Source[path] = h
			return nil
		}); err != nil {
			return err
		}
	}
	if !*pure {
		rawBuild, err := os.ReadFile(filepath.Join(filepath.Dir(*backend), "manifest.json"))
		if err != nil {
			return err
		}
		var build struct {
			ABI    int    `json:"abi_version"`
			Family string `json:"cuda_family"`
		}
		if err = json.Unmarshal(rawBuild, &build); err != nil {
			return err
		}
		if build.ABI != 4 {
			return fmt.Errorf("check requires an ABI4 build manifest")
		}
		if build.Family != "none" {
			m.Status = "runtime_held"
			return fmt.Errorf("CUDA backend loading requires the separate admitted CUDA capture/replay command; CPU check does not initialize CUDA")
		}
		paths := []string{*backend, filepath.Join(filepath.Dir(*backend), "manifest.json")}
		if *oldBackend != "" {
			paths = append(paths, *oldBackend, filepath.Join(filepath.Dir(*oldBackend), "manifest.json"))
		}
		if *weight != "" {
			paths = append(paths, *weight, filepath.Join(filepath.Dir(*backend), "real-weight.json"))
		}
		for _, path := range paths {
			h, err := hash(path)
			if err != nil {
				return err
			}
			m.Inputs[path] = h
		}
	}
	cmd := exec.Command("git", "rev-parse", "HEAD")
	head, err := cmd.Output()
	if err != nil {
		return err
	}
	m.Head = strings.TrimSpace(string(head))
	execute := func(label string, overrides []string, args ...string) error {
		r, err := runner.Execute(*out, label, overrides, args...)
		m.Commands = append(m.Commands, r)
		if err == nil {
			fmt.Printf("%s passed\n", label)
		}
		return err
	}
	if err := execute("go-version", nil, "go", "version"); err != nil {
		return err
	}
	if err := execute("ordinary", []string{"NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_WEIGHT=", "NANOPORE_TENSOR_OLD_BACKEND=", "NANOPORE_TENSOR_CUDA=", "NANOPORE_TENSOR_RUNTIME_MANIFEST=", "NANOPORE_TENSOR_CUDA_CASE="}, "go", "test", "-count=1", "-v", "./..."); err != nil {
		return err
	}
	if err := execute("nocgo", []string{"CGO_ENABLED=0", "NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_WEIGHT=", "NANOPORE_TENSOR_OLD_BACKEND=", "NANOPORE_TENSOR_CUDA=", "NANOPORE_TENSOR_RUNTIME_MANIFEST=", "NANOPORE_TENSOR_CUDA_CASE="}, "go", "test", "-count=1", "-v", "./..."); err != nil {
		return err
	}
	if !*pure {
		if *oldBackend != "" {
			if err := execute("real-abi3-rejection", []string{"NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_OLD_BACKEND=" + *oldBackend}, "go", "test", "-count=1", "-v", "./internal/tensor", "-run", "^TestNativeRejectOldABI$"); err != nil {
				return err
			}
		}
		native := []string{"CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND=" + *backend, "NANOPORE_TENSOR_WEIGHT=" + *weight, "NANOPORE_TENSOR_OLD_BACKEND=", "NANOPORE_TENSOR_CUDA=", "NANOPORE_TENSOR_RUNTIME_MANIFEST=", "NANOPORE_TENSOR_CUDA_CASE="}
		if err := execute("native", native, "go", "test", "-count=1", "-v", "./internal/tensor"); err != nil {
			return err
		}
		if err := execute("c-compiler", nil, "cc", "--version"); err != nil {
			return err
		}
		probe := filepath.Join(*out, "abi-probe")
		if err := execute("compile-abi-probe", nil, "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Iinternal/tensor", "verification/tensor/abi_probe.c", "-ldl", "-o", probe); err != nil {
			return err
		}
		m.Inputs[probe], err = hash(probe)
		if err != nil {
			return err
		}
		if err := execute("abi-probe", nil, probe, *backend); err != nil {
			return err
		}
		if err := execute("race", native, "go", "test", "-count=1", "-race", "./internal/tensor"); err != nil {
			return err
		}
	} else {
		if err := execute("race", []string{"NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_OLD_BACKEND=", "NANOPORE_TENSOR_CUDA="}, "go", "test", "-count=1", "-race", "./internal/tensor", "./verification/tensor/..."); err != nil {
			return err
		}
	}
	if err := execute("vet", nil, "go", "vet", "./internal/tensor", "./verification/tensor/..."); err != nil {
		return err
	}
	m.Status = "passed"
	fmt.Println(filepath.Join(*out, "manifest.json"))
	return nil
}
