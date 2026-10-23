// Command check records the tensor package's ordinary, unsupported, and native gates.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type commandReceipt struct {
	Args     []string `json:"command"`
	Env      []string `json:"environment_overrides"`
	Started  string   `json:"started_utc"`
	Finished string   `json:"finished_utc"`
	Exit     int      `json:"exit_code"`
	Log      string   `json:"log"`
	SHA      string   `json:"log_sha256"`
}
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
		os.Exit(1)
	}
}
func run() (ret error) {
	backend := flag.String("backend", "", "compiled tensor library")
	weight := flag.String("weight", "", "optional real Dorado weight, captured by builder -weight")
	oldBackend := flag.String("old-backend", "", "retained ABI 1 library for actual version rejection")
	out := flag.String("out", filepath.Join("artifacts", "tensor", time.Now().UTC().Format("20060102T150405.000000000Z")+"-checks"), "new receipt directory")
	flag.Parse()
	if *backend == "" {
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
	if _, err := os.Stat(*out); err == nil {
		return fmt.Errorf("output exists; preserve attempts with a fresh -out")
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
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
	paths := []string{*backend, filepath.Join(filepath.Dir(*backend), "manifest.json")}
	if *oldBackend != "" {
		paths = append(paths, *oldBackend)
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
	cmd := exec.Command("git", "rev-parse", "HEAD")
	head, err := cmd.Output()
	if err != nil {
		return err
	}
	m.Head = strings.TrimSpace(string(head))
	execute := func(label string, overrides []string, args ...string) error {
		r := commandReceipt{Args: args, Env: overrides, Started: time.Now().UTC().Format(time.RFC3339Nano), Log: filepath.Join(*out, label+".log"), Exit: -1}
		cmd := exec.Command(args[0], args[1:]...)
		env := os.Environ()
		for _, kv := range overrides {
			key, _, _ := strings.Cut(kv, "=")
			filtered := env[:0]
			for _, prior := range env {
				if !strings.HasPrefix(prior, key+"=") {
					filtered = append(filtered, prior)
				}
			}
			env = append(filtered, kv)
		}
		cmd.Env = env
		data, err := cmd.CombinedOutput()
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if cmd.ProcessState != nil {
			r.Exit = cmd.ProcessState.ExitCode()
		}
		if writeErr := os.WriteFile(r.Log, data, 0644); writeErr != nil {
			return writeErr
		}
		sum := sha256.Sum256(data)
		r.SHA = hex.EncodeToString(sum[:])
		m.Commands = append(m.Commands, r)
		if err != nil {
			return fmt.Errorf("%s failed: %v; see %s", label, err, r.Log)
		}
		fmt.Printf("%s passed\n", label)
		return nil
	}
	if err := execute("go-version", nil, "go", "version"); err != nil {
		return err
	}
	if err := execute("ordinary", []string{"NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_WEIGHT=", "NANOPORE_TENSOR_OLD_BACKEND="}, "go", "test", "-count=1", "-v", "./internal/tensor"); err != nil {
		return err
	}
	if err := execute("nocgo", []string{"CGO_ENABLED=0", "NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_WEIGHT=", "NANOPORE_TENSOR_OLD_BACKEND="}, "go", "test", "-count=1", "-v", "./internal/tensor"); err != nil {
		return err
	}
	native := []string{"CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND=" + *backend, "NANOPORE_TENSOR_WEIGHT=" + *weight, "NANOPORE_TENSOR_OLD_BACKEND=" + *oldBackend}
	if err := execute("native", native, "go", "test", "-count=1", "-v", "./internal/tensor"); err != nil {
		return err
	}
	if err := execute("race", native, "go", "test", "-count=1", "-race", "./internal/tensor"); err != nil {
		return err
	}
	m.Status = "passed"
	fmt.Println(filepath.Join(*out, "manifest.json"))
	return nil
}
