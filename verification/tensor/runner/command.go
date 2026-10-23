package runner

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type CommandReceipt struct {
	Args        []string `json:"command"`
	Environment []string `json:"environment_overrides"`
	Started     string   `json:"started_utc"`
	Finished    string   `json:"finished_utc"`
	Exit        int      `json:"exit_code"`
	Stdout      string   `json:"stdout"`
	Stderr      string   `json:"stderr"`
	StdoutSHA   string   `json:"stdout_sha256"`
	StderrSHA   string   `json:"stderr_sha256"`
}

func SHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func Execute(dir, label string, overrides []string, args ...string) (CommandReceipt, error) {
	r := CommandReceipt{Args: args, Environment: overrides, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(dir, label+".stdout"), Stderr: filepath.Join(dir, label+".stderr")}
	stdout, err := os.Create(r.Stdout)
	if err != nil {
		return r, err
	}
	defer stdout.Close()
	stderr, err := os.Create(r.Stderr)
	if err != nil {
		return r, err
	}
	defer stderr.Close()
	cmd := exec.Command(args[0], args[1:]...)
	env := os.Environ()
	for _, kv := range overrides {
		key, _, _ := strings.Cut(kv, "=")
		clean := env[:0]
		for _, prior := range env {
			if !strings.HasPrefix(prior, key+"=") {
				clean = append(clean, prior)
			}
		}
		env = append(clean, kv)
	}
	cmd.Env = env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err = cmd.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	r.StdoutSHA, _ = SHA(r.Stdout)
	r.StderrSHA, _ = SHA(r.Stderr)
	if err != nil {
		return r, fmt.Errorf("%s failed: %v; see %s", label, err, r.Stderr)
	}
	return r, nil
}
func NewAttempt(path string) (string, error) {
	root, err := os.Getwd()
	if err != nil {
		return "", err
	}
	out, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	base := filepath.Join(root, "artifacts", "tensor")
	rel, err := filepath.Rel(base, out)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("new tensor attempts must be below %s", base)
	}
	if err = os.MkdirAll(filepath.Dir(out), 0755); err != nil {
		return "", err
	}
	actual, err := filepath.EvalSymlinks(filepath.Dir(out))
	if err != nil {
		return "", err
	}
	if actual != filepath.Dir(out) {
		return "", fmt.Errorf("attempt parents cannot be symlinks")
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return "", fmt.Errorf("fresh attempt required: %w", err)
	}
	return out, nil
}
