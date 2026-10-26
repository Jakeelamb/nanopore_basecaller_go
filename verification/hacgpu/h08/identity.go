package h08

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func readIdentityPlan(pin Pin) (p IdentityPlan, err error) {
	if err = pinnedJSON(pin, &p); err != nil {
		return
	}
	if p.Schema != "nanopore-h08-identity-metadata/v1" || len(p.Seeds) == 0 || filepath.Base(p.Executable.Path) != "nvidia-smi" {
		return p, fmt.Errorf("metadata tool contract differs")
	}
	found := false
	for _, seed := range p.Seeds {
		found = found || seed == p.Executable
	}
	if !found {
		return p, fmt.Errorf("metadata executable not in prospective tool closure")
	}
	closure, e := nativeproof.CloseELF(p.Seeds, p.SearchDirs)
	if e != nil || !reflect.DeepEqual(closure, p.Closure) {
		return p, fmt.Errorf("metadata prospective ELF/runtime closure differs: %v", e)
	}
	return
}

func executeIdentity(ctx context.Context, out, name string, p IdentityPlan, ordinal int) (r runner.CommandReceipt, value string, err error) {
	// This clean metadata process is separate from the audited numerical child.
	// In particular it cannot inherit the private audit protocol or loader flags.
	args := []string{p.Executable.Path, "-i", strconv.Itoa(ordinal), "--query-gpu=index,uuid,pci.bus_id,driver_version", "--format=csv,noheader,nounits"}
	env := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "LD_LIBRARY_PATH=", "LD_PRELOAD=", "LD_AUDIT="}
	r = runner.CommandReceipt{Args: args, Environment: env, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, name+".stdout"), Stderr: filepath.Join(out, name+".stderr")}
	stdout, err := os.Create(r.Stdout)
	if err != nil {
		return r, "", err
	}
	defer stdout.Close()
	stderr, err := os.Create(r.Stderr)
	if err != nil {
		return r, "", err
	}
	defer stderr.Close()
	commandCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	command := exec.CommandContext(commandCtx, args[0], args[1:]...)
	command.Env, command.Stdout, command.Stderr = env, stdout, stderr
	err = command.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if command.ProcessState != nil {
		r.Exit = command.ProcessState.ExitCode()
	}
	r.StdoutSHA, _ = runner.SHA(r.Stdout)
	r.StderrSHA, _ = runner.SHA(r.Stderr)
	if err != nil {
		return r, "", err
	}
	b, err := os.ReadFile(r.Stdout)
	return r, strings.TrimSpace(string(b)), err
}

func validateIdentityCommand(r runner.CommandReceipt, out, name string, p IdentityPlan, ordinal int, want string) error {
	args := []string{p.Executable.Path, "-i", strconv.Itoa(ordinal), "--query-gpu=index,uuid,pci.bus_id,driver_version", "--format=csv,noheader,nounits"}
	env := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "LD_LIBRARY_PATH=", "LD_PRELOAD=", "LD_AUDIT="}
	a, e := time.Parse(time.RFC3339Nano, r.Started)
	b, f := time.Parse(time.RFC3339Nano, r.Finished)
	if e != nil || f != nil || b.Before(a) || r.Exit != 0 || !reflect.DeepEqual(args, r.Args) || !reflect.DeepEqual(env, r.Environment) || r.Stdout != filepath.Join(out, name+".stdout") || r.Stderr != filepath.Join(out, name+".stderr") {
		return fmt.Errorf("metadata command/exit/clocks differ")
	}
	body, e := readPin(Pin{Path: r.Stdout, SHA256: r.StdoutSHA})
	if e != nil {
		return e
	}
	if _, e = readPin(Pin{Path: r.Stderr, SHA256: r.StderrSHA}); e != nil {
		return e
	}
	if strings.TrimSpace(string(body)) != want {
		return fmt.Errorf("metadata source UUID differs")
	}
	return nil
}
