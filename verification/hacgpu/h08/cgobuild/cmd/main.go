package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func run() error {
	mode := flag.String("mode", "plan", "plan, queries, build, cc, tool, forbidden-cxx, observe-compiled, or check-controller")
	observationRequest := flag.String("observation-request", "", "independently selected observation request")
	observationRequestSHA := flag.String("observation-request-sha", "", "independently selected observation request digest")
	role := flag.String("role", string(cgobuild.H08Command), "fixed h08-command or cpu13-tensor-tests build role")
	session := flag.String("session", "", "immutable native session")
	sessionSHA := flag.String("session-sha", "", "independent native session digest")
	controller := flag.String("controller", "", "sealed CGO0 interposer custody")
	controllerSHA := flag.String("controller-sha", "", "sealed interposer custody digest")
	plan := flag.String("plan", "", "reviewed current Cgo compiler plan")
	planSHA := flag.String("plan-sha", "", "reviewed Cgo compiler plan digest")
	queries := flag.String("queries", "", "accepted finite metadata-query receipt")
	queriesSHA := flag.String("queries-sha", "", "accepted metadata-query receipt digest")
	native := flag.Bool("native-admitted", false, "coordinator separately admitted this exact native phase")
	root := flag.String("root", "/home/jake/Projects/nanopore_basecaller_go", "fixed consumer checkout")
	goRoot := flag.String("goroot", "/usr/lib/go", "selected Go1.27.3 installation")
	compiler := flag.String("compiler-plan", "", "accepted immutable closedbuild plan005")
	compilerSHA := flag.String("compiler-plan-sha", "", "independent accepted compiler-plan digest")
	proof := flag.String("backend-proof", "", "accepted genuine typed backend002 reader proof")
	proofSHA := flag.String("backend-proof-sha", "", "independent accepted typed proof digest")
	out := flag.String("out", "", "fresh artifacts/tensor plan directory")
	admitted := flag.Bool("host-admitted", false, "coordinator admitted this Go-only preparation")
	flag.Parse()
	sp := cgobuild.Pin{Path: *session, SHA256: *sessionSHA}
	switch *mode {
	case "observe-compiled":
		if !*admitted {
			return fmt.Errorf("%w: host observation admission required", runner.ErrAdmissionBlocked)
		}
		if err := runner.CheckRoot(); err != nil {
			return err
		}
		if _, err := closedbuild.Limits(); err != nil {
			return err
		}
		if *observationRequest == "" || *observationRequestSHA == "" || *out == "" || len(flag.Args()) != 0 {
			return fmt.Errorf("explicit observation request path/hash and fresh out required")
		}
		pin, err := cgobuild.ObserveCompiled(cgobuild.Pin{Path: *observationRequest, SHA256: *observationRequestSHA}, *out)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(pin)
	case "cc", "forbidden-cxx":
		args := flag.Args()
		if *mode == "forbidden-cxx" {
			args = append([]string{"-unadmitted-cxx-request"}, args...)
		}
		code, err := cgobuild.InterposeCC(sp, args)
		if err != nil {
			return helperError{err}
		}
		return nativeExit(code)
	case "tool":
		if err := cgobuild.InterposeTool(sp, flag.Args()); err != nil {
			return helperError{err}
		}
		return nil
	case "check-controller":
		return cgobuild.CheckController(cgobuild.Pin{Path: *controller, SHA256: *controllerSHA})
	case "queries", "build":
		_, err := cgobuild.Execute(cgobuild.DriverOptions{Plan: cgobuild.Pin{Path: *plan, SHA256: *planSHA}, Controller: cgobuild.Pin{Path: *controller, SHA256: *controllerSHA}, Queries: cgobuild.Pin{Path: *queries, SHA256: *queriesSHA}, Phase: *mode, Out: *out, Admitted: *native})
		return err
	case "plan":
	default:
		return fmt.Errorf("unknown Cgo custody mode")
	}
	if !*admitted {
		return fmt.Errorf("%w: host preparation admission required", runner.ErrAdmissionBlocked)
	}
	if *out == "" {
		return fmt.Errorf("fresh -out required")
	}
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return err
	}
	defer lock.Close()
	limits, err := closedbuild.Limits()
	if err != nil {
		return err
	}
	start := time.Now().UTC().Format(time.RFC3339Nano)
	absolute, err := filepath.Abs(*out)
	if err != nil {
		return err
	}
	p, err := cgobuild.Prepare(cgobuild.Options{Role: cgobuild.BuildRole(*role), Root: *root, GoRoot: *goRoot, WorkRoot: filepath.Join(absolute, "work"), CompilerPlan: cgobuild.Pin{Path: *compiler, SHA256: *compilerSHA}, BackendProof: cgobuild.Pin{Path: *proof, SHA256: *proofSHA}})
	if err != nil {
		return err
	}
	dir, err := runner.NewAttempt(absolute)
	if err != nil {
		return err
	}
	if err = os.Mkdir(filepath.Join(dir, "sources"), 0700); err != nil {
		return err
	}
	snapshots := map[string]cgobuild.Pin{}
	for _, path := range slices.Sorted(maps.Keys(p.Inputs)) {
		body, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		actual, e := closedbuild.PinFile(path)
		if e != nil || actual.SHA256 != p.Inputs[path] {
			return fmt.Errorf("current plan input changed: %s", path)
		}
		dest := filepath.Join(dir, "sources", actual.SHA256)
		f, e := os.OpenFile(dest, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0400)
		if os.IsExist(e) {
		} else if e != nil {
			return e
		} else {
			_, e = f.Write(body)
			ce := f.Close()
			if e != nil {
				return e
			}
			if ce != nil {
				return ce
			}
		}
		s, e := closedbuild.PinFile(dest)
		if e != nil || s.SHA256 != actual.SHA256 {
			return fmt.Errorf("plan snapshot differs: %s", path)
		}
		snapshots[path] = s
	}
	if err = closedbuild.Write(filepath.Join(dir, "plan.json"), p); err != nil {
		return err
	}
	if err = closedbuild.Write(filepath.Join(dir, "snapshots.json"), snapshots); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	binary, err := closedbuild.PinFile(exe)
	if err != nil {
		return err
	}
	return closedbuild.Write(filepath.Join(dir, "receipt.json"), struct {
		Schema, Status, Started, Finished string
		Executable                        cgobuild.Pin
		Argv                              []string
		Limits                            map[string]string
		NativeCommands                    int
	}{"nanopore-h08-cgo-plan-receipt/v1", p.Status, start, time.Now().UTC().Format(time.RFC3339Nano), binary, slices.Clone(os.Args), limits, 0})
}

type nativeExit int

func (e nativeExit) Error() string { return fmt.Sprintf("native exit %d", e) }

type helperError struct{ error }

func main() {
	if path, sha := os.Getenv("H08_SESSION"), os.Getenv("H08_SESSION_SHA"); path != "" && sha != "" && (len(os.Args) < 2 || os.Args[1] != "-mode") {
		code, err := cgobuild.InterposeCC(cgobuild.Pin{Path: path, SHA256: sha}, os.Args[1:])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(125)
		}
		os.Exit(code)
	}
	if err := run(); err != nil {
		var exit nativeExit
		if errors.As(err, &exit) {
			os.Exit(int(exit))
		}
		var helper helperError
		if errors.As(err, &helper) {
			fmt.Fprintln(os.Stderr, helper.error)
			os.Exit(125)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}
