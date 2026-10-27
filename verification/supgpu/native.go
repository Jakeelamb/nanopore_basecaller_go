package supgpu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// RunBuild is a future admitted native phase. Merely preparing, testing or
// importing this package never calls it. Failed commands retain their receipts.
func RunBuild(planPath, planSHA string, admitted bool) (b Build, err error) {
	b = Build{Schema: "nanopore-supgpu-build/v2", Status: "failed", Inputs: []Pin{}, Depfiles: []Pin{}, LinkMaps: []Pin{}, Commands: []runner.CommandReceipt{}, PreflightCommands: []runner.CommandReceipt{}, PreDepfiles: []Pin{}, Dependencies: [][]string{}, Objects: []Pin{}}
	if !admitted {
		return b, fmt.Errorf("SUP native build requires separate coordinator admission")
	}
	var p BuildPlan
	if err = readPinnedJSON(planPath, planSHA, &p); err != nil {
		return b, err
	}
	if p.Schema != "nanopore-supgpu-build-plan/v2" || len(p.Commands) != nativeCommandCount {
		return b, fmt.Errorf("invalid finite build plan")
	}
	if err = verifyPlan(p, true); err != nil {
		return b, err
	}
	if _, err = ReadPrepared(p.Prepared.Path, p.Prepared.SHA256); err != nil {
		return b, err
	}
	b.PreparedSHA256 = p.Prepared.SHA256
	b.Compiler = p.Compiler
	b.Plan, err = pin(planPath)
	if err != nil {
		return b, err
	}
	for _, v := range append(append([]Pin{}, p.Inputs...), p.Compiler, p.Generated) {
		if err = verifyPin("", v); err != nil {
			return b, err
		}
	}
	out := filepath.Dir(p.Reference)
	if err = runner.CheckRoot(); err != nil {
		return b, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return b, err
	}
	defer lock.Close()
	defer func() {
		save := writeJSON(filepath.Join(out, "build.json"), b)
		if err == nil {
			err = save
		}
	}()
	seen := map[string]bool{}
	add := func(v Pin) {
		if !seen[v.Path] {
			b.Inputs = append(b.Inputs, v)
			seen[v.Path] = true
		}
	}
	for _, v := range p.Inputs {
		add(v)
	}
	add(p.Generated)
	add(p.Prepared)
	for i, q := range p.Queries {
		receipt, e := executeCompiler(p, p.Commands[0].Directory, out, fmt.Sprintf("query-%02d", i), q.Argv)
		b.PreflightCommands = append(b.PreflightCommands, receipt)
		if e != nil {
			return b, e
		}
		raw, e := os.ReadFile(receipt.Stdout)
		if e != nil {
			return b, e
		}
		if e = sameQueryResult(string(raw), q, p.Commands[0].Directory); e != nil {
			return b, e
		}
	}
	for i, c := range p.Preprocess {
		receipt, e := executeCompiler(p, c.Directory, out, fmt.Sprintf("deps-%02d", i), c.Argv)
		b.PreflightCommands = append(b.PreflightCommands, receipt)
		if e != nil {
			return b, e
		}
		v, e := pin(c.Depfile)
		if e != nil {
			return b, e
		}
		b.PreDepfiles = append(b.PreDepfiles, v)
		deps, e := sortedDependencies(c.Depfile, c.Directory)
		if e != nil {
			return b, e
		}
		b.Dependencies = append(b.Dependencies, deps)
		for _, path := range deps {
			v, e := pin(path)
			if e != nil {
				return b, e
			}
			add(v)
		}
	}
	beforePath := filepath.Join(out, "before-compile-inputs.json")
	if err = writeJSON(beforePath, b.Inputs); err != nil {
		return b, err
	}
	b.BeforeInputs, err = pin(beforePath)
	if err != nil {
		return b, err
	}
	for i, c := range p.Commands {
		for _, v := range b.Inputs {
			if err = verifyPin("", v); err != nil {
				return b, err
			}
		}
		receipt, e := executeCompiler(p, c.Directory, out, fmt.Sprintf("build-%02d", i), c.Argv)
		b.Commands = append(b.Commands, receipt)
		if e != nil {
			return b, e
		}
		if c.Depfile != "" {
			v, e := pin(c.Depfile)
			if e != nil {
				return b, e
			}
			b.Depfiles = append(b.Depfiles, v)
			deps, e := sortedDependencies(c.Depfile, c.Directory)
			if e != nil {
				return b, e
			}
			if !reflect.DeepEqual(deps, b.Dependencies[i]) {
				return b, fmt.Errorf("compile dependency set differs from before-compile closure")
			}
			object, e := pin(c.Argv[len(c.Argv)-1])
			if e != nil {
				return b, e
			}
			b.Objects = append(b.Objects, object)
		}
		if c.LinkMap != "" {
			v, e := pin(c.LinkMap)
			if e != nil {
				return b, e
			}
			b.LinkMaps = append(b.LinkMaps, v)
			trace, e := os.ReadFile(receipt.Stdout)
			if e != nil {
				return b, e
			}
			if e = verifyLinkInputs(string(trace), c.Directory, b.Inputs, b.Objects, false); e != nil {
				return b, e
			}
			linkMap, e := os.ReadFile(c.LinkMap)
			if e != nil {
				return b, e
			}
			if e = verifyLinkInputs(string(linkMap), c.Directory, b.Inputs, b.Objects, true); e != nil {
				return b, e
			}
		}
		for _, v := range append(append([]Pin{}, b.Inputs...), b.Objects...) {
			if err = verifyPin("", v); err != nil {
				return b, err
			}
		}
	}
	b.Reference, err = pin(p.Reference)
	if err != nil {
		return b, err
	}
	b.Observer, err = pin(p.Observer)
	if err != nil {
		return b, err
	}
	b.Audit, err = pin(p.Audit)
	if err != nil {
		return b, err
	}
	if err = verifyAuditELF(b.Audit); err != nil {
		return b, err
	}
	b.Status = "built"
	if err = verifyBuild(b, p.Prepared.SHA256); err != nil {
		b.Status = "failed"
		return b, err
	}
	return b, nil
}

func executeCompiler(p BuildPlan, cwd, out, label string, args []string) (runner.CommandReceipt, error) {
	if err := verifyCompilerRuntime(p); err != nil {
		return runner.CommandReceipt{}, err
	}
	r, err := execute(cwd, out, label, args)
	closureErr := verifyCompilerRuntime(p)
	if err == nil {
		err = closureErr
	}
	return r, err
}

func execute(cwd, out, label string, args []string) (r runner.CommandReceipt, err error) {
	r = runner.CommandReceipt{Args: args, Environment: []string{}, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
	stdout, err := os.OpenFile(r.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return r, err
	}
	defer stdout.Close()
	stderr, err := os.OpenFile(r.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return r, err
	}
	defer stderr.Close()
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = cwd
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Env = nativeEnvironment()
	r.Environment = append(append(r.Environment, "SUPGPU_COMMAND_CWD="+cwd), cmd.Env...)
	err = cmd.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	if p, e := pin(r.Stdout); e == nil {
		r.StdoutSHA = p.SHA256
	} else if err == nil {
		err = e
	}
	if p, e := pin(r.Stderr); e == nil {
		r.StderrSHA = p.SHA256
	} else if err == nil {
		err = e
	}
	return r, err
}

func nativeEnvironment() []string {
	return []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"}
}

// RunCapture executes each separately linked producer once. A failed import is
// repaired through Seal using these saved files, never by repeating inference.
func RunCapture(preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA, out string, admitted bool) (Capture, error) {
	if !admitted {
		return Capture{}, fmt.Errorf("SUP native capture requires separate coordinator admission")
	}
	if _, err := runGraphPair(preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA, out, "capture"); err != nil {
		return Capture{}, err
	}
	return Seal(out, preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA)
}

type AttestationRun struct {
	Schema      string                  `json:"schema"`
	RuntimePlan Pin                     `json:"runtime_plan"`
	Commands    []runner.CommandReceipt `json:"commands"`
	Guards      []Pin                   `json:"guards"`
}

func RunRuntimeAttest(preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA, out string, admitted bool) (AttestationRun, error) {
	if !admitted {
		return AttestationRun{}, fmt.Errorf("SUP loader attestation requires separate coordinator admission")
	}
	result, err := runGraphPair(preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA, out, "attest-only")
	if _, e := os.Stat(out); e == nil {
		if e = writeJSON(filepath.Join(out, "attestation-run.json"), result); e != nil && err == nil {
			err = e
		}
	}
	return result, err
}
func runGraphPair(preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA, out, action string) (AttestationRun, error) {
	result := AttestationRun{Schema: "nanopore-supgpu-loader-attestation/v1", Commands: []runner.CommandReceipt{}, Guards: []Pin{}}
	p, err := ReadPrepared(preparedPath, preparedSHA)
	if err != nil {
		return result, err
	}
	var b Build
	if err = readPinnedJSON(buildPath, buildSHA, &b); err != nil {
		return result, err
	}
	if err = verifyBuild(b, preparedSHA); err != nil {
		return result, err
	}
	var plan BuildPlan
	if err = readPinnedJSON(b.Plan.Path, b.Plan.SHA256, &plan); err != nil {
		return result, err
	}
	if err = verifyPlan(plan, false); err != nil {
		return result, err
	}
	runtimePin, err := pin(runtimePath)
	if err != nil {
		return result, err
	}
	if runtimePin.SHA256 != runtimeSHA {
		return result, fmt.Errorf("runtime plan changed")
	}
	buildPin, err := pin(buildPath)
	if err != nil {
		return result, err
	}
	runtime, err := readRuntimePlan(runtimePin, b, buildPin, true)
	if err != nil {
		return result, err
	}
	result.RuntimePlan = runtimePin
	if err = os.Mkdir(out, 0755); err != nil {
		return result, err
	}
	if err = runner.CheckRoot(); err != nil {
		return result, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return result, err
	}
	defer lock.Close()
	commands := []runner.CommandReceipt{}
	for _, v := range []struct{ mode, path string }{{"reference", b.Reference.Path}, {"observer", b.Observer.Path}} {
		args := []string{v.path, v.mode, filepath.Dir(p.Config.Path), p.Input.Path, filepath.Join(out, v.mode), fmt.Sprint(p.Geometry.Batch), fmt.Sprint(p.Geometry.Chunk), action}
		r, e := executeAttested("/home/jake/Projects/nanopore_basecaller_go", out, v.mode, action, args, runtime, runtimePin, readCodeMaps)
		commands = append(commands, r)
		result.Commands = commands
		// Persist actual command evidence before any parsing, including failure.
		if err = writeJSON(filepath.Join(out, v.mode+"-command.json"), r); err != nil {
			return result, err
		}
		guard, pinErr := pin(filepath.Join(out, v.mode+"-guard.json"))
		if pinErr == nil {
			result.Guards = append(result.Guards, guard)
		}
		if e != nil {
			return result, e
		}
		if pinErr != nil {
			return result, pinErr
		}
	}
	result.RuntimePlan = runtimePin
	result.Commands = commands
	return result, nil
}

func rebase(p Pin, prefix string) Pin { p.Path = filepath.Join(prefix, p.Path); return p }

// Seal projects retained raw native output. It is host-only and can be rerun
// into a fresh copy after an observer/importer repair without producer replay.
func Seal(out, preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA string) (Capture, error) {
	c := Capture{Schema: CaptureSchema, PreparedSHA256: preparedSHA, BuildSHA256: buildSHA, UseI8: true, UseF8: true, Events: []Event{}, Runtime: []Pin{}, RuntimeMaps: []Pin{}, Commands: []runner.CommandReceipt{}, Boundaries: []Boundary{}, Guards: []Pin{}}
	p, err := ReadPrepared(preparedPath, preparedSHA)
	if err != nil {
		return c, err
	}
	c.DeviceUUID = p.DeviceUUID
	var b Build
	if err = readPinnedJSON(buildPath, buildSHA, &b); err != nil {
		return c, err
	}
	if err = verifyBuild(b, preparedSHA); err != nil {
		return c, err
	}
	c.RuntimePlan, err = pin(runtimePath)
	if err != nil {
		return c, err
	}
	if c.RuntimePlan.SHA256 != runtimeSHA {
		return c, fmt.Errorf("runtime plan changed")
	}
	buildPin, err := pin(buildPath)
	if err != nil {
		return c, err
	}
	runtime, err := readRuntimePlan(c.RuntimePlan, b, buildPin, false)
	if err != nil {
		return c, err
	}
	for _, v := range []struct {
		path, prefix string
		dest         *Pin
	}{{"reference/scores.json", "reference", &c.Reference}, {"observer/staged.json", "observer", &c.Staged}} {
		raw, e := os.ReadFile(filepath.Join(out, v.path))
		if e != nil {
			return c, e
		}
		var t Pin
		if e = decodeClosed(raw, &t); e != nil {
			return c, e
		}
		*v.dest = rebase(t, v.prefix)
	}
	f, err := os.Open(filepath.Join(out, "observer/events.jsonl"))
	if err != nil {
		return c, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	for scanner.Scan() {
		var e Event
		if err = decodeClosed(scanner.Bytes(), &e); err != nil {
			return c, err
		}
		for i := range e.Tensors {
			e.Tensors[i].Data = rebase(e.Tensors[i].Data, "observer")
			e.Tensors[i].Backing = rebase(e.Tensors[i].Backing, "observer")
		}
		c.Events = append(c.Events, e)
	}
	if err = scanner.Err(); err != nil {
		return c, err
	}
	coarse, err := os.Open(filepath.Join(out, "reference/coarse.jsonl"))
	if err != nil {
		return c, err
	}
	defer coarse.Close()
	cs := bufio.NewScanner(coarse)
	for cs.Scan() {
		var b Boundary
		if err = decodeClosed(cs.Bytes(), &b); err != nil {
			return c, err
		}
		b.Data = rebase(b.Data, "reference")
		c.Boundaries = append(c.Boundaries, b)
	}
	if err = cs.Err(); err != nil {
		return c, err
	}
	seen := map[string]bool{}
	for _, mode := range []string{"reference", "observer"} {
		raw, e := os.ReadFile(filepath.Join(out, mode+"-command.json"))
		if e != nil {
			return c, e
		}
		var cmd runner.CommandReceipt
		if e = decodeClosed(raw, &cmd); e != nil {
			return c, e
		}
		c.Commands = append(c.Commands, cmd)
		guardPin, e := pin(filepath.Join(out, mode+"-guard.json"))
		if e != nil {
			return c, e
		}
		c.Guards = append(c.Guards, guardPin)
		var guard GuardReceipt
		if e = readPinnedJSON(guardPin.Path, guardPin.SHA256, &guard); e != nil {
			return c, e
		}
		actual, e := verifyGuard(guard, runtime, c.RuntimePlan, mode, "capture")
		if e != nil {
			return c, e
		}
		for _, object := range actual {
			if !seen[object.Path] {
				seen[object.Path] = true
				c.Runtime = append(c.Runtime, object)
			}
		}
		path := filepath.Join(out, mode, "runtime.maps")
		v, e := pin(path)
		if e != nil {
			return c, e
		}
		c.RuntimeMaps = append(c.RuntimeMaps, v)

	}
	if err = verifyCapture(out, p, c); err != nil {
		return c, err
	}
	if err = requireCaptureRuntime(c.Runtime, runtime.Policy); err != nil {
		return c, err
	}
	if err = verifyCaptureCommands(out, p, b, c, runtime); err != nil {
		return c, err
	}
	if err = writeJSON(filepath.Join(out, "capture.json"), c); err != nil {
		return c, err
	}
	return c, nil
}

// Marshal is used by the finite command for reviewable host output.
func Marshal(v any) ([]byte, error) { return json.MarshalIndent(v, "", "  ") }
