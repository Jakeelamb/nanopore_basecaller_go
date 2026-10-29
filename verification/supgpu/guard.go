package supgpu

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type GateRequest struct {
	Schema    string `json:"schema"`
	Kind      string `json:"kind"`
	Sequence  uint64 `json:"sequence"`
	Cookie    uint64 `json:"cookie"`
	Path      string `json:"path"`
	Namespace int64  `json:"namespace"`
	PID       int    `json:"pid"`
}
type GateDecision struct {
	Request  GateRequest `json:"request"`
	Maps     Pin         `json:"maps"`
	Object   Pin         `json:"object"`
	Code     []Pin       `json:"code"`
	Approved bool        `json:"approved"`
	Error    string      `json:"error"`
}
type GuardReceipt struct {
	Schema       string              `json:"schema"`
	RuntimePlan  Pin                 `json:"runtime_plan"`
	Role         string              `json:"role"`
	Action       string              `json:"action"`
	PID          int                 `json:"pid"`
	StartupMaps  Pin                 `json:"startup_maps"`
	StartupError string              `json:"startup_error"`
	Decisions    []GateDecision      `json:"decisions"`
	Group        ProcessGroupCleanup `json:"process_group"`
	Error        string              `json:"error"`
}

type ProcessGroupMember struct {
	PID   int    `json:"pid"`
	State string `json:"state"`
	Stat  string `json:"proc_stat"`
}
type ProcessGroupObservation struct {
	At      string               `json:"observed_utc"`
	Members []ProcessGroupMember `json:"members"`
}
type ProcessGroupCleanup struct {
	PGID         int                       `json:"pgid"`
	Observations []ProcessGroupObservation `json:"observations"`
	Kills        []string                  `json:"sigkill_utc"`
	Empty        bool                      `json:"empty"`
	Error        string                    `json:"error"`
}

// A reaped leader and drained pipes do not prove that its process group ended.
func observeProcessGroup(pgid int) (ProcessGroupObservation, error) {
	o := ProcessGroupObservation{At: time.Now().UTC().Format(time.RFC3339Nano), Members: []ProcessGroupMember{}}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return o, err
	}
	o.Members, err = scanProcessGroup(pgid, entries, os.ReadFile)
	return o, err
}

func scanProcessGroup(pgid int, entries []os.DirEntry, readFile func(string) ([]byte, error)) ([]ProcessGroupMember, error) {
	members := []ProcessGroupMember{}
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || !entry.IsDir() {
			continue
		}
		raw, err := readFile(filepath.Join("/proc", entry.Name(), "stat"))
		if os.IsNotExist(err) || errors.Is(err, syscall.ESRCH) {
			continue
		}
		if err != nil {
			return members, err
		}
		end := strings.LastIndexByte(string(raw), ')')
		if end < 0 {
			return members, fmt.Errorf("invalid process stat for %d", pid)
		}
		fields := strings.Fields(string(raw[end+1:]))
		if len(fields) < 3 {
			return members, fmt.Errorf("short process stat for %d", pid)
		}
		group, err := strconv.Atoi(fields[2])
		if err != nil {
			return members, err
		}
		if group == pgid {
			members = append(members, ProcessGroupMember{PID: pid, State: fields[0], Stat: string(raw)})
		}
	}
	sort.Slice(members, func(i, j int) bool { return members[i].PID < members[j].PID })
	return members, nil
}

func killProcessGroup(g *ProcessGroupCleanup) error {
	o, err := observeProcessGroup(g.PGID)
	g.Observations = append(g.Observations, o)
	// Kill regardless of the observation result or the leader's Wait state.
	g.Kills = append(g.Kills, time.Now().UTC().Format(time.RFC3339Nano))
	if e := syscall.Kill(-g.PGID, syscall.SIGKILL); e != nil && e != syscall.ESRCH {
		return e
	}
	return err
}

func finishProcessGroup(g *ProcessGroupCleanup) (err error) {
	defer func() {
		if err != nil {
			g.Error = err.Error()
		}
	}()
	o, err := observeProcessGroup(g.PGID)
	g.Observations = append(g.Observations, o)
	if err != nil {
		killProcessGroup(g)
		return err
	}
	if len(o.Members) == 0 {
		g.Empty = true
		return nil
	}
	if err = killProcessGroup(g); err != nil {
		return err
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		o, err = observeProcessGroup(g.PGID)
		if err != nil || len(o.Members) == 0 || time.Now().After(deadline) {
			g.Observations = append(g.Observations, o)
			if err != nil {
				return err
			}
			g.Empty = len(o.Members) == 0
			if !g.Empty {
				return fmt.Errorf("target process group %d remains after SIGKILL", g.PGID)
			}
			return fmt.Errorf("target process group %d outlived direct child", g.PGID)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

type mapObservation struct {
	Raw   []byte
	Names []string
	VDSO  bool
}
type mapReader func(int, RuntimePolicy) (mapObservation, error)

func readCodeMaps(pid int, p RuntimePolicy) (mapObservation, error) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return mapObservation{}, err
	}
	return parseCodeMaps(raw, p)
}
func parseCodeMaps(raw []byte, p RuntimePolicy) (mapObservation, error) {
	o := mapObservation{Raw: raw, Names: []string{}}
	known := map[string]bool{p.Reference.Path: true, p.Observer.Path: true, p.Audit.Path: true}
	for _, c := range append(append([]RuntimeCode{}, p.Allowed...), p.Alternates...) {
		known[c.Object.Path] = true
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		f := strings.Fields(line)
		if len(f) < 5 {
			continue
		}
		executable := strings.Contains(f[1], "x")
		name := ""
		if len(f) > 5 {
			name = strings.Join(f[5:], " ")
		}
		if name == "[vdso]" {
			o.VDSO = true
			continue
		}
		if name == "[vsyscall]" {
			continue
		}
		if !filepath.IsAbs(name) {
			if executable {
				return o, fmt.Errorf("anonymous/unrecognized executable mapping: %s", line)
			}
			continue
		}
		deleted := strings.HasSuffix(name, " (deleted)")
		path := strings.TrimSuffix(name, " (deleted)")
		actual, err := filepath.EvalSymlinks(path)
		if !executable && !known[path] && (err != nil || !known[actual]) {
			continue
		}
		if deleted {
			return o, fmt.Errorf("deleted mapped object: %s", name)
		}
		if err != nil {
			return o, err
		}
		if !seen[actual] {
			seen[actual] = true
			o.Names = append(o.Names, actual)
		}
	}
	sort.Strings(o.Names)
	return o, nil
}

func supPhaseNames(role, action string) []string {
	if action == "attest-only" {
		return []string{"pre-model"}
	}
	if role == "reference" {
		return []string{"pre-model", "pre-forward", "post-full", "post-staged"}
	}
	return []string{"pre-model", "pre-forward", "post-observed"}
}

type gateState struct {
	hello         bool
	auditSequence uint64
	phase         int
	opened        map[uint64]Pin
	approved      []Pin
}

func decideGate(q GateRequest, channel string, p RuntimePolicy, role, action string, pid int, s *gateState, o mapObservation) (GateDecision, error) {
	return decideGatePhases(q, channel, p, role, action, pid, s, o, supPhaseNames(role, action))
}

func decideGatePhases(q GateRequest, channel string, p RuntimePolicy, role, action string, pid int, s *gateState, o mapObservation, phases []string) (GateDecision, error) {
	d := GateDecision{Request: q, Code: []Pin{}}
	bad := func(err error) (GateDecision, error) { d.Error = err.Error(); return d, err }
	if q.Schema != "nanopore-supgpu-loader-gate/v1" || q.PID != pid || q.Namespace != 0 {
		return bad(fmt.Errorf("loader gate schema/pid/namespace differs"))
	}
	// All actually executable images must be policy members, even during startup.
	for _, name := range o.Names {
		known := false
		root := p.Reference
		if role == "observer" {
			root = p.Observer
		}
		if name == root.Path || name == p.Audit.Path {
			known = true
		}
		for _, c := range p.Allowed {
			known = known || name == c.Object.Path
		}
		if !known {
			for _, c := range p.Alternates {
				if name == c.Object.Path {
					return bad(fmt.Errorf("known wrong runtime tier rejected: %s", name))
				}
			}
			return bad(fmt.Errorf("unknown mapped code rejected: %s", name))
		}
	}
	if channel == "audit" {
		if q.Sequence != s.auditSequence+1 {
			return bad(fmt.Errorf("audit request sequence changed"))
		}
		s.auditSequence = q.Sequence
		switch q.Kind {
		case "hello":
			if s.hello || q.Sequence != 1 || q.Path != "" || q.Cookie != 0 {
				return bad(fmt.Errorf("invalid audit startup"))
			}
			if !containsString(o.Names, p.Audit.Path) {
				return bad(fmt.Errorf("required auditor not actually mapped"))
			}
			if err := verifyPin("", p.Audit); err != nil {
				return bad(err)
			}
			s.hello = true
			d.Object = p.Audit
		case "open":
			if !s.hello || q.Cookie == 0 {
				return bad(fmt.Errorf("object arrived before audit startup"))
			}
			if _, exists := s.opened[q.Cookie]; exists {
				return bad(fmt.Errorf("duplicate loader object cookie"))
			}
			if q.Path == "linux-vdso.so.1" {
				if !o.VDSO {
					return bad(fmt.Errorf("unobserved kernel vdso"))
				}
				s.opened[q.Cookie] = Pin{}
				break
			}
			path := q.Path
			if path == "" {
				path = p.Reference.Path
				if role == "observer" {
					path = p.Observer.Path
				}
			}
			object, err := canonicalPin(path)
			if err != nil {
				return bad(err)
			}
			d.Object = object
			if !containsString(o.Names, object.Path) {
				return bad(fmt.Errorf("loader object is not actually mapped: %s", object.Path))
			}
			if err = requireSelectedCode(object, p, role); err != nil {
				return bad(err)
			}
			s.opened[q.Cookie] = object
		case "close":
			if !s.hello || q.Path != "" {
				return bad(fmt.Errorf("invalid loader close"))
			}
			object, ok := s.opened[q.Cookie]
			if !ok {
				return bad(fmt.Errorf("unknown loader close cookie"))
			}
			d.Object = object
			delete(s.opened, q.Cookie)
		default:
			return bad(fmt.Errorf("unexpected audit request kind"))
		}
	} else {
		names := phases
		if q.Kind != "phase" || !s.hello || q.Cookie != 0 || q.Sequence != uint64(s.phase+1) || s.phase >= len(names) || q.Path != names[s.phase] {
			return bad(fmt.Errorf("model phase arrived without the required audit/phase order"))
		}
		for _, name := range o.Names {
			object, err := canonicalPin(name)
			if err != nil {
				return bad(err)
			}
			d.Code = append(d.Code, object)
		}
		if err := requireSelectedMappings(d.Code, p, role, true); err != nil {
			return bad(err)
		}
		s.phase++
	}
	if d.Object.Path != "" && channel == "audit" && q.Kind != "close" {
		s.approved = append(s.approved, d.Object)
	}
	d.Approved = true
	return d, nil
}

func completeGuard(s gateState, p RuntimePolicy, role string, phases []string, mode guardMode) error {
	if !s.hello || s.phase != len(phases) {
		return fmt.Errorf("target exited without complete loader/model gates")
	}
	if mode == metadataGuard {
		return requireSelectedMappings(s.approved, p, role, true)
	}
	return nil
}

type gateEvent struct {
	channel string
	line    []byte
	err     error
}

// executeAttested is used only after the public exact-phase admission guards.
// Tests inject a map reader and a static Go protocol peer, never a native DSO.
func executeAttested(cwd, out, role, action string, args []string, r RuntimePlan, runtimePin Pin, readMaps mapReader) (receipt runner.CommandReceipt, err error) {
	return executeGuardCore(context.Background(), cwd, out, role, action, args, r.Policy, runtimePin, supPhaseNames(role, action), phaseGuard, readMaps)
}

func executeGuardCore(ctx context.Context, cwd, out, role, action string, args []string, policy RuntimePolicy, runtimePin Pin, phases []string, mode guardMode, readMaps mapReader) (receipt runner.CommandReceipt, err error) {
	receipt = runner.CommandReceipt{Args: args, Environment: append([]string{"SUPGPU_COMMAND_CWD=" + cwd, "SUPGPU_GATE_FDS=3,4,5,6"}, modelEnvironment(policy)...), Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, role+".stdout"), Stderr: filepath.Join(out, role+".stderr")}
	g := GuardReceipt{Schema: "nanopore-supgpu-loader-transcript/v1", RuntimePlan: runtimePin, Role: role, Action: action, Decisions: []GateDecision{}}
	defer func() {
		receipt.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		for _, v := range []struct {
			path string
			sha  *string
		}{{receipt.Stdout, &receipt.StdoutSHA}, {receipt.Stderr, &receipt.StderrSHA}} {
			if p, e := pin(v.path); e == nil {
				*v.sha = p.SHA256
			} else if err == nil {
				err = e
			}
		}
		if err != nil {
			g.Error = err.Error()
		}
		if e := writeJSON(filepath.Join(out, role+"-guard.json"), g); e != nil && err == nil {
			err = e
		}
	}()
	stdout, e := os.OpenFile(receipt.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if e != nil {
		return receipt, e
	}
	defer stdout.Close()
	stderr, e := os.OpenFile(receipt.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if e != nil {
		return receipt, e
	}
	defer stderr.Close()
	if e = os.Mkdir(filepath.Join(out, role+"-guard-maps"), 0755); e != nil {
		return receipt, e
	}
	requests := []*os.File{}
	replies := []*os.File{}
	child := []*os.File{}
	defer func() {
		for _, f := range append(append(requests, replies...), child...) {
			f.Close()
		}
	}()
	for i := 0; i < 2; i++ {
		rd, wr, e := os.Pipe()
		if e != nil {
			return receipt, e
		}
		ackrd, ackwr, e := os.Pipe()
		if e != nil {
			rd.Close()
			wr.Close()
			return receipt, e
		}
		requests = append(requests, rd)
		replies = append(replies, ackwr)
		child = append(child, wr, ackrd)
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = cwd
	cmd.Env = modelEnvironment(policy)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.ExtraFiles = child
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	if e = ctx.Err(); e != nil {
		return receipt, e
	}
	if e = cmd.Start(); e != nil {
		return receipt, e
	}
	g.PID = cmd.Process.Pid
	g.Group = ProcessGroupCleanup{PGID: g.PID, Observations: []ProcessGroupObservation{}, Kills: []string{}}
	receipt.Environment = append(receipt.Environment, fmt.Sprintf("SUPGPU_TARGET_PID=%d", g.PID))
	if raw, e := os.ReadFile(fmt.Sprintf("/proc/%d/maps", g.PID)); e == nil {
		path := filepath.Join(out, role+"-startup.maps")
		if e = os.WriteFile(path, raw, 0644); e == nil {
			g.StartupMaps, e = pin(path)
		}
		if e != nil {
			g.StartupError = e.Error()
		}
	} else {
		g.StartupError = e.Error()
	}
	for _, f := range child {
		f.Close()
	}
	stream := make(chan gateEvent, 2)
	var wg sync.WaitGroup
	for i, rd := range requests {
		wg.Add(1)
		go func(i int, rd *os.File) {
			defer wg.Done()
			sc := bufio.NewScanner(rd)
			sc.Buffer(make([]byte, 4096), 8192)
			for sc.Scan() {
				stream <- gateEvent{[]string{"audit", "phase"}[i], append([]byte{}, sc.Bytes()...), nil}
			}
			if e := sc.Err(); e != nil {
				stream <- gateEvent{err: e}
			}
		}(i, rd)
	}
	go func() { wg.Wait(); close(stream) }()
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
	s := gateState{opened: map[uint64]Pin{}}
	var failure error
	var grace <-chan time.Time
	waited := false
	count := 0
	cancelled := ctx.Done()
	for stream != nil || !waited {
		select {
		case <-cancelled:
			cancelled = nil
			failure = errors.Join(failure, ctx.Err(), killProcessGroup(&g.Group))
			for _, rd := range requests {
				rd.Close()
			}
			grace = time.After(5 * time.Second)
		case event, ok := <-stream:
			if !ok {
				stream = nil
				continue
			}
			var q GateRequest
			d := GateDecision{Code: []Pin{}}
			problem := event.err
			if problem == nil {
				problem = decodeClosed(event.line, &q)
				d.Request = q
			}
			var o mapObservation
			if problem == nil {
				o, problem = readMaps(g.PID, policy)
			}
			mapPath := filepath.Join(out, role+"-guard-maps", fmt.Sprintf("%04d.maps", count))
			count++
			if len(o.Raw) > 0 {
				if w := os.WriteFile(mapPath, o.Raw, 0644); w != nil && problem == nil {
					problem = w
				}
				if p, w := pin(mapPath); w == nil {
					d.Maps = p
				} else if problem == nil {
					problem = w
				}
			}
			mapPin := d.Maps
			if problem == nil && failure == nil {
				d, problem = decideGatePhases(q, event.channel, policy, role, action, g.PID, &s, o, phases)
				d.Maps = mapPin
			}
			if problem == nil && failure != nil {
				problem = failure
			}
			if problem != nil {
				d.Approved = false
				d.Error = problem.Error()
				if failure == nil {
					failure = problem
					grace = time.After(5 * time.Second)
				}
			}
			g.Decisions = append(g.Decisions, d)
			// Each decision and map is durably projected before its approval.
			if e := writeGateProgress(filepath.Join(out, role+"-guard-progress.json"), g); e != nil {
				if failure == nil {
					failure = e
					grace = time.After(5 * time.Second)
				}
				problem = e
			}
			index := 0
			if event.channel == "phase" {
				index = 1
			}
			ack := "deny\n"
			if problem == nil {
				ack = fmt.Sprintf("allow %s %d\n", q.Kind, q.Sequence)
			}
			if _, e := replies[index].WriteString(ack); e != nil && failure == nil {
				failure = e
				grace = time.After(5 * time.Second)
			}
		case e := <-wait:
			waited = true
			wait = nil
			receipt.Exit = cmd.ProcessState.ExitCode()
			if e != nil && failure == nil {
				failure = e
			}
			if stream != nil {
				grace = time.After(5 * time.Second)
			}
		case <-grace:
			if e := killProcessGroup(&g.Group); e != nil && failure == nil {
				failure = e
			}
			for _, rd := range requests {
				rd.Close()
			}
			grace = nil
			if failure == nil {
				failure = fmt.Errorf("gate pipes did not drain after process exit")
			}
		}
	}
	if e := finishProcessGroup(&g.Group); e != nil && failure == nil {
		failure = e
	}
	if failure == nil {
		failure = completeGuard(s, policy, role, phases, mode)
	}
	return receipt, failure
}

func writeGateProgress(path string, g GuardReceipt) error {
	raw, err := Marshal(g)
	if err != nil {
		return err
	}
	tmp := path + ".next"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	_, err = f.Write(append(raw, '\n'))
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func verifyGuard(g GuardReceipt, r RuntimePlan, runtimePin Pin, role, action string) ([]Pin, error) {
	return replayGuardCore(g, r.Policy, runtimePin, role, action, supPhaseNames(role, action), phaseGuard)
}

func replayGuardCore(g GuardReceipt, policy RuntimePolicy, runtimePin Pin, role, action string, phases []string, mode guardMode) ([]Pin, error) {
	if g.Schema != "nanopore-supgpu-loader-transcript/v1" || g.RuntimePlan != runtimePin || g.Role != role || g.Action != action || g.PID <= 0 || g.Error != "" {
		return nil, fmt.Errorf("incomplete/mismatched loader transcript")
	}
	if g.Group.PGID != g.PID || !g.Group.Empty || g.Group.Error != "" || len(g.Group.Kills) != 0 || len(g.Group.Observations) == 0 || len(g.Group.Observations[len(g.Group.Observations)-1].Members) != 0 {
		return nil, fmt.Errorf("loader process group did not finish cleanly")
	}
	if g.StartupError != "" {
		return nil, fmt.Errorf("startup map observation failed: %s", g.StartupError)
	}
	if err := verifyPin("", g.StartupMaps); err != nil {
		return nil, err
	}
	s := gateState{opened: map[uint64]Pin{}}
	actual := []Pin{}
	seen := map[string]bool{}
	for _, d := range g.Decisions {
		if !d.Approved || d.Error != "" {
			return nil, fmt.Errorf("rejected loader decision cannot qualify capture")
		}
		if err := verifyPin("", d.Maps); err != nil {
			return nil, err
		}
		raw, err := os.ReadFile(d.Maps.Path)
		if err != nil {
			return nil, err
		}
		o, err := parseCodeMaps(raw, policy)
		if err != nil {
			return nil, err
		}
		channel := "audit"
		if d.Request.Kind == "phase" {
			channel = "phase"
		}
		want, err := decideGatePhases(d.Request, channel, policy, role, action, g.PID, &s, o, phases)
		if err != nil {
			return nil, err
		}
		want.Maps = d.Maps
		if !reflect.DeepEqual(want, d) {
			return nil, fmt.Errorf("loader decision differs from retained maps/policy")
		}
		pins := append(append([]Pin{}, d.Code...), d.Object)
		for _, p := range pins {
			if p.Path != "" && !seen[p.Path] {
				seen[p.Path] = true
				actual = append(actual, p)
			}
		}
	}
	if err := completeGuard(s, policy, role, phases, mode); err != nil {
		return nil, err
	}
	return actual, nil
}

func verifyCaptureGuardFiles(dir string, c Capture, r RuntimePlan) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if len(c.Guards) != 2 {
		return fmt.Errorf("both process loader transcripts are required")
	}
	actual := []Pin{}
	seen := map[string]bool{}
	for i, role := range []string{"reference", "observer"} {
		p := c.Guards[i]
		if p.Path != filepath.Join(abs, role+"-guard.json") {
			return fmt.Errorf("loader transcript binding changed")
		}
		var g GuardReceipt
		if err := readPinnedJSON(p.Path, p.SHA256, &g); err != nil {
			return err
		}
		pins, err := verifyGuard(g, r, c.RuntimePlan, role, "capture")
		if err != nil {
			return err
		}
		for j, d := range g.Decisions {
			if d.Maps.Path != filepath.Join(abs, role+"-guard-maps", fmt.Sprintf("%04d.maps", j)) {
				return fmt.Errorf("actual process map receipt binding changed")
			}
		}
		for _, pin := range pins {
			if !seen[pin.Path] {
				seen[pin.Path] = true
				actual = append(actual, pin)
			}
		}
	}
	if !reflect.DeepEqual(actual, c.Runtime) {
		return fmt.Errorf("runtime image set differs from loader transcripts")
	}
	return requireCaptureRuntime(actual, r.Policy)
}
