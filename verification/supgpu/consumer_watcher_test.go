package supgpu

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Host-only peers: no native backend, device, stock lock or qualification.
func TestConsumerWatcherPeer(t *testing.T) {
	at := -1
	for i, arg := range os.Args {
		if arg == "--" {
			at = i + 1
			break
		}
	}
	if at < 0 {
		return
	}
	a := os.Args[at:]
	if a[0] == "identity" || a[0] == "graph" {
		if os.Getenv("SUPGPU_OVERLAP_CHILD") == "1" {
			time.Sleep(8 * time.Second)
			os.Exit(0)
		}
		time.Sleep(250 * time.Millisecond)
		os.Exit(0)
	}
	mode := a[0]
	exe, _ := os.Executable()
	if mode == "overlap" {
		children := []*exec.Cmd{}
		for _, role := range []string{"identity", "graph"} {
			child := exec.Command(exe, "-test.run=^TestConsumerWatcherPeer$", "--", role)
			child.Env = append(os.Environ(), "SUPGPU_OVERLAP_CHILD=1")
			child.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			if err := child.Start(); err != nil {
				os.Exit(54)
			}
			children = append(children, child)
			if role == "identity" {
				// Keep the first child alive and let it be observed before
				// starting graph. The receipt must prove that ordering.
				time.Sleep(500 * time.Millisecond)
			}
		}
		// Reap both children even when the watcher refuses graph and signals
		// both private groups. A serial launch/Wait cannot reach overlap.
		for _, child := range children {
			child.Wait()
		}
		os.Exit(0)
	}
	order := []string{"identity", "graph", "identity"}
	switch mode {
	case "wrong-order":
		order = []string{"graph", "identity", "identity"}
	case "extra":
		order = append(order, "graph")
	case "missing":
		order = order[:2]
	case "duplicate":
		order = []string{"identity", "identity", "graph", "identity"}
	}
	for _, role := range order {
		child := exec.Command(exe, "-test.run=^TestConsumerWatcherPeer$", "--", role)
		if mode != "nonprivate" {
			child.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		}
		if err := child.Start(); err != nil {
			os.Exit(51)
		}
		if mode == "failure" || mode == "timeout" {
			// Leave an observed role alive across parent failure/cancellation.
			time.Sleep(100 * time.Millisecond)
			if mode == "failure" {
				os.Exit(53)
			}
			for {
				time.Sleep(time.Hour)
			}
		}
		if err := child.Wait(); err != nil {
			os.Exit(52)
		}
	}
	time.Sleep(100 * time.Millisecond)
	os.Exit(0)
}

func consumerWatchPlan(t *testing.T, mode string) map[string]any {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	image, err := canonicalPin(exe)
	if err != nil {
		t.Fatal(err)
	}
	args := func(role string) []string {
		return []string{image.Path, "-test.run=^TestConsumerWatcherPeer$", "--", role}
	}
	roles := []map[string]any{}
	for i, name := range []string{"before-identity", "graph", "after-identity"} {
		role := "identity"
		if i == 1 {
			role = "graph"
		}
		roles = append(roles, map[string]any{"name": name, "argv": args(role), "sha256": image.SHA256, "limit_seconds": 2})
	}
	return map[string]any{"schema": "nanopore-sup-consumer-watch/v1", "controller_argv": args(mode), "controller_sha256": image.SHA256, "controller_limit_seconds": 4, "outer_limit_seconds": 64, "roles": roles}
}

func watchConsumerPlan(t *testing.T, plan map[string]any) (int, []byte) {
	t.Helper()
	dir := t.TempDir()
	planPath, script, report := filepath.Join(dir, "plan.json"), filepath.Join(dir, "watch.py"), filepath.Join(dir, "receipt.json")
	if err := writeJSON(planPath, plan); err != nil {
		t.Fatal(err)
	}
	p, err := pin(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(script, []byte(roleWatcherSource), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	started := time.Now().UTC()
	raw, err := exec.CommandContext(ctx, "/usr/bin/python3", script, planPath, report, p.SHA256).CombinedOutput()
	finished := time.Now().UTC()
	body, _ := os.ReadFile(report)
	// Preserve the producer output before projecting assertions, including
	// failed attempts. The external runner owns the enclosing process cap.
	if evidence := os.Getenv("SUPGPU_GUARD_TEST_EVIDENCE"); evidence != "" {
		dest := filepath.Join(evidence, t.Name())
		if err := os.MkdirAll(dest, 0755); err != nil {
			t.Fatal(err)
		}
		planRaw, readErr := os.ReadFile(planPath)
		if readErr != nil {
			t.Fatal(readErr)
		}
		for name, data := range map[string][]byte{"plan.json": planRaw, "receipt.json": body, "combined-output": raw} {
			if err := os.WriteFile(filepath.Join(dest, name), data, 0644); err != nil {
				t.Fatal(err)
			}
		}
		if saveErr := writeJSON(filepath.Join(dest, "attempt.json"), map[string]any{"started": started, "finished": finished, "plan_sha256": p.SHA256, "process_state": fmt.Sprint(err), "context_error": fmt.Sprint(ctx.Err())}); saveErr != nil {
			t.Fatal(saveErr)
		}
	}
	code := 0
	if err != nil {
		exit, ok := err.(*exec.ExitError)
		if !ok || ctx.Err() != nil {
			t.Fatalf("watcher did not terminate under test cap: %v %s", err, raw)
		}
		code = exit.ExitCode()
	}
	return code, body
}

func TestConsumerWatcherPlanReplacementBoundary(t *testing.T) {
	dir := t.TempDir()
	if evidence := os.Getenv("SUPGPU_GUARD_TEST_EVIDENCE"); evidence != "" {
		dir = filepath.Join(evidence, t.Name())
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	script := filepath.Join(dir, "watch.py")
	approved, replacement := filepath.Join(dir, "approved.json"), filepath.Join(dir, "replacement.json")
	active, report := filepath.Join(dir, "active.json"), filepath.Join(dir, "receipt.json")
	if err := os.WriteFile(script, []byte(roleWatcherSource), 0600); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(approved, consumerWatchPlan(t, "success")); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(replacement, consumerWatchPlan(t, "wrong-order")); err != nil {
		t.Fatal(err)
	}
	p, err := pin(approved)
	if err != nil {
		t.Fatal(err)
	}
	q, err := pin(replacement)
	if err != nil {
		t.Fatal(err)
	}
	// Run the real __main__ path. Swap the pathname exactly when the first
	// plan digest is returned, before the old entry point's second read.
	// The digest hook performs the swap; Popen captures which plan reaches
	// the actual launch boundary without launching a controller.
	control := `import hashlib,json,os,pathlib,runpy,subprocess,sys
script,approved,replacement,active,report,expected=sys.argv[1:]
a=pathlib.Path(approved).read_bytes(); b=pathlib.Path(replacement).read_bytes()
pathlib.Path(active).write_bytes(a)
real_sha=hashlib.sha256; swapped=False; observed=None
class Digest:
    def __init__(self,*args,**kwargs): self.inner=real_sha(*args,**kwargs)
    def update(self,data): self.inner.update(data)
    def hexdigest(self):
        global swapped
        digest=self.inner.hexdigest()
        if digest==expected and not swapped:
            os.replace(replacement,active); swapped=True
        return digest
class LaunchObserved(Exception): pass
def launch(argv,**kwargs):
    global observed
    observed=argv
    raise LaunchObserved()
hashlib.sha256=Digest; subprocess.Popen=launch
sys.argv=[script,active,report,expected]
try: runpy.run_path(script,run_name='__main__')
except LaunchObserved: pass
finally:
    pathlib.Path(report).write_text(json.dumps({'swapped':swapped,'observed_controller_argv':observed,'approved_sha256':expected,'replacement_sha256':real_sha(b).hexdigest()}))
assert swapped and pathlib.Path(active).read_bytes()==b
assert observed==json.loads(a)['controller_argv'], 'decoded unapproved replacement at real entry boundary'
`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := time.Now().UTC()
	raw, runErr := exec.CommandContext(ctx, "/usr/bin/python3", "-c", control, script, approved, replacement, active, report, p.SHA256).CombinedOutput()
	finished := time.Now().UTC()
	if err := os.WriteFile(filepath.Join(dir, "combined-output"), raw, 0644); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(filepath.Join(dir, "attempt.json"), map[string]any{"started": started, "finished": finished, "approved_sha256": p.SHA256, "replacement_sha256": q.SHA256, "process_state": fmt.Sprint(runErr)}); err != nil {
		t.Fatal(err)
	}
	if runErr != nil {
		t.Fatalf("original-byte entry boundary failed: %v %s (artifacts %s)", runErr, raw, dir)
	}
}

// Import the real watcher and inject only the stat boundary. Metadata reads
// still use this host-only helper's own process; no controller is launched.
func TestConsumerWatcherProcessIdentityExitBoundary(t *testing.T) {
	dir := t.TempDir()
	if evidence := os.Getenv("SUPGPU_GUARD_TEST_EVIDENCE"); evidence != "" {
		dir = filepath.Join(evidence, t.Name())
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	script := filepath.Join(dir, "watch.py")
	if err := os.WriteFile(script, []byte(roleWatcherSource), 0600); err != nil {
		t.Fatal(err)
	}
	control := `import importlib.util,os,sys
s=importlib.util.spec_from_file_location('watch',sys.argv[1]); w=importlib.util.module_from_spec(s); s.loader.exec_module(w)
pid=os.getpid(); stable=w.process_identity(pid,with_hash=True)
assert stable is not None, 'helper process unavailable'
first={k:stable[k] for k in ('pid','state','ppid','pgid','sid','start_ticks','raw')}
original=w.process_stat
for hashed in (False,True):
    expected=dict(stable)
    if not hashed: del expected['image_sha256']
    cases=[('initial-missing',[None],False,None),
           ('initial-zombie',[dict(first,state='Z')],False,None),
           ('second-missing',[first,None],False,None),
           ('second-zombie',[first,dict(first,state='Z')],False,None),
           ('stable',[first,first],False,expected)]
    for key in ('start_ticks','pgid','sid'):
        for state in (first['state'],'Z'):
            changed=dict(first,state=state); changed[key]+=1
            cases.append(('changed-'+key+'-'+state,[first,changed],True,None))
    for name,records,rejected,want in cases:
        answers=iter(None if r is None else dict(r) for r in records)
        w.process_stat=lambda _:next(answers)
        try:
            try: actual=w.process_identity(pid,with_hash=hashed)
            except w.OwnershipError:
                assert rejected, (name,hashed,'unexpected ownership error')
            else:
                assert not rejected, (name,hashed,'accepted changed identity')
                assert actual==want, (name,hashed,actual,want)
        finally: w.process_stat=original
        print(name,'hashed='+str(hashed),'passed')
`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := time.Now().UTC()
	raw, runErr := exec.CommandContext(ctx, "/usr/bin/python3", "-c", control, script).CombinedOutput()
	finished := time.Now().UTC()
	if err := os.WriteFile(filepath.Join(dir, "combined-output"), raw, 0644); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(filepath.Join(dir, "attempt.json"), map[string]any{"started": started, "finished": finished, "process_state": fmt.Sprint(runErr), "context_error": fmt.Sprint(ctx.Err())}); err != nil {
		t.Fatal(err)
	}
	if runErr != nil {
		t.Fatalf("process identity exit boundary failed: %v %s (artifacts %s)", runErr, raw, dir)
	}
}

func TestConsumerWatcherOrderedBookendsAndRejections(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	unrelated := exec.Command(exe, "-test.run=^TestRoleWatcherPeer$", "--", "sleep", filepath.Join(t.TempDir(), "ready"), "unused")
	unrelated.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := unrelated.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		unrelated.Process.Kill()
		unrelated.Wait()
	}()
	for _, mode := range []string{"success", "overlap", "wrong-order", "extra", "missing", "duplicate", "nonprivate", "failure", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			plan := consumerWatchPlan(t, mode)
			if mode == "timeout" {
				plan["controller_limit_seconds"] = 1
				for _, role := range plan["roles"].([]map[string]any) {
					role["limit_seconds"] = 1
				}
			}
			code, raw := watchConsumerPlan(t, plan)
			if (mode == "success") != (code == 0) {
				t.Fatalf("unexpected exit %d: %s", code, raw)
			}
			var r struct {
				Error string
				Roles []struct {
					PID, PGID, Slot int
					Start           int64 `json:"start_ticks"`
					Cgroup          string
				}
				Groups        map[string][]json.RawMessage `json:"members_after_cleanup"`
				Authenticated []struct {
					Accepted     bool
					ExpectedRole string `json:"expected_role"`
					Identity     struct {
						PID, PGID, SID int
						Start          int64 `json:"start_ticks"`
						Image, Cgroup  string
						SHA            string `json:"image_sha256"`
						Argv           []string
					}
				} `json:"authenticated_children"`
				Signals []struct {
					PID         int
					Pidfd, Sent bool
				} `json:"signals"`
				OwnershipErrors []json.RawMessage `json:"ownership_errors"`
				ControllerExit  *int              `json:"controller_exit"`
			}
			if err := json.Unmarshal(raw, &r); err != nil {
				t.Fatalf("missing raw receipt: %v %s", err, raw)
			}
			for _, group := range r.Groups {
				if len(group) != 0 {
					t.Fatalf("surviving owned group: %s", raw)
				}
			}
			seen := map[int]bool{}
			for i, role := range r.Roles {
				if role.PID <= 0 || role.PID != role.PGID || role.Start <= 0 || role.Cgroup == "" || role.Slot != i || seen[role.PID] {
					t.Fatalf("invalid observed slot identity: %s", raw)
				}
				seen[role.PID] = true
			}
			if mode == "success" && len(r.Roles) != 3 {
				t.Fatalf("bookends not both observed: %s", raw)
			}
			if mode == "overlap" {
				if code != 125 || r.Error != "next ordered slot started before prior slot finished" || len(r.Roles) != 1 || len(r.Authenticated) != 2 || len(r.OwnershipErrors) != 0 || r.ControllerExit == nil {
					t.Fatalf("overlap branch/terminal cleanup not proved: %s", raw)
				}
				before, refused := r.Authenticated[0], r.Authenticated[1]
				id := refused.Identity
				graph := plan["roles"].([]map[string]any)[1]
				members, tracked := r.Groups[strconv.Itoa(id.PID)]
				if !before.Accepted || before.Identity.PID != r.Roles[0].PID || refused.Accepted || refused.ExpectedRole != "graph" || id.PID <= 0 || id.PGID != id.PID || id.SID <= 0 || id.Start <= 0 || id.Cgroup == "" || id.Image != graph["argv"].([]string)[0] || id.SHA != graph["sha256"].(string) || !reflectStrings(id.Argv, graph["argv"].([]string)) || !tracked || len(members) != 0 {
					t.Fatalf("refused authenticated child identity/empty group missing: %s", raw)
				}
				for _, pid := range []int{before.Identity.PID, id.PID} {
					signaled := false
					for _, event := range r.Signals {
						if event.PID == pid && event.Pidfd && event.Sent {
							signaled = true
						}
					}
					if !signaled {
						t.Fatalf("owned overlap child was not signaled through pidfd: %s", raw)
					}
				}
			}
			if err := unrelated.Process.Signal(syscall.Signal(0)); err != nil {
				t.Fatalf("unrelated process terminated: %v", err)
			}
		})
	}
}

// Pure boundary controls use synthetic records only to test refusal. The
// positive observation proof above comes from actual private-group children.
func TestConsumerWatcherStrictPlanAndSelector(t *testing.T) {
	plan := consumerWatchPlan(t, "success")
	body, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "watch.py")
	if err := os.WriteFile(script, []byte(roleWatcherSource), 0600); err != nil {
		t.Fatal(err)
	}
	control := `import importlib.util,json,sys,copy,hashlib,os
s=importlib.util.spec_from_file_location('watch',sys.argv[1]); w=importlib.util.module_from_spec(s); s.loader.exec_module(w)
p=json.loads(sys.argv[2]); w.validate_sup_plan(p)
def reject(f):
    try: f()
    except (ValueError,RuntimeError,TypeError): return
    raise AssertionError('accepted invalid boundary')
for key,ceiling in [('controller_limit_seconds',2100),('outer_limit_seconds',2400)]:
    for bad in [None,True,False,-1,0,1.5,ceiling+1,'1']:
        q=copy.deepcopy(p); q[key]=bad; reject(lambda:w.validate_sup_plan(q))
    q=copy.deepcopy(p); del q[key]; reject(lambda:w.validate_sup_plan(q))
for bad in [None,True,-1,0,1.5,901,'1',5]:
    q=copy.deepcopy(p); q['roles'][0]['limit_seconds']=bad; reject(lambda:w.validate_sup_plan(q))
q=copy.deepcopy(p); del q['roles'][0]['limit_seconds']; reject(lambda:w.validate_sup_plan(q))
q=copy.deepcopy(p); q['outer_limit_seconds']=63; reject(lambda:w.validate_sup_plan(q))
for key in ['pid','PID','identity','Schema']:
    q=copy.deepcopy(p); q[key]=1; reject(lambda:w.validate_sup_plan(q))
for key in p:
    q=copy.deepcopy(p); q[key]=None; reject(lambda:w.validate_sup_plan(q))
for key,bad in [('name','before-identity'),('pid',123),('argv',None),('sha256','A'*64)]:
    q=copy.deepcopy(p); q['roles'][1][key]=bad; reject(lambda:w.validate_sup_plan(q))
reject(lambda:json.loads('{"schema":1,"schema":2}',object_pairs_hook=w.unique_object))
for mutate in ['unknown','duplicate','case','null']:
    raw=json.dumps(p)
    if mutate=='unknown': raw=raw[:-1]+',"pid":123}'
    if mutate=='duplicate': raw=raw[:-1]+',"schema":"nanopore-sup-consumer-watch/v1"}'
    if mutate=='case': raw=raw.replace('"controller_sha256"','"Controller_sha256"')
    if mutate=='null': raw=raw.replace('"outer_limit_seconds": 64','"outer_limit_seconds": null')
    path=sys.argv[1]+'.'+mutate+'.json'
    with open(path,'w') as f: f.write(raw)
    reject(lambda:w.read_plan(path,hashlib.sha256(raw.encode()).hexdigest()))
path=sys.argv[1]+'.bounded.json'; raw=json.dumps(p).encode()
with open(path,'wb') as f: f.write(raw)
assert w.read_plan(path,hashlib.sha256(raw).hexdigest())==p
reject(lambda:w.read_plan(path,'0'*64))
reject(lambda:w.read_plan(path))
with open(path,'wb') as f: f.write(b' '*(1024*1024+1))
reject(lambda:w.read_plan(path,hashlib.sha256(b' '*(1024*1024+1)).hexdigest()))
fifo=sys.argv[1]+'.fifo'; os.mkfifo(fifo)
reject(lambda:w.read_plan(fifo,'0'*64))
r=p['roles'][0]; c={'pid':10,'sid':10}; a={'pid':11,'pgid':11,'ppid':10,'sid':10,'cgroup':'own','argv':r['argv'],'image':r['argv'][0],'image_sha256':r['sha256']}
assert w.select_role(p['roles'],True,0,a,c,'own') is r
assert w.select_role(p['roles'],True,2,a,c,'own') is p['roles'][2]
reject(lambda:w.select_role(p['roles'],False,0,a,c,'own'))
for key,bad in [('pgid',10),('ppid',9),('sid',9),('cgroup','other'),('image_sha256','0'*64),('argv',['wrong'])]:
    b=copy.deepcopy(a); b[key]=bad; reject(lambda:w.select_role(p['roles'],True,0,b,c,'own'))
reject(lambda:w.select_role(p['roles'],True,1,a,c,'own'))
reject(lambda:w.select_role(p['roles'],True,3,a,c,'own'))
`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw, err := exec.CommandContext(ctx, "/usr/bin/python3", "-c", control, script, string(body)).CombinedOutput()
	if err != nil {
		t.Fatalf("strict boundary controls failed: %v %s", err, raw)
	}
	for _, field := range []string{"controller_sha256", "role_sha256", "legacy_role_limit", "legacy_controller_limit"} {
		t.Run(field, func(t *testing.T) {
			q := consumerWatchPlan(t, "success")
			switch field {
			case "controller_sha256":
				q[field] = strings.Repeat("0", 64)
			case "role_sha256":
				q["roles"].([]map[string]any)[0]["sha256"] = strings.Repeat("0", 64)
			default:
				delete(q, "schema")
				delete(q, "controller_sha256")
				delete(q, "outer_limit_seconds")
				if field == "legacy_role_limit" {
					q["roles"].([]map[string]any)[0]["limit_seconds"] = 181
				} else {
					q["controller_limit_seconds"] = 561
				}
			}
			if code, _ := watchConsumerPlan(t, q); code == 0 {
				t.Fatal("accepted changed SHA or widened legacy limit " + strconv.Itoa(code))
			}
		})
	}
}
