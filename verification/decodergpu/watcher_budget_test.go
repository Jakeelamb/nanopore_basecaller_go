package decodergpu

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestNumericWatchPlanCrossesWatcher019(t *testing.T) {
	if os.Getenv("D01_WATCHER_HOST_TEST") != "1" {
		t.Skip("explicit pinned host Python watcher admission opt-in")
	}
	watcher, err := filepath.Abs("../supgpu/tools/watch_roles.py")
	if err != nil {
		t.Fatal(err)
	}
	if err = (File{Path: watcher, SHA256: roleWatcherSHA}).verify(); err != nil {
		t.Fatal(err)
	}
	python, err := filepath.EvalSymlinks("/usr/bin/python3")
	if err != nil {
		t.Fatal(err)
	}
	image, err := artifactPin(python)
	if err != nil {
		t.Fatal(err)
	}
	r := RuntimePlan{WorkingDirectory: "/source", CaptureDirectory: "/capture", Consumer: File{Path: python}, ConsumerPreparation: File{Path: "/consumer/preparation.json"}}
	b := CompilerPhase{Output: image}
	p := ClosedBuildPlan{Schema: "nanopore-decoder-closed-build/v3", Preparation: File{Path: "/producer/preparation.json"}}
	positive := expectedWatchPlan(r, File{Path: "/runtime.json", SHA256: "runtime"}, b, p)
	harness := `import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("pinned_watch",sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=json.load(open(sys.argv[2]));calls=[]
class ReachedSpawn(Exception):pass
def stop(*args,**kwargs):calls.append(args);raise ReachedSpawn()
module.subprocess.Popen=stop
try:module.run(plan,sys.argv[2]+".unused");raise AssertionError("watcher unexpectedly returned")
except ReachedSpawn:
 assert sys.argv[3]=="spawn" and len(calls)==1
 print("exact-controller-Popen-boundary")
except ValueError as error:
 assert sys.argv[3]!="spawn" and not calls
 assert sys.argv[3] in str(error),(sys.argv[3],str(error))
 print("rejected-before-controller-Popen")
`
	for _, kind := range []string{"numeric540", "legacy240", "controller900", "controller561", "native181"} {
		t.Run(kind, func(t *testing.T) {
			plan := positive
			plan.Roles = append([]WatchRole(nil), positive.Roles...)
			want := "spawn"
			switch kind {
			case "legacy240":
				q := p
				q.Schema = "nanopore-decoder-closed-build/v2"
				plan = expectedWatchPlan(r, File{Path: "/runtime.json", SHA256: "runtime"}, b, q)
			case "controller900":
				plan.Limit = 900
				want = "controller limit must fit outer600s cleanup budget"
			case "controller561":
				plan.Limit = 561
				want = "controller limit must fit outer600s cleanup budget"
			case "native181":
				plan.Roles[0].Limit = 181
				want = "exact role argv and bounded lifetime required"
			}
			if kind == "numeric540" && plan.Limit != 540 {
				t.Errorf("numeric plan must fit unchanged watcher: got %d", plan.Limit)
			}
			if kind == "legacy240" && plan.Limit != 240 {
				t.Fatal("legacy controller budget changed")
			}
			path := filepath.Join(t.TempDir(), "watch-plan.json")
			raw, e := json.Marshal(plan)
			if e != nil {
				t.Fatal(e)
			}
			if e = os.WriteFile(path, raw, 0600); e != nil {
				t.Fatal(e)
			}
			cmd := exec.Command(python, "-I", "-S", "-B", "-c", harness, watcher, path, want)
			cmd.Env = []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "PYTHONDONTWRITEBYTECODE=1"}
			out, e := cmd.CombinedOutput()
			if e != nil {
				t.Fatalf("actual watcher serialized-plan boundary: %v\n%s", e, out)
			}
			marker := "rejected-before-controller-Popen"
			if want == "spawn" {
				marker = "exact-controller-Popen-boundary"
			}
			if !strings.Contains(string(out), marker) {
				t.Fatal("missing exact watcher boundary", string(out))
			}
			t.Logf("serialized plan %s: %s", raw, out)
		})
	}
}

func TestNumericWatcherTimeoutAndCleanupPeers(t *testing.T) {
	if os.Getenv("D01_WATCHER_HOST_TEST") != "1" {
		t.Skip("explicit host watcher sentinel opt-in")
	}
	watcher, err := filepath.Abs("../supgpu/tools/watch_roles.py")
	if err != nil {
		t.Fatal(err)
	}
	if err = (File{Path: watcher, SHA256: roleWatcherSHA}).verify(); err != nil {
		t.Fatal(err)
	}
	python, err := filepath.EvalSymlinks("/usr/bin/python3")
	if err != nil {
		t.Fatal(err)
	}
	image, err := artifactPin(python)
	if err != nil {
		t.Fatal(err)
	}
	r := RuntimePlan{WorkingDirectory: "/source", CaptureDirectory: "/capture", Consumer: image, ConsumerPreparation: File{Path: "/preparation.json"}}
	b := CompilerPhase{Output: image}
	p := ClosedBuildPlan{Schema: "nanopore-decoder-closed-build/v3", Preparation: File{Path: "/producer/preparation.json"}}
	plan := expectedWatchPlan(r, File{Path: "/runtime.json", SHA256: "runtime"}, b, p)
	harness := `import importlib.util,json,sys,copy,pathlib
spec=importlib.util.spec_from_file_location("watch",sys.argv[1]);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
plan=json.load(open(sys.argv[2]));mode=sys.argv[3];out=sys.argv[2]+".receipt";calls=[];sent=[]
cgroup=pathlib.Path('/proc/self/cgroup').read_text();hz=w.os.sysconf('SC_CLK_TCK')
def identity(pid,argv,pgid,parent):
 return dict(pid=pid,state='S',ppid=parent,pgid=pgid,sid=100,start_ticks=hz,image=argv[0],argv=argv,image_sha256=plan['roles'][0]['sha256'],cgroup=cgroup,raw='sentinel')
live={100:identity(100,plan['controller_argv'],100,1),200:identity(200,plan['roles'][0]['argv'],200,100)}
class Peer:
 pid=100
 def poll(self):return None if 100 in live else 1
 def wait(self,timeout):
  if 100 in live:raise w.subprocess.TimeoutExpired('sentinel',timeout)
  return 1
def popen(argv,**kwargs):
 assert argv==plan['controller_argv'] and kwargs==dict(start_new_session=True)
 calls.append(argv);return Peer()
w.subprocess.Popen=popen
w.process_identity=lambda pid,with_hash=False:copy.deepcopy(live.get(pid))
w.group_members=lambda pgid:[copy.deepcopy(v) for v in live.values() if v['pgid']==pgid]
w.thread_children=lambda pid:{100:[200]}
w.os.pidfd_open=lambda pid,flags:pid+1000
w.os.close=lambda fd:None
def signal(fd,sig):
 pid=fd-1000;sent.append((pid,sig))
 if mode!='cleanup-failure':live.pop(pid,None)
w.signal.pidfd_send_signal=signal
w.time.sleep=lambda seconds:None
ticks=iter([0,541]);w.time.monotonic=lambda:next(ticks)
w.time.clock_gettime=lambda clock:182 if mode=='native-timeout' else 2
if mode=='output-failure':pathlib.Path(out).write_text('immutable prior receipt')
try:
 result=w.run(plan,out)
 assert mode!='output-failure'
 receipt=json.load(open(out))
 assert len(calls)==1 and sent and receipt['timeout'] and result==124
 if mode=='cleanup-failure':
  assert receipt['cleanup_error'] and any(receipt['members_after_cleanup'].values())
 else:
  assert not any(receipt['members_after_cleanup'].values()) and not live
  assert receipt['controller_exit']==1 and not receipt['ownership_errors']
 if mode=='native-timeout':assert 'birth-relative' in receipt['error']
 else:assert 'controller exceeded' in receipt['error']
 print(json.dumps(receipt,sort_keys=True))
except FileExistsError:
 assert mode=='output-failure' and not live and len(calls)==1 and sent
 assert pathlib.Path(out).read_text()=='immutable prior receipt'
 print('output failure preserved prior receipt after empty owned-group cleanup')
`
	for _, mode := range []string{"controller-timeout", "native-timeout", "cleanup-failure", "output-failure"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "plan.json")
			raw, e := json.Marshal(plan)
			if e != nil {
				t.Fatal(e)
			}
			if e = os.WriteFile(path, raw, 0600); e != nil {
				t.Fatal(e)
			}
			cmd := exec.Command(python, "-I", "-S", "-B", "-c", harness, watcher, path, mode)
			cmd.Env = []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C"}
			out, e := cmd.CombinedOutput()
			if e != nil {
				t.Fatalf("watcher peer failed: %v\n%s", e, out)
			}
			t.Log(string(out))
		})
	}
}
