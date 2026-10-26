package supgpu

import (
	"bufio"
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

func TestSelectedRuntimePolicyRejectsKnownWrongTier(t *testing.T) {
	dir := t.TempDir()
	qualified := filepath.Join(dir, "qualified")
	if err := os.Mkdir(qualified, 0700); err != nil {
		t.Fatal(err)
	}
	q := runtimeTestFile(t, qualified, "libcudnn.so.9.13.1", []byte("qualified pinned image"))
	wrong := runtimeTestFile(t, dir, "libcudnn.so.9.27.0", []byte("known pinned wrong tier"))
	onlySystem := runtimeTestFile(t, dir, "libcudnn_ext.so.9.27.0", []byte("system-only wrong tier"))
	root := runtimeTestFile(t, dir, "target", []byte("target"))
	audit := runtimeTestFile(t, dir, "audit", []byte("audit"))
	p, err := selectRuntimePolicy([]RuntimeCode{{Object: q, SONAME: "libcudnn.so.9", Needed: []string{}}, {Object: wrong, SONAME: "libcudnn.so.9", Needed: []string{}}, {Object: onlySystem, SONAME: "libcudnn_ext.so.9", Needed: []string{}}}, map[string]string{q.Path: q.SHA256}, qualified, root, root, audit)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Allowed) != 1 || p.Allowed[0].Object != q || len(p.Alternates) != 2 {
		t.Fatalf("tier selection differs: %+v", p)
	}
	if err = requireSelectedCode(q, p, "reference"); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []Pin{wrong, onlySystem} {
		if err = requireSelectedCode(bad, p, "reference"); err == nil || !strings.Contains(err.Error(), "known wrong runtime tier") {
			t.Fatalf("wrong known tier accepted: %v", err)
		}
	}
	p.Allowed[0].Startup = true
	if err = requireSelectedMappings([]Pin{root, audit}, p, "reference", true); err == nil {
		t.Fatal("missing selected dependency accepted before model")
	}
	if err = requireSelectedMappings([]Pin{root, audit, q}, p, "reference", true); err != nil {
		t.Fatal(err)
	}
}

// Static Go peer for the private pipes. This simulates callback/phase order;
// it is not a native la_objopen or real DSO-constructor test.
func TestGuardProtocolPeer(t *testing.T) {
	marker := ""
	late := ""
	for i, arg := range os.Args {
		if arg == "--" && i+2 < len(os.Args) {
			marker = os.Args[i+1]
			late = os.Args[i+2]
		}
	}
	if marker == "" {
		return
	}
	auditOut := os.NewFile(3, "audit-request")
	auditIn := bufio.NewReader(os.NewFile(4, "audit-approval"))
	phaseOut := os.NewFile(5, "phase-request")
	phaseIn := bufio.NewReader(os.NewFile(6, "phase-approval"))
	send := func(kind string, seq, cookie uint64, path string, phase bool) bool {
		q := GateRequest{"nanopore-supgpu-loader-gate/v1", kind, seq, cookie, path, 0, os.Getpid()}
		out := auditOut
		in := auditIn
		if phase {
			out = phaseOut
			in = phaseIn
		}
		raw, _ := json.Marshal(q)
		if _, err := out.Write(append(raw, '\n')); err != nil {
			return false
		}
		ack, err := in.ReadString('\n')
		return err == nil && ack == fmt.Sprintf("allow %s %d\n", kind, seq)
	}
	if !send("hello", 1, 0, "", false) || !send("open", 2, 11, "", false) || !send("phase", 1, 0, "pre-model", true) || !send("open", 3, 12, late, false) {
		os.Exit(24)
	}
	if err := os.WriteFile(marker, []byte("model/initializer boundary crossed after approval"), 0600); err != nil {
		os.Exit(25)
	}
	if !send("close", 4, 12, "", false) || !send("close", 5, 11, "", false) {
		os.Exit(26)
	}
	os.Exit(0)
}

func TestLoaderPipesRejectUnknownAndWrongKnownTierBeforeModel(t *testing.T) {
	for _, role := range []string{"reference", "observer"} {
		for _, kind := range []string{"allowed", "unknown", "wrong-known-tier"} {
			t.Run(role+"/"+kind, func(t *testing.T) {
				dir := t.TempDir()
				exe, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				root, err := pin(exe)
				if err != nil {
					t.Fatal(err)
				}
				audit := runtimeTestFile(t, dir, "audit-placeholder", []byte("inert metadata, never loaded"))
				allowed := runtimeTestFile(t, dir, "qualified.so", []byte("selected inert image"))
				wrong := runtimeTestFile(t, dir, "wrong-known.so", []byte("known alternate inert image"))
				unknown := runtimeTestFile(t, dir, "unknown.so", []byte("unknown inert image"))
				p := RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Reference: root, Observer: root, Audit: audit, Allowed: []RuntimeCode{{Object: allowed}}, Alternates: []RuntimeCode{{Object: wrong}}, Directories: []string{dir}}
				late := allowed
				if kind == "unknown" {
					late = unknown
				}
				if kind == "wrong-known-tier" {
					late = wrong
				}
				marker := filepath.Join(dir, "model-started")
				calls := 0
				reader := func(pid int, p RuntimePolicy) (mapObservation, error) {
					calls++
					names := []string{root.Path, audit.Path}
					if calls >= 4 {
						names = append(names, late.Path)
					}
					var raw strings.Builder
					for i, name := range names {
						fmt.Fprintf(&raw, "%x-%x r-xp 00000000 00:00 1 %s\n", i*4096, (i+1)*4096, name)
					}
					return parseCodeMaps([]byte(raw.String()), p)
				}
				r, err := executeAttested(dir, dir, role, "attest-only", []string{exe, "-test.run=^TestGuardProtocolPeer$", "--", marker, late.Path}, RuntimePlan{Policy: p}, Pin{}, reader)
				_, modelErr := os.Stat(marker)
				if kind == "allowed" {
					if err != nil || modelErr != nil || r.Exit != 0 {
						t.Fatalf("positive protocol failed: %v marker=%v exit=%d", err, modelErr, r.Exit)
					}
				} else {
					if err == nil || !os.IsNotExist(modelErr) {
						t.Fatalf("negative crossed model/initializer: %v %v", err, modelErr)
					}
					want := "unknown mapped code"
					if kind == "wrong-known-tier" {
						want = "known wrong runtime tier"
					}
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("wrong rejection reason: %v", err)
					}
				}
				var g GuardReceipt
				gp, err := pin(filepath.Join(dir, role+"-guard.json"))
				if err != nil {
					t.Fatal(err)
				}
				if err = readPinnedJSON(gp.Path, gp.SHA256, &g); err != nil {
					t.Fatal(err)
				}
				if len(g.Decisions) < 4 || g.Decisions[3].Maps.Path == "" {
					t.Fatal("startup/late-load failure map evidence missing")
				}
				if kind == "allowed" {
					if _, err = verifyGuard(g, RuntimePlan{Policy: p}, Pin{}, role, "attest-only"); err != nil {
						t.Fatal(err)
					}
					g.Group.Empty = false
					if _, err = verifyGuard(g, RuntimePlan{Policy: p}, Pin{}, role, "attest-only"); err == nil || !strings.Contains(err.Error(), "process group") {
						t.Fatalf("missing group-empty evidence accepted: %v", err)
					}
				} else {
					if _, err = verifyGuard(g, RuntimePlan{Policy: p}, Pin{}, role, "attest-only"); err == nil {
						t.Fatal("failed loader evidence accepted by importer")
					}
				}
			})
		}
	}
}

func TestAnonymousExecutableMappingIsNotSilentlyExcluded(t *testing.T) {
	if _, err := parseCodeMaps([]byte("1000-2000 rwxp 00000000 00:00 0\n"), RuntimePolicy{}); err == nil {
		t.Fatal("unknown executable mapping accepted")
	}
}

// This static Go peer creates a real descendant in the target's process group.
// It deliberately outlives the leader, with either inherited or closed gate FDs.
func TestGuardDescendantPeer(t *testing.T) {
	var args []string
	for i, arg := range os.Args {
		if arg == "--" {
			args = os.Args[i+1:]
			break
		}
	}
	if len(args) != 4 {
		return
	}
	mode, ready, pipes, outcome := args[0], args[1], args[2], args[3]
	if mode == "descendant" || mode == "unrelated" {
		if pipes == "closed" {
			for fd := 3; fd <= 6; fd++ {
				syscall.Close(fd)
			}
		}
		if err := os.WriteFile(ready, []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			os.Exit(31)
		}
		for {
			time.Sleep(time.Hour)
		}
	}
	if mode != "leader" {
		os.Exit(32)
	}
	auditOut := os.NewFile(3, "audit-request")
	auditIn := bufio.NewReader(os.NewFile(4, "audit-approval"))
	phaseOut := os.NewFile(5, "phase-request")
	phaseIn := bufio.NewReader(os.NewFile(6, "phase-approval"))
	send := func(kind string, seq, cookie uint64, path string, phase bool) bool {
		q := GateRequest{"nanopore-supgpu-loader-gate/v1", kind, seq, cookie, path, 0, os.Getpid()}
		out, in := auditOut, auditIn
		if phase {
			out, in = phaseOut, phaseIn
		}
		raw, _ := json.Marshal(q)
		if _, err := out.Write(append(raw, '\n')); err != nil {
			return false
		}
		ack, err := in.ReadString('\n')
		return err == nil && ack == fmt.Sprintf("allow %s %d\n", kind, seq)
	}
	if !send("hello", 1, 0, "", false) || !send("open", 2, 11, "", false) || !send("phase", 1, 0, "pre-model", true) {
		os.Exit(33)
	}
	exe, _ := os.Executable()
	child := exec.Command(exe, "-test.run=^TestGuardDescendantPeer$", "--", "descendant", ready, pipes, outcome)
	child.ExtraFiles = []*os.File{auditOut, os.NewFile(4, "audit-approval"), phaseOut, os.NewFile(6, "phase-approval")}
	if err := child.Start(); err != nil {
		os.Exit(34)
	}
	for deadline := time.Now().Add(3 * time.Second); ; {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			os.Exit(35)
		}
		time.Sleep(time.Millisecond)
	}
	if outcome == "reject" {
		if send("open", 3, 12, ready, false) {
			os.Exit(36)
		}
		os.Exit(24)
	}
	os.Exit(0)
}

func TestLoaderCleansDescendantsAfterLeaderExit(t *testing.T) {
	for _, role := range []string{"reference", "observer"} {
		for _, mode := range []struct{ pipes, outcome string }{{"held", "reject"}, {"closed", "reject"}, {"closed", "exit"}} {
			t.Run(role+"/"+mode.pipes+"/"+mode.outcome, func(t *testing.T) {
				dir := t.TempDir()
				exe, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				root, err := pin(exe)
				if err != nil {
					t.Fatal(err)
				}
				audit := runtimeTestFile(t, dir, "audit-placeholder", []byte("inert metadata, not loaded"))
				p := RuntimePolicy{Reference: root, Observer: root, Audit: audit, Allowed: []RuntimeCode{}, Alternates: []RuntimeCode{}, Directories: []string{dir}}
				reader := func(pid int, p RuntimePolicy) (mapObservation, error) {
					return parseCodeMaps([]byte(fmt.Sprintf("1000-2000 r-xp 00000000 00:00 1 %s\n2000-3000 r-xp 00000000 00:00 2 %s\n", root.Path, audit.Path)), p)
				}
				unrelatedReady := filepath.Join(dir, "unrelated.pid")
				unrelated := exec.Command(exe, "-test.run=^TestGuardDescendantPeer$", "--", "unrelated", unrelatedReady, "closed", "exit")
				unrelated.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
				if err = unrelated.Start(); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { unrelated.Process.Kill(); unrelated.Wait() })
				ready := filepath.Join(dir, "descendant.pid")
				started := time.Now()
				r, runErr := executeAttested(dir, dir, role, "attest-only", []string{exe, "-test.run=^TestGuardDescendantPeer$", "--", "leader", ready, mode.pipes, mode.outcome}, RuntimePlan{Policy: p}, Pin{}, reader)
				raw, err := os.ReadFile(ready)
				if err != nil {
					t.Fatalf("descendant never started: %v; guard=%v", err, runErr)
				}
				pid, err := strconv.Atoi(string(raw))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { syscall.Kill(pid, syscall.SIGKILL) })
				if runErr == nil {
					t.Fatal("leader exit with a surviving descendant was accepted")
				}
				if time.Since(started) > 12*time.Second {
					t.Fatalf("descendant cleanup exceeded bound: %s", time.Since(started))
				}
				if err = syscall.Kill(pid, 0); err != syscall.ESRCH {
					t.Fatalf("descendant %d survived guard return: %v; guard=%v", pid, err, runErr)
				}
				if err = syscall.Kill(unrelated.Process.Pid, 0); err != nil {
					t.Fatalf("unrelated process was killed: %v", err)
				}
				var g GuardReceipt
				gp, err := pin(filepath.Join(dir, role+"-guard.json"))
				if err != nil {
					t.Fatal(err)
				}
				if err = readPinnedJSON(gp.Path, gp.SHA256, &g); err != nil || g.Error == "" {
					t.Fatalf("failure receipt missing: %v %+v", err, g)
				}
				if g.Group.PGID != g.PID || !g.Group.Empty || len(g.Group.Kills) == 0 || len(g.Group.Observations) < 2 || len(g.Group.Observations[len(g.Group.Observations)-1].Members) != 0 {
					t.Fatalf("group cleanup evidence missing: %+v", g.Group)
				}
				observedDescendant := false
				for _, o := range g.Group.Observations {
					for _, member := range o.Members {
						observedDescendant = observedDescendant || member.PID == pid
					}
				}
				if !observedDescendant || syscall.Kill(-g.PID, 0) != syscall.ESRCH {
					t.Fatalf("actual descendant/group emptiness not established: %+v", g.Group)
				}
				if _, err = verifyGuard(g, RuntimePlan{Policy: p}, Pin{}, role, "attest-only"); err == nil {
					t.Fatal("descendant failure accepted by import")
				}
				retainGuardTestEvidence(t, dir, role+"-"+mode.pipes+"-"+mode.outcome)
				t.Logf("leader=%d descendant=%d gone=true group_empty=true observations=%d sigkills=%d unrelated=%d alive=true exit=%d elapsed=%s error=%v", g.PID, pid, len(g.Group.Observations), len(g.Group.Kills), unrelated.Process.Pid, r.Exit, time.Since(started), runErr)
			})
		}
	}
}

func retainGuardTestEvidence(t *testing.T, dir, name string) {
	t.Helper()
	root := os.Getenv("SUPGPU_GUARD_TEST_EVIDENCE")
	if root == "" {
		return
	}
	out := filepath.Join(root, name)
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(out, 0700); err != nil {
		t.Fatal(err)
	}
	copies := []Snapshot{}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil || rel == "." {
			return err
		}
		copyPath := filepath.Join(out, rel)
		if info.IsDir() {
			return os.Mkdir(copyPath, 0700)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err = os.WriteFile(copyPath, raw, 0600); err != nil {
			return err
		}
		p, err := pin(copyPath)
		if err == nil {
			copies = append(copies, Snapshot{Original: path, Copy: p})
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if err = writeJSON(filepath.Join(out, "retained.json"), copies); err != nil {
		t.Fatal(err)
	}
}
