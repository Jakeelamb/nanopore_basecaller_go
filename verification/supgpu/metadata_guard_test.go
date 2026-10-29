package supgpu

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMetadataContractIsClosed(t *testing.T) {
	good := PhaseContract{Schema: "nanopore-audited-metadata-contract/v1", Role: "observer", Action: "device-identity", Names: []string{}}
	if err := validatePhaseContract(good); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*PhaseContract){
		func(c *PhaseContract) { c.Names = nil },
		func(c *PhaseContract) { c.Names = []string{"pre-model"} },
		func(c *PhaseContract) { c.Role = "reference" },
		func(c *PhaseContract) { c.Action = "sup-tiny" },
		func(c *PhaseContract) { c.Schema = "nanopore-audited-phase-contract/v1" },
	} {
		bad := good
		change(&bad)
		if validatePhaseContract(bad) == nil {
			t.Fatal("metadata variant escaped closed shape")
		}
	}
	for _, raw := range []string{
		`{"schema":"nanopore-audited-metadata-contract/v1","role":"observer","action":"device-identity","names":null}`,
		`{"schema":"nanopore-audited-metadata-contract/v1","role":"observer","action":"device-identity"}`,
	} {
		var c PhaseContract
		if decodeClosed([]byte(raw), &c) == nil {
			t.Fatal("missing/null metadata names accepted")
		}
	}
}

// Inert Go protocol peer: tests core live/replay symmetry, not native loading.
func TestMetadataProtocolPeer(t *testing.T) {
	var args []string
	for i, s := range os.Args {
		if s == "--metadata-peer" {
			args = os.Args[i+1:]
			break
		}
	}
	if len(args) != 2 {
		return
	}
	rd := bufio.NewReader(os.NewFile(4, "approval"))
	wr := os.NewFile(3, "request")
	seq := uint64(0)
	send := func(kind, path string, cookie uint64) {
		seq++
		q := GateRequest{Schema: "nanopore-supgpu-loader-gate/v1", Kind: kind, Sequence: seq, Path: path, Cookie: cookie, PID: os.Getpid()}
		b, _ := json.Marshal(q)
		if _, e := wr.Write(append(b, '\n')); e != nil {
			os.Exit(51)
		}
		line, e := rd.ReadString('\n')
		if e != nil || line != fmt.Sprintf("allow %s %d\n", kind, seq) {
			os.Exit(52)
		}
	}
	send("hello", "", 0)
	if args[0] == "wait" {
		time.Sleep(time.Hour)
		os.Exit(53)
	}
	if args[0] != "hello-only" {
		if args[0] != "omit-root" {
			send("open", "", 1)
		}
		if args[0] != "omit-startup" {
			send("open", args[1], 2)
			send("close", "", 2)
		}
	}
	if args[0] == "phase" {
		c, e := OpenPhaseClient()
		if e != nil {
			os.Exit(54)
		}
		if c.Checkpoint("pre-model") != nil {
			os.Exit(55)
		}
		c.Close()
	}
	os.Exit(0)
}

func metadataFixture(t *testing.T, kind string) (Pin, Pin, GuardCommand, mapReader) {
	t.Helper()
	pp, cp, cmd, reader := genericGuardFixture(t, []string{"pre-model"})
	var p GuardPlan
	if e := readPinnedJSON(pp.Path, pp.SHA256, &p); e != nil {
		t.Fatal(e)
	}
	c := PhaseContract{Schema: "nanopore-audited-metadata-contract/v1", Role: "observer", Action: "device-identity", Names: []string{}}
	replaceMetadataJSON(t, cp.Path, c)
	cp, _ = pin(cp.Path)
	p.Contract = cp
	replaceMetadataJSON(t, pp.Path, p)
	pp, _ = pin(pp.Path)
	cmd.Args = []string{p.Policy.Observer.Path, "-test.run=^TestMetadataProtocolPeer$", "--", "--metadata-peer", kind, p.Policy.Allowed[0].Object.Path}
	return pp, cp, cmd, reader
}

func replaceMetadataJSON(t *testing.T, path string, value any) {
	t.Helper()
	b, e := Marshal(value)
	if e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(path, append(b, '\n'), 0600); e != nil {
		t.Fatal(e)
	}
}

func TestMetadataLiveAndReplayRequireStartupHistory(t *testing.T) {
	for _, kind := range []string{"complete", "hello-only", "omit-root", "omit-startup", "phase"} {
		t.Run(kind, func(t *testing.T) {
			p, c, cmd, maps := metadataFixture(t, kind)
			r, e := executeGuarded(context.Background(), p, c, cmd, maps)
			_, re := ReplayGuarded(p, c, cmd, r)
			if kind == "complete" {
				if e != nil || re != nil {
					t.Fatalf("legitimate closed-provider history failed: %v / %v", e, re)
				}
			} else if e == nil || re == nil {
				t.Fatal("incomplete/unexpected metadata gates accepted")
			}
		})
	}
}

func TestMetadataRehashedTruncationAndInvocationReject(t *testing.T) {
	p, c, cmd, maps := metadataFixture(t, "complete")
	r, e := executeGuarded(context.Background(), p, c, cmd, maps)
	if e != nil {
		t.Fatal(e)
	}
	var original GuardReceipt
	if e = readPinnedJSON(r.Transcript.Path, r.Transcript.SHA256, &original); e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"hello-only", "omit-root", "omit-startup", "object", "maps", "environment", "argv", "receipt"} {
		t.Run(kind, func(t *testing.T) {
			g := original
			g.Decisions = append([]GateDecision{}, original.Decisions...)
			mutant := r
			switch kind {
			case "hello-only":
				g.Decisions = g.Decisions[:1]
			case "omit-root":
				g.Decisions = append(g.Decisions[:1], g.Decisions[2:]...)
			case "omit-startup":
				g.Decisions = g.Decisions[:2]
			case "object":
				g.Decisions[2].Object.SHA256 = strings.Repeat("0", 64)
			case "maps":
				path := filepath.Join(cmd.Directory, "changed.maps")
				if e := os.WriteFile(path, []byte("1000-2000 r-xp 0 00:00 1 /unselected/provider.so\n"), 0600); e != nil {
					t.Fatal(e)
				}
				g.Decisions[2].Maps, _ = pin(path)
			case "environment":
				mutant.Command.Environment = append([]string{}, r.Command.Environment...)
				mutant.Command.Environment[2] = "LD_AUDIT="
			case "argv":
				mutant.Command.Args = []string{"/other"}
			case "receipt":
				mutant.Plan.SHA256 = strings.Repeat("0", 64)
			}
			replaceMetadataJSON(t, r.Transcript.Path, g)
			mutant.Transcript, _ = pin(r.Transcript.Path)
			if _, e := ReplayGuarded(p, c, cmd, mutant); e == nil {
				t.Fatal("rehashed mutation accepted")
			}
		})
	}
}

func TestGuardContextRejectsBeforeOutputAndRetainsCancellation(t *testing.T) {
	p, c, cmd, maps := metadataFixture(t, "wait")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, ctx := range []context.Context{nil, ctx} {
		if _, e := executeGuarded(ctx, p, c, cmd, maps); e == nil {
			t.Fatal("invalid context accepted")
		}
		if _, e := os.Stat(cmd.Output); !os.IsNotExist(e) {
			t.Fatal("invalid context created output")
		}
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	cancelOnMap := func(pid int, policy RuntimePolicy) (mapObservation, error) {
		o, e := maps(pid, policy)
		cancel() // A real child is already started and blocked on its audit pipe.
		return o, e
	}
	r, e := executeGuarded(ctx, p, c, cmd, cancelOnMap)
	if e == nil || r.Command.Exit == 0 {
		t.Fatal("cancelled peer qualified")
	}
	var g GuardReceipt
	if e := readPinnedJSON(r.Transcript.Path, r.Transcript.SHA256, &g); e != nil {
		t.Fatal(e)
	}
	if g.Error == "" || len(g.Group.Kills) == 0 || !g.Group.Empty {
		t.Fatal("cancellation failure/cleanup not retained")
	}
}
