package supgpu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var decoderPhases = []string{"pre-model", "pre-forward", "post-back-guide", "post-beam-search", "post-posterior", "post-staged"}

// A CGO0 Go protocol peer plus injected inert maps, not an auditor/DSO load.
func TestGenericGuardProtocolPeer(t *testing.T) {
	var args []string
	for i, arg := range os.Args {
		if arg == "--" {
			args = os.Args[i+1:]
			break
		}
	}
	if len(args) != 2 {
		return
	}
	request, approval := os.NewFile(3, "audit-request"), bufio.NewReader(os.NewFile(4, "audit-approval"))
	for i, kind := range []string{"hello", "open"} {
		q := GateRequest{Schema: "nanopore-supgpu-loader-gate/v1", Kind: kind, Sequence: uint64(i + 1), PID: os.Getpid()}
		if kind == "open" {
			q.Cookie = 11
		}
		raw, _ := json.Marshal(q)
		if _, err := request.Write(append(raw, '\n')); err != nil {
			os.Exit(41)
		}
		if line, err := approval.ReadString('\n'); err != nil || line != fmt.Sprintf("allow %s %d\n", kind, i+1) {
			os.Exit(42)
		}
	}
	client, err := OpenPhaseClient()
	if err != nil {
		os.Exit(43)
	}
	for _, phase := range strings.Split(args[0], ",") {
		if err := client.Checkpoint(phase); err != nil {
			os.Exit(44)
		}
	}
	client.Close()
	if err := os.WriteFile(args[1], []byte("fake child completed its submitted sequence"), 0600); err != nil {
		os.Exit(45)
	}
	os.Exit(0)
}

func genericGuardFixture(t *testing.T, phases []string) (Pin, Pin, GuardCommand, mapReader) {
	t.Helper()
	dir := t.TempDir()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	root, err := pin(exe)
	if err != nil {
		t.Fatal(err)
	}
	audit := runtimeTestFile(t, dir, "inert-audit", []byte("not loaded: private fake maps only"))
	provider := runtimeTestFile(t, dir, "provider.so", []byte("selected inert provider"))
	source := runtimeTestFile(t, dir, "caller-source.go", []byte("pinned caller source"))
	contract := PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "reference", Action: "decoder", Names: append([]string{}, phases...)}
	cp := filepath.Join(dir, "phases.json")
	if err := writeJSON(cp, contract); err != nil {
		t.Fatal(err)
	}
	phasePin, _ := pin(cp)
	policy := RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: root, Observer: root, Audit: audit, Allowed: []RuntimeCode{{Object: provider, SONAME: "provider.so", Needed: []string{}, Tier: "frozen-test-selected", Startup: true}}, Alternates: []RuntimeCode{}, Directories: []string{dir}}
	p := GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Policy: policy, Contract: phasePin, Inputs: []Pin{source}}
	path := filepath.Join(dir, "guard-plan.json")
	if err := writeJSON(path, p); err != nil {
		t.Fatal(err)
	}
	planPin, _ := pin(path)
	command := GuardCommand{Directory: dir, Output: filepath.Join(dir, "output"), Args: []string{root.Path, "-test.run=^TestGenericGuardProtocolPeer$", "--", strings.Join(phases, ","), filepath.Join(dir, "finished")}}
	reader := func(pid int, policy RuntimePolicy) (mapObservation, error) {
		raw := fmt.Sprintf("1000-2000 r-xp 00000000 00:00 1 %s\n2000-3000 r-xp 00000000 00:00 2 %s\n3000-4000 r-xp 00000000 00:00 3 %s\n", root.Path, audit.Path, provider.Path)
		return parseCodeMaps([]byte(raw), policy)
	}
	return planPin, phasePin, command, reader
}

func TestGenericGuardExactPhasesAndGoClient(t *testing.T) {
	for _, names := range [][]string{decoderPhases, {"pre-model", "pre-forward", "post-forward"}, {"pre-model"}} {
		t.Run(fmt.Sprint(len(names)), func(t *testing.T) {
			plan, phases, command, reader := genericGuardFixture(t, names)
			run, err := executeGuarded(plan, phases, command, reader)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = ReplayGuarded(plan, phases, command, run); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGenericGuardRejectsWrongAndIncompleteLivePhases(t *testing.T) {
	for _, kind := range []string{"wrong", "reordered", "truncated", "extra"} {
		t.Run(kind, func(t *testing.T) {
			plan, phases, command, reader := genericGuardFixture(t, decoderPhases)
			actual := append([]string{}, decoderPhases...)
			switch kind {
			case "wrong":
				actual[2] = "post-other-stage"
			case "reordered":
				actual[2], actual[3] = actual[3], actual[2]
			case "truncated":
				actual = actual[:5]
			case "extra":
				actual = append(actual, "unapproved-extra")
			}
			command.Args[3] = strings.Join(actual, ",")
			run, err := executeGuarded(plan, phases, command, reader)
			if err == nil {
				t.Fatal("inexact actual phase sequence qualified")
			}
			if _, err = ReplayGuarded(plan, phases, command, run); err == nil {
				t.Fatal("failed phase transcript qualified replay")
			}
		})
	}
}

func TestGenericGuardReplayCannotSelfAuthorize(t *testing.T) {
	plan, phases, command, reader := genericGuardFixture(t, decoderPhases)
	run, err := executeGuarded(plan, phases, command, reader)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(run.Transcript.Path)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"truncated", "reordered", "missing-map", "wrong-object-hash", "group-not-empty", "forged-contract"} {
		t.Run(kind, func(t *testing.T) {
			defer os.WriteFile(run.Transcript.Path, raw, 0600)
			var g GuardReceipt
			if err := decodeClosed(raw, &g); err != nil {
				t.Fatal(err)
			}
			mutant := run
			switch kind {
			case "truncated":
				g.Decisions = g.Decisions[:len(g.Decisions)-1]
			case "reordered":
				g.Decisions[4], g.Decisions[5] = g.Decisions[5], g.Decisions[4]
			case "missing-map":
				g.Decisions[0].Maps = Pin{}
			case "wrong-object-hash":
				g.Decisions[1].Object.SHA256 = strings.Repeat("0", 64)
			case "group-not-empty":
				g.Group.Empty = false
			case "forged-contract":
				short := PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "reference", Action: "decoder", Names: decoderPhases[:5]}
				path := filepath.Join(command.Directory, "forged-short-contract.json")
				if err := writeJSON(path, short); err != nil {
					t.Fatal(err)
				}
				mutant.Contract, _ = pin(path)
			}
			changed, _ := Marshal(g)
			if err := os.WriteFile(run.Transcript.Path, append(changed, '\n'), 0600); err != nil {
				t.Fatal(err)
			}
			mutant.Transcript, _ = pin(run.Transcript.Path)
			if _, err := ReplayGuarded(plan, phases, command, mutant); err == nil {
				t.Fatal("rehashed/forged receipt selected its own expectations")
			}
		})
	}
}

func TestGenericGuardSourceAndContractFailBeforeChild(t *testing.T) {
	for _, kind := range []string{"changed-source", "omitted-input", "wrong-contract", "wrong-command"} {
		t.Run(kind, func(t *testing.T) {
			plan, phases, command, _ := genericGuardFixture(t, decoderPhases)
			var p GuardPlan
			if err := readPinnedJSON(plan.Path, plan.SHA256, &p); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "changed-source":
				if err := os.WriteFile(p.Inputs[0].Path, []byte("changed source"), 0600); err != nil {
					t.Fatal(err)
				}
			case "omitted-input":
				p.Inputs = []Pin{}
			case "wrong-contract":
				p.Contract.SHA256 = strings.Repeat("0", 64)
			case "wrong-command":
				command.Args[0] = "/must/not/execute"
			}
			changed, _ := Marshal(p)
			if err := os.WriteFile(plan.Path, append(changed, '\n'), 0600); err != nil {
				t.Fatal(err)
			}
			plan, _ = pin(plan.Path)
			if _, err := ExecuteGuarded(plan, phases, command); err == nil {
				t.Fatal("invalid custody/contract reached child")
			}
			if _, err := os.Stat(command.Output); !os.IsNotExist(err) {
				t.Fatal("invalid custody/contract created output before rejection")
			}
		})
	}
}

func TestPhaseContractRequiresFirstUniqueBoundedNames(t *testing.T) {
	good := PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "reference", Action: "decoder", Names: append([]string{}, decoderPhases...)}
	if err := validatePhaseContract(good); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"first", "duplicate", "empty", "long", "too-many", "role"} {
		t.Run(kind, func(t *testing.T) {
			bad := good
			bad.Names = append([]string{}, good.Names...)
			switch kind {
			case "first":
				bad.Names[0] = "pre-forward"
			case "duplicate":
				bad.Names[1] = bad.Names[0]
			case "empty":
				bad.Names = []string{}
			case "long":
				bad.Names[1] = strings.Repeat("x", 65)
			case "too-many":
				for i := len(bad.Names); i < 33; i++ {
					bad.Names = append(bad.Names, fmt.Sprintf("phase-%d", i))
				}
			case "role":
				bad.Role = "unchecked-role"
			}
			if reflect.DeepEqual(good, bad) || validatePhaseContract(bad) == nil {
				t.Fatal("invalid phase contract accepted")
			}
		})
	}
}
