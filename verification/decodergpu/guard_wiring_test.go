package decodergpu

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestDecoderSelectedPolicyPreservesQualifiedProviders(t *testing.T) {
	dir := t.TempDir()
	root := testFile(t, dir, "decoder", "fake decoder metadata")
	audit := testFile(t, dir, "audit.so", "fake audit metadata")
	cupti := testFile(t, dir, "libcupti.so", "fake CUPTI metadata")
	libc := testFile(t, dir, "libc.so", "fake CPU metadata")
	wrong := testFile(t, dir, "wrong.so", "wrong CPU metadata")
	rootPin, _ := guardPin(root)
	auditPin, _ := guardPin(audit)
	libcPin, _ := guardPin(libc)
	wrongPin, _ := guardPin(wrong)
	closure := nativeproof.ELFClosure{Nodes: []nativeproof.ELFMetadata{{Pin: nativeproof.Pin{Path: root.Path, SHA256: root.SHA256}, Needed: []string{"libcupti.so.13"}}, {Pin: nativeproof.Pin{Path: cupti.Path, SHA256: cupti.SHA256}, SONAME: "libcupti.so.13", Needed: []string{"libc.so.6"}}, {Pin: nativeproof.Pin{Path: libc.Path, SHA256: libc.SHA256}, SONAME: "libc.so.6"}}}
	qualification := supgpu.RuntimePolicy{Allowed: []supgpu.RuntimeCode{{Object: libcPin, SONAME: "libc.so.6", Tier: "p01-observed"}}, Alternates: []supgpu.RuntimeCode{{Object: wrongPin, SONAME: "libc.so.6", Tier: "wrong"}}}
	policy, err := selectedDecoderPolicy(closure, rootPin, auditPin, qualification, []string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if policy.Reference != rootPin || policy.Observer != rootPin || policy.Audit != auditPin || len(policy.Allowed) != 2 || !policy.Allowed[0].Startup || !policy.Allowed[1].Startup || policy.Allowed[1].Tier != "p01-observed" {
		t.Fatalf("selected startup policy differs: %+v", policy)
	}
	if policy.Allowed[1].Needed == nil || len(policy.Alternates) != 1 || policy.Alternates[0].Needed == nil {
		t.Fatal("dependency-free selected/rejected providers must serialize empty arrays")
	}
	for _, kind := range []string{"qualified-tier", "rejected-tier", "duplicate-soname", "missing-startup", "missing-cupti", "torch"} {
		t.Run(kind, func(t *testing.T) {
			c := closure
			c.Nodes = slices.Clone(closure.Nodes)
			q := qualification
			q.Allowed = slices.Clone(qualification.Allowed)
			switch kind {
			case "qualified-tier":
				q.Allowed[0].Object = wrongPin
			case "rejected-tier":
				q.Alternates = []supgpu.RuntimeCode{{Object: libcPin, SONAME: "libc.so.6", Tier: "wrong"}}
			case "duplicate-soname":
				c.Nodes = append(c.Nodes, c.Nodes[2])
			case "missing-startup":
				c.Nodes[1].Needed = []string{"missing.so"}
			case "missing-cupti":
				c.Nodes = c.Nodes[:1]
			case "torch":
				c.Nodes[2].SONAME = "libtorch_cpu.so"
			}
			if _, e := selectedDecoderPolicy(c, rootPin, auditPin, q, []string{dir}); e == nil {
				t.Fatal("unsupported runtime/provider policy accepted")
			}
		})
	}
}

func TestDecoderSerializedGuardPlanCrossesSharedBoundary(t *testing.T) {
	dir := t.TempDir()
	binary := testFile(t, dir, "decoder", "fake decoder metadata")
	audit := testFile(t, dir, "audit.so", "fake auditor metadata")
	provider := testFile(t, dir, "provider.so", "fake runtime metadata")
	binaryPin, _ := guardPin(binary)
	auditPin, _ := guardPin(audit)
	providerPin, _ := guardPin(provider)
	r := RuntimePlan{WorkingDirectory: dir, CaptureDirectory: filepath.Join(dir, "capture"), Custody: map[string]string{binary.Path: binary.SHA256}, Policy: supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: binaryPin, Observer: binaryPin, Audit: auditPin, Allowed: []supgpu.RuntimeCode{{Object: providerPin, SONAME: "provider.so", Needed: []string{}, Tier: "fake-host"}}, Alternates: []supgpu.RuntimeCode{}, Directories: []string{dir}}}
	phasePath := filepath.Join(dir, "phases.json")
	if err := writeJSON(phasePath, decoderPhases()); err != nil {
		t.Fatal(err)
	}
	r.PhaseContract, _ = artifactPin(phasePath)
	b := CompilerPhase{Output: binary}
	p := ClosedBuildPlan{Preparation: File{Path: filepath.Join(dir, "preparation.json")}}
	for _, kind := range []string{"positive", "null-selected-needed", "null-alternate-needed", "changed-contract"} {
		t.Run(kind, func(t *testing.T) {
			g, err := makeGuardPlan(r)
			if err != nil {
				t.Fatal(err)
			}
			g.Policy.Allowed = slices.Clone(g.Policy.Allowed)
			switch kind {
			case "null-selected-needed":
				g.Policy.Allowed[0].Needed = nil
			case "null-alternate-needed":
				g.Policy.Alternates = []supgpu.RuntimeCode{{Object: binaryPin, SONAME: "provider.so", Needed: nil, Tier: "fake-rejected"}}
			case "changed-contract":
				g.Contract.SHA256 = strings.Repeat("0", 64)
			}
			guardPath := filepath.Join(dir, kind+"-guard.json")
			if err = writeJSON(guardPath, g); err != nil {
				t.Fatal(err)
			}
			trial := r
			trial.Guard, _ = artifactPin(guardPath)
			err = checkSerializedGuard(trial, b, p)
			if kind == "positive" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil {
				t.Fatal("malformed serialized shared guard was accepted")
			} else {
				want := map[string]string{"null-selected-needed": "policy: allowed: needed: null []string", "null-alternate-needed": "policy: rejected_alternates: needed: null []string", "changed-contract": "guard plan differs from independent phase/custody contract"}[kind]
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("wrong serialized-boundary rejection: %v; want %q", err, want)
				}
			}
		})
	}
}

func TestDecoderPhasesAndImmutableProducerCommand(t *testing.T) {
	want := []string{"pre-model", "pre-forward", "post-back-guide", "post-beam-search", "post-posterior", "post-staged"}
	if !reflect.DeepEqual(decoderPhases().Names, want) {
		t.Fatal("six original phases differ")
	}
	first := decoderPhases()
	first.Names[0] = "changed"
	if !reflect.DeepEqual(decoderPhases().Names, want) {
		t.Fatal("shared mutable phase contract")
	}
	r := RuntimePlan{WorkingDirectory: "/source", CaptureDirectory: "/capture", Consumer: File{Path: "/current/consumer"}, ConsumerPreparation: File{Path: "/current/preparation.json"}}
	b := CompilerPhase{Output: File{Path: "/old/decoder"}}
	p := ClosedBuildPlan{Preparation: File{Path: "/old/preparation.json"}}
	command := decoderCommand(r, b, p)
	if command.Args[2] != "/old/params.bin" || command.Args[4] != "/old/scores-original.f16" || command.Args[5] != "/capture/native" || command.Output != "/capture/guard" {
		t.Fatalf("consumer selected new numerical inputs: %+v", command)
	}
	watch := expectedWatchPlan(r, File{Path: "/runtime.json", SHA256: "independent"}, b, p)
	if watch.Limit != 240 || len(watch.Roles) != 1 || watch.Roles[0].Limit != 180 || !reflect.DeepEqual(watch.Roles[0].Args, command.Args) {
		t.Fatal("watcher does not supervise exact single producer")
	}
}

func TestDecoderSupervisorRequiresExactIdentityAndEmptyCleanup(t *testing.T) {
	r := RuntimePlan{WorkingDirectory: "/source", CaptureDirectory: "/capture", Consumer: File{Path: "/current/consumer", SHA256: "consumer"}, ConsumerPreparation: File{Path: "/current/preparation.json"}}
	b := CompilerPhase{Output: File{Path: "/old/decoder", SHA256: "decoder"}}
	p := ClosedBuildPlan{Preparation: File{Path: "/old/preparation.json"}}
	c := CaptureReceipt{ControllerPID: 100}
	expected := expectedWatchPlan(r, File{Path: "/runtime.json", SHA256: "independent"}, b, p)
	data := map[string]any{"schema": "nanopore-native-role-watch/v1", "plan": expected, "controller_pid": 100, "controller_pgid": 100, "controller_exit": 0, "controller_identity": map[string]any{"pid": 100, "pgid": 100, "sid": 100, "image": r.Consumer.Path, "argv": expected.Controller, "image_sha256": r.Consumer.SHA256, "start_ticks": 1, "cgroup": "owned"}, "roles": []map[string]any{{"name": "decoder", "pid": 200, "pgid": 200, "sid": 100, "argv": expected.Roles[0].Args, "image_sha256": "decoder", "limit_seconds": 180, "start_ticks": 2, "cgroup": "owned"}}, "members_after_cleanup": map[string]any{"100": []any{}, "200": []any{}}, "own_cgroup": "owned", "signal_policy": "current-identity-revalidated-pidfd-members-only"}
	bytes, _ := json.Marshal(data)
	var positive watchObservation
	if err := json.Unmarshal(bytes, &positive); err != nil {
		t.Fatal(err)
	}
	if err := validateWatchObservation(positive, expected, r, b, p, c, 200); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"pid", "image", "argv", "cleanup", "timeout", "ownership", "signals", "limit", "cgroup", "missing-role"} {
		t.Run(kind, func(t *testing.T) {
			var o watchObservation
			if err := json.Unmarshal(bytes, &o); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "pid":
				o.Roles[0].PID++
			case "image":
				o.Roles[0].SHA256 = "other"
			case "argv":
				o.Roles[0].Args[1] = "other"
			case "cleanup":
				o.Members["200"] = []json.RawMessage{json.RawMessage(`{"pid":200}`)}
			case "timeout":
				o.Timeout = true
			case "ownership":
				o.OwnershipErrors = []json.RawMessage{json.RawMessage(`{}`)}
			case "signals":
				o.Signals = []json.RawMessage{json.RawMessage(`{}`)}
			case "limit":
				o.Roles[0].Limit = 181
			case "cgroup":
				o.Roles[0].Cgroup = "other"
			case "missing-role":
				o.Roles = nil
			}
			if e := validateWatchObservation(o, expected, r, b, p, c, 200); e == nil {
				t.Fatal("unqualified supervision accepted")
			}
		})
	}
}

func TestDecoderRawBuffersKeepProspectiveMasks(t *testing.T) {
	dir := t.TempDir()
	contract := FixedContract()
	var manifest strings.Builder
	for _, stage := range contract.Stages {
		for _, buffer := range contract.Buffers {
			body := make([]byte, buffer.Bytes)
			name := stage.Name + "-" + buffer.Name + ".bin"
			file := testFile(t, dir, name, string(body))
			fmt.Fprintf(&manifest, "%s\t%s\t%d\t%s\t%s\n", stage.Name, buffer.Name, buffer.Bytes, file.SHA256, name)
		}
	}
	path := filepath.Join(dir, "buffers.tsv")
	if err := os.WriteFile(path, []byte(manifest.String()), 0600); err != nil {
		t.Fatal(err)
	}
	rows, _, err := readNativeBufferRows(dir, contract, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 32 || rows[1].ComparedBytes != 0 || rows[9].ComparedBytes != N*12 || rows[3].ComparedBytes != GuidesBytes {
		t.Fatal("source-defined mask/extent differs")
	}
	file := filepath.Join(dir, "back-guide-aux.bin")
	body, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	body[GuidesBytes] = 7
	if err = os.WriteFile(file, body, 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, e := readNativeBufferRows(dir, contract, map[string]string{}); e == nil {
		t.Fatal("raw excluded-byte custody tamper accepted")
	}
	newPin, err := artifactPin(file)
	if err != nil {
		t.Fatal(err)
	}
	updated := strings.Replace(manifest.String(), rows[3].Raw.SHA256+"\tback-guide-aux.bin", newPin.SHA256+"\tback-guide-aux.bin", 1)
	if err = os.WriteFile(path, []byte(updated), 0600); err != nil {
		t.Fatal(err)
	}
	again, _, err := readNativeBufferRows(dir, contract, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if again[3].Raw.SHA256 == rows[3].Raw.SHA256 || again[3].ComparedSHA256 != rows[3].ComparedSHA256 || !reflect.DeepEqual(again[3].Compare, rows[3].Compare) {
		t.Fatal("raw/history change altered prospective initialized-byte comparison")
	}
}

func TestDecoderEventsRequireOriginalFourCallsOnce(t *testing.T) {
	dir := t.TempDir()
	var events strings.Builder
	for _, stage := range FixedContract().Stages {
		for _, state := range []string{"before-call", "captured"} {
			fmt.Fprintf(&events, "{\"stage\":%q,\"state\":%q,\"utc\":\"2026-10-25T00:00:00Z\"}\n", stage.Name, state)
		}
	}
	events.WriteString("{\"stage\":\"all\",\"state\":\"completed\",\"utc\":\"2026-10-25T00:00:00Z\"}\n")
	path := filepath.Join(dir, "events.jsonl")
	if err := os.WriteFile(path, []byte(events.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if err := readNativeEvents(dir, FixedContract()); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(events.String()+events.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if err := readNativeEvents(dir, FixedContract()); err == nil {
		t.Fatal("duplicate numerical producer events accepted")
	}
}
