package graph

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestClosedBoundaryRejectsDuplicateUnknownNullMissingAndTrailing(t *testing.T) {
	type wire struct {
		Schema string
		Values []string
	}
	for _, raw := range []string{`null`, `{}`, `{"Schema":"a","Schema":"b","Values":[]}`, `{"schema":"a","Values":[]}`, `{"Schema":null,"Values":[]}`, `{"Schema":"a","Values":null}`, `{"Schema":"a","Values":[null]}`, `{"Schema":"a","Values":[],"extra":0}`, `{"Schema":"a","Values":[]} {}`} {
		var out wire
		if e := decodeStrict([]byte(raw), &out); e == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	var out wire
	if e := decodeStrict([]byte(`{"Schema":"a","Values":[]}`), &out); e != nil {
		t.Fatal(e)
	}
}
func TestPinnedBoundaryRejectsFIFOAndSymlinksAndWrongExtent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "input.json")
	b := []byte(`{"Schema":"a","Values":[]}`)
	if e := os.WriteFile(path, b, 0600); e != nil {
		t.Fatal(e)
	}
	type wire struct {
		Schema string
		Values []string
	}
	var out wire
	pin := nativeproof.Pin{Path: path, SHA256: digest(b)}
	if e := readJSON(pin, &out); e != nil {
		t.Fatal(e)
	}
	alias := filepath.Join(dir, "alias")
	if e := os.Symlink(path, alias); e != nil {
		t.Fatal(e)
	}
	pin.Path = alias
	if e := readJSON(pin, &out); e == nil {
		t.Fatal("symlink accepted")
	}
	fifo := filepath.Join(dir, "fifo")
	if e := syscall.Mkfifo(fifo, 0600); e != nil {
		t.Fatal(e)
	}
	pin.Path = fifo
	if e := readJSON(pin, &out); e == nil {
		t.Fatal("FIFO accepted")
	}
	if _, e := readSized(path, supgpu.Pin{Path: path, SHA256: digest(b), Bytes: int64(len(b) + 1)}, 64<<20); e == nil {
		t.Fatal("wrong byte extent accepted")
	}
	pin.Path = path
	pin.SHA256 = strings.Repeat("0", 64)
	if e := readJSON(pin, &out); e == nil {
		t.Fatal("changed JSON bytes accepted")
	}
}
func TestOutputContainmentRejectsAliasesOverlapsAndInput(t *testing.T) {
	root := t.TempDir()
	base := filepath.Join(root, "artifacts")
	if e := os.Mkdir(base, 0700); e != nil {
		t.Fatal(e)
	}
	out := filepath.Join(base, "run")
	if e := validateOutputs(root, []string{out}, nil); e != nil {
		t.Fatal(e)
	}
	for _, outputs := range [][]string{{base}, {filepath.Join(root, "else")}, {out, out}, {filepath.Join(base, "a"), filepath.Join(base, "a", "b")}} {
		if e := validateOutputs(root, outputs, nil); e == nil {
			t.Fatalf("accepted %q", outputs)
		}
	}
	if e := validateOutputs(root, []string{out}, []supgpu.Pin{{Path: filepath.Join(out, "input")}}); e == nil {
		t.Fatal("output contains input")
	}
}

func TestSUPNumericalAdmissionRejectsMetadataVariant(t *testing.T) {
	c := supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "sup-tiny", Names: []string{"pre-model", "pre-forward", "post-forward"}}
	if !numericalContract(c) {
		t.Fatal("ordinary SUP phases rejected")
	}
	c.Schema = "nanopore-audited-metadata-contract/v1"
	if numericalContract(c) {
		t.Fatal("stolen metadata schema accepted for numerical admission")
	}
	c.Action = "device-identity"
	c.Names = []string{}
	if numericalContract(c) {
		t.Fatal("metadata contract bypassed numerical admission")
	}
}

// Root supplies genuine producer artifacts. A skip is no reader/runtime proof.
// Run after role integration, from the selected root; this test never launches
// a process, loads a backend, queries a device, or acquires the host lock.
func TestGenuineSUPAdmissionReaderOptIn(t *testing.T) {
	path, sha := os.Getenv("SUP_GRAPH_ADMISSION"), os.Getenv("SUP_GRAPH_ADMISSION_SHA256")
	if path == "" && sha == "" {
		t.Skip("genuine externally selected SUP admission not supplied; unqualified")
	}
	if path == "" || sha == "" {
		t.Fatal("both independent fixture pins required")
	}
	a, e := readAdmission(nativeproof.Pin{Path: path, SHA256: sha})
	if e != nil {
		t.Fatal(e)
	}
	if len(a.request.backend.Runtime) != 23 || len(a.request.witness.capture.Events) != 191 {
		t.Fatal("genuine fixture membership differs")
	}
	// Mutating an authenticated in-memory policy is a negative check, never a
	// substitute for the external fixture's positive admission chain.
	p := a.plan
	p.Policy.Allowed = append([]supgpu.RuntimeCode{}, p.Policy.Allowed...)
	target := a.request.backend.Runtime[0].Pin.Path
	for i, c := range p.Policy.Allowed {
		if c.Object.Path == target {
			p.Policy.Allowed = append(p.Policy.Allowed[:i], p.Policy.Allowed[i+1:]...)
			break
		}
	}
	if e = backendPolicy(a.request.backend, p); e == nil {
		t.Fatal("omitted actual provider accepted")
	}
	p = a.plan
	p.Policy.Allowed = append([]supgpu.RuntimeCode{}, p.Policy.Allowed...)
	for i, c := range p.Policy.Allowed {
		if c.Object.Path == a.request.backend.Output.Path {
			p.Policy.Allowed[i].SONAME = "fabricated.so"
		}
	}
	if e = backendPolicy(a.request.backend, p); e == nil {
		t.Fatal("fabricated backend SONAME accepted")
	}
	changed := a.request.wire.Backend
	changed.Output.SHA256 = strings.Repeat("0", 64)
	if _, e := sizedPin(changed.Output); e == nil {
		t.Fatal("changed backend pin accepted")
	}
}

func TestGenuineSUPNumericalReplayOptIn(t *testing.T) {
	ap, as := os.Getenv("SUP_GRAPH_ADMISSION"), os.Getenv("SUP_GRAPH_ADMISSION_SHA256")
	rp, rs := os.Getenv("SUP_GRAPH_RECEIPT"), os.Getenv("SUP_GRAPH_RECEIPT_SHA256")
	if rp == "" && rs == "" {
		t.Skip("genuine runtime wrapper not supplied; no numerical qualification")
	}
	if ap == "" || as == "" || rp == "" || rs == "" {
		t.Fatal("all independently selected admission/wrapper pins required")
	}
	a, e := readAdmission(nativeproof.Pin{Path: ap, SHA256: as})
	if e != nil {
		t.Fatal(e)
	}
	if _, e = Replay(nativeproof.Pin{Path: ap, SHA256: as}, nativeproof.Pin{Path: rp, SHA256: rs}); e != nil {
		t.Fatal(e)
	}
	var r childWire
	pin, e := pinFile(filepath.Join(a.request.wire.Out, "consumer.json"))
	if e != nil {
		t.Fatal(e)
	}
	if e = readJSON(pin, &r); e != nil {
		t.Fatal(e)
	}
	for _, mutate := range []func(*Receipt){
		func(n *Receipt) { n.ScoreSHA256 = strings.Repeat("0", 64) },
		func(n *Receipt) { n.Comparisons[0].GoSHA256 = strings.Repeat("0", 64) },
		func(n *Receipt) { n.Comparisons[0].BackingSHA256 = strings.Repeat("0", 64) },
		func(n *Receipt) { n.Comparisons[0].Bytes++ },
		func(n *Receipt) { n.Comparisons[0].ContractSHA256 = strings.Repeat("0", 64) },
		func(n *Receipt) { n.Comparisons[0], n.Comparisons[1] = n.Comparisons[1], n.Comparisons[0] },
		func(n *Receipt) { n.EventContracts[0] = strings.Repeat("0", 64) },
	} {
		n := r.Numerical
		n.Comparisons = append([]Comparison{}, n.Comparisons...)
		n.EventContracts = append([]string{}, n.EventContracts...)
		mutate(&n)
		if e = validateComparisons(a.request.witness, n); e == nil {
			t.Fatal("corrupted genuine comparison accepted")
		}
	}
}
