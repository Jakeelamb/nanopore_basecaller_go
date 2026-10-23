package gpuoracle

import (
	"encoding/json"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClosedContract(t *testing.T) {
	valid := `{"Schema":"nanopore-gpu-oracle/v1","Stage":"prepared","Source":[],"Inputs":[],"RangeDeclaration":{"Path":"x","SHA256":"hash"}}`
	for _, raw := range []string{`{}`, `null`, strings.Replace(valid, `"Inputs":[]`, `"Inputs":null`, 1), strings.Replace(valid, `"Source":[]`, `"Source":[],"Source":[]`, 1), strings.Replace(valid, `"Path":"x"`, `"Path":"x","path":"x"`, 1), valid + `{}`} {
		var p Prepared
		if err := Decode([]byte(raw), &p); err == nil {
			t.Fatalf("accepted invalid contract %s", raw)
		}
	}
	var p Prepared
	if err := Decode([]byte(valid), &p); err != nil {
		t.Fatal(err)
	}
}
func TestAdmissionRejectsBeforeLock(t *testing.T) {
	calls := 0
	qualify := func() error { calls++; return os.ErrInvalid }
	if _, err := admit(false, qualify, "missing"); err == nil || calls != 0 {
		t.Fatal("missing admission reached qualification")
	}
	if _, err := admit(true, qualify, "missing"); err == nil || calls != 1 {
		t.Fatal("invalid qualification admitted")
	}
}

func TestProductionAdmissionRejectsBeforeAcquisition(t *testing.T) {
	calls := 0
	qualify := func() error { calls++; return os.ErrInvalid }
	if _, err := Admit(false, qualify); err == nil || calls != 0 {
		t.Fatal("missing production intake reached qualification")
	}
	if _, err := Admit(true, qualify); err == nil || calls != 1 {
		t.Fatal("invalid production qualification admitted")
	}
}

func TestMissingLockLaunchesZeroChildren(t *testing.T) {
	children := 0
	launch := func() error {
		release, err := admit(true, func() error { return nil }, filepath.Join(t.TempDir(), "missing.lock"))
		if err != nil {
			return err
		}
		defer release()
		children++
		return nil
	}
	if err := launch(); err == nil || children != 0 {
		t.Fatal("missing lock allowed native launch")
	}
}
func TestRequiredTestAccounting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.json")
	for _, raw := range []string{"", `{"Action":"skip","Package":"p","Test":"TestGate"}`, `{"Action":"pass","Package":"p","Test":"TestOther"}`} {
		if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		if err := RequireTests(path, "p", []string{"TestGate"}); err == nil {
			t.Fatalf("accepted %q", raw)
		}
	}
	if err := RequireTests(path, "p", nil); err == nil {
		t.Fatal("empty gate accepted")
	}
}
func TestAllocationCoverage(t *testing.T) {
	pin := Pin{"source", strings.Repeat("a", 64)}
	a := Allocation{"output", 8, []Range{{0, 4, "live", pin}, {4, 4, "uninitialized", pin}}}
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	a.Ranges[1].Offset = 3
	if err := a.Validate(); err == nil {
		t.Fatal("overlap accepted")
	}
}
func TestChangedPayload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "payload")
	if err := os.WriteFile(path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := (Pin{path, strings.Repeat("a", 64)}).Check(); err == nil {
		t.Fatal("corrupt payload accepted")
	}
}

// This control acquires only a newly created temporary lock when tests are admitted.
func TestHeldTemporaryLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host.lock")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	release, err := admit(true, func() error { return nil }, path)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	children := 0
	second, err := admit(true, func() error { return nil }, path)
	if err == nil {
		second()
		children++
		t.Fatal("held temporary lock admitted")
	}
	if children != 0 {
		t.Fatal("native child launched")
	}
}

func TestContractIdentity(t *testing.T) {
	cases := []struct {
		value  any
		stage  Stage
		target func() any
	}{
		{Prepared{Schema: Schema, Stage: StagePrepared, Source: []Pin{}, Inputs: []Pin{}}, StagePrepared, func() any { return new(Prepared) }},
		{Frozen{Schema: Schema, Stage: StageFrozen, Producer: []Pin{}, Commands: []runner.CommandReceipt{}, Payloads: []Pin{}}, StageFrozen, func() any { return new(Frozen) }},
		{Imported{Schema: Schema, Stage: StageImported, Consumer: []Pin{}, RequiredTests: []string{}}, StageImported, func() any { return new(Imported) }},
		{Accepted{Schema: Schema, Stage: StageAccepted}, StageAccepted, func() any { return new(Accepted) }},
	}
	for _, c := range cases {
		raw, err := json.Marshal(c.value)
		if err != nil {
			t.Fatal(err)
		}
		if err := Decode(raw, c.target()); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []string{
			strings.Replace(string(raw), Schema, "unknown/v2", 1),
			strings.Replace(string(raw), `"Stage":"`+string(c.stage)+`"`, `"Stage":"wrong"`, 1),
			strings.Replace(string(raw), `"Schema":"`+Schema+`",`, "", 1),
			strings.Replace(string(raw), `"Stage":"`+string(c.stage)+`"`, `"Stage":null`, 1),
		} {
			if err := Decode([]byte(bad), c.target()); err == nil {
				t.Fatalf("accepted identity %s", bad)
			}
		}
	}
	// Valid wire identity and structure do not establish valid evidence pins.
	if err := (Prepared{Schema: Schema, Stage: StagePrepared}).ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
}
