package h08

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestH08GuardPinRejectsFIFO(t *testing.T) {
	const childKey = "H08_GUARD_PIN_FIFO_CHILD"
	if path := os.Getenv(childKey); path != "" {
		if _, err := guardPin(Pin{Path: path, SHA256: strings.Repeat("0", 64)}); err == nil || !strings.Contains(err.Error(), "regular guard input required") {
			t.Fatalf("FIFO must reject as nonregular: %v", err)
		}
		return
	}
	path := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(path, 0600); err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// A regressed blocking open is killed and reaped in the self-child; no
	// goroutine can remain blocked on the FIFO in this test process.
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestH08GuardPinRejectsFIFO$", "-test.count=1")
	cmd.Env = append(os.Environ(), childKey+"="+path)
	cmd.WaitDelay = time.Second
	output, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("FIFO rejection blocked: %v: %s", ctx.Err(), output)
	}
	if err != nil {
		t.Fatalf("FIFO rejection child failed: %v: %s", err, output)
	}
}

func TestH08GuardPinRejectsDirectory(t *testing.T) {
	if _, err := guardPin(Pin{Path: t.TempDir(), SHA256: strings.Repeat("0", 64)}); err == nil || !strings.Contains(err.Error(), "regular guard input required") {
		t.Fatalf("directory accepted: %v", err)
	}
}

func TestH08GuardAcceptsSameSizedMetadataAsSharedGuard(t *testing.T) {
	dir := t.TempDir()
	inert := hostJSON(t, filepath.Join(dir, "inert.json"), map[string]string{"fixture": "metadata only; no native artifact"})
	sized := func(p Pin) supgpu.Pin {
		t.Helper()
		body, err := readPin(p)
		if err != nil {
			t.Fatal(err)
		}
		return supgpu.Pin{Path: p.Path, SHA256: p.SHA256, Bytes: int64(len(body))}
	}
	contract := hostJSON(t, filepath.Join(dir, "contract.json"), supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"}})
	plan := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Contract: sized(contract), Policy: supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: sized(inert), Observer: sized(inert), Audit: sized(inert), Allowed: []supgpu.RuntimeCode{}, Alternates: []supgpu.RuntimeCode{}, Directories: []string{dir}}, Inputs: []supgpu.Pin{sized(inert)}}
	pp := hostJSON(t, filepath.Join(dir, "plan.json"), plan)
	if _, _, err := supgpu.ReadGuardPlan(sized(pp), sized(contract)); err != nil {
		t.Fatalf("shared guard rejected valid sized metadata: %v", err)
	}
	a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: inert, Plan: pp, Contract: contract, ParentBuild: inert, ParentBuildSelection: inert, Command: supgpu.GuardCommand{Directory: dir, Output: filepath.Join(dir, "never-executed")}}
	if err := validateGuardInputs(a, inert, inert, nil, "hac-cpu13"); err != nil {
		t.Fatalf("H08 must accept the same sized metadata as shared guard: %v", err)
	}
}

func TestH08GuardPinAuthenticatesSize(t *testing.T) {
	dir := t.TempDir()
	p := hostJSON(t, filepath.Join(dir, "object.json"), map[string]string{"inert": "metadata"})
	got, err := guardPin(p)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p.Path)
	if err != nil || got.Bytes != int64(len(b)) || got.Bytes == 0 {
		t.Fatalf("actual authenticated size missing: %+v: %v", got, err)
	}
	if err = os.WriteFile(p.Path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = guardPin(p); err == nil {
		t.Fatal("changed input accepted")
	}
	if err = os.Remove(p.Path); err != nil {
		t.Fatal(err)
	}
	if _, err = guardPin(p); err == nil {
		t.Fatal("missing input accepted")
	}
	if err = os.WriteFile(p.Path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = guardPin(Pin{Path: p.Path, SHA256: digest(nil)}); err == nil {
		t.Fatal("zero-byte guard input accepted")
	}
}

func TestH08GuardRejectsSizedMetadataMutations(t *testing.T) {
	for _, mutation := range []string{"contract-zero", "contract-size", "role", "action", "phase", "policy", "target", "custody-zero", "custody-size", "custody-missing", "custody-duplicate"} {
		t.Run(mutation, func(t *testing.T) {
			dir := t.TempDir()
			object := hostJSON(t, filepath.Join(dir, "inert.json"), map[string]string{"inert": "metadata"})
			c := supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"}}
			switch mutation {
			case "role":
				c.Role = "reference"
			case "action":
				c.Action = "hac-all-active"
			case "phase":
				c.Names = []string{"pre-model", "post-forward"}
			}
			cp := hostJSON(t, filepath.Join(dir, "contract.json"), c)
			p := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Contract: hostGuardPin(t, cp), Policy: hostGuardPolicy(t, object, dir), Inputs: []supgpu.Pin{hostGuardPin(t, object)}}
			switch mutation {
			case "contract-zero":
				p.Contract.Bytes = 0
			case "contract-size":
				p.Contract.Bytes++
			case "policy":
				p.Policy.Lazy = "unaudited"
			case "target":
				p.Policy.Observer = hostGuardPin(t, cp)
			case "custody-zero":
				p.Inputs[0].Bytes = 0
			case "custody-size":
				p.Inputs[0].Bytes++
			case "custody-missing":
				p.Inputs = []supgpu.Pin{hostGuardPin(t, cp)}
			case "custody-duplicate":
				p.Inputs = append(p.Inputs, p.Inputs[0])
			}
			pp := hostJSON(t, filepath.Join(dir, "plan.json"), p)
			a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: object, ParentBuild: object, ParentBuildSelection: object, Plan: pp, Contract: cp, Command: supgpu.GuardCommand{Directory: dir, Output: filepath.Join(dir, "never-executed")}}
			if err := validateGuardInputs(a, object, object, nil, "hac-cpu13"); err == nil {
				t.Fatalf("accepted %s", mutation)
			}
		})
	}
}
