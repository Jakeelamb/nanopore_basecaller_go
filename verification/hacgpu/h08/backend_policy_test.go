package h08

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

// Uses a separately pinned retained typed proof; it never replays its producer.
// Copy this file alone onto old production to expose accepted fake SONAMEs.
func TestBackendPolicyGenuineMetadata(t *testing.T) {
	keys := []string{"H08_UNNAMED_PROOF", "H08_UNNAMED_PROOF_SHA256", "H08_UNNAMED_BACKEND", "H08_UNNAMED_BACKEND_SHA256", "H08_UNNAMED_POLICY", "H08_UNNAMED_POLICY_SHA256", "H08_UNNAMED_CONTROLS"}
	set := 0
	for _, k := range keys {
		if os.Getenv(k) != "" {
			set++
		}
	}
	if set == 0 {
		t.Skip("independent unnamed backend/policy/proof pins and controls directory absent")
	}
	if set != len(keys) {
		t.Fatal("partial H08_UNNAMED environment")
	}
	for _, k := range []string{keys[0], keys[2], keys[4], keys[6]} {
		p := os.Getenv(k)
		canonical, err := filepath.EvalSymlinks(p)
		if err != nil || !filepath.IsAbs(p) || canonical != p {
			t.Fatalf("canonical absolute %s required: %v", k, err)
		}
	}
	var proof struct {
		Schema             string
		Backend            closedbuild.BackendArtifact
		ConsumerExecutable Pin
		RuntimeHeld        bool
	}
	if err := pinnedJSON(Pin{Path: os.Getenv(keys[0]), SHA256: os.Getenv(keys[1])}, &proof); err != nil {
		t.Fatal(err)
	}
	b := proof.Backend
	expected := Pin{Path: os.Getenv(keys[2]), SHA256: os.Getenv(keys[3])}
	if proof.Schema != "nanopore-h08-backend-proof/v1" || !proof.RuntimeHeld || b.Schema != "nanopore-tensor-backend-artifact/v1" || !b.RuntimeHeld || b.Output != expected || len(b.Runtime) != 23 {
		t.Fatal("selected proof/backend contract differs")
	}
	output, err := nativeproof.InspectELF(expected.Path)
	if err != nil || output.Pin.Path != expected.Path || output.Pin.SHA256 != expected.SHA256 || output.SONAME != "" || len(output.Needed) != 12 {
		t.Fatalf("exact genuine unnamed backend differs: %+v: %v", output, err)
	}
	var p supgpu.RuntimePolicy
	// Decode the authenticated original without calling the production validator:
	// the fake-basename regression must reach old validateBackendPolicy itself.
	if err := pinnedJSON(Pin{Path: os.Getenv(keys[4]), SHA256: os.Getenv(keys[5])}, &p); err != nil {
		t.Fatal(err)
	}
	index := -1
	for i, c := range p.Allowed {
		if c.Object.Path == expected.Path && c.Object.SHA256 == expected.SHA256 {
			index = i
		}
	}
	if index < 0 || p.Allowed[index].SONAME != "" || !slices.Equal(p.Allowed[index].Needed, output.Needed) {
		t.Fatal("truthful genuine backend policy required")
	}
	provider := -1
	for i, c := range p.Allowed {
		if c.SONAME == output.Needed[0] {
			provider = i
		}
	}
	if provider < 0 {
		t.Fatal("selected first Needed provider required")
	}
	dir, err := os.MkdirTemp(os.Getenv(keys[6]), "h08-unnamed-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained controls: %s", dir)
	for _, kind := range []string{"positive", "fake-basename", "named-as-empty", "provider-needed", "backend-needed", "missing-provider", "size", "hash"} {
		t.Run(kind, func(t *testing.T) {
			q := p
			q.Allowed = slices.Clone(p.Allowed)
			switch kind {
			case "fake-basename":
				q.Allowed[index].SONAME = filepath.Base(expected.Path)
			case "named-as-empty":
				q.Allowed[provider].SONAME = ""
			case "provider-needed":
				q.Allowed[provider].Needed = []string{"invented.so"}
			case "backend-needed":
				q.Allowed[index].Needed = []string{}
			case "missing-provider":
				q.Allowed = slices.Delete(q.Allowed, provider, provider+1)
			case "size":
				q.Allowed[index].Object.Bytes++
			case "hash":
				q.Allowed[index].Object.SHA256 = ""
			}
			plan := supgpu.GuardPlan{Policy: q}
			path := filepath.Join(dir, kind+"-plan.json")
			raw, err := json.MarshalIndent(plan, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
			if err != nil {
				t.Fatal(err)
			}
			_, writeErr := f.Write(raw)
			closeErr := f.Close()
			if writeErr != nil || closeErr != nil {
				t.Fatalf("retain plan: %v: %v", writeErr, closeErr)
			}
			pin, err := closedbuild.PinFile(path)
			if err != nil {
				t.Fatal(err)
			}
			err = validateBackendPolicy(RuntimeAdmission{Plan: pin}, b)
			if kind == "positive" {
				if err != nil {
					t.Fatalf("genuine truthful backend policy rejected: %v", err)
				}
			} else if err == nil {
				t.Fatalf("accepted rehashed %s", kind)
			}
		})
	}
}
