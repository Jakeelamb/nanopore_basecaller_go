package graph

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestSUPUUIDFormatsAndDeviceFields(t *testing.T) {
	source := "00112233445566778899aabbccddeeff"
	p := identityWire{ExpectedPCI: "00000000:01:00.0", ExpectedDriver: "580.1"}
	good := "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:01:00.0, 580.1\n"
	if e := checkIdentityBody([]byte(good), p, 0, source); e != nil {
		t.Fatal(e)
	}
	for _, s := range []string{"GPU-00112233-4455-6677-8899-aabbccddeeff", "00112233-4455-6677-8899-aabbccddeeff", "00112233445566778899AABBCCDDEEFF", "", source + "0"} {
		if _, e := sourceUUID(s); e == nil {
			t.Fatalf("source accepted %q", s)
		}
	}
	for _, body := range []string{"", good + good, "1, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:01:00.0, 580.1", "0, " + source + ", 00000000:01:00.0, 580.1", "0, GPU-00112233-4455-6677-8899-aabbccddeef0, 00000000:01:00.0, 580.1", "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:02:00.0, 580.1", "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:01:00.0, "} {
		if e := checkIdentityBody([]byte(body), p, 0, source); e == nil {
			t.Fatalf("accepted metadata %q", body)
		}
	}
}

type metadataFixtureCase struct {
	Identity nativeproof.Pin
	Plan     supgpu.Pin
}
type metadataFixtureWire struct {
	Schema                              string
	Audit, Contract                     supgpu.Pin
	ProviderA, ProviderB                nativeproof.Pin
	Matched, Mismatched                 metadataFixtureCase
	Directory, Output, MarkerA, MarkerB string
}

// Controller builds/selects this real lazy-loader fixture under its native
// unit. This opt-in launches processes and needs the same external stock lock
// and finite watcher admission as any other shared guard execution.
func TestGenuineMetadataProviderBoundaryOptIn(t *testing.T) {
	path, sha := os.Getenv("SUP_METADATA_FIXTURE"), os.Getenv("SUP_METADATA_FIXTURE_SHA256")
	if path == "" && sha == "" {
		t.Skip("genuine metadata fixture unselected; no native qualification")
	}
	if path == "" || sha == "" {
		t.Fatal("independent fixture path and SHA256 required")
	}
	var fixture metadataFixtureWire
	if e := readJSON(nativeproof.Pin{Path: path, SHA256: sha}, &fixture); e != nil {
		t.Fatal(e)
	}
	if fixture.Schema != "nanopore-sup-metadata-provider-fixture/v1" || fixture.MarkerA == fixture.MarkerB {
		t.Fatal("fixture schema/markers differ")
	}
	if e := requireFresh(fixture.MarkerA, fixture.MarkerB); e != nil {
		t.Fatal(e)
	}
	if fixture.ProviderA.Path == fixture.ProviderB.Path || fixture.ProviderA.SHA256 == fixture.ProviderB.SHA256 {
		t.Fatal("distinct A/B providers required")
	}
	for _, v := range []nativeproof.Pin{fixture.ProviderA, fixture.ProviderB} {
		if _, e := sizedPin(v); e != nil {
			t.Fatal(e)
		}
		m, e := nativeproof.InspectELF(v.Path)
		if e != nil || m.Pin != v || m.SONAME != "libmetadata-provider.so" {
			t.Fatal("genuine same-SONAME provider required", e)
		}
	}
	for i, selected := range []metadataFixtureCase{fixture.Matched, fixture.Mismatched} {
		p, e := readIdentity(selected.Identity)
		if e != nil {
			t.Fatal(e)
		}
		plan, c, e := supgpu.ReadGuardPlan(selected.Plan, fixture.Contract)
		if e != nil {
			t.Fatal(e)
		}
		if c.Schema != "nanopore-audited-metadata-contract/v1" {
			t.Fatal("metadata schema required")
		}
		if e := identityPolicy(p, plan, fixture.Audit); e != nil {
			t.Fatal(e)
		}
		if len(p.LazyProviders) != 1 || p.LazyProviders[0] != fixture.ProviderA || !slices.Equal(p.Args, identityArgs(p, 0)) {
			t.Fatal("explicit A provider and exact metadata argv required")
		}
		label := "matched"
		if i == 1 {
			label = "mismatched"
		}
		w := admissionWire{MetadataGuardPlan: selected.Plan, MetadataContract: fixture.Contract, Command: supgpu.GuardCommand{Directory: fixture.Directory, Output: fixture.Output + "-" + label}}
		if !strings.HasPrefix(w.Command.Output, filepath.Join(fixture.Directory, "artifacts")+string(os.PathSeparator)) {
			t.Fatal("fixture outputs must be under selected artifacts")
		}
		r, e := executeIdentity(context.Background(), w, "before", p, 0, "00112233445566778899aabbccddeeff")
		if i == 0 {
			if e != nil {
				t.Fatal(e)
			}
			if _, e := os.Stat(fixture.MarkerA); e != nil {
				t.Fatal("matched constructor did not run", e)
			}
			if e := validateIdentity(r, w, "before", p, 0, "00112233445566778899aabbccddeeff"); e != nil {
				t.Fatal(e)
			}
		} else {
			if e == nil {
				t.Fatal("unselected B provider accepted")
			}
			if _, e := os.Stat(fixture.MarkerB); !os.IsNotExist(e) {
				t.Fatal("unselected B constructor ran or marker unavailable")
			}
			var g supgpu.GuardReceipt
			if e := readJSON(unsized(r.Transcript), &g); e != nil {
				t.Fatal(e)
			}
			denied := false
			for _, d := range g.Decisions {
				denied = denied || (!d.Approved && d.Request.Kind == "open" && d.Request.Path == fixture.ProviderB.Path && (strings.Contains(d.Error, "runtime tier rejected") || strings.Contains(d.Error, "unknown mapped code rejected")))
			}
			if !denied {
				t.Fatal("failure did not retain an actual unselected-provider denial")
			}
		}
	}
}

func TestMetadataPolicyBindsExplicitClosureAndEnvironment(t *testing.T) {
	dir := t.TempDir()
	makePin := func(name string) nativeproof.Pin {
		path := filepath.Join(dir, name)
		if e := os.WriteFile(path, []byte(name), 0600); e != nil {
			t.Fatal(e)
		}
		p, e := pinFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return p
	}
	root, provider, audit := makePin("nvidia-smi"), makePin("provider"), makePin("auditor")
	rp, _ := sizedPin(root)
	pp, _ := sizedPin(provider)
	ap, _ := sizedPin(audit)
	p := identityWire{Executable: root, SearchDirs: []string{dir}, LazyProviders: []nativeproof.Pin{provider}, Closure: nativeproof.ELFClosure{Nodes: []nativeproof.ELFMetadata{{Pin: root}, {Pin: provider, SONAME: "provider.so", Needed: []string{}}}, Bindings: []nativeproof.ELFBinding{}}}
	plan := supgpu.GuardPlan{Policy: supgpu.RuntimePolicy{Reference: rp, Observer: rp, Audit: ap, Directories: []string{dir}, Allowed: []supgpu.RuntimeCode{{Object: pp, SONAME: "provider.so", Needed: []string{}, Tier: "explicit", Startup: false}}}}
	p.Environment = supgpu.SelectedModelEnvironment(plan.Policy)
	if e := identityPolicy(p, plan, ap); e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"environment", "directories", "root", "audit", "missing", "extra", "hash", "soname", "needed", "startup"} {
		t.Run(kind, func(t *testing.T) {
			q := p
			candidate := plan
			candidate.Policy.Allowed = append([]supgpu.RuntimeCode{}, plan.Policy.Allowed...)
			switch kind {
			case "environment":
				q.Environment = []string{"LD_AUDIT="}
			case "directories":
				candidate.Policy.Directories = []string{filepath.Join(dir, "other")}
			case "root":
				candidate.Policy.Observer = pp
			case "audit":
				candidate.Policy.Audit = pp
			case "missing":
				candidate.Policy.Allowed = nil
			case "extra":
				candidate.Policy.Allowed = append(candidate.Policy.Allowed, supgpu.RuntimeCode{Object: ap})
			case "hash":
				candidate.Policy.Allowed[0].Object.SHA256 = strings.Repeat("0", 64)
			case "soname":
				candidate.Policy.Allowed[0].SONAME = "fabricated.so"
			case "needed":
				candidate.Policy.Allowed[0].Needed = []string{"fabricated.so"}
			case "startup":
				candidate.Policy.Allowed[0].Startup = true
			}
			if identityPolicy(q, candidate, ap) == nil {
				t.Fatal("changed metadata selection accepted")
			}
		})
	}
}
