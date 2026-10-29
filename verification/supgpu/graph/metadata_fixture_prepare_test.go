package graph

import (
	"bytes"
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

// Every field is required. SearchDirs are explicit canonical directories, not
// paths discovered from the host cache or a native command.
type metadataPrepareOptions struct {
	Schema, Root, Output                            string
	Audit                                           supgpu.Pin
	MatchedCLI, MismatchedCLI, ProviderA, ProviderB nativeproof.Pin
	MarkerA, MarkerB                                string
	SearchDirs                                      []string
	TimeoutSeconds                                  int
}

// This constructor only reads/hashes ELF metadata and writes fixture JSON. It
// never executes a selected file, invokes a compiler, or acquires a host lock.
func TestPrepareMetadataProviderFixtureOptIn(t *testing.T) {
	path, sha := os.Getenv("SUP_METADATA_PREPARE"), os.Getenv("SUP_METADATA_PREPARE_SHA256")
	if path == "" && sha == "" {
		t.Skip("metadata fixture construction unselected; no native qualification")
	}
	if path == "" || sha == "" {
		t.Fatal("independent preparation options path and SHA256 required")
	}
	optionPin := nativeproof.Pin{Path: path, SHA256: sha}
	var o metadataPrepareOptions
	if e := readJSON(optionPin, &o); e != nil {
		t.Fatal(e)
	}
	root, e := currentRoot()
	if e != nil {
		t.Fatal(e)
	}
	if o.Schema != "nanopore-sup-metadata-prepare/v1" || o.Root != root || o.TimeoutSeconds < 1 || o.TimeoutSeconds > 60 || len(o.SearchDirs) == 0 {
		t.Fatal("exact current root, preparation schema, explicit search dirs and finite timeout required")
	}
	inputs := []supgpu.Pin{}
	addInput := func(p nativeproof.Pin) supgpu.Pin {
		t.Helper()
		v, e := sizedPin(p)
		if e != nil {
			t.Fatal(e)
		}
		for _, before := range inputs {
			if before.Path == v.Path {
				if before != v {
					t.Fatal("conflicting custody pin", v.Path)
				}
				return v
			}
		}
		inputs = append(inputs, v)
		return v
	}
	addInput(optionPin)
	if addInput(unsized(o.Audit)) != o.Audit {
		t.Fatal("auditor extent differs")
	}
	for _, name := range []string{"metadata-cli.c", "metadata-provider.c", "metadata_before_test.go.txt"} {
		p, e := pinFile(filepath.Join(root, "verification/supgpu/graph/testdata", name))
		if e != nil {
			t.Fatal(e)
		}
		addInput(p)
	}
	self, e := pinFile(filepath.Join(root, "verification/supgpu/graph/metadata_fixture_prepare_test.go"))
	if e != nil {
		t.Fatal(e)
	}
	addInput(self)
	selected := []nativeproof.Pin{o.MatchedCLI, o.MismatchedCLI, o.ProviderA, o.ProviderB, unsized(o.Audit)}
	metadata := make([]nativeproof.ELFMetadata, len(selected))
	infos := []os.FileInfo{}
	for i, p := range selected {
		addInput(p)
		info, e := os.Stat(p.Path)
		if e != nil {
			t.Fatal(e)
		}
		for _, before := range infos {
			if os.SameFile(info, before) {
				t.Fatal("selected ELF paths must be separate files")
			}
		}
		infos = append(infos, info)
		m, e := nativeproof.InspectELF(p.Path)
		if e != nil || m.Pin != p {
			t.Fatal("selected ELF differs", p.Path, e)
		}
		metadata[i] = m
		if i < 2 && (filepath.Base(p.Path) != "nvidia-smi" || info.Mode().Perm()&0111 == 0 || m.Interpreter == "") {
			t.Fatal("real executable metadata CLI required", p.Path)
		}
		if i >= 2 && (m.Type != elf.ET_DYN || m.Interpreter != "") {
			t.Fatal("provider/auditor must be shared ELF", p.Path)
		}
		if i == 2 || i == 3 {
			if filepath.Base(p.Path) != "libmetadata-provider.so" || m.SONAME != "libmetadata-provider.so" {
				t.Fatal("same-SONAME A/B providers required")
			}
		}
		if i > 0 && (m.Class != metadata[0].Class || m.Data != metadata[0].Data || m.Machine != metadata[0].Machine) {
			t.Fatal("selected ELF ABI differs")
		}
	}
	if o.ProviderA.SHA256 == o.ProviderB.SHA256 || o.MatchedCLI.SHA256 == o.MismatchedCLI.SHA256 || len(metadata[4].Needed) != 0 {
		t.Fatal("distinct native A/B and CLI bytes, dependency-free auditor required")
	}
	// Authenticate compile-selected literals without loading or running an ELF.
	for _, v := range []struct {
		pin                nativeproof.Pin
		literal, forbidden string
	}{
		{o.MatchedCLI, o.ProviderA.Path, o.ProviderB.Path},
		{o.MismatchedCLI, o.ProviderB.Path, o.ProviderA.Path},
		{o.ProviderA, o.MarkerA, o.MarkerB},
		{o.ProviderB, o.MarkerB, o.MarkerA},
	} {
		p := addInput(v.pin)
		body, e := readSized(p.Path, p, 64<<20)
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Contains(body, append([]byte(v.literal), 0)) || bytes.Contains(body, append([]byte(v.forbidden), 0)) {
			t.Fatal("compiled selected literal differs", v.pin.Path)
		}
	}
	seenDirs := map[string]bool{}
	for _, dir := range o.SearchDirs {
		if e := canonical(dir); e != nil {
			t.Fatal(e)
		}
		info, e := os.Stat(dir)
		if e != nil || !info.IsDir() || seenDirs[dir] || strings.ContainsAny(dir, ":\x00\r\n") {
			t.Fatal("unique canonical loader directory required", dir)
		}
		seenDirs[dir] = true
	}
	if !seenDirs[filepath.Dir(o.ProviderA.Path)] || seenDirs[filepath.Dir(o.ProviderB.Path)] {
		t.Fatal("explicit A search directory and separate unselected B directory required")
	}
	outputs := []string{o.Output, o.MarkerA, o.MarkerB}
	if e := validateOutputs(root, outputs, inputs); e != nil {
		t.Fatal(e)
	}
	if e := requireFresh(outputs...); e != nil {
		t.Fatal(e)
	}
	// Close both actual roots with A only. B is evidence, never a policy seed.
	identities := make([]identityWire, 2)
	policies := make([]supgpu.RuntimePolicy, 2)
	for i, cli := range []nativeproof.Pin{o.MatchedCLI, o.MismatchedCLI} {
		seeds := []nativeproof.Pin{cli, o.ProviderA}
		closure, e := nativeproof.CloseELF(seeds, o.SearchDirs)
		if e != nil {
			t.Fatal(e)
		}
		// Empty dynamic lists have the same ELF meaning, but closed JSON rejects
		// null. Keep all real metadata and bindings, changing only nil to [].
		if closure.Bindings == nil {
			closure.Bindings = []nativeproof.ELFBinding{}
		}
		for j := range closure.Nodes {
			m := &closure.Nodes[j]
			m.Needed = append([]string{}, m.Needed...)
			m.RPath = append([]string{}, m.RPath...)
			m.RunPath = append([]string{}, m.RunPath...)
			if m.Pin == o.ProviderB {
				t.Fatal("B entered selected closure")
			}
			addInput(m.Pin)
		}
		for j := range closure.Bindings {
			closure.Bindings[j].SearchDirs = append([]string{}, closure.Bindings[j].SearchDirs...)
		}
		startup := map[string]bool{cli.Path: true}
		for changed := true; changed; {
			changed = false
			for _, b := range closure.Bindings {
				if startup[b.From] && !startup[b.Target.Path] {
					startup[b.Target.Path] = true
					changed = true
				}
			}
		}
		if startup[o.ProviderA.Path] {
			t.Fatal("fixture A must be optional, not a startup dependency")
		}
		rootPin := addInput(cli)
		policy := supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: rootPin, Observer: rootPin, Audit: o.Audit, Allowed: []supgpu.RuntimeCode{}, Alternates: []supgpu.RuntimeCode{}, Directories: slices.Clone(o.SearchDirs)}
		for _, m := range closure.Nodes {
			if m.Pin == cli {
				continue
			}
			policy.Allowed = append(policy.Allowed, supgpu.RuntimeCode{Object: addInput(m.Pin), SONAME: m.SONAME, Needed: append([]string{}, m.Needed...), Tier: "metadata-selected", Startup: startup[m.Pin.Path]})
		}
		p := identityWire{Schema: "nanopore-sup-tiny-device-plan/v1", Executable: cli, Seeds: seeds, LazyProviders: []nativeproof.Pin{o.ProviderA}, SearchDirs: slices.Clone(o.SearchDirs), Closure: closure, Environment: supgpu.SelectedModelEnvironment(policy), ExpectedPCI: "00000000:01:00.0", ExpectedDriver: "580.1", TimeoutSeconds: o.TimeoutSeconds}
		p.Args = identityArgs(p, 0)
		identities[i], policies[i] = p, policy
	}
	// Recheck containment with the complete closure before creating any output.
	if e := validateOutputs(root, outputs, inputs); e != nil {
		t.Fatal(e)
	}
	if e := os.Mkdir(o.Output, 0700); e != nil {
		t.Fatal(e)
	}
	generated := []supgpu.Pin{}
	write := func(name string, value any) supgpu.Pin {
		t.Helper()
		path := filepath.Join(o.Output, name)
		if e := writeJSON(path, value); e != nil {
			t.Fatal(e)
		}
		p, e := pinFile(path)
		if e != nil {
			t.Fatal(e)
		}
		v, e := sizedPin(p)
		if e != nil {
			t.Fatal(e)
		}
		generated = append(generated, v)
		return v
	}
	contract := write("metadata-contract.json", supgpu.PhaseContract{Schema: "nanopore-audited-metadata-contract/v1", Role: "observer", Action: "device-identity", Names: []string{}})
	cases := make([]metadataFixtureCase, 2)
	for i, label := range []string{"matched", "mismatched"} {
		ip := write(label+"-identity.json", identities[i])
		custody := append([]supgpu.Pin{}, inputs...)
		custody = append(custody, ip, contract)
		plan := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Policy: policies[i], Contract: contract, Inputs: custody}
		pp := write(label+"-plan.json", plan)
		p, e := readIdentity(unsized(ip))
		if e != nil {
			t.Fatal(e)
		}
		checked, c, e := supgpu.ReadGuardPlan(pp, contract)
		if e != nil {
			t.Fatal(e)
		}
		if c.Schema != "nanopore-audited-metadata-contract/v1" || c.Names == nil || len(c.Names) != 0 {
			t.Fatal("metadata contract roundtrip differs")
		}
		if e := identityPolicy(p, checked, o.Audit); e != nil {
			t.Fatal(e)
		}
		cases[i] = metadataFixtureCase{Identity: unsized(ip), Plan: pp}
	}
	fixture := metadataFixtureWire{Schema: "nanopore-sup-metadata-provider-fixture/v1", Audit: o.Audit, Contract: contract, ProviderA: o.ProviderA, ProviderB: o.ProviderB, Matched: cases[0], Mismatched: cases[1], Directory: root, Output: filepath.Join(o.Output, "control"), MarkerA: o.MarkerA, MarkerB: o.MarkerB}
	runOutputs := []string{fixture.Output + "-original", o.MarkerA, o.MarkerB}
	for _, label := range []string{"matched", "mismatched"} {
		for _, suffix := range []string{"", "-identity", "-identity-before", "-identity-after"} {
			runOutputs = append(runOutputs, fixture.Output+"-"+label+suffix)
		}
	}
	custody := append(append([]supgpu.Pin{}, inputs...), generated...)
	if e := validateOutputs(root, runOutputs, custody); e != nil {
		t.Fatal(e)
	}
	if e := requireFresh(runOutputs...); e != nil {
		t.Fatal(e)
	}
	fp := write("metadata-fixture.json", fixture)
	var roundtrip metadataFixtureWire
	if e := readJSON(unsized(fp), &roundtrip); e != nil {
		t.Fatal(e)
	}
	for _, v := range append(append([]supgpu.Pin{}, inputs...), generated...) {
		actual, e := sizedPin(unsized(v))
		if e != nil || actual != v {
			t.Fatal("final publication custody differs", v.Path, e)
		}
	}
	result := struct {
		Schema          string
		Fixture         nativeproof.Pin
		Inputs, Outputs []supgpu.Pin
	}{"nanopore-sup-metadata-prepared/v1", unsized(fp), inputs, generated}
	b, e := json.Marshal(result)
	if e != nil {
		t.Fatal(e)
	}
	fmt.Println(string(b))
}
