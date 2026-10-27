package decodergpu

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestLineageSharedSerializedBoundaryAndProspectivePolicy(t *testing.T) {
	dir := t.TempDir()
	binary := testFile(t, dir, "probe", "fake host metadata")
	audit := testFile(t, dir, "audit.so", "fake host auditor")
	provider := testFile(t, dir, "provider.so", "fake host provider")
	binaryPin, _ := guardPin(binary)
	auditPin, _ := guardPin(audit)
	providerPin, _ := guardPin(provider)
	r := LineageRuntime{WorkingDirectory: dir, Output: filepath.Join(dir, "output"), Custody: map[string]string{binary.Path: binary.SHA256}, Policy: supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: binaryPin, Observer: binaryPin, Audit: auditPin, Allowed: []supgpu.RuntimeCode{{Object: providerPin, SONAME: "provider.so", Needed: []string{}, Tier: "fake-host"}}, Alternates: []supgpu.RuntimeCode{}, Directories: []string{dir}}}
	phasePath := filepath.Join(dir, "phases.json")
	if err := writeJSON(phasePath, lineagePhaseContract()); err != nil {
		t.Fatal(err)
	}
	r.Phases, _ = artifactPin(phasePath)
	b := CompilerPhase{Output: binary}
	g, err := lineageGuardPlan(r)
	if err != nil {
		t.Fatal(err)
	}
	guardPath := filepath.Join(dir, "guard.json")
	if err = writeJSON(guardPath, g); err != nil {
		t.Fatal(err)
	}
	r.Guard, _ = artifactPin(guardPath)
	if err = lineageSerializedGuard(r, b); err != nil {
		t.Fatal(err)
	}
	g.Policy.Allowed[0].Needed = nil
	badPath := filepath.Join(dir, "null-needed.json")
	if err = writeJSON(badPath, g); err != nil {
		t.Fatal(err)
	}
	r.Guard, _ = artifactPin(badPath)
	if err = lineageSerializedGuard(r, b); err == nil || !strings.Contains(err.Error(), "policy: allowed: needed: null []string") {
		t.Fatalf("wrong serialized policy rejection: %v", err)
	}
	if !reflect.DeepEqual(lineagePhaseContract().Names, []string{"pre-model", "pre-load", "post-load", "post-lookup", "post-cleanup"}) {
		t.Fatal("exact finite phase contract differs")
	}
	for _, phase := range []string{"query", "headers", "build"} {
		if _, err = RunLineageCompiler(CompilerPhaseOptions{Phase: phase}); err == nil {
			t.Fatal("unadmitted compiler command accepted")
		}
	}
	if _, err = RunLineageLoad(LineageLoadOptions{}); err == nil {
		t.Fatal("unadmitted load accepted")
	}
	p := LineagePlan{Recipe: LineageRecipe{Inputs: map[string]string{"driver": "a"}}, Compiler: CompilerClosure{Inputs: map[string]string{"compiler": "b"}}, Inputs: map[string]string{"libkoi.a": "provenance"}}
	if _, ok := lineageLinkInputs(p)["libkoi.a"]; ok {
		t.Fatal("provenance archive entered linker allowance")
	}
	for _, name := range []string{"libkoi.a", "libcudart.so", "libtorch.so", "libc10.so", "libcudnn.so"} {
		if err = lineageLinkMap("LOAD "+filepath.Join(dir, name), dir, map[string]string{}, map[string]string{}); err == nil {
			t.Fatal("forbidden link input accepted")
		}
	}
	c, o := lineageFixture()
	c.ExpectedOutput.SHA256 = strings.Repeat("0", 64)
	if _, err = BindLineage(c, o); err == nil {
		t.Fatal("expected whole output was replaced after observation")
	}
}
func TestLineageBothOriginalInputsRetainArchitectureAmbiguity(t *testing.T) {
	a := LineageLoadReceipt{Status: "guarded_load_lineage_reproduced_only", Binding: LineageBinding{Architecture: "sm_120", Output: File{"/a", lineageOutputSHA}}}
	b := a
	b.Binding.Architecture = "sm_120a"
	b.Binding.Output.Path = "/b"
	status, err := CombineLineage(a, b)
	if err != nil || status != "both_original_inputs_reproduce_output_source_architecture_ambiguous" {
		t.Fatalf("ambiguity discarded: %s %v", status, err)
	}
	b.Binding.Architecture = "sm_120"
	if _, err = CombineLineage(a, b); err == nil {
		t.Fatal("one original counted twice")
	}
	b.Binding.Architecture = "sm_120a"
	b.Status = "load_lineage_pending_supervisor"
	if _, err = CombineLineage(a, b); err == nil {
		t.Fatal("pending supervision was accepted")
	}
}
