package closedbuild

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func artifactFile(t *testing.T, path string, body []byte) Pin {
	t.Helper()
	if e := os.WriteFile(path, body, 0600); e != nil {
		t.Fatal(e)
	}
	p, e := PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}

func historicalFixture(t *testing.T) (Controller, string) {
	t.Helper()
	d := t.TempDir()
	if e := os.Mkdir(filepath.Join(d, "sources"), 0700); e != nil {
		t.Fatal(e)
	}
	source := artifactFile(t, filepath.Join(d, "old.go"), []byte("package old\n"))
	tool := artifactFile(t, filepath.Join(d, "old-go-tool"), []byte("historical tool bytes"))
	list, _ := json.Marshal(goPackage{Dir: d, GoFiles: []string{"old.go"}})
	lp := artifactFile(t, filepath.Join(d, "list.json"), list)
	c := Controller{Schema: "nanopore-tensor-closed-controller/v1", Status: "built", Lists: []Pin{lp, lp}, Commands: make([]runner.CommandReceipt, 5), Binaries: map[string]Pin{"closed": {}, "maintained": {}}, Inputs: map[string]string{source.Path: source.SHA256, tool.Path: tool.SHA256}, Tools: map[string]string{tool.Path: tool.SHA256}, Snapshots: map[string]Pin{}}
	for path, want := range c.Inputs {
		body, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		c.Snapshots[path] = artifactFile(t, filepath.Join(d, "sources", want), body)
	}
	return c, d
}

func TestHistoricalProducerUsesCompleteSnapshotsInsteadOfMutableOriginals(t *testing.T) {
	c, d := historicalFixture(t)
	if e := producerSources(c, d); e != nil {
		t.Fatal(e)
	}
	for path := range c.Inputs {
		if e := os.Remove(path); e != nil {
			t.Fatal(e)
		}
	}
	if e := producerSources(c, d); e != nil {
		t.Fatalf("historical originals were re-read: %v", e)
	}
	for _, mutation := range []struct {
		Name   string
		Change func(*Controller)
	}{
		{"omitted snapshot", func(c *Controller) {
			for key := range c.Snapshots {
				delete(c.Snapshots, key)
				break
			}
		}},
		{"source binding", func(c *Controller) {
			for key, p := range c.Snapshots {
				c.Snapshots[key+".wrong"] = p
				delete(c.Snapshots, key)
				break
			}
		}},
		{"hash binding", func(c *Controller) {
			for key, p := range c.Snapshots {
				p.SHA256 = strings.Repeat("0", 64)
				c.Snapshots[key] = p
				break
			}
		}},
		{"self-consistent foreign source", func(c *Controller) {
			for key := range c.Inputs {
				if strings.HasSuffix(key, ".go") {
					delete(c.Inputs, key)
					delete(c.Snapshots, key)
				}
			}
		}},
	} {
		t.Run(mutation.Name, func(t *testing.T) {
			copy := c
			copy.Inputs = maps.Clone(c.Inputs)
			copy.Snapshots = maps.Clone(c.Snapshots)
			mutation.Change(&copy)
			if producerSources(copy, d) == nil {
				t.Fatal("invalid historical binding accepted")
			}
		})
	}
	for _, p := range c.Snapshots {
		artifactFile(t, p.Path, []byte("changed snapshot"))
		break
	}
	if producerSources(c, d) == nil {
		t.Fatal("modified immutable snapshot accepted")
	}
}

func TestArtifactCommandRejectsChangedRawLogAndRehashedArgv(t *testing.T) {
	d := t.TempDir()
	out := artifactFile(t, filepath.Join(d, "stdout"), []byte("actual output\n"))
	errlog := artifactFile(t, filepath.Join(d, "stderr"), nil)
	args, env := []string{"/compiler", "-M"}, []string{"LANG=C"}
	r := runner.CommandReceipt{Args: args, Environment: env, Exit: 0, Started: "2026-10-25T00:00:01Z", Finished: "2026-10-25T00:00:02Z", Stdout: out.Path, StdoutSHA: out.SHA256, Stderr: errlog.Path, StderrSHA: errlog.SHA256}
	check := func(r runner.CommandReceipt) error {
		return artifactCommand(r, args, env, "2026-10-25T00:00:00Z", "2026-10-25T00:00:03Z")
	}
	if e := check(r); e != nil {
		t.Fatal(e)
	}
	copy := r
	copy.Args = []string{"/compiler", "-MM"}
	if check(copy) == nil {
		t.Fatal("self-rehashed manifest argv accepted")
	}
	artifactFile(t, out.Path, []byte("modified output\n"))
	if check(r) == nil {
		t.Fatal("modified raw log accepted")
	}
}

func TestArtifactIndependentOutputAndPlanAnchorsRejectRehashedManifest(t *testing.T) {
	d := t.TempDir()
	output := Pin{Path: filepath.Join(d, "library.so"), SHA256: strings.Repeat("a", 64)}
	plan := Pin{Path: filepath.Join(d, "plan.json"), SHA256: strings.Repeat("b", 64)}
	for _, mutation := range []string{"wrong output", "wrong plan", "failed status"} {
		t.Run(mutation, func(t *testing.T) {
			b := Build{Schema: "nanopore-tensor-closed-build/v1", Status: "built_outputs_unexecuted", Plan: plan, Outputs: map[string]Pin{"backend": output}, RuntimeHeld: true}
			switch mutation {
			case "wrong output":
				b.Outputs["backend"] = Pin{Path: output.Path + ".foreign", SHA256: output.SHA256}
			case "wrong plan":
				b.Plan.SHA256 = strings.Repeat("c", 64)
			case "failed status":
				b.Status = "failed"
			}
			body, _ := json.Marshal(b)
			bp := artifactFile(t, filepath.Join(d, strings.ReplaceAll(mutation, " ", "-")+".json"), body)
			_, e := ReadArtifact(ArtifactOptions{Build: bp, Plan: plan, Producer: plan, Output: output})
			if e == nil || !strings.Contains(e.Error(), "build/output differs") {
				t.Fatalf("rehashed manifest did not reach anchor guard: %v", e)
			}
		})
	}
}

func TestLinkMapAndTraceResolveSameCanonicalInputs(t *testing.T) {
	d := t.TempDir()
	p := artifactFile(t, filepath.Join(d, "input.o"), []byte("object"))
	a, e := linkPaths(p.Path+"\n", d, false)
	if e != nil {
		t.Fatal(e)
	}
	b, e := linkPaths("LOAD "+p.Path+"\n", d, true)
	if e != nil || !maps.Equal(a, b) {
		t.Fatalf("trace/map differs: %v", e)
	}
	if _, e = linkPaths("not a LOAD record\n", d, true); e == nil {
		t.Fatal("empty map accepted")
	}
}
