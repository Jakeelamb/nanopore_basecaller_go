package decodergpu

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

type proofFixture struct {
	anchor                      File
	inputs                      map[string]string
	alias, target, alternate    string
	image, snapshot, membership File
	loads                       int
}

func newProofFixture(t *testing.T) *proofFixture {
	t.Helper()
	dir := t.TempDir()
	f := &proofFixture{anchor: testFile(t, dir, "proof.json", `{"proof":1}`)}
	f.image = testFile(t, dir, "module.cubin", "image")
	f.snapshot = testFile(t, dir, "consumer.copy", "source")
	f.membership = testFile(t, dir, "membership.json", `{"members":[1]}`)
	first := testFile(t, dir, "first-source", "same source bytes")
	other := testFile(t, dir, "other-source", "same source bytes")
	f.alias, f.target, f.alternate = filepath.Join(dir, "source-alias"), first.Path, other.Path
	if err := os.Symlink(f.target, f.alias); err != nil {
		t.Fatal(err)
	}
	f.inputs = map[string]string{f.anchor.Path: f.anchor.SHA256, f.image.Path: f.image.SHA256, f.snapshot.Path: f.snapshot.SHA256, f.membership.Path: f.membership.SHA256, f.target: first.SHA256}
	return f
}

func (f *proofFixture) read(anchor File) (numericLineage, error) {
	f.loads++
	var metadata map[string]int
	if anchor != f.anchor {
		return numericLineage{}, fmt.Errorf("fixture anchor changed")
	}
	if err := readJSON(anchor, &metadata); err != nil {
		return numericLineage{}, err
	}
	if err := checkPins(f.inputs); err != nil {
		return numericLineage{}, err
	}
	actual, err := canonical(f.alias)
	if err != nil || actual != f.target {
		return numericLineage{}, fmt.Errorf("fixture source alias/membership changed")
	}
	proof := numericLineage{Inputs: maps.Clone(f.inputs), Context: LineageContext{Environment: []string{"frozen-environment"}}}
	proof.Report.Binding.Output = f.image
	return proof, nil
}

func TestNumericProofEpochsRequireFreshPostAuthentication(t *testing.T) {
	t.Run("fifteen-requests-two-complete-walks", func(t *testing.T) {
		f := newProofFixture(t)
		p := &numericProofs{reader: f.read}
		for i := 0; i < 6; i++ {
			if _, err := p.historical(f.anchor); err != nil {
				t.Fatal(err)
			}
		}
		calls := 0
		if _, err := p.native(func() (supgpu.GuardedRun, error) {
			calls++
			if _, e := p.historical(f.anchor); e == nil {
				t.Fatal("pre proof borrowed during child")
			}
			if e := p.finish(); e == nil {
				t.Fatal("proof finished while child active")
			}
			return supgpu.GuardedRun{}, nil
		}, nil); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 9; i++ {
			if _, err := p.historical(f.anchor); err != nil {
				t.Fatal(err)
			}
		}
		if f.loads != 2 || calls != 1 {
			t.Fatalf("one full pre/post walk required: loads=%d child=%d", f.loads, calls)
		}
		if err := p.finish(); err != nil || f.loads != 2 {
			t.Fatal("post authentication repeated", err, f.loads)
		}
		if _, err := p.native(func() (supgpu.GuardedRun, error) { calls++; return supgpu.GuardedRun{}, nil }, nil); err == nil || calls != 1 {
			t.Fatal("second external execution was accepted")
		}
		q := &numericProofs{reader: f.read}
		if _, err := q.historical(f.anchor); err != nil {
			t.Fatal(err)
		}
		if err := q.finish(); err != nil {
			t.Fatal(err)
		}
		if _, err := q.native(func() (supgpu.GuardedRun, error) { calls++; return supgpu.GuardedRun{}, nil }, nil); err == nil || calls != 1 {
			t.Fatal("closed invocation started a child")
		}
	})
	for _, kind := range []string{"metadata", "image", "consumer-snapshot", "source-alias", "membership"} {
		for _, failed := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/failed-child=%t", kind, failed), func(t *testing.T) {
				child := errors.New("native child failed")
				f := newProofFixture(t)
				p := &numericProofs{reader: f.read}
				if _, err := p.historical(f.anchor); err != nil {
					t.Fatal(err)
				}
				_, err := p.native(func() (supgpu.GuardedRun, error) {
					if kind == "source-alias" {
						if e := os.Remove(f.alias); e != nil {
							t.Fatal(e)
						}
						if e := os.Symlink(f.alternate, f.alias); e != nil {
							t.Fatal(e)
						}
					} else {
						file := map[string]File{"metadata": f.anchor, "image": f.image, "consumer-snapshot": f.snapshot, "membership": f.membership}[kind]
						info, e := os.Stat(file.Path)
						if e != nil {
							t.Fatal(e)
						}
						data, e := os.ReadFile(file.Path)
						if e != nil {
							t.Fatal(e)
						}
						data[len(data)-1] ^= 1
						if e = os.WriteFile(file.Path, data, 0600); e != nil {
							t.Fatal(e)
						}
						if e = os.Chtimes(file.Path, info.ModTime(), info.ModTime()); e != nil {
							t.Fatal(e)
						}
						after, e := os.Stat(file.Path)
						if e != nil || after.Size() != info.Size() || !after.ModTime().Equal(info.ModTime()) {
							t.Fatal("same size/time control not effective", e)
						}
					}
					if failed {
						return supgpu.GuardedRun{}, child
					}
					return supgpu.GuardedRun{}, nil
				}, nil)
				if errors.Is(err, child) != failed {
					t.Fatalf("child failure lost: %v", err)
				}
				if err == nil || f.loads != 2 {
					t.Fatalf("post-child mutation accepted: %v loads=%d", err, f.loads)
				}
				if _, e := p.historical(f.anchor); e == nil || f.loads != 2 {
					t.Fatal("denied post epoch reused or retried proof")
				}
			})
		}
	}
}

func TestNumericProofCacheCannotAliasOrContaminate(t *testing.T) {
	f := newProofFixture(t)
	p := &numericProofs{reader: f.read}
	first, err := p.historical(f.anchor)
	if err != nil {
		t.Fatal(err)
	}
	delete(first.Inputs, f.image.Path)
	first.Context.Environment[0] = "changed"
	first.Report.Binding.Output.Path = "different"
	second, err := p.historical(f.anchor)
	if err != nil {
		t.Fatal(err)
	}
	if second.Inputs[f.image.Path] != f.image.SHA256 || second.Context.Environment[0] != "frozen-environment" || second.Report.Binding.Output != f.image || f.loads != 1 {
		t.Fatal("borrower contaminated immutable cache")
	}
	copied := testFile(t, t.TempDir(), "copy.json", `{"proof":1}`)
	if copied.SHA256 != f.anchor.SHA256 {
		t.Fatal("same SHA path control differs")
	}
	for _, anchor := range []File{copied, {Path: f.anchor.Path, SHA256: strings.Repeat("0", 64)}} {
		if _, e := p.historical(anchor); e == nil || f.loads != 1 {
			t.Fatal("different exact anchor reused accepted entry", e, f.loads)
		}
	}
	if _, e := p.read(f.anchor, numericProofMode(99)); e == nil || f.loads != 1 {
		t.Fatal("validation mode aliased accepted entry")
	}
	q := &numericProofs{reader: f.read}
	if _, e := q.historical(f.anchor); e != nil || f.loads != 2 {
		t.Fatal("proof was reused across invocations", e, f.loads)
	}
	failed := errors.New("first load failed")
	tries := 0
	retry := &numericProofs{reader: func(a File) (numericLineage, error) {
		tries++
		if tries == 1 {
			return numericLineage{}, failed
		}
		return f.read(a)
	}}
	if _, e := retry.historical(f.anchor); !errors.Is(e, failed) {
		t.Fatal(e)
	}
	if _, e := retry.historical(f.anchor); e != nil || tries != 2 {
		t.Fatal("failed read became accepted cache entry", e, tries)
	}
	if _, e := readNumericLineage(f.anchor); e == nil {
		t.Fatal("standalone strict reader trusted private fixture cache")
	}
}

func TestNumericProofFailedChildKeepsPostFailure(t *testing.T) {
	f := newProofFixture(t)
	p := &numericProofs{reader: f.read}
	if _, err := p.historical(f.anchor); err != nil {
		t.Fatal(err)
	}
	child := errors.New("native child failed")
	post := errors.New("post proof changed")
	original := p.reader
	p.reader = func(a File) (numericLineage, error) {
		value, e := original(a)
		if e != nil {
			return numericLineage{}, fmt.Errorf("%w: %v", post, e)
		}
		return value, nil
	}
	_, err := p.native(func() (supgpu.GuardedRun, error) {
		if e := os.WriteFile(f.image.Path, []byte("other"), 0600); e != nil {
			t.Fatal(e)
		}
		return supgpu.GuardedRun{}, child
	}, nil)
	if !errors.Is(err, child) || !errors.Is(err, post) || f.loads != 2 {
		t.Fatalf("child/post failure lost: %v loads=%d", err, f.loads)
	}
}

func TestNumericProofCurrentChecksStayFresh(t *testing.T) {
	f := newProofFixture(t)
	p := &numericProofs{reader: f.read}
	if _, err := p.historical(f.anchor); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	source := testFile(t, dir, "current.go", "current consumer")
	bindings, copies, err := snapshotPins(map[string]string{source.Path: source.SHA256}, filepath.Join(dir, "native-inputs"))
	if err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(dir, "native-inputs", "current.go")
	if err = os.WriteFile(alias, []byte("current consumer"), 0400); err != nil {
		t.Fatal(err)
	}
	copies[alias] = source.SHA256
	producer := ClosedBuildPlan{Directory: dir, SourceBindings: bindings, SnapshotPins: copies}
	inputs := map[string]string{source.Path: source.SHA256}
	if err = os.WriteFile(source.Path, []byte("changed consumer"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = p.historical(f.anchor); err != nil || f.loads != 1 {
		t.Fatal(err, f.loads)
	}
	if err = checkProducerInputs(inputs, producer, false); err != nil {
		t.Fatal("historical snapshot rejected", err)
	}
	if err = checkProducerInputs(inputs, producer, true); err == nil {
		t.Fatal("primed historical cache bypassed current source bytes")
	}
	controller := testFile(t, dir, "consumer-binary", "current program")
	if err = os.WriteFile(controller.Path, []byte("another program"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = controller.verify(); err == nil {
		t.Fatal("primed proof bypassed executing consumer identity")
	}
}

func TestNumericProofPostRuntimeChecksRunOnEveryChildOutcome(t *testing.T) {
	for _, childFails := range []bool{false, true} {
		for _, proofFails := range []bool{false, true} {
			t.Run(fmt.Sprintf("child=%t/proof=%t", childFails, proofFails), func(t *testing.T) {
				f := newProofFixture(t)
				p := &numericProofs{reader: f.read}
				if _, err := p.historical(f.anchor); err != nil {
					t.Fatal(err)
				}
				child, runtime := errors.New("child failed"), errors.New("current runtime changed")
				checked := false
				_, err := p.native(func() (supgpu.GuardedRun, error) {
					if proofFails {
						if e := os.WriteFile(f.image.Path, []byte("other"), 0600); e != nil {
							t.Fatal(e)
						}
					}
					if childFails {
						return supgpu.GuardedRun{}, child
					}
					return supgpu.GuardedRun{}, nil
				}, func() error {
					checked = true
					if p.epoch != afterExternal || f.loads != 2 {
						t.Fatal("runtime checked before fresh post epoch")
					}
					return runtime
				})
				if !checked || !errors.Is(err, runtime) || errors.Is(err, child) != childFails {
					t.Fatalf("post runtime/child error lost: %v", err)
				}
				if proofFails && (p.denied == nil || !errors.Is(err, p.denied)) {
					t.Fatal("post proof error lost", err)
				}
			})
		}
	}
}

func TestPinnedMetadataDecodesAuthenticatedBytes(t *testing.T) {
	dir := t.TempDir()
	anchor := testFile(t, dir, "metadata.json", `{"value":1}`)
	raw, err := readPinnedMetadata(anchor, maxPinnedMetadataBytes)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(anchor.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(anchor.Path, []byte(`{"value":2}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.Chtimes(anchor.Path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	var decoded map[string]int
	if err = json.Unmarshal(raw, &decoded); err != nil || decoded["value"] != 1 {
		t.Fatal("parse was not bound to authenticated buffer", err, decoded)
	}
	if err = readJSON(anchor, &decoded); err == nil {
		t.Fatal("same-size/time changed metadata accepted")
	}
	fresh := testFile(t, dir, "bounded.json", `{"value":1}`)
	if _, err = readPinnedMetadata(fresh, 2); err == nil {
		t.Fatal("oversized metadata accepted")
	}
	alias := filepath.Join(dir, "alias.json")
	if err = os.Symlink(fresh.Path, alias); err != nil {
		t.Fatal(err)
	}
	if _, err = readPinnedMetadata(File{Path: alias, SHA256: fresh.SHA256}, maxPinnedMetadataBytes); err == nil {
		t.Fatal("noncanonical metadata alias accepted")
	}
}
