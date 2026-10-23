package runledger

import (
	"errors"
	"math"
	"testing"
)

func ledger(t *testing.T, ids ...string) *Ledger {
	t.Helper()
	requests := make([]Request, len(ids))
	for i, id := range ids {
		requests[i] = Request{"file", id}
	}
	l, err := New([]File{{"file", []string{"parent", "other"}}}, requests)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func reject(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected rejection")
	}
}

func TestReconcileParentsChildrenAndWriters(t *testing.T) {
	l := ledger(t, "parent", "other", "missing")
	must(t, l.Load(0, 100, 80))
	must(t, l.AddChunks(0, 3))
	must(t, l.SetChildren(0, []Child{{"child", "parent", true, ""}, {"filtered", "parent", false, "empty"}}))
	must(t, l.OpenWriter("reads.bam"))
	must(t, l.RecordWrite(0, Write{"child", "parent", "reads.bam"}))
	must(t, l.CloseWriter("reads.bam", 1, ""))
	must(t, l.Complete(0))
	must(t, l.Load(1, 20, 10))
	must(t, l.AddChunks(1, 1))
	must(t, l.SetChildren(1, []Child{{"other", "", false, "empty"}}))
	must(t, l.Complete(1))
	must(t, l.Stop(2, Absent, "not in input"))
	r, err := l.Finalize()
	must(t, err)
	if r.RawSamples != 120 || r.AdmittedSamples != 90 || r.Chunks != 4 || r.Children != 3 || r.Kept != 1 || r.Written != 1 {
		t.Fatalf("wrong totals %+v", r)
	}
	if r.Parents[0].State != ChildBearing || r.Parents[1].State != ZeroSurvivor || r.Parents[2].State != Absent {
		t.Fatalf("wrong dispositions %+v", r.Parents)
	}
	reject(t, l.Load(0, 1, 1))
}

func TestDuplicateFileUUIDAndCorruptInventory(t *testing.T) {
	for _, files := range [][]File{
		{{"file", nil}, {"file", nil}}, {{"", nil}}, {{"file", []string{"p", "p"}}}, {{"file", []string{""}}},
	} {
		_, err := New(files, nil)
		reject(t, err)
	}
	_, err := New(nil, []Request{{"unknown", "parent"}})
	reject(t, err)
}

func TestMissingSelectionAndDuplicateRequests(t *testing.T) {
	l := ledger(t, "missing", "parent", "parent")
	reject(t, l.Load(0, 1, 1))
	must(t, l.Stop(0, Absent, "missing selected ID"))
	must(t, l.Stop(1, Rejected, "unsupported metadata"))
	must(t, l.Stop(2, Failed, "cancelled before load"))
	r, err := l.Finalize()
	must(t, err)
	if len(r.Parents) != 3 || r.Parents[1].Request != r.Parents[2].Request {
		t.Fatal("duplicate request lost")
	}
}

func TestCorruptRowUnsupportedMetadataAndCancellation(t *testing.T) {
	for _, reason := range []string{"corrupt row", "unsupported metadata", "cancelled"} {
		t.Run(reason, func(t *testing.T) {
			l := ledger(t, "parent")
			must(t, l.Load(0, 20, 10))
			must(t, l.AddChunks(0, 2))
			must(t, l.Stop(0, Failed, reason))
			r, err := l.Finalize()
			must(t, err)
			if r.Parents[0].Reason != reason || r.Chunks != 2 {
				t.Fatal("failure lost partial work")
			}
			reject(t, l.Stop(0, Failed, reason))
		})
	}
}

func TestRenamedFilteredDuplicateAndWrongLineageWrites(t *testing.T) {
	l := ledger(t, "parent")
	must(t, l.Load(0, 20, 10))
	must(t, l.SetChildren(0, []Child{{"child", "parent", true, ""}, {"filtered", "parent", false, "empty"}}))
	must(t, l.OpenWriter("reads.bam"))
	for _, w := range []Write{{"renamed", "parent", "reads.bam"}, {"child", "wrong", "reads.bam"}, {"filtered", "parent", "reads.bam"}, {"child", "parent", "unknown"}} {
		reject(t, l.RecordWrite(0, w))
	}
	reject(t, l.Complete(0))
	must(t, l.RecordWrite(0, Write{"child", "parent", "reads.bam"}))
	reject(t, l.RecordWrite(0, Write{"child", "parent", "reads.bam"}))
	reject(t, l.Complete(0))
	must(t, l.CloseWriter("reads.bam", 1, ""))
	must(t, l.Complete(0))
	_, err := l.Finalize()
	must(t, err)
}

func TestWriterFailureRetainsPartialOutput(t *testing.T) {
	l := ledger(t, "parent")
	must(t, l.Load(0, 20, 10))
	must(t, l.SetChildren(0, []Child{{"parent", "", true, ""}}))
	must(t, l.OpenWriter("reads.fastq"))
	must(t, l.RecordWrite(0, Write{"parent", "", "reads.fastq"}))
	reject(t, l.CloseWriter("reads.fastq", 1, "flush failed"))
	reject(t, l.Complete(0))
	must(t, l.Stop(0, Failed, "writer flush failed"))
	r, err := l.Finalize()
	must(t, err)
	if r.Written != 1 || r.Writers[0].Failure == "" || r.Parents[0].State != Failed {
		t.Fatal("writer failure lost")
	}
}

func TestAllFilteredAndEmptyChildrenAreProcessed(t *testing.T) {
	for _, children := range [][]Child{nil, {{"parent", "", false, "empty"}}} {
		l := ledger(t, "parent")
		must(t, l.Load(0, 20, 10))
		must(t, l.SetChildren(0, children))
		must(t, l.Complete(0))
		r, err := l.Finalize()
		must(t, err)
		if r.Parents[0].State != ZeroSurvivor || r.RawSamples != 20 || r.Written != 0 {
			t.Fatal("empty output erased processed work")
		}
	}
}

func TestRejectInvalidTransitionsAndUnfinishedRun(t *testing.T) {
	l := ledger(t, "parent")
	reject(t, l.Load(-1, 0, 0))
	reject(t, l.Load(1, 0, 0))
	reject(t, l.Load(0, 1, 2))
	reject(t, l.Stop(0, Absent, "exists"))
	reject(t, l.Stop(0, Failed, ""))
	reject(t, l.Stop(0, Loaded, "invalid"))
	_, err := l.Finalize()
	reject(t, err)
	must(t, l.Load(0, 20, 10))
	reject(t, l.Load(0, 20, 10))
	reject(t, l.AddChunks(0, 0))
	reject(t, l.Complete(0))
	_, err = l.Finalize()
	reject(t, err)
	must(t, l.OpenWriter("unused"))
	reject(t, l.OpenWriter("unused"))
	must(t, l.Stop(0, Failed, "cancelled"))
	_, err = l.Finalize()
	reject(t, err)
	must(t, l.CloseWriter("unused", 0, ""))
	_, err = l.Finalize()
	must(t, err)
	reject(t, l.OpenWriter("late"))
	reject(t, l.CloseWriter("unused", 0, ""))
	reject(t, l.RecordWrite(0, Write{}))
	_, err = l.Finalize()
	reject(t, err)
}

func TestChildInventoryRejectionIsAtomicAndCopied(t *testing.T) {
	l := ledger(t, "parent", "other")
	must(t, l.Load(0, 1, 1))
	must(t, l.Load(1, 1, 1))
	for _, children := range [][]Child{
		{{"child", "wrong", true, ""}}, {{"parent", "parent", true, ""}},
		{{"child", "parent", true, "reason"}}, {{"child", "parent", false, ""}},
		{{"child", "parent", true, ""}, {"child", "parent", true, ""}},
	} {
		reject(t, l.SetChildren(0, children))
	}
	children := []Child{{"child", "parent", false, "empty"}}
	must(t, l.SetChildren(0, children))
	children[0].Keep = true
	reject(t, l.SetChildren(0, nil))
	reject(t, l.SetChildren(1, []Child{{"child", "other", false, "empty"}}))
	must(t, l.SetChildren(1, nil))
	must(t, l.Complete(0))
	must(t, l.Complete(1))
	_, err := l.Finalize()
	must(t, err)
}

func TestCountOverflowRejected(t *testing.T) {
	l := ledger(t, "parent", "other")
	must(t, l.Load(0, math.MaxUint64, math.MaxUint64))
	must(t, l.AddChunks(0, math.MaxUint64))
	reject(t, l.AddChunks(0, 1))
	must(t, l.Stop(0, Failed, "cancelled"))
	must(t, l.Load(1, 1, 1))
	must(t, l.Stop(1, Failed, "cancelled"))
	_, err := l.Finalize()
	reject(t, err)
}

func TestRepeatedSelectionRetainsRepeatedOutput(t *testing.T) {
	l := ledger(t, "parent", "parent")
	must(t, l.OpenWriter("reads.fastq"))
	for i := 0; i < 2; i++ {
		must(t, l.Load(i, 20, 10))
		must(t, l.SetChildren(i, []Child{{"parent", "", true, ""}}))
		must(t, l.RecordWrite(i, Write{"parent", "", "reads.fastq"}))
	}
	must(t, l.CloseWriter("reads.fastq", 2, ""))
	for i := 0; i < 2; i++ {
		must(t, l.Complete(i))
	}
	r, err := l.Finalize()
	must(t, err)
	if len(r.Parents) != 2 || r.Written != 2 || r.RawSamples != 40 {
		t.Fatal("repeated selection collapsed")
	}
}

func TestInventoryIsCopied(t *testing.T) {
	files := []File{{"file", []string{"parent"}}}
	requests := []Request{{"file", "parent"}}
	l, err := New(files, requests)
	must(t, err)
	files[0].ParentIDs[0] = "changed"
	requests[0].ParentID = "changed"
	must(t, l.Stop(0, Rejected, "unsupported metadata"))
	r, err := l.Finalize()
	must(t, err)
	if r.Files[0].ParentIDs[0] != "parent" || r.Parents[0].Request.ParentID != "parent" {
		t.Fatal("borrowed inventory mutated ledger")
	}
}

func TestCountMismatchRetainsPhysicalClosure(t *testing.T) {
	for _, failure := range []string{"", "flush failed"} {
		l := ledger(t, "parent")
		must(t, l.Load(0, 20, 10))
		must(t, l.SetChildren(0, []Child{{"parent", "", true, ""}}))
		must(t, l.OpenWriter("out"))
		must(t, l.RecordWrite(0, Write{"parent", "", "out"}))
		closureErr := l.CloseWriter("out", 0, failure)
		reject(t, closureErr)
		var closure *WriterClosureError
		if !errors.As(closureErr, &closure) || closure.Observed != 0 || closure.Recorded != 1 || closure.Failure != failure {
			t.Fatalf("missing typed closure observation %v", closureErr)
		}
		reject(t, l.CloseWriter("out", 1, ""))
		reject(t, l.RecordWrite(0, Write{"parent", "", "out"}))
		reject(t, l.Complete(0))
		_, err := l.Finalize()
		reject(t, err)
		must(t, l.Stop(0, Failed, "cancelled after close failure"))
		r, err := l.Finalize()
		must(t, err)
		w := r.Writers[0]
		if !w.Closed || !w.Mismatch || w.Records != 0 || w.Recorded != 1 || w.Failure != failure || r.Written != 1 || r.Parents[0].State != Failed {
			t.Fatalf("lost physical closure or failure %+v", r)
		}
	}
}

func TestChunksFrozenAfterChildInventory(t *testing.T) {
	for _, children := range [][]Child{nil, {{"parent", "", true, ""}}} {
		l := ledger(t, "parent")
		must(t, l.Load(0, 20, 10))
		must(t, l.AddChunks(0, 1))
		must(t, l.SetChildren(0, children))
		reject(t, l.AddChunks(0, 1))
		if len(children) > 0 {
			must(t, l.OpenWriter("out"))
			must(t, l.RecordWrite(0, Write{"parent", "", "out"}))
			must(t, l.CloseWriter("out", 1, ""))
			reject(t, l.AddChunks(0, 1))
		}
		must(t, l.Complete(0))
		r, err := l.Finalize()
		must(t, err)
		if r.Chunks != 1 {
			t.Fatal("late chunks changed accounting")
		}
	}
}
