package h05

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

func TestScratchComparisonUsesEntireInitializedSourceTemp(t *testing.T) {
	dir, attempt := t.TempDir(), t.TempDir()
	const size = 1089920
	native := make([]byte, 7782400)
	for i := 0; i < size; i++ {
		native[i] = byte(i % 251)
	}
	path := filepath.Join(dir, "source-backing.bin")
	if err := os.WriteFile(path, native, 0600); err != nil {
		t.Fatal(err)
	}
	stage := hacgpu.Stage{Name: "flstm0", BackingFile: "source-backing.bin", BackingSHA: sum(native)}
	p := prepared{capture: hacgpu.Capture{Batch: 64, TOut: 32, Reservation: 7782400, Stages: []hacgpu.Stage{stage}}, conv: hacgpu.Stage{OffsetBytes: 5357568, Bytes: 2424832}, scratch: model.GPUView{Offset: 2424832}}
	backing := make([]byte, 2424832+size)
	copy(backing[p.scratch.Offset:], native[:size])
	rec, err := observeScratch(p, dir, attempt, 0, backing)
	if err != nil || !rec.Equal || rec.Bytes != size {
		t.Fatalf("source temp comparison %+v %v", rec, err)
	}
	backing[len(backing)-1] ^= 1
	rec, err = observeScratch(p, dir, attempt, 0, backing)
	if err != nil || rec.Equal {
		t.Fatalf("workspace final byte was excluded %+v %v", rec, err)
	}
	native[0] ^= 1
	if err := os.WriteFile(path, native, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := observeScratch(p, dir, attempt, 0, backing); err == nil {
		t.Fatal("changed source backing accepted")
	}
}

func TestCacheOrderEvidenceRetainsRecomputableMismatch(t *testing.T) {
	dir, attempt := t.TempDir(), t.TempDir()
	row := 64 * 1024
	native := make([]byte, 37*row)
	source := hacgpu.Stage{Name: "flstm0", DataFile: "source.bin", DataSHA: sum(native), UnwrittenT: [][2]int{{1, 1}, {35, 2}}}
	if err := os.WriteFile(filepath.Join(dir, "source.bin"), native, 0600); err != nil {
		t.Fatal(err)
	}
	p := prepared{capture: hacgpu.Capture{Batch: 64, TOut: 32, Stages: []hacgpu.Stage{source}}}
	mutant := append([]byte(nil), native...)
	mutant[2*row] = 1
	mutant[row] = 99
	rec, err := retainCacheOrderEvidence(p, dir, attempt, mutant)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Equal || rec.ComparedBytes != 34*row || rec.GotSHA == rec.NativeSHA || rec.FullSHA != sum(mutant) || !reflect.DeepEqual(rec.ExcludedRows, source.UnwrittenT) {
		t.Fatalf("incomplete mutant evidence %+v", rec)
	}
	body, err := json.Marshal(Control{Name: "cache-order", Separated: true, Stage: &rec})
	if err != nil {
		t.Fatal(err)
	}
	var control Control
	if err := json.Unmarshal(body, &control); err != nil {
		t.Fatal(err)
	}
	if control.Stage == nil {
		t.Fatal("control lost stage record")
	}
	if err := verifyControlEvidence(p, dir, *control.Stage); err != nil {
		t.Fatal(err)
	}
	bad := rec
	bad.GotSHA = "changed"
	if err := verifyControlEvidence(p, dir, bad); err == nil {
		t.Fatal("changed comparison digest accepted")
	}
	mutant[3*row] ^= 1
	if err := os.WriteFile(rec.CandidateFile, mutant, 0600); err != nil {
		t.Fatal(err)
	}
	if err := verifyControlEvidence(p, dir, rec); err == nil {
		t.Fatal("mutated retained payload accepted")
	}
}
