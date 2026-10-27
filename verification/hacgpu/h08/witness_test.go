package h08

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func TestFixedStageContractsAreProspectiveAndComplete(t *testing.T) {
	stages, err := stageContracts()
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 13 || stages[0].Name != "input" || stages[12].Name != "scores" {
		t.Fatal("missing graph stages")
	}
	for _, s := range stages {
		if _, err = definedRanges(s); err != nil {
			t.Fatal(s.Name, err)
		}
	}
	count := func(s StageContract) int64 {
		var n int64
		for _, r := range s.Defined {
			n += r.Bytes
		}
		return n
	}
	if count(stages[2]) != 64*192*16*2 || count(stages[3]) != 32*64*1024 || count(stages[4]) != 35*64*1024 || count(stages[12]) != 4194304 {
		t.Fatal("wrong source-initialized byte extents")
	}
	if stages[4].NativeUnwritten[0] != [2]int{1, 1} || stages[4].Excluded[0] != [2]int{35, 2} {
		t.Fatal("preserved initialized row1 was silently excluded")
	}
	if stages[3].Offset != 5357568 || stages[11].Offset != 524288 {
		t.Fatal("working-memory alias offsets differ")
	}
}

func TestFixedStageMaskRejectsDefinedChangesAndKeepsDeclaredExclusions(t *testing.T) {
	s := StageContract{Name: "fixture", Layout: "NTC", DType: "Half", Shape: []int64{2, 3, 1}, Bytes: 12, Excluded: [][2]int{{1, 1}}}
	var err error
	s.Defined, err = definedRanges(s)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte{10, 0, 11, 0, 12, 0, 20, 0, 21, 0, 22, 0}
	before, err := selectedBytes(s, body)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Clone(body)
	changed[2] ^= 1
	after, err := selectedBytes(s, changed)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("prospectively excluded row compared")
	}
	changed[0] ^= 1
	after, err = selectedBytes(s, changed)
	if err != nil || bytes.Equal(before, after) {
		t.Fatal("defined byte mutation accepted")
	}
	first, err := stageSelection(s, body, true)
	if err != nil || !bytes.Equal(first, []byte{10, 0, 12, 0}) {
		t.Fatalf("first NTC slot: %v %v", first, err)
	}
	s = StageContract{Name: "fixture", Layout: "CUTLASS_TNC_I8", DType: "Char", Shape: []int64{3, 2, 1}, Bytes: 6, Excluded: [][2]int{{1, 1}}}
	s.Defined, err = definedRanges(s)
	if err != nil {
		t.Fatal(err)
	}
	first, err = stageSelection(s, []byte{10, 20, 11, 21, 12, 22}, true)
	if err != nil || !bytes.Equal(first, []byte{10, 12}) {
		t.Fatalf("first CUTLASS slot: %v %v", first, err)
	}
	s.Excluded = [][2]int{{0, 3}}
	if _, err = definedRanges(s); err == nil {
		t.Fatal("empty comparison silently accepted")
	}
}

func TestFixedCacheComparatorRejectsSameShapedForeignPayload(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	contract := CacheContract{Role: "flstm0.dn-ih-cache", DType: tensor.DTypeI8, Shape: []int64{2, 2}, Bytes: 4, SHA256: digest(data)}
	c := model.HACGPUCacheTensor{Role: contract.Role, DType: contract.DType, Shape: []int64{2, 2}, Data: data}
	if err := verifyCache(c, contract); err != nil {
		t.Fatal(err)
	}
	c.Data = []byte{5, 6, 7, 8}
	if err := verifyCache(c, contract); err == nil {
		t.Fatal("foreign same-shaped cache accepted")
	}
	c.Data = data
	c.Shape = []int64{1, 4}
	if err := verifyCache(c, contract); err == nil {
		t.Fatal("shape metadata collapsed into element count")
	}
}

func TestFixedConsumerAdmissionPrecedesLockAndNativeLoader(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "absent")
	if _, err := Consume(context.Background(), ConsumerOptions{}); err == nil {
		t.Fatal("unadmitted native phase accepted")
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("unadmitted phase created an attempt")
	}
	if _, err := Consume(context.Background(), ConsumerOptions{}); !errors.Is(err, ErrRuntimeHeld) {
		t.Fatalf("structural runtime hold missing: %v", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("held phase created an attempt")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Consume(ctx, ConsumerOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled phase reached native boundary: %v", err)
	}
}

func TestFixedRetainedInputControlUsesIndependentDigest(t *testing.T) {
	p := filepath.Join(t.TempDir(), "copy.bin")
	original := []byte{1, 2}
	if err := os.WriteFile(p, original, 0600); err != nil {
		t.Fatal(err)
	}
	pin := Pin{Path: p, SHA256: digest(original)}
	if _, err := readPin(pin); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte{2, 2}, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readPin(pin); err == nil {
		t.Fatal("changed retained input accepted")
	}
}

func TestBackendSelectionKeepsOutputIdentityEvenWithEqualBytes(t *testing.T) {
	dir := t.TempDir()
	first, second := filepath.Join(dir, "admitted.so"), filepath.Join(dir, "foreign.so")
	for _, path := range []string{first, second} {
		if err := os.WriteFile(path, []byte("same bytes"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	p, err := closedbuild.PinFile(first)
	if err != nil {
		t.Fatal(err)
	}
	_, err = AuthenticateBackend(closedbuild.ArtifactOptions{Output: p}, second)
	if err == nil || !strings.Contains(err.Error(), "selected backend differs") {
		t.Fatalf("foreign output reached producer reader: %v", err)
	}
}
