package gpubatch

import (
	"reflect"
	"testing"
)

func TestHACLSTMAuxiliaryGeometry(t *testing.T) {
	cfg := hacConfig(true)
	cfg.ChunkSize = 12288
	a, err := NewAux(cfg, []int{768, 1536, 12288})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := a.ChunkTable(), [][2]int{{0, 128}, {128, 256}, {384, 2048}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("table = %v want %v", got, want)
	}
	if got, want := a.Intervals(), [][2]int{{0, 768}, {768, 2304}, {2304, 14592}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("intervals = %v want %v", got, want)
	}
	if a.Total() != 2432 || a.Max() != 131072 {
		t.Fatalf("total=%d max=%d", a.Total(), a.Max())
	}
	if got, want := a.SizesOut(), []int{128, 256, 2048}; !reflect.DeepEqual(got, want) {
		t.Fatalf("out sizes = %v want %v", got, want)
	}
	if a.TLSTM() != 2050 {
		t.Fatalf("T_lstm = %d", a.TLSTM())
	}
	chunkSum, outLayout, fwd, bwd, err := a.LSTMLayout()
	if err != nil || chunkSum != 2432 || outLayout != 131264 || fwd != 131200 || bwd != 131264 {
		t.Fatalf("layout = %d %d %d %d %v", chunkSum, outLayout, fwd, bwd, err)
	}
	if ws, err := a.Workspace(); err != nil || ws != 787584 {
		t.Fatalf("workspace = %d %v", ws, err)
	}
	restored, err := a.RestoredIntervals()
	if err != nil || restored[0] != [2]int{0, 4608} {
		t.Fatalf("restored interval = %v %v", restored, err)
	}
	if _, _, _, _, err := a.LSTMLayout(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.DeviceTable(); err == nil {
		t.Fatal("HAC device table accepted")
	}
	if _, err := a.QKVRopeLUTSize(); err == nil {
		t.Fatal("HAC QKV LUT accepted")
	}
}

func TestSUPAuxiliaryGeometry(t *testing.T) {
	cfg := supConfig(true)
	cfg.ChunkSize = 12288
	a, err := NewAux(cfg, []int{768, 1536, 12288})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := a.ChunkTable(), [][2]int{{0, 1}, {1, 2}, {3, 16}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("table = %v want %v", got, want)
	}
	if got, want := a.Intervals(), [][2]int{{0, 768}, {768, 2304}, {2304, 14592}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("intervals = %v want %v", got, want)
	}
	if a.Total() != 19 || a.Max() != 512 {
		t.Fatalf("total=%d max=%d", a.Total(), a.Max())
	}
	table, err := a.DeviceTable()
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{0, 768, 768, 1536, 2304, 12288}; !reflect.DeepEqual(table, want) {
		t.Fatalf("device table = %v want %v", table, want)
	}
	if size, err := a.QKVRopeLUTSize(); err != nil || size != 20 {
		t.Fatalf("QKV LUT = %d %v", size, err)
	}
	if _, _, _, _, err := a.LSTMLayout(); err == nil {
		t.Fatal("SUP LSTM layout accepted")
	}
	if _, err := a.RestoredIntervals(); err == nil {
		t.Fatal("SUP restored intervals accepted")
	}
}

func TestAuxRejectsInvalidChunkUnits(t *testing.T) {
	cfg := supConfig(true)
	for _, sizes := range [][]int{nil, {}, {0}, {767}, {1536 + 1}, {12288 + 768}} {
		if _, err := NewAux(cfg, sizes); err == nil {
			t.Fatalf("sizes %v accepted", sizes)
		}
	}
	over := make([]int, 33)
	for i := range over {
		over[i] = 768
	}
	if _, err := NewAux(cfg, over); err == nil {
		t.Fatal("total granularity beyond maximum accepted")
	}
}

func TestAuxRejectsFixedBatch(t *testing.T) {
	if _, err := NewAux(supConfig(false), []int{768}); err == nil {
		t.Fatal("fixed batch auxiliary data accepted")
	}
}

func TestAuxTablesDoNotAlias(t *testing.T) {
	cfg := supConfig(true)
	a, err := NewAux(cfg, []int{768})
	if err != nil {
		t.Fatal(err)
	}
	table := a.ChunkTable()
	table[0][0] = 99
	if a.ChunkTable()[0][0] != 0 {
		t.Fatal("chunk table aliases auxiliary state")
	}
	intervals := a.Intervals()
	intervals[0][1] = 99
	if a.Intervals()[0][1] != 768 {
		t.Fatal("intervals alias auxiliary state")
	}
}

func TestAuxDistinctBatchesDoNotShareTables(t *testing.T) {
	cfg := supConfig(true)
	first, err := NewAux(cfg, []int{768})
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewAux(cfg, []int{768, 768})
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(first.ChunkTable(), second.ChunkTable()) {
		t.Fatal("two batches share a table")
	}
}
