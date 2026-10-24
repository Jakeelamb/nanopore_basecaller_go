package h03

import (
	"bytes"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

// hacV6Params is the pinned convolution stack of
// dna_r10.4.1_e8.2_400bps_hac@v6.0.0, in source order.
func hacV6Params() []model.SmallConvParams {
	return []model.SmallConvParams{
		{CIn: 1, COut: 16, WinLen: 5, Stride: 1, Activation: model.SmallConvSwish, OutputLayout: model.GPUStorageNTC},
		{CIn: 16, COut: 16, WinLen: 5, Stride: 1, Activation: model.SmallConvSwish, OutputLayout: model.GPUStorageNTC},
		{CIn: 16, COut: 1024, WinLen: 19, Stride: 6, Activation: model.SmallConvTanh, OutputLayout: model.GPUStorageCUTLASSI8},
	}
}

func pinnedPlan(t *testing.T) (*model.HACSmallConvPlan, *model.HACLastConvPlan) {
	t.Helper()
	stack, err := model.PlanHACSmallConvs(64, 192, 1, 7782400, hacV6Params())
	if err != nil {
		t.Fatal(err)
	}
	last, err := model.PlanHACLastConv(stack, true)
	if err != nil {
		t.Fatal(err)
	}
	return stack, last
}

func conv2Stage() hacgpu.Stage {
	return hacgpu.Stage{
		Name: "conv2", DType: "Char", Layout: "CUTLASS_TNC_I8",
		Shape: []int64{37, 64, 1024}, OffsetBytes: 5357568, Bytes: 2424832,
		UnwrittenT: [][2]int{{0, 1}, {33, 4}},
	}
}

// TestRecordLastConvPinnedAlias pins the interval arithmetic that places the
// last allocation over conv0's: conv2's allocation starts at 5357568, conv0's at
// 7389184, one row is 65536 bytes, so row 31 is the first shared row and the six
// conv0 rows occupy rows 31..36. The view writes rows 1..32, so it overwrites
// rows 31 and 32 and leaves 33..36 holding conv0's output.
func TestRecordLastConvPinnedAlias(t *testing.T) {
	stack, last := pinnedPlan(t)
	rec, err := recordLastConv(stack, last, conv2Stage())
	if err != nil {
		t.Fatal(err)
	}
	if rec.OutputOffsetBytes != 5357568 {
		t.Fatalf("output offset %d", rec.OutputOffsetBytes)
	}
	if rec.ViewOffsetElements != 65536 {
		t.Fatalf("view offset %d elements", rec.ViewOffsetElements)
	}
	if !slices.Equal(rec.ViewShape, []int64{64, 32, 1024}) {
		t.Fatalf("view shape %v", rec.ViewShape)
	}
	if !slices.Equal(rec.ViewStrides, []int64{1024, 65536, 1}) {
		t.Fatalf("view strides %v", rec.ViewStrides)
	}
	if !sameUnwrittenRows(rec.SourceUnwritten, [][2]int{{0, 1}, {33, 4}}) {
		t.Fatalf("source unwritten %v", rec.SourceUnwritten)
	}
	if rec.InNStride != 3360 || rec.OutNStride != 1024 || rec.OutTStride != 65536 {
		t.Fatalf("strides in_n %d out_n %d out_t %d", rec.InNStride, rec.OutNStride, rec.OutTStride)
	}
	if len(rec.Aliases) != 1 {
		t.Fatalf("%d aliases, want 1", len(rec.Aliases))
	}
	a := rec.Aliases[0]
	if a.Layer != 0 || a.FirstSharedRow != 31 || a.SharedRows != 6 {
		t.Fatalf("alias %+v", a)
	}
	if !sameUnwrittenRows(a.OverwrittenRows, [][2]int{{31, 2}}) {
		t.Fatalf("overwritten rows %v", a.OverwrittenRows)
	}
	if !sameUnwrittenRows(a.UntouchedRows, [][2]int{{33, 4}}) {
		t.Fatalf("untouched rows %v", a.UntouchedRows)
	}
}

// TestRecordLastConvRejectsChangedCapture proves the plan's exclusion set is
// declared from the source and must equal the capture's before execution. A
// capture that declared a different set is a rejection, not a new mask.
func TestRecordLastConvRejectsChangedCapture(t *testing.T) {
	stack, last := pinnedPlan(t)
	stage := conv2Stage()
	stage.UnwrittenT = [][2]int{{0, 1}, {34, 3}}
	if _, err := recordLastConv(stack, last, stage); err == nil {
		t.Fatal("a capture with a different unwritten set was accepted")
	}
}

// TestTransposeHalf pins the exact byte relation between the H02 packing
// ([W*C_in, C_out]) and the H03 packing ([C_out, W*C_in]).
func TestTransposeHalf(t *testing.T) {
	const rows, cols = 3, 2
	src := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	got := transposeHalf(src, rows, cols)
	want := []byte{0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11}
	if !bytes.Equal(got, want) {
		t.Fatalf("transpose %v, want %v", got, want)
	}
	if !bytes.Equal(transposeHalf(got, cols, rows), src) {
		t.Fatal("transpose is not an involution")
	}
}

// TestCheckUnwrittenPinnedLayout builds the declared allocation and proves each
// unwritten row is accounted for by the preparation it was declared with: row 0
// still fill, rows 33..36 still holding conv0's output, and the two rows the view
// covers no longer holding it.
func TestCheckUnwrittenPinnedLayout(t *testing.T) {
	stack, last := pinnedPlan(t)
	rec, err := recordLastConv(stack, last, conv2Stage())
	if err != nil {
		t.Fatal(err)
	}
	alias := rec.Aliases[0]
	const rowBytes = 64 * 1024
	earlier := bytes.Repeat([]byte{0x11}, int(alias.EarlierBytes))
	whole := bytes.Repeat([]byte{FillPattern}, int(stack.ReservationBytes))
	copy(whole[rec.OutputOffsetBytes+alias.FirstSharedRow*rowBytes:], earlier)
	// Rows 31 and 32 are written by the view, so they no longer hold conv0.
	for i := int64(0); i < 2; i++ {
		row := whole[rec.OutputOffsetBytes+(alias.FirstSharedRow+i)*rowBytes:]
		for j := range row[:rowBytes] {
			row[j] = byte(j)
		}
	}
	got, err := checkUnwritten(whole, earlier, alias, last, last.Output.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if !got.RowZeroFill || !got.StillFill {
		t.Fatalf("receipt %+v", got)
	}
	if !sameUnwrittenRows(got.AliasedRows, [][2]int{{33, 1}, {34, 1}, {35, 1}, {36, 1}}) {
		t.Fatalf("aliased rows %v", got.AliasedRows)
	}
	if got.AliasedBytes != 4*rowBytes {
		t.Fatalf("aliased bytes %d", got.AliasedBytes)
	}

	// A changed aliased row is a rejection: the row is declared unwritten but it
	// is not the earlier output any more.
	broken := append([]byte(nil), whole...)
	broken[rec.OutputOffsetBytes+34*rowBytes] ^= 0xff
	if _, err := checkUnwritten(broken, earlier, alias, last, last.Output.Bytes); err == nil {
		t.Fatal("a changed aliased row was accepted")
	}

	// So is a written row 0.
	broken = append([]byte(nil), whole...)
	broken[rec.OutputOffsetBytes] = 0
	if _, err := checkUnwritten(broken, earlier, alias, last, last.Output.Bytes); err == nil {
		t.Fatal("a written row 0 was accepted")
	}

	// And so is a written row that still holds the earlier output.
	broken = append([]byte(nil), whole...)
	copy(broken[rec.OutputOffsetBytes+31*rowBytes:], earlier[:rowBytes])
	if _, err := checkUnwritten(broken, earlier, alias, last, last.Output.Bytes); err == nil {
		t.Fatal("an overwritten row that still matches was accepted")
	}
}

// TestCheckFillSpans pins the only span that must still be the fill pattern for
// the declared case: between conv1's allocation and the last allocation, nothing
// writes.
func TestCheckFillSpans(t *testing.T) {
	stack, last := pinnedPlan(t)
	whole := bytes.Repeat([]byte{FillPattern}, int(stack.ReservationBytes))
	rec, err := checkFill(whole, stack, last)
	if err != nil {
		t.Fatal(err)
	}
	if !rec.BackingStillFill {
		t.Fatal("fill proof failed")
	}
	if !slices.Equal(rec.FillSpans, [][2]int64{{430080, 5423104}}) {
		t.Fatalf("fill spans %v", rec.FillSpans)
	}
	whole[3000000] = 0
	if _, err := checkFill(whole, stack, last); err == nil {
		t.Fatal("a written byte outside every declared view was accepted")
	}
}

func TestSubtract(t *testing.T) {
	got := subtract(0, 100, [][2]int64{{10, 20}, {50, 60}})
	want := [][2]int64{{0, 10}, {20, 50}, {60, 100}}
	if !slices.Equal(got, want) {
		t.Fatalf("subtract %v, want %v", got, want)
	}
	if got := subtract(0, 10, [][2]int64{{-5, 15}}); len(got) != 0 {
		t.Fatalf("fully covered span left %v", got)
	}
}

// TestVisitWrittenRowsCutlassOrder pins the CUTLASS row order: the time axis
// leads, so a time row is one contiguous N*C block and the declared unwritten
// time rows are skipped wherever they appear.
func TestVisitWrittenRowsCutlassOrder(t *testing.T) {
	stage := hacgpu.Stage{
		DType: "Char", Layout: "CUTLASS_TNC_I8",
		Shape: []int64{4, 2, 3}, Bytes: 4 * 2 * 3,
		UnwrittenT: [][2]int{{0, 1}, {3, 1}},
	}
	payload := make([]byte, 24)
	for i := range payload {
		payload[i] = byte(i)
	}
	got, err := visitWrittenRows(stage, payload)
	if err != nil {
		t.Fatal(err)
	}
	// Time rows 1 and 2 are written, each 2*3 bytes.
	want := append(append([]byte{}, payload[6:12]...), payload[12:18]...)
	if !bytes.Equal(got, want) {
		t.Fatalf("written rows %v, want %v", got, want)
	}
}

// TestVisitWrittenRowsNTCOrder pins the NTC order: batches are contiguous
// blocks and the time axis is the middle one.
func TestVisitWrittenRowsNTCOrder(t *testing.T) {
	stage := hacgpu.Stage{
		DType: "Half", Layout: "NTC",
		Shape: []int64{2, 3, 2}, Bytes: 2 * 3 * 2 * 2,
		UnwrittenT: [][2]int{{0, 1}, {2, 1}},
	}
	payload := make([]byte, 24)
	for i := range payload {
		payload[i] = byte(i)
	}
	got, err := visitWrittenRows(stage, payload)
	if err != nil {
		t.Fatal(err)
	}
	want := append(append([]byte{}, payload[4:8]...), payload[16:20]...)
	if !bytes.Equal(got, want) {
		t.Fatalf("written rows %v, want %v", got, want)
	}
}
