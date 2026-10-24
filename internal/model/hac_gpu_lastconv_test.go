package model

import (
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

// TestPackLastConvWeightsLayout pins the source's transpose chain
//
//	weight [C_out, C_in, W] -> permute({2,1,0}) -> flatten(0,1) -> to(kF16)
//	                        -> transpose(0,1) -> contiguous   (ConvStack.cpp:230-236)
//
// The result is [C_out, W*C_in] and element [co][w][ci] lives at
// co*(W*C_in)+w*C_in+ci. A packer that emitted the H02 layout instead would put
// each value at a different index, so the mapping is asserted directly rather
// than only through a digest.
func TestPackLastConvWeightsLayout(t *testing.T) {
	const cOut, cIn, win = 2, 3, 3
	weight := make([]float32, cOut*cIn*win)
	for i := range weight {
		weight[i] = float32(i + 1)
	}
	got, err := PackLastConvWeights(weight, cOut, cIn, win)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != cOut*win*cIn {
		t.Fatalf("packed %d elements, want %d", len(got), cOut*win*cIn)
	}
	for co := 0; co < cOut; co++ {
		for w := 0; w < win; w++ {
			for ci := 0; ci < cIn; ci++ {
				want := signal.Float16Bits(weight[(co*cIn+ci)*win+w])
				idx := co*(win*cIn) + w*cIn + ci
				if bits := got[idx]; bits != want {
					t.Fatalf("packed[%d][%d][%d] = %#04x, want %#04x", co, w, ci, bits, want)
				}
			}
		}
	}
}

func TestPackLastConvWeightsRejectsShape(t *testing.T) {
	if _, err := PackLastConvWeights([]float32{1, 2, 3}, 2, 3, 3); err == nil {
		t.Fatal("short weight accepted")
	}
	if _, err := PackLastConvWeights(nil, 0, 1, 1); err == nil {
		t.Fatal("zero geometry accepted")
	}
}

// TestPlanHACLastConvPinnedGeometry pins the final layer's placement and view
// for the declared HAC v6 case. The allocation is [37, 64, 1024] I8 at offset
// 5357568; the view starts one row in and transposes, so its strides are
// [1024, 65536, 1] and it is not contiguous.
func TestPlanHACLastConvPinnedGeometry(t *testing.T) {
	stack, err := PlanHACSmallConvs(64, 192, 1, 7782400, hacV6Convs())
	if err != nil {
		t.Fatal(err)
	}
	plan, err := PlanHACLastConv(stack, true)
	if err != nil {
		t.Fatal(err)
	}
	if plan.InputOffsetBytes != 0 || plan.OutputOffsetBytes != 5357568 {
		t.Fatalf("input offset %d output offset %d", plan.InputOffsetBytes, plan.OutputOffsetBytes)
	}
	if got := plan.Output.Rows; got != 37 {
		t.Fatalf("allocation rows %d, want T_out+5=37", got)
	}
	if got := plan.OutputViewOffsetElements(); got != 65536 {
		t.Fatalf("view offset %d elements, want 64*1024", got)
	}
	if got := plan.OutputViewShape(); !slices.Equal(got, []int64{64, 32, 1024}) {
		t.Fatalf("view shape %v", got)
	}
	if got := plan.OutputViewStrides(); !slices.Equal(got, []int64{1024, 65536, 1}) {
		t.Fatalf("view strides %v", got)
	}
	if got := plan.UnwrittenRows(); !slices.Equal(got, [][2]int{{0, 1}, {33, 4}}) {
		t.Fatalf("unwritten rows %v, want [[0,1],[33,4]]", got)
	}

	g, err := plan.Geometry(DefaultLastConvDevOptions())
	if err != nil {
		t.Fatal(err)
	}
	if g.N != 64 || g.TIn != 192 || g.CIn != 16 || g.COut != 1024 || g.Window != 19 ||
		g.Stride != 6 || g.Padding != 9 {
		t.Fatalf("geometry %+v", g)
	}
	if g.InNStride != 3360 || g.OutNStride != 1024 || g.OutTStride != 65536 {
		t.Fatalf("strides in_n %d out_n %d out_t %d", g.InNStride, g.OutNStride, g.OutTStride)
	}
	if g.Activation != 3 || g.InType != 5 || g.OutType != 2 || g.AccumType != 5 {
		t.Fatalf("types act %d in %d out %d accum %d", g.Activation, g.InType, g.OutType, g.AccumType)
	}
}

// TestPlanHACLastConvRevLSTMStart pins the other narrow branch: when the output
// feeds a reverse LSTM the view starts at row 1, otherwise row 2.
func TestPlanHACLastConvRevLSTMStart(t *testing.T) {
	stack, err := PlanHACSmallConvs(64, 192, 1, 7782400, hacV6Convs())
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := PlanHACLastConv(stack, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := fwd.OutputViewOffsetElements(); got != 2*64*1024 {
		t.Fatalf("forward view offset %d elements, want 2*N*C", got)
	}
	if got := fwd.UnwrittenRows(); !slices.Equal(got, [][2]int{{0, 2}, {34, 3}}) {
		t.Fatalf("forward unwritten rows %v", got)
	}
}

// TestPlanHACLastConvRejects proves the plan refuses a stack it does not model
// rather than guessing: a non-CUTLASS last layout, an input length that is not
// `wm.T - 2*padding`, and a channel mismatch.
func TestPlanHACLastConvRejects(t *testing.T) {
	base := hacV6Convs()
	if _, err := PlanHACLastConv(nil, true); err == nil {
		t.Fatal("nil plan accepted")
	}

	ntc := append([]SmallConvParams(nil), base...)
	ntc[2].OutputLayout = GPUStorageNTC
	stack, err := PlanHACSmallConvs(64, 192, 1, 7782400, ntc)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PlanHACLastConv(stack, true); err == nil {
		t.Fatal("non-CUTLASS last layer accepted")
	}

	// A T_in that is not `wm.T - 2*padding` is a different call, not a warning.
	stack, err = PlanHACSmallConvs(64, 192, 1, 7782400, base)
	if err != nil {
		t.Fatal(err)
	}
	stack.Layers[2].TIn++
	if _, err := PlanHACLastConv(stack, true); err == nil {
		t.Fatal("T_in disagreement accepted")
	}

	// A last layer whose C_in does not follow the previous allocation.
	bad := append([]SmallConvParams(nil), base...)
	bad[2].CIn = 32
	if _, err := PlanHACSmallConvs(64, 192, 1, 7782400, bad); err == nil {
		t.Fatal("channel mismatch accepted by the stack plan")
	}
}

// TestLastConvDevOptionsDefaults pins the source defaults. A zero value must not
// silently select a path: cutlass_conv defaults to true.
func TestLastConvDevOptionsDefaults(t *testing.T) {
	d := DefaultLastConvDevOptions()
	if !d.CutlassConv || !d.KoiConv || d.KoiConvF32 {
		t.Fatalf("defaults %s", d)
	}
	if (LastConvDevOptions{}).CutlassConv {
		t.Fatal("zero value must not select the CUTLASS branch")
	}
}

// TestInputPaddingBands pins the two bands per batch and proves they cover
// exactly rows [0,padding) and [rows-padding,rows), which is what
// `in.slice(1,0,padding)=0` and `in.slice(1,-padding,None)=0` write.
func TestInputPaddingBands(t *testing.T) {
	const batch, rows, c, padding = 3, 10, 4, 2
	bands := inputPaddingBands(batch, rows, c, padding)
	if len(bands) != 2*batch {
		t.Fatalf("%d bands, want %d", len(bands), 2*batch)
	}
	covered := map[uint64]bool{}
	for _, b := range bands {
		for i := uint64(0); i < b[1]; i++ {
			if covered[b[0]+i] {
				t.Fatalf("element %d covered twice", b[0]+i)
			}
			covered[b[0]+i] = true
		}
	}
	if want := uint64(2 * padding * c * batch); uint64(len(covered)) != want {
		t.Fatalf("covered %d elements, want %d", len(covered), want)
	}
	for n := 0; n < batch; n++ {
		for _, row := range []int64{0, 1, rows - 2, rows - 1} {
			for col := int64(0); col < c; col++ {
				idx := uint64((int64(n)*rows+row)*c + col)
				if !covered[idx] {
					t.Fatalf("batch %d row %d col %d not covered", n, row, col)
				}
			}
		}
		for row := int64(padding); row < rows-padding; row++ {
			for col := int64(0); col < c; col++ {
				idx := uint64((int64(n)*rows+row)*c + col)
				if covered[idx] {
					t.Fatalf("batch %d interior row %d col %d covered", n, row, col)
				}
			}
		}
	}
}

// TestNewHACLastConvRefusesWindowPath proves the unported selection is refused
// at construction instead of silently running a different convolution.
func TestNewHACLastConvRefusesWindowPath(t *testing.T) {
	dev := DefaultLastConvDevOptions()
	dev.CutlassConv = false
	if _, err := NewHACLastConv(nil, nil, hacV6Convs()[2], dev, nil, nil); err == nil {
		t.Fatal("window/half-GEMM path accepted")
	}
}
