package h02

import (
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

// syntheticCapture is the declared case's conv-stack shape, taken from the H01
// capture geometry rather than from the plan, so a plan that drifted would not
// agree with it.
func syntheticCapture() hacgpu.Capture {
	return hacgpu.Capture{
		Schema: hacgpu.CaptureSchema, Status: "captured",
		Case: "hac-v6-n64-chunk192", Batch: 64, Chunk: 192, Features: 1,
		Convs: 3, Stride: 6, TOut: 32, Reservation: 7782400, BackingElems: 3891200,
		Stages: []hacgpu.Stage{
			{Name: "conv0", Kind: "convolution", Layer: 0, Layout: "NTC",
				Shape: []int64{64, 192, 16}, OffsetBytes: 7389184, Bytes: 393216},
			{Name: "conv1", Kind: "convolution", Layer: 1, Layout: "NTC",
				Shape: []int64{64, 210, 16}, OffsetBytes: 0, Bytes: 430080,
				UnwrittenT: [][2]int{{0, 9}, {201, 9}}},
			{Name: "conv2", Kind: "convolution", Layer: 2, Layout: "CUTLASS_TNC_I8",
				Shape: []int64{37, 64, 1024}, OffsetBytes: 5357568, Bytes: 2424832},
		},
	}
}

func syntheticPlan(t *testing.T) (*model.HACSmallConvPlan, []ConvSpec) {
	t.Helper()
	convs := []ConvSpec{
		{Index: 0, CIn: 1, COut: 16, WinLen: 5, Stride: 1, Padding: 2, Activation: "swish"},
		{Index: 1, CIn: 16, COut: 16, WinLen: 5, Stride: 1, Padding: 2, Activation: "swish"},
		{Index: 2, CIn: 16, COut: 1024, WinLen: 19, Stride: 6, Padding: 9, Activation: "tanh"},
	}
	params, err := ConvParams(convs)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := model.PlanHACSmallConvs(64, 192, 1, 7782400, params)
	if err != nil {
		t.Fatal(err)
	}
	return plan, convs
}

// TestRecordPlanAgreesWithTheCapture proves the plan's placement, extent, crop
// and layout all reproduce what the native graph did. This is the differential
// check on the planner; the byte comparison is separate.
func TestRecordPlanAgreesWithTheCapture(t *testing.T) {
	plan, convs := syntheticPlan(t)
	layers, err := recordPlan(plan, syntheticCapture(), convs)
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != 3 {
		t.Fatalf("%d layers", len(layers))
	}
	if layers[1].CropOffsetElements != 144 || layers[1].BatchStride != 3360 ||
		layers[1].OutputTPadding != 9 || layers[1].CaptureLayout != "NTC" {
		t.Fatalf("conv1 plan %+v", layers[1])
	}
	if layers[2].CaptureLayout != "CUTLASS_TNC_I8" || layers[2].Executes {
		t.Fatalf("conv2 plan %+v", layers[2])
	}
}

// TestRecordPlanRejectsDrift proves every recorded field is load-bearing: a
// capture that moved a layer, resized it, or used a different layout is a
// rejection, so the plan cannot silently diverge from the oracle.
func TestRecordPlanRejectsDrift(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*hacgpu.Capture)
		want   string
	}{
		{"offset", func(c *hacgpu.Capture) { c.Stages[0].OffsetBytes += 256 }, "plan offset"},
		{"extent", func(c *hacgpu.Capture) { c.Stages[0].Bytes += 256 }, "plan extent"},
		{"layout", func(c *hacgpu.Capture) { c.Stages[2].Layout = "CUTLASS_TNC_F16" }, "plan layout"},
		{"shape", func(c *hacgpu.Capture) { c.Stages[1].Shape = []int64{64, 208, 16} }, "does not describe the captured"},
		{"missing stage", func(c *hacgpu.Capture) { c.Stages = c.Stages[:2] }, "no conv2 stage"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			plan, convs := syntheticPlan(t)
			c := syntheticCapture()
			tt.mutate(&c)
			_, err := recordPlan(plan, c, convs)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
		})
	}
}

// TestWrittenRowsAreTheDeclaredRowsOnly proves the comparison mask comes from
// the capture's declared unwritten rows, never from observed mismatches.
func TestWrittenRowsAreTheDeclaredRowsOnly(t *testing.T) {
	stage := hacgpu.Stage{Name: "conv1", Layout: "NTC", Shape: []int64{2, 6, 1}, Bytes: 2 * 6 * 1 * 2,
		UnwrittenT: [][2]int{{0, 2}, {5, 1}}}
	observed := []byte("AABBCCDDEEFFGGHHIIJJKKLL")
	payload := []byte("aabbccddeeffgghhiijjkkll")
	written, err := writtenBytes(stage, observed, payload)
	if err != nil {
		t.Fatal(err)
	}
	// Rows 2,3,4 of each batch: "CCDDEE" and "IIJJKK".
	if string(written) != "CCDDEEIIJJKK" {
		t.Fatalf("written rows %q", written)
	}
	expected, err := writtenPayload(stage, payload)
	if err != nil {
		t.Fatal(err)
	}
	if string(expected) != "ccddeeiijjkk" {
		t.Fatalf("written payload %q", expected)
	}
	bad := append([]byte(nil), observed...)
	bad[0] = 'Z' // an unwritten row
	if _, err = writtenBytes(stage, bad, payload); err != nil {
		t.Fatal(err)
	}
	bad[6] = 'Z' // the first written row
	written, err = writtenBytes(stage, bad, payload)
	if err != nil {
		t.Fatal(err)
	}
	if string(written) == "CCDDEEIIJJKK" {
		t.Fatal("a written row change was masked out")
	}
}
