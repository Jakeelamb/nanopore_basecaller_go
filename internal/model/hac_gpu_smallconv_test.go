package model

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// hacV6Convs is the pinned convolution stack of
// dna_r10.4.1_e8.2_400bps_hac@v6.0.0/config.toml, in source order. The third
// layer is the CUTLASS width-19 stride-6 layer; it is planned but not executed
// by H02, and it is what gives the second layer nine rows of output padding
// (ConvStack.cpp:143-148).
func hacV6Convs() []SmallConvParams {
	return []SmallConvParams{
		{CIn: 1, COut: 16, WinLen: 5, Stride: 1, Activation: SmallConvSwish, OutputLayout: GPUStorageNTC},
		{CIn: 16, COut: 16, WinLen: 5, Stride: 1, Activation: SmallConvSwish, OutputLayout: GPUStorageNTC},
		{CIn: 16, COut: 1024, WinLen: 19, Stride: 6, Activation: SmallConvTanh, OutputLayout: GPUStorageCUTLASSI8},
	}
}

// TestPlanHACSmallConvs pins the working-memory placement H01 observed for the
// declared case: batch 64, chunk 192, whole-model reservation 7782400 bytes,
// input front, conv0 back, conv1 front, conv2 back.
func TestPlanHACSmallConvs(t *testing.T) {
	plan, err := PlanHACSmallConvs(64, 192, 1, 7782400, hacV6Convs())
	if err != nil {
		t.Fatal(err)
	}
	if plan.InputOffsetBytes != 0 {
		t.Fatalf("input offset %d", plan.InputOffsetBytes)
	}
	if plan.Input.Bytes != 24576 {
		t.Fatalf("input extent %d, want 24576", plan.Input.Bytes)
	}
	want := []struct {
		offset    int64
		bytes     int64
		tIn, tOut int64
		rows      int64
		padding   int64
		executes  bool
	}{
		{7389184, 393216, 192, 192, 192, 0, true},
		{0, 430080, 192, 192, 210, 9, true},
		{5357568, 2424832, 192, 32, 37, 0, false},
	}
	if len(plan.Layers) != len(want) {
		t.Fatalf("%d layers, want %d", len(plan.Layers), len(want))
	}
	for i, w := range want {
		l := plan.Layers[i]
		if l.OffsetBytes != w.offset || l.Geometry.Bytes != w.bytes || l.TIn != w.tIn ||
			l.TOut != w.tOut || l.Geometry.Rows != w.rows || l.Params.OutputTPadding != w.padding ||
			l.Executes != w.executes {
			t.Fatalf("layer %d = offset %d bytes %d t_in %d t_out %d rows %d padding %d executes %v",
				i, l.OffsetBytes, l.Geometry.Bytes, l.TIn, l.TOut, l.Geometry.Rows,
				l.Params.OutputTPadding, l.Executes)
		}
	}
}

// TestHACSmallConvCrop pins the cropped view the kernel writes into: the crop
// starts nine rows into the allocation and keeps the allocation's batch stride,
// so the padding rows stay between batches (ConvStack.cpp:241-242).
func TestHACSmallConvCrop(t *testing.T) {
	plan, err := PlanHACSmallConvs(64, 192, 1, 7782400, hacV6Convs())
	if err != nil {
		t.Fatal(err)
	}
	l := plan.Layers[1]
	if got := l.CropOffsetElements(); got != 144 {
		t.Fatalf("crop offset %d elements, want 9*16=144", got)
	}
	if got := l.CropShape(); !slices.Equal(got, []int64{64, 192, 16}) {
		t.Fatalf("crop shape %v", got)
	}
	if got := l.CropStrides(); !slices.Equal(got, []int64{3360, 16, 1}) {
		t.Fatalf("crop strides %v, want the parent batch stride 210*16", got)
	}
	if got := l.BatchStride(); got != 3360 {
		t.Fatalf("batch stride %d, want 3360", got)
	}
	g, err := l.SmallConvGeometry()
	if err != nil {
		t.Fatal(err)
	}
	if g.N != 64 || g.CIn != 16 || g.COut != 16 || g.TIn != 192 || g.Window != 5 ||
		g.Stride != 1 || g.Padding != 2 || g.OutNStride != 3360 {
		t.Fatalf("geometry %+v", g)
	}
	if g.Activation.String() != "swish" {
		t.Fatalf("activation %s", g.Activation)
	}
	// The first layer is not padded: its crop is the whole allocation.
	if got := plan.Layers[0].CropOffsetElements(); got != 0 {
		t.Fatalf("unpadded crop offset %d", got)
	}
	if got := plan.Layers[0].CropStrides(); !slices.Equal(got, []int64{192 * 16, 16, 1}) {
		t.Fatalf("unpadded crop strides %v", got)
	}
}

// TestPlanHACSmallConvsRejections pins the plan's own guards. Each case is a
// declared-from-source invariant, not a mask over observed mismatches.
func TestPlanHACSmallConvsRejections(t *testing.T) {
	base := func() []SmallConvParams { return hacV6Convs() }
	cases := []struct {
		name        string
		batch       int64
		chunk       int64
		features    int64
		reservation int64
		layers      []SmallConvParams
		want        error
	}{
		{"zero batch", 0, 192, 1, 7782400, base(), ErrGPUPlanLayout},
		{"zero chunk", 64, 0, 1, 7782400, base(), ErrGPUPlanLayout},
		{"zero features", 64, 192, 0, 7782400, base(), ErrGPUPlanLayout},
		{"zero reservation", 64, 192, 1, 0, base(), ErrGPUPlanLayout},
		{"one layer", 64, 192, 1, 7782400, base()[:1], ErrGPUPlanLayout},
		{"broken channels", 64, 192, 1, 7782400, func() []SmallConvParams {
			l := base()
			l[1].CIn = 15
			return l
		}(), ErrGPUPlanLayout},
		{"broken stride", 64, 193, 1, 7782400, base(), ErrGPUPlanLayout},
		{"undersized reservation", 64, 192, 1, 400000, base(), ErrGPUPlanUndersized},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := PlanHACSmallConvs(tt.batch, tt.chunk, tt.features, tt.reservation, tt.layers); !errors.Is(err, tt.want) {
				t.Fatalf("plan error = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestPlanHACSmallConvsPaddingComesFromTheNextLayer pins ConvStack.cpp:147:
// only a layer followed by a CUTLASS layer is padded, and by the *next* layer's
// window/2. A plan that hardcoded nine would survive a different pinned model;
// this one must not.
func TestPlanHACSmallConvsPaddingComesFromTheNextLayer(t *testing.T) {
	layers := hacV6Convs()
	layers[2].WinLen = 7
	plan, err := PlanHACSmallConvs(64, 192, 1, 7782400, layers)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.Layers[1].Params.OutputTPadding; got != 3 {
		t.Fatalf("padding %d, want the last window/2 = 3", got)
	}
	// Without a CUTLASS tail there is no padding at all.
	plain := hacV6Convs()
	plain[2].OutputLayout = GPUStorageNTC
	plan, err = PlanHACSmallConvs(64, 192, 1, 7782400, plain)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.Layers[1].Params.OutputTPadding; got != 0 {
		t.Fatalf("padding %d, want 0 without a CUTLASS tail", got)
	}
}

// TestPackSmallConvWeights pins the exact permutation
// `permute({2,1,0}).contiguous().flatten(0,1)` (ConvStack.cpp:230-233):
// element [co][ci][w] is written at (w*C_in+ci)*C_out+co.
func TestPackSmallConvWeights(t *testing.T) {
	weight := make([]float32, 2*2*3)
	for i := range weight {
		weight[i] = float32(i)
	}
	got, err := PackSmallConvWeights(weight, 2, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint16{
		0x0000, 0x4600, 0x4200, 0x4880, // w=0: co0ci0=0, co1ci0=6, co0ci1=3, co1ci1=9
		0x3c00, 0x4700, 0x4400, 0x4900, // w=1: 1, 7, 4, 10
		0x4000, 0x4800, 0x4500, 0x4980, // w=2: 2, 8, 5, 11
	}
	if !slices.Equal(got, want) {
		t.Fatalf("packed weights %#04x, want %#04x", got, want)
	}
	// The flattened order is [w][ci] rows, each row holding C_out columns.
	if len(got) != 6*2 {
		t.Fatalf("packed length %d, want W*C_in*C_out = 12", len(got))
	}
}

func TestPackSmallConvRejections(t *testing.T) {
	if _, err := PackSmallConvWeights(make([]float32, 5), 2, 2, 3); !errors.Is(err, ErrGPUPlanLayout) {
		t.Fatalf("short weight: %v", err)
	}
	if _, err := PackSmallConvWeights(nil, 0, 2, 3); !errors.Is(err, ErrGPUPlanLayout) {
		t.Fatalf("zero C_out: %v", err)
	}
	if _, err := PackSmallConvBias(nil); !errors.Is(err, ErrGPUPlanLayout) {
		t.Fatalf("empty bias: %v", err)
	}
}

// TestPackSmallConvHalfCast pins the round-to-nearest-even cast the source gets
// from `.to(kF16)`. Every vector is an exact halfway case, so a truncating or
// ties-away cast is distinguished.
func TestPackSmallConvHalfCast(t *testing.T) {
	values := []float32{
		1.0,                              // 0x3c00
		1 + 1.0/2048,                     // tie at 0.5 ulp -> even 0x3c00
		1 + 1.0/1024,                     // 0x3c01
		1 + 1.5/1024,                     // tie at 1.5 ulp -> even 0x3c02
		1 + 2.5/1024,                     // tie at 2.5 ulp -> even 0x3c02
		float32(math.Ldexp(1, -25)),      // tie between 0 and min subnormal -> 0x0000
		float32(3 * math.Ldexp(1, -25)),  // tie at 1.5 subnormals -> even 0x0002
		float32(65504),                   // 0x7bff
		float32(65536),                   // above the rounding boundary -> 0x7c00
		math.Float32frombits(0x80000000), // -0 -> 0x8000
	}
	want := []uint16{0x3c00, 0x3c00, 0x3c01, 0x3c02, 0x3c02, 0x0000, 0x0002, 0x7bff, 0x7c00, 0x8000}
	got, err := PackSmallConvBias(values)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("half cast %#04x, want %#04x", got, want)
	}
}

// TestHalfBytes pins the little-endian upload form of a half tensor.
func TestHalfBytes(t *testing.T) {
	got := HalfBytes([]uint16{0x3c00, 0xbc00})
	if !slices.Equal(got, []byte{0x00, 0x3c, 0x00, 0xbc}) {
		t.Fatalf("half bytes %x", got)
	}
}

// TestSmallConvActivationMapping pins get_koi_activation (ConvStack.cpp:21-31):
// an unknown activation is an error, never a default.
func TestSmallConvActivationMapping(t *testing.T) {
	if a, err := SmallConvSwish.KoiActivation(); err != nil || a.String() != "swish" {
		t.Fatalf("swish -> %v %v", a, err)
	}
	if a, err := SmallConvTanh.KoiActivation(); err != nil || a.String() != "tanh" {
		t.Fatalf("tanh -> %v %v", a, err)
	}
	if _, err := SmallConvActivation(0).KoiActivation(); err == nil {
		t.Fatal("zero activation accepted")
	}
	if _, err := SmallConvActivation(9).KoiActivation(); err == nil {
		t.Fatal("unknown activation accepted")
	}
}
