package model

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestSUPGPUFixedPlanAndImmutableDescriptors(t *testing.T) {
	p, err := PlanSUPGPU(FixedSUPGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	calls := p.Calls()
	if len(calls) != 134 {
		t.Fatalf("calls=%d", len(calls))
	}
	for i := 0; i < 5; i++ {
		if calls[i].Primitive != tensor.SUPConvSiLU || calls[i].Selector != i || calls[i].Operands[0].Tags[1] != tensor.TagC {
			t.Fatal("fixed NCT conv route changed")
		}
	}
	for layer := 0; layer < 18; layer++ {
		for stage, op := range []tensor.SUPPrimitive{tensor.SUPQKVRotary, tensor.SUPMaskedAttention, tensor.SUPProjection, tensor.SUPNorm1, tensor.SUPSwiGLU, tensor.SUPFC2, tensor.SUPNorm2} {
			c := calls[6+layer*7+stage]
			if c.Layer != layer || c.Primitive != op {
				t.Fatalf("layer%d stage%d replaced", layer, stage)
			}
		}
	}
	last := calls[len(calls)-1]
	if !slices.Equal(last.Operands[2].Shape, []int64{32, 128, 4096}) {
		t.Fatal("final score geometry changed")
	}
	calls[0].Operands[0].Shape[0] = 1
	calls[0].Operands[0].Tags[0] = 'W'
	if p.Calls()[0].Operands[0].Shape[0] != 32 || p.Calls()[0].Operands[0].Tags[0] != 'N' {
		t.Fatal("plan aliases caller snapshots")
	}
	for _, p := range []SUPGPUProfile{{}, {1, 768, 64, true, true, false, 4}, {32, 769, 64, true, true, false, 4}, {32, 768, 64, false, true, false, 4}, {32, 768, 64, true, false, false, 4}, {32, 768, 64, true, true, true, 4}, {32, 768, 64, true, true, false, 3}, {math.MaxInt64, math.MaxInt64, math.MaxInt64, true, true, false, 4}} {
		if _, err := PlanSUPGPU(p); err == nil {
			t.Fatal("unsupported profile accepted", p)
		}
	}
}

func TestSUPGPUTileRoundTripAndBounds(t *testing.T) {
	for _, geometry := range []tensor.SUPGeometry{tensor.SUPGeometryTiny, tensor.SUPGeometryFull, tensor.SUPGeometryShort} {
		d, err := geometry.Dimensions()
		if err != nil {
			t.Fatal(err)
		}
		for _, lane := range []int64{8, 16} {
			shape, strides, err := SUPNTCTileStrides(geometry, lane)
			if err != nil {
				t.Fatal(err)
			}
			seen := make([]bool, d.Batch*d.EncoderT*512)
			for n := int64(0); n < shape[0]; n++ {
				for tb := int64(0); tb < shape[1]; tb++ {
					for cb := int64(0); cb < shape[2]; cb++ {
						for t0 := int64(0); t0 < 16; t0++ {
							for c0 := int64(0); c0 < lane; c0++ {
								got := n*strides[0] + tb*strides[1] + cb*strides[2] + t0*strides[3] + c0
								want := (n*d.EncoderT+tb*16+t0)*512 + cb*lane + c0
								if got != want || seen[got] {
									t.Fatal("tile lost or aliased a logical element")
								}
								seen[got] = true
							}
						}
					}
				}
			}
			for _, ok := range seen {
				if !ok {
					t.Fatal("tile omitted an element")
				}
			}
		}
		if _, _, err := SUPNTCTileStrides(geometry, 32); err == nil {
			t.Fatal("unselected tile lane accepted")
		}
	}
	if _, _, err := SUPNTCTileStrides(tensor.SUPGeometry(99), 8); err == nil {
		t.Fatal("unknown geometry accepted")
	}
}

func TestSUPGPUShortFixedPlanAndConstructor(t *testing.T) {
	p, err := PlanSUPGPU(ShortFixedSUPGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	if p.Profile() != ShortFixedSUPGPUProfile() || p.Geometry() != tensor.SUPGeometryShort || p.dimensions.Batch != 32 || p.dimensions.Chunk != 6144 || p.dimensions.EncoderT != 512 {
		t.Fatal("short profile was aliased to another geometry")
	}
	calls := p.Calls()
	if len(calls) != 134 {
		t.Fatalf("short graph has %d calls", len(calls))
	}
	for i, want := range []int64{6144, 6144, 2048, 1024, 512} {
		if !slices.Equal(calls[i].Operands[3].Shape, []int64{32, []int64{64, 64, 128, 128, 512}[i], want}) {
			t.Fatalf("short convolution%d has wrong output: %v", i, calls[i].Operands[3].Shape)
		}
	}
	for index, shapes := range map[int][][]int64{
		5:   {{32, 512, 512}, {32, 512, 512}, {32, 512}},
		6:   {{32, 32, 32, 16, 16}, {32, 512}, {3, 8, 4, 32, 16, 16}, {3, 8, 64}, {32, 8, 16, 8}, {32, 32, 3, 8, 8, 16, 8}, {1}},
		7:   {{32, 32, 3, 8, 8, 16, 8}, {32, 32, 8, 8, 16, 8}},
		8:   {{1024, 64, 16, 8}, {32, 64, 16, 8}, {512}, {1024, 64, 16, 8}, {1}},
		9:   {{32, 32, 64, 16, 8}, {32, 32, 64, 16, 8}, {512}, {32, 32, 32, 16, 16}},
		10:  {{1024, 32, 16, 16}, {256, 32, 16, 16}, {1024, 128, 16, 16}, {1}},
		11:  {{1024, 128, 16, 16}, {32, 128, 16, 16}, {1024, 64, 16, 8}, {1}},
		12:  {{32, 32, 64, 16, 8}, {32, 32, 64, 16, 8}, {512}, {32, 32, 32, 16, 16}, {32, 512}},
		132: {{32, 512, 512}, {1024, 512}, {1024}, {32, 512, 1024}},
		133: {{32, 1024, 512}, {4096, 512}, {32, 1024, 4096}},
	} {
		for j, shape := range shapes {
			if !slices.Equal(calls[index].Operands[j].Shape, shape) {
				t.Fatalf("short call%d operand%d differs: %v", index, j, calls[index].Operands[j].Shape)
			}
		}
	}
	full, err := PlanSUPGPU(FullFixedSUPGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	for j, call := range full.Calls() {
		if call.Name != calls[j].Name || call.Layer != calls[j].Layer || call.Primitive != calls[j].Primitive || call.Selector != calls[j].Selector {
			t.Fatal("short graph changed model operation order")
		}
		for k, operand := range call.Operands {
			short := calls[j].Operands[k]
			if operand.DType != short.DType || !slices.Equal(operand.Tags, short.Tags) || operand.Scaled != short.Scaled || operand.ScaleTag != short.ScaleTag || operand.Write != short.Write {
				t.Fatal("short graph changed numerical or scale policy")
			}
		}
	}
	if _, err := NewSUPGPU(context.Background(), nil, nil, SUPGPUWeights{}); !errors.Is(err, tensor.ErrClosed) {
		t.Fatalf("short constructor rejected geometry before owner admission: %v", err)
	}
	for _, mutate := range []func(*SUPGPUProfile){func(p *SUPGPUProfile) { p.Batch = 64 }, func(p *SUPGPUProfile) { p.Chunk = 12288 }, func(p *SUPGPUProfile) { p.EncoderT = 1024 }, func(p *SUPGPUProfile) { p.UseI8 = false }, func(p *SUPGPUProfile) { p.UseF8 = false }, func(p *SUPGPUProfile) { p.UseUSwiGLU = true }, func(p *SUPGPUProfile) { p.RemoveBits = 3 }} {
		profile := ShortFixedSUPGPUProfile()
		mutate(&profile)
		if _, err := PlanSUPGPU(profile); err == nil {
			t.Fatal("foreign short profile admitted", profile)
		}
	}
}

func TestSUPGPUFullFixedPlanDimensionsAndRejections(t *testing.T) {
	full, err := PlanSUPGPU(FullFixedSUPGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	tiny, err := PlanSUPGPU(FixedSUPGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	if full.Geometry() != tensor.SUPGeometryFull || full.Profile() != FullFixedSUPGPUProfile() || len(full.Calls()) != 134 {
		t.Fatal("full fixed geometry did not select the production graph")
	}
	calls, small := full.Calls(), tiny.Calls()
	wantTimes := []int64{12288, 12288, 4096, 2048, 1024}
	for i, want := range wantTimes {
		if !slices.Equal(calls[i].Operands[3].Shape, []int64{32, []int64{64, 64, 128, 128, 512}[i], want}) {
			t.Fatal("full convolution output differs", i, calls[i].Operands)
		}
	}
	want := map[int][][]int64{
		5:   {{32, 1024, 512}, {32, 1024, 512}, {32, 1024}},
		6:   {{32, 64, 32, 16, 16}, {32, 1024}, {3, 8, 4, 32, 16, 16}, {3, 8, 64}, {64, 8, 16, 8}, {32, 64, 3, 8, 8, 16, 8}, {1}},
		7:   {{32, 64, 3, 8, 8, 16, 8}, {32, 64, 8, 8, 16, 8}},
		8:   {{2048, 64, 16, 8}, {32, 64, 16, 8}, {512}, {2048, 64, 16, 8}, {1}},
		9:   {{32, 64, 64, 16, 8}, {32, 64, 64, 16, 8}, {512}, {32, 64, 32, 16, 16}},
		10:  {{2048, 32, 16, 16}, {256, 32, 16, 16}, {2048, 128, 16, 16}, {1}},
		11:  {{2048, 128, 16, 16}, {32, 128, 16, 16}, {2048, 64, 16, 8}, {1}},
		12:  {{32, 64, 64, 16, 8}, {32, 64, 64, 16, 8}, {512}, {32, 64, 32, 16, 16}, {32, 1024}},
		132: {{32, 1024, 512}, {1024, 512}, {1024}, {32, 1024, 1024}},
		133: {{32, 2048, 512}, {4096, 512}, {32, 2048, 4096}},
	}
	for index, shapes := range want {
		for j, shape := range shapes {
			if !slices.Equal(calls[index].Operands[j].Shape, shape) {
				t.Fatal("full operand differs", index, j, calls[index].Operands[j].Shape, shape)
			}
		}
	}
	for i, c := range calls {
		if c.Name != small[i].Name || c.Layer != small[i].Layer || c.Primitive != small[i].Primitive || c.Selector != small[i].Selector {
			t.Fatal("full geometry changed model order", i)
		}
		for j, operand := range c.Operands {
			if operand.DType != small[i].Operands[j].DType || !slices.Equal(operand.Tags, small[i].Operands[j].Tags) || operand.Write != small[i].Operands[j].Write || operand.ScaleTag != small[i].Operands[j].ScaleTag || operand.Scaled != small[i].Operands[j].Scaled {
				t.Fatal("full geometry changed dtype, tags, mutability or scale policy", i, j)
			}
		}
	}
	for _, mutate := range []func(*SUPGPUProfile){
		func(p *SUPGPUProfile) { p.Batch = 512 },
		func(p *SUPGPUProfile) { p.Chunk = 768 },
		func(p *SUPGPUProfile) { p.EncoderT = 64 },
		func(p *SUPGPUProfile) { p.EncoderT = 512 },
		func(p *SUPGPUProfile) { p.UseI8 = false },
		func(p *SUPGPUProfile) { p.UseF8 = false },
		func(p *SUPGPUProfile) { p.UseUSwiGLU = true },
		func(p *SUPGPUProfile) { p.RemoveBits = 3 },
	} {
		profile := FullFixedSUPGPUProfile()
		mutate(&profile)
		if _, err = PlanSUPGPU(profile); err == nil {
			t.Fatal("unsupported full profile accepted", profile)
		}
	}
}

func syntheticSUPEncoder() SUPEncoderHalfWeights {
	fill := func(n int) []uint16 {
		x := make([]uint16, n)
		for i := range x {
			x[i] = 0x3401 + uint16(i%37)
		}
		return x
	}
	return SUPEncoderHalfWeights{fill(3 * 512 * 512), fill(512 * 512), fill(512), fill(4096 * 512), fill(512 * 2048), fill(512), fill(512)}
}
func TestSUPGPUEncoderCachesPreserveSourceQuantizationOrder(t *testing.T) {
	w := syntheticSUPEncoder()
	cache, err := PackSUPEncoderGPU(w)
	if err != nil {
		t.Fatal(err)
	}
	before, after := cache.Snapshot(false), cache.Snapshot(true)
	if len(before) != 10 || len(after) != 10 {
		t.Fatal("incomplete caches")
	}
	for i, b := range before {
		a := after[i]
		round := b.Name == "qkv_half" || b.Name == "proj_weight" || b.Name == "norm1_weight" || b.Name == "norm2_weight" || b.Name == "fc1_half"
		if round {
			for j := 0; j < len(b.Data); j += 2 {
				want := (binary.LittleEndian.Uint16(b.Data[j:]) + 8) & 0xfff0
				if binary.LittleEndian.Uint16(a.Data[j:]) != want {
					t.Fatal("remove_bits ordering differs", b.Name, j)
				}
			}
		} else if !slices.Equal(a.Data, b.Data) {
			t.Fatal("quantized cache was rounded", b.Name)
		}
	}
	q, err := prepareSUPQKVF16Weights(w.QKV)
	if err != nil {
		t.Fatal(err)
	}
	want, err := BuildSUPQKVInt8(q.interleavedHalf)
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range want.Tiled {
		if before[1].Data[i] != byte(x) {
			t.Fatal("QKV quantized from rounded input")
		}
	}
	before[0].Data[0] ^= 0xff
	before[0].Shape[0] = 99
	if cache.Snapshot(false)[0].Shape[0] != 3 || cache.Snapshot(false)[0].Data[0] == before[0].Data[0] {
		t.Fatal("cache snapshot mutated private state")
	}
	w.Norm1[0] = 0x7bff
	if _, err := PackSUPEncoderGPU(w); err == nil {
		t.Fatal("remove_bits overflow to infinity accepted")
	}
}
func TestSUPGPUWeightRolesAndCancellationStayClosed(t *testing.T) {
	manifest, err := WeightManifest(SUP)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = supGPURoles(manifest); err != nil {
		t.Fatal(err)
	}
	slices.Reverse(manifest)
	if _, err = supGPURoles(manifest); err != nil {
		t.Fatal("roles depend on ordering", err)
	}
	manifest[0].SHA256 = manifest[1].SHA256
	if _, err = supGPURoles(manifest); err == nil {
		t.Fatal("foreign archive accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = LoadSUPGPUWeights(ctx, nil, "/must/not/read"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err = NewSUPGPU(ctx, nil, nil, SUPGPUWeights{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	m := &SUPGPU{}
	if r, err := m.Forward(ctx, nil, nil, SUPGPUPlan{}); r != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("partial result or native access after cancellation: %v %v", r, err)
	}
	for _, dtype := range []tensor.DType{tensor.DTypeF16, tensor.DTypeF32, tensor.DTypeE4M3} {
		data := []byte{0xff, 0x7f}
		if dtype == tensor.DTypeF32 {
			data = []byte{0, 0, 0x80, 0x7f}
		}
		if err = validateSUPValues(dtype, data, false); err == nil {
			t.Fatal("nonfinite accepted", dtype)
		}
	}
}

func TestSUPGPURoPEUsesHalfPowerReciprocalAndProduct(t *testing.T) {
	freq := SUPRoPEFrequencies()
	if len(freq) != 4096*32 {
		t.Fatal(len(freq))
	}
	for _, d := range []int{0, 1, 9, 31} {
		power := signal.Float32FromHalf(signal.Float16Bits(float32(math.Pow(10000, float64(float32(2*d)/64)))))
		inv := signal.Float32FromHalf(signal.Float16Bits(1 / power))
		for _, t0 := range []int{0, 1, 3, 2049, 4095} {
			pos := signal.Float32FromHalf(signal.Float16Bits(float32(t0)))
			want := signal.Float16Bits(pos * inv)
			if freq[t0*32+d] != want {
				t.Fatal("half frequency policy changed")
			}
		}
	}
	if freq[2049*32] != signal.Float16Bits(2048) {
		t.Fatal("half arange conversion was skipped")
	}
}

func TestSUPGPUEncoderCancellationAndFailureStopAtFirstOperation(t *testing.T) {
	for stop := 0; stop < 7; stop++ {
		for _, cancelled := range []bool{false, true} {
			ctx, cancel := context.WithCancel(context.Background())
			count := 0
			sentinel := errors.New("failed native operation")
			err := runSUPEncoderStages(ctx, 4, func(stage supEncoderStage) error {
				if stage.op != supEncoderStages[count].op {
					t.Fatal("operation order changed")
				}
				count++
				if count == stop+1 {
					if cancelled {
						cancel()
						return nil
					}
					return sentinel
				}
				return nil
			})
			cancel()
			if count != stop+1 {
				t.Fatalf("later operation ran after failure/cancel at %d", stop)
			}
			if cancelled && !errors.Is(err, context.Canceled) || !cancelled && !errors.Is(err, sentinel) {
				t.Fatal("failed operation was reported as success", err)
			}
		}
	}
	count := 0
	if err := runSUPEncoderStages(context.Background(), 0, func(supEncoderStage) error { count++; return nil }); err != nil || count != 7 {
		t.Fatal("successful layer did not execute exactly seven operations", count, err)
	}
}
