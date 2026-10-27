package supgpu

import (
	"fmt"
	"strings"
)

type tensorSpec struct {
	Name, DType string
	Shape       []int64
	Tags        []int
	ScaleTag    int
}
type eventSpec struct {
	Name    string
	Layer   int
	Tensors []tensorSpec
}

func spec(name, dtype string, shape []int64, tags string) tensorSpec {
	t := []int{}
	for _, r := range tags {
		t = append(t, int(r))
	}
	return tensorSpec{name, dtype, shape, t, 0}
}

// eventRegistry is the closed prospective fixed/FP8 graph contract. The cache
// snapshots bracket the source's remove_bits call during initialization. Later
// primitive observations reuse those allocations; they never rebuild a cache.
func eventRegistry(g Geometry) []eventSpec {
	n, t := int64(g.Batch), int64(g.EncoderT)
	out := []eventSpec{}
	add := func(name string, layer int, ts ...tensorSpec) { out = append(out, eventSpec{name, layer, ts}) }
	add("input", -1, spec("input", "Half", []int64{n, 1, int64(g.Chunk)}, ""))
	length := int64(g.Chunk)
	for i, cs := range []struct{ c, s int64 }{{64, 1}, {64, 1}, {128, 3}, {128, 2}, {512, 2}} {
		length = (length + cs.s - 1) / cs.s
		add(fmt.Sprintf("conv%d", i), -1, spec("output", "Half", []int64{n, cs.c, length}, ""))
	}
	add("encoder.input", -1, spec("half", "Half", []int64{n, t / 16, 64, 16, 8}, "NTCtc"), spec("int8", "Char", []int64{n, t / 16, 32, 16, 16}, "NTCtc"), spec("scale", "Float", []int64{n, t}, ""))
	caches := make([][]tensorSpec, 18)
	for layer := 0; layer < 18; layer++ {
		cache := []tensorSpec{
			spec("qkv_half", "Half", []int64{3, 8, 4, 64, 16, 8}, ""),
			spec("qkv_int8", "Char", []int64{3, 8, 4, 32, 16, 16}, ""),
			spec("qkv_scale", "Float", []int64{3, 8, 64}, ""),
			spec("sincos", "Half", []int64{4096 / 16, 8, 16, 8}, ""),
			spec("proj_weight", "Half", []int64{32, 64, 16, 8}, ""),
			spec("proj_bias", "Half", []int64{512}, ""),
			spec("norm1_weight", "Half", []int64{512}, ""), spec("norm2_weight", "Half", []int64{512}, ""),
			spec("fc1_half", "Half", []int64{256, 64, 16, 8}, ""),
			spec("fc1_fp8", "Float8_e4m3fn", []int64{256, 32, 16, 16}, ""),
			spec("fc2_fp8", "Float8_e4m3fn", []int64{32, 128, 16, 16}, ""),
		}
		caches[layer] = cache
	}
	for layer := 0; layer < 18; layer++ {
		prefix := fmt.Sprintf("encoder.%02d.", layer)
		add(prefix+"cache.before_remove_bits", layer, caches[layer]...)
		add(prefix+"cache.after_remove_bits", layer, caches[layer]...)
		half := spec("residual", "Half", []int64{n, t / 16, 64, 16, 8}, "NTCtc")
		i8 := spec("int8", "Char", []int64{n, t / 16, 32, 16, 16}, "NTCtc")
		i8.ScaleTag = int('C')
		scale := spec("scale", "Float", []int64{n, t}, "")
		counter := spec("counter", "Int", []int64{4}, "")
		add(prefix+"enter", layer, half, i8, scale, counter)
		qkv := spec("qkv", "Half", []int64{n, t / 16, 3, 8, 8, 16, 8}, "NTQHDtd")
		qkv.Tags[2] = 0 // KOI_DIM_MAT_Q_K_V from the pinned koi.h.
		attn := spec("attention", "Half", []int64{n, t / 16, 8, 8, 16, 8}, "NTHDtd")
		proj := spec("projection", "Half", []int64{n, t / 16, 64, 16, 8}, "NTCtc")
		rms := spec("rms_fp8", "Float8_e4m3fn", []int64{n, t / 16, 32, 16, 16}, "NTCtc")
		fc1 := spec("swiglu", "Float8_e4m3fn", []int64{n * t / 16, 128, 16, 16}, "MNmn")
		fc2 := spec("fc2", "Half", []int64{n, t / 16, 64, 16, 8}, "NTCtc")
		add(prefix+"qkv", layer, qkv, counter)
		add(prefix+"attention", layer, attn)
		add(prefix+"projection", layer, proj, counter)
		add(prefix+"norm1", layer, half, rms)
		add(prefix+"swiglu", layer, fc1, counter)
		add(prefix+"fc2", layer, fc2, counter)
		add(prefix+"norm2", layer, half, i8, scale)
	}
	add("encoder.output", -1, spec("output", "Half", []int64{n, t, 512}, ""))
	add("upsample", -1, spec("output", "Half", []int64{n, t * 2, 512}, ""))
	add("crf.weight", -1, spec("weight", "Half", []int64{4096, 512}, ""))
	add("scores", -1, spec("output", "Half", []int64{n, t * 2, 4096}, ""))
	return out
}

// EventNames exposes ordering without allowing a caller to mutate the policy.
func EventNames(g Geometry) []string {
	out := []string{}
	for _, e := range eventRegistry(g) {
		out = append(out, e.Name)
	}
	return out
}

func descriptorRegistry(event string, g Geometry) []Descriptor {
	n, t := int64(g.Batch), int64(g.EncoderT)
	m := n * t / 16
	d := func(name, dtype string, shape []int64, tags string, scale int) Descriptor {
		s := spec(name, dtype, shape, tags)
		strides := make([]int64, len(shape))
		stride := int64(1)
		for i := len(shape) - 1; i >= 0; i-- {
			strides[i] = stride
			stride *= shape[i]
		}
		scaleType := "None"
		if scale != 0 {
			scaleType = "Float"
		}
		return Descriptor{name, dtype, shape, strides, s.Tags, scale, scaleType}
	}
	if !strings.HasPrefix(event, "encoder.") {
		return []Descriptor{}
	}
	stage := event[strings.LastIndex(event, ".")+1:]
	half := func(name string) Descriptor { return d(name, "Half", []int64{n, t / 16, 64, 16, 8}, "NTCtc", 0) }
	qkv := func(name string) Descriptor {
		v := d(name, "Half", []int64{n, t / 16, 3, 8, 8, 16, 8}, "NTQHDtd", 0)
		v.Tags[2] = 0
		return v
	}
	switch stage {
	case "qkv":
		w := d("weights", "Char", []int64{3, 8, 4, 32, 16, 16}, "QHDCdc", int('C'))
		w.Tags[0] = 0
		return []Descriptor{d("input", "Char", []int64{n, t / 16, 32, 16, 16}, "NTCtc", int('C')), w, d("sincos", "Half", []int64{t / 16, 8, 16, 8}, "TDtd", 0), qkv("output")}
	case "attention":
		return []Descriptor{qkv("input"), d("output", "Half", []int64{n, t / 16, 8, 8, 16, 8}, "NTHDtd", 0)}
	case "projection":
		return []Descriptor{d("input", "Half", []int64{m, 64, 16, 8}, "MKmk", 0), d("weight", "Half", []int64{32, 64, 16, 8}, "NKnk", 0), d("bias", "Half", []int64{512}, "N", 0), d("output", "Half", []int64{m, 64, 16, 8}, "MNmn", 0)}
	case "norm1":
		return []Descriptor{half("input"), half("residual"), d("weight", "Half", []int64{512}, "C", 0), half("output"), d("fp8", "Float8_e4m3fn", []int64{n, t / 16, 32, 16, 16}, "NTCtc", 0)}
	case "swiglu":
		return []Descriptor{d("input", "Float8_e4m3fn", []int64{m, 32, 16, 16}, "MKmk", 0), d("weight", "Float8_e4m3fn", []int64{256, 32, 16, 16}, "NKnk", 0), d("output", "Float8_e4m3fn", []int64{m, 128, 16, 16}, "MNmn", 0)}
	case "fc2":
		return []Descriptor{d("input", "Float8_e4m3fn", []int64{m, 128, 16, 16}, "MKmk", 0), d("weight", "Float8_e4m3fn", []int64{32, 128, 16, 16}, "NKnk", 0), d("output", "Half", []int64{m, 64, 16, 8}, "MNmn", 0)}
	case "norm2":
		return []Descriptor{half("input"), half("residual"), d("weight", "Half", []int64{512}, "C", 0), half("output"), d("int8", "Char", []int64{n, t / 16, 32, 16, 16}, "NTCtc", int('C'))}
	default:
		return []Descriptor{}
	}
}
