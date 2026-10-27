package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type supGPUBuffer struct {
	dtype tensor.DType
	shape []int64
	tags  []tensor.KoiTag
	data  []byte
}

type SUPGPUBufferSnapshot struct {
	Name     string
	DType    tensor.DType
	Shape    []int64
	Strides  []int64
	ScaleTag tensor.KoiTag
	Tags     []tensor.KoiTag
	Data     []byte
}

func (b supGPUBuffer) snapshot(name string) SUPGPUBufferSnapshot {
	return SUPGPUBufferSnapshot{Name: name, DType: b.dtype, Shape: slices.Clone(b.shape), Strides: supContiguous(b.shape), Tags: slices.Clone(b.tags), Data: slices.Clone(b.data)}
}
func supHalfBuffer(half []uint16, shape []int64, tags string) supGPUBuffer {
	b := supGPUBuffer{dtype: tensor.DTypeF16, shape: slices.Clone(shape), data: make([]byte, 2*len(half))}
	for _, c := range tags {
		b.tags = append(b.tags, tensor.KoiTag(c))
	}
	for i, h := range half {
		binary.LittleEndian.PutUint16(b.data[2*i:], h)
	}
	return b
}
func supByteBuffer(data []byte, dtype tensor.DType, shape []int64, tags string) supGPUBuffer {
	b := supGPUBuffer{dtype: dtype, shape: slices.Clone(shape), data: slices.Clone(data)}
	for _, c := range tags {
		b.tags = append(b.tags, tensor.KoiTag(c))
	}
	return b
}
func supF32Buffer(data []float32, shape []int64) supGPUBuffer {
	b := supGPUBuffer{dtype: tensor.DTypeF32, shape: slices.Clone(shape), tags: make([]tensor.KoiTag, len(shape)), data: make([]byte, 4*len(data))}
	for i, x := range data {
		binary.LittleEndian.PutUint32(b.data[i*4:], math.Float32bits(x))
	}
	return b
}
func supRound(half []uint16) []uint16 {
	out := slices.Clone(half)
	for i, h := range out {
		out[i] = (h + 8) & 0xfff0
	}
	return out
}
func supHalf(values []float32) ([]uint16, error) {
	out := make([]uint16, len(values))
	for i, v := range values {
		h := signal.Float16Bits(v)
		if h&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("%w: nonfinite half at %d", ErrGPUWeights, i)
		}
		out[i] = h
	}
	return out, nil
}

type SUPEncoderHalfWeights struct{ QKV, Projection, ProjectionBias, FC1, FC2, Norm1, Norm2 []uint16 }
type SUPEncoderGPUCache struct{ before, after map[string]supGPUBuffer }

// Snapshot returns independent copies. The source's half remove_bits is
// applied only after I8/FP8 quantization and never changes an FP8 buffer.
func (c SUPEncoderGPUCache) Snapshot(after bool) []SUPGPUBufferSnapshot {
	m := c.before
	if after {
		m = c.after
	}
	names := []string{"qkv_half", "qkv_int8", "qkv_scale", "proj_weight", "proj_bias", "norm1_weight", "norm2_weight", "fc1_half", "fc1_fp8", "fc2_fp8"}
	out := make([]SUPGPUBufferSnapshot, 0, len(names))
	for _, n := range names {
		out = append(out, m[n].snapshot(n))
	}
	return out
}

func PackSUPEncoderGPU(w SUPEncoderHalfWeights) (SUPEncoderGPUCache, error) {
	lengths := []int{3 * 512 * 512, 512 * 512, 512, 4096 * 512, 512 * 2048, 512, 512}
	values := [][]uint16{w.QKV, w.Projection, w.ProjectionBias, w.FC1, w.FC2, w.Norm1, w.Norm2}
	for role, v := range values {
		if len(v) != lengths[role] {
			return SUPEncoderGPUCache{}, fmt.Errorf("%w: SUP encoder role %d shape differs", ErrGPUWeights, role)
		}
		for i, h := range v {
			if h&0x7c00 == 0x7c00 {
				return SUPEncoderGPUCache{}, fmt.Errorf("%w: SUP encoder role%d element%d nonfinite", ErrGPUWeights, role, i)
			}
		}
	}
	q, err := prepareSUPQKVF16Weights(w.QKV)
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	qi, err := BuildSUPQKVInt8(q.interleavedHalf)
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	qh, err := permuteContiguous(q.interleavedHalf, []int64{3, 8, 4, 16, 64, 8}, []int{0, 1, 2, 4, 3, 5})
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	proj, err := permuteContiguous(w.Projection, []int64{32, 16, 64, 8}, []int{0, 2, 1, 3})
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	fp8, err := PackSUPFeedForwardFP8(w.FC1, w.FC2)
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	interleaved := make([]uint16, len(w.FC1))
	shuffle := [16]int{0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}
	for block := 0; block < 256; block++ {
		for lane := 0; lane < 16; lane++ {
			src := (block%2)*2048 + (block/2)*16 + shuffle[lane]
			copy(interleaved[(block*16+lane)*512:(block*16+lane+1)*512], w.FC1[src*512:(src+1)*512])
		}
	}
	fc1, err := permuteContiguous(interleaved, []int64{256, 16, 64, 8}, []int{0, 2, 1, 3})
	if err != nil {
		return SUPEncoderGPUCache{}, err
	}
	i8bytes := make([]byte, len(qi.Tiled))
	for i, v := range qi.Tiled {
		i8bytes[i] = byte(v)
	}
	c := SUPEncoderGPUCache{before: map[string]supGPUBuffer{}, after: map[string]supGPUBuffer{}}
	c.before["qkv_int8"] = supByteBuffer(i8bytes, tensor.DTypeI8, []int64{3, 8, 4, 32, 16, 16}, "\x00HDCdc")
	c.before["qkv_scale"] = supF32Buffer(qi.InverseScale, []int64{3, 8, 64})
	c.before["proj_weight"] = supHalfBuffer(proj, []int64{32, 64, 16, 8}, "NKnk")
	c.before["proj_bias"] = supHalfBuffer(w.ProjectionBias, []int64{512}, "N")
	c.before["norm1_weight"] = supHalfBuffer(w.Norm1, []int64{512}, "C")
	c.before["norm2_weight"] = supHalfBuffer(w.Norm2, []int64{512}, "C")
	c.before["fc1_half"] = supHalfBuffer(fc1, []int64{256, 64, 16, 8}, "NKnk")
	c.before["fc1_fp8"] = supByteBuffer(fp8.fc1, tensor.DTypeE4M3, []int64{256, 32, 16, 16}, "NKnk")
	c.before["fc2_fp8"] = supByteBuffer(fp8.fc2, tensor.DTypeE4M3, []int64{32, 128, 16, 16}, "NKnk")
	// qkv_half is a retained source cache, not the selected I8 descriptor.
	c.before["qkv_half"] = supHalfBuffer(qh, []int64{3, 8, 4, 64, 16, 8}, "\x00HDCdc")
	for name, b := range c.before {
		if name == "qkv_half" || name == "proj_weight" || name == "norm1_weight" || name == "norm2_weight" || name == "fc1_half" {
			h := make([]uint16, len(b.data)/2)
			for i := range h {
				h[i] = binary.LittleEndian.Uint16(b.data[2*i:])
			}
			b = supHalfBuffer(supRound(h), b.shape, stringTags(b.tags))
		}
		c.after[name] = b
	}
	for _, b := range c.after {
		if err := validateSUPBuffer(b); err != nil {
			return SUPEncoderGPUCache{}, err
		}
	}
	return c, nil
}
func stringTags(tags []tensor.KoiTag) string {
	b := make([]byte, len(tags))
	for i, v := range tags {
		b[i] = byte(v)
	}
	return string(b)
}

// SUPRoPEFrequencies reproduces the half casts before reciprocal and multiply.
// Sin and cos remain individual pinned CUDA math operations.
func SUPRoPEFrequencies() []uint16 {
	out := make([]uint16, 4096*32)
	for d := 0; d < 32; d++ {
		power := signal.Float16Bits(float32(math.Pow(10000, float64(float32(2*d)/64))))
		inv := signal.Float16Bits(1 / signal.Float32FromHalf(power))
		for t := 0; t < 4096; t++ {
			pos := signal.Float32FromHalf(signal.Float16Bits(float32(t)))
			out[t*32+d] = signal.Float16Bits(pos * signal.Float32FromHalf(inv))
		}
	}
	return out
}

type SUPGPUWeights struct {
	spec     Spec
	receipts []WeightReceipt
	convs    [5][2]supGPUBuffer
	encoders [18]SUPEncoderGPUCache
	upsample [2]supGPUBuffer
	crf      supGPUBuffer
}

func (w SUPGPUWeights) Receipts() []WeightReceipt {
	out := slices.Clone(w.receipts)
	for i := range out {
		out[i].Shape = slices.Clone(out[i].Shape)
	}
	return out
}

func supGPURoles(manifest []WeightReceipt) (map[string]WeightReceipt, error) {
	authoritative, err := WeightManifest(SUP)
	if err != nil {
		return nil, err
	}
	if len(manifest) != 139 {
		return nil, fmt.Errorf("%w: SUP needs 139 archives", ErrGPUWeights)
	}
	want := map[string]WeightReceipt{}
	for _, p := range authoritative {
		want[p.Name] = p
	}
	out := map[string]WeightReceipt{}
	for _, p := range manifest {
		a, ok := want[p.Name]
		if !ok || p.SHA256 != a.SHA256 || !slices.Equal(p.Shape, a.Shape) {
			return nil, fmt.Errorf("%w: SUP role %s differs", ErrGPUWeights, p.Name)
		}
		if _, dup := out[p.Name]; dup {
			return nil, fmt.Errorf("%w: duplicate SUP role", ErrGPUWeights)
		}
		out[p.Name] = p
	}
	return out, nil
}

func LoadSUPGPUWeights(ctx context.Context, e *tensor.Engine, directory string) (out SUPGPUWeights, err error) {
	if ctx == nil {
		return out, fmt.Errorf("nil SUP context")
	}
	if err = ctx.Err(); err != nil {
		return out, err
	}
	if e == nil {
		return out, tensor.ErrClosed
	}
	out.spec, err = Inspect(directory)
	if err != nil {
		return out, err
	}
	pinned, _ := Pinned(SUP)
	if out.spec != pinned {
		return SUPGPUWeights{}, fmt.Errorf("%w: pinned SUP config required", ErrGPUWeights)
	}
	out.receipts, err = WeightManifest(SUP)
	if err != nil {
		return out, err
	}
	if _, err = supGPURoles(out.receipts); err != nil {
		return out, err
	}
	read := func(pin WeightReceipt) ([]uint16, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		w, err := loadWeight(e, filepath.Join(directory, pin.Name), pin)
		if err != nil {
			return nil, err
		}
		v, readErr := w.Data()
		if err = errors.Join(readErr, w.Close()); err != nil {
			return nil, err
		}
		return supHalf(v)
	}
	for i := 0; i < 5; i++ {
		for j := 0; j < 2; j++ {
			p := out.receipts[2*i+j]
			h, err := read(p)
			if err != nil {
				return SUPGPUWeights{}, err
			}
			tags := "NCW"
			if j == 1 {
				tags = "C"
			}
			out.convs[i][j] = supHalfBuffer(h, p.Shape, tags)
		}
	}
	for layer := 0; layer < 18; layer++ {
		var h [7][]uint16
		for j := range h {
			h[j], err = read(out.receipts[10+7*layer+j])
			if err != nil {
				return SUPGPUWeights{}, err
			}
		}
		out.encoders[layer], err = PackSUPEncoderGPU(SUPEncoderHalfWeights{h[0], h[1], h[2], h[3], h[4], h[5], h[6]})
		if err != nil {
			return SUPGPUWeights{}, fmt.Errorf("SUP encoder %d: %w", layer, err)
		}
	}
	for j := 0; j < 2; j++ {
		p := out.receipts[136+j]
		h, err := read(p)
		if err != nil {
			return SUPGPUWeights{}, err
		}
		tags := "NC"
		if j == 1 {
			tags = "C"
		}
		out.upsample[j] = supHalfBuffer(h, p.Shape, tags)
	}
	h, err := read(out.receipts[138])
	if err != nil {
		return SUPGPUWeights{}, err
	}
	for i, v := range h {
		h[i] = signal.Float16Bits(signal.Float32FromHalf(v) * 5)
	}
	out.crf = supHalfBuffer(h, []int64{4096, 512}, "NC")
	return out, ctx.Err()
}
