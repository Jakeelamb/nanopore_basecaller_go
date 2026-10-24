package h02

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

// recordPlan turns the Go plan into the receipt's recorded layer table and
// cross-checks every convolution against the capture. A plan whose offset,
// shape or layout differs from the oracle is a rejection, not a warning: the
// placement is what keeps the nine padding rows in place.
func recordPlan(plan *model.HACSmallConvPlan, capture hacgpu.Capture, convs []ConvSpec) ([]LayerPlan, error) {
	out := make([]LayerPlan, 0, len(plan.Layers))
	for i, l := range plan.Layers {
		stage, ok := capture.Stage(fmt.Sprintf("conv%d", i))
		if !ok {
			return nil, fmt.Errorf("capture has no conv%d stage", i)
		}
		rec := LayerPlan{
			Index: l.Index, CIn: l.Params.CIn, COut: l.Params.COut,
			WinLen: l.Params.WinLen, Stride: l.Params.Stride, Padding: l.Params.Padding(),
			Activation: l.Params.Activation.String(), Layout: l.Params.OutputLayout.String(),
			OutputTPadding: l.Params.OutputTPadding, TIn: l.TIn, TOut: l.TOut,
			OffsetBytes: l.OffsetBytes, Bytes: l.Geometry.Bytes,
			CropOffsetElements: l.CropOffsetElements(), CropShape: l.CropShape(),
			CropStrides: l.CropStrides(), BatchStride: l.BatchStride(),
			Executes:           l.Executes,
			CaptureOffsetBytes: stage.OffsetBytes, CaptureShape: stage.Shape,
			CaptureLayout: stage.Layout,
		}
		if rec.OffsetBytes != rec.CaptureOffsetBytes {
			return nil, fmt.Errorf("conv%d plan offset %d differs from the captured %d",
				i, rec.OffsetBytes, rec.CaptureOffsetBytes)
		}
		// The captured shape is the whole allocation: for NTC that is
		// [N, T_out+2*padding, C_out], and a CUTLASS layer reserves
		// [T_out+2*padding+5, N, C_out] (WorkingMemory.cpp:87-94).
		var want []int64
		switch l.Params.OutputLayout {
		case model.GPUStorageNTC:
			want = []int64{l.Geometry.N, l.Geometry.Rows, l.Geometry.C}
		case model.GPUStorageCUTLASSI8, model.GPUStorageCUTLASSF16:
			want = []int64{l.Geometry.Rows, l.Geometry.N, l.Geometry.C}
		default:
			return nil, fmt.Errorf("conv%d plan layout %s has no declared captured shape", i, rec.Layout)
		}
		if !sameShape(want, stage.Shape) {
			return nil, fmt.Errorf("conv%d plan allocation %v (crop %v, padding %d) does not describe the captured %v",
				i, want, rec.CropShape, rec.OutputTPadding, stage.Shape)
		}
		if rec.Layout != stage.Layout {
			return nil, fmt.Errorf("conv%d plan layout %s differs from the captured %s",
				i, rec.Layout, stage.Layout)
		}
		if rec.Bytes != stage.Bytes {
			return nil, fmt.Errorf("conv%d plan extent %d differs from the captured %d",
				i, rec.Bytes, stage.Bytes)
		}
		out = append(out, rec)
	}
	return out, nil
}

// replay runs the two declared convolutions once and compares every written
// byte with the capture.
func replay(dir string, r *Receipt, plan *model.HACSmallConvPlan, convs []ConvSpec, capture hacgpu.Capture) error {
	if len(plan.Layers) < 2 {
		return fmt.Errorf("the HAC stack needs at least two planned convolutions")
	}
	for i := 0; i < 2; i++ {
		if !plan.Layers[i].Executes {
			return fmt.Errorf("conv%d is planned but not marked executable", i)
		}
	}
	executed := plan.Layers[:2]

	e, err := tensor.Open(r.Backend)
	if err != nil {
		return err
	}
	defer e.Close()
	d, err := e.NewCUDADevice(r.Ordinal)
	if err != nil {
		return err
	}
	s, err := d.NewStream()
	if err != nil {
		return err
	}

	weights, err := LoadConvWeights(e, r.Model, convs)
	if err != nil {
		return err
	}
	r.Inputs[filepath.Join("internal", "model", "hac_weights.json")] = weights.ManifestSHA

	// Pack in Go, then prove the Go bytes against the verification-only packing
	// oracle that reproduces ConvStack.cpp:230-234 natively.
	if r.Packing, err = packAndVerify(d, s, executed, weights); err != nil {
		return err
	}

	// One backing allocation, exactly the reserved extent, pre-filled with the
	// declared pattern so an unwritten element can be proven untouched.
	if plan.ReservationBytes%2 != 0 {
		return fmt.Errorf("reservation %d is not a whole number of F16 elements", plan.ReservationBytes)
	}
	fill := bytes.Repeat([]byte{FillPattern}, int(plan.ReservationBytes))
	backing, err := d.NewStorage(s, tensor.DTypeF16,
		[]int64{plan.ReservationBytes / 2}, []tensor.KoiTag{tensor.TagN}, fill)
	if err != nil {
		return err
	}
	defer backing.Close()

	views, err := bindViews(backing, plan, executed)
	if err != nil {
		return err
	}

	// The input is the exact bytes the native graph consumed.
	inputPayload, err := capture.RawPayload(r.Capture, "input")
	if err != nil {
		return err
	}
	if int64(len(inputPayload)) != plan.Input.Bytes {
		return fmt.Errorf("captured input is %d bytes, the plan reserves %d", len(inputPayload), plan.Input.Bytes)
	}
	if err = views.input.Upload(s, inputPayload); err != nil {
		return err
	}

	caches := make([]*model.HACSmallConv, len(executed))
	for i, l := range executed {
		w := weights.Layers[i]
		cache, cerr := model.NewHACSmallConv(d, s, l.Params, w.Weight, w.Bias)
		if cerr != nil {
			return cerr
		}
		caches[i] = cache
		defer cache.Close()
	}

	// conv0: unpadded, so the crop is the whole allocation.
	if err = caches[0].Run(s, views.input, views.alloc[0], executed[0]); err != nil {
		return fmt.Errorf("conv0: %w", err)
	}
	if r.Stages, err = appendStage(r.Stages, r.Capture, capture, "conv0", views.alloc[0], s); err != nil {
		return err
	}

	// The conv1 allocation aliases the input buffer the graph has now consumed
	// (WorkingMemory::next reuses the front of the reservation without clearing
	// it). Pre-fill it with the declared pattern so the rows the kernel must not
	// write are observable. Nothing reads this allocation before conv1 writes it.
	if err = views.alloc[1].Fill(s, []byte{FillPattern, FillPattern}); err != nil {
		return err
	}
	r.DeclaredPreparation = append(r.DeclaredPreparation,
		"conv1-allocation-pre-filled-with-"+fmt.Sprintf("%#02x", FillPattern))

	// conv1: reads conv0's output, writes the cropped view nine rows into its
	// own allocation.
	if err = caches[1].Run(s, views.alloc[0], views.crop[1], executed[1]); err != nil {
		return fmt.Errorf("conv1: %w", err)
	}
	if r.Stages, err = appendStage(r.Stages, r.Capture, capture, "conv1", views.alloc[1], s); err != nil {
		return err
	}
	if r.Padding, err = checkPadding(r.Padding, 1, views.alloc[1], s, executed[1].Params); err != nil {
		return err
	}

	// Lifetime: the input and conv0 views are no longer needed; closing them must
	// not disturb conv1's retained output. Then close conv1's crop view and prove
	// the parent allocation still holds the same bytes.
	conv1Before, err := views.alloc[1].Download(s)
	if err != nil {
		return err
	}
	for _, name := range []string{"input", "conv0-allocation"} {
		v := views.input
		if name == "conv0-allocation" {
			v = views.alloc[0]
		}
		if cerr := v.Close(); cerr != nil {
			return cerr
		}
		r.Lifetime.ClosedViews = append(r.Lifetime.ClosedViews, name)
	}
	if cerr := views.crop[1].Close(); cerr != nil {
		return cerr
	}
	r.Lifetime.ClosedViews = append(r.Lifetime.ClosedViews, "conv1-crop")
	conv1After, err := views.alloc[1].Download(s)
	if err != nil {
		return err
	}
	if !bytes.Equal(conv1Before, conv1After) {
		return fmt.Errorf("conv1 allocation changed after its crop view was released")
	}
	r.Lifetime.ParentStillEqual = true

	// The conv2 allocation is not executed, so its whole extent must still hold
	// the declared fill pattern: proof the two kernels wrote nothing outside
	// their own views.
	whole, err := backing.Download(s)
	if err != nil {
		return err
	}
	if r.Lifetime.BackingStillFill, err = stillFill(whole, plan, executed); err != nil {
		return err
	}
	return nil
}

// views are the working-memory views the plan binds to the single backing.
type convViews struct {
	input *tensor.Storage
	alloc []*tensor.Storage
	crop  []*tensor.Storage
}

func bindViews(backing *tensor.Storage, plan *model.HACSmallConvPlan, executed []model.HACSmallConvLayer) (convViews, error) {
	var v convViews
	input, err := backing.View(uint64(plan.InputOffsetBytes/2), []int64{plan.Input.N, plan.Input.T, plan.Input.C},
		[]int64{plan.Input.T * plan.Input.C, plan.Input.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return v, err
	}
	v.input = input
	v.alloc = make([]*tensor.Storage, len(executed))
	v.crop = make([]*tensor.Storage, len(executed))
	for i, l := range executed {
		shape := []int64{l.Geometry.N, l.Geometry.Rows, l.Geometry.C}
		strides := []int64{l.Geometry.Rows * l.Geometry.C, l.Geometry.C, 1}
		alloc, aerr := backing.View(uint64(l.OffsetBytes/2), shape, strides,
			[]tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
		if aerr != nil {
			return v, aerr
		}
		v.alloc[i] = alloc
		crop, cerr := alloc.View(l.CropOffsetElements(), l.CropShape(), l.CropStrides(),
			[]tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
		if cerr != nil {
			return v, cerr
		}
		v.crop[i] = crop
	}
	return v, nil
}

// appendStage compares the bytes the Go caller produced with the captured stage
// payload, excluding only the rows the source declares it never writes.
func appendStage(stages []Stage, captureDir string, capture hacgpu.Capture, name string, got *tensor.Storage, s *tensor.Stream) ([]Stage, error) {
	stage, ok := capture.Stage(name)
	if !ok {
		return stages, fmt.Errorf("capture has no stage %q", name)
	}
	payload, err := capture.RawPayload(captureDir, name)
	if err != nil {
		return stages, err
	}
	observed, err := got.Download(s)
	if err != nil {
		return stages, err
	}
	if int64(len(observed)) != stage.Bytes {
		return stages, fmt.Errorf("stage %q observed %d bytes, captured %d", name, len(observed), stage.Bytes)
	}
	written, err := writtenBytes(stage, observed, payload)
	if err != nil {
		return stages, fmt.Errorf("stage %q: %w", name, err)
	}
	expected, err := writtenPayload(stage, payload)
	if err != nil {
		return stages, fmt.Errorf("stage %q: %w", name, err)
	}
	rec := Stage{
		Name: name, CaptureSHA: stage.DataSHA, GoSHA: hashBytes(observed),
		Shape: stage.Shape, WrittenBytes: int64(len(written)),
		GoWrittenSHA: hashBytes(written), CapWrittenSHA: hashBytes(expected),
		UnwrittenRows: stage.UnwrittenT, Equal: bytes.Equal(written, expected),
	}
	if !rec.Equal {
		return stages, fmt.Errorf("stage %q differs from the capture in a written element", name)
	}
	return append(stages, rec), nil
}

// writtenBytes returns the concatenation of every row the source writes, in the
// order the mask visits them.
func writtenBytes(stage hacgpu.Stage, observed, payload []byte) ([]byte, error) {
	if len(observed) != len(payload) {
		return nil, fmt.Errorf("observed %d bytes, payload %d", len(observed), len(payload))
	}
	out := make([]byte, 0, len(payload))
	if err := visitWrittenRows(stage, func(offset, length int) {
		out = append(out, observed[offset:offset+length]...)
	}); err != nil {
		return nil, err
	}
	return out, nil
}

func writtenPayload(stage hacgpu.Stage, payload []byte) ([]byte, error) {
	out := make([]byte, 0, len(payload))
	err := visitWrittenRows(stage, func(offset, length int) {
		out = append(out, payload[offset:offset+length]...)
	})
	return out, err
}

// visitWrittenRows walks the rows the source declares it writes. For NTC the
// time axis is the middle one and batches are contiguous blocks; for a CUTLASS
// layer the time axis leads (WorkingMemory.cpp:87-94). The mask is derived from
// the stage's declared `unwritten_t` only.
func visitWrittenRows(stage hacgpu.Stage, visit func(offset, length int)) error {
	if len(stage.Shape) != 3 {
		return fmt.Errorf("stage shape %v is not rank three", stage.Shape)
	}
	// Half only: every stage this consumer compares is F16.
	const elemSize = 2
	var n, t, c int
	var timeLeading bool
	switch stage.Layout {
	case "NTC":
		n, t, c = int(stage.Shape[0]), int(stage.Shape[1]), int(stage.Shape[2])
	case "CUTLASS_TNC_I8", "CUTLASS_TNC_F16":
		t, n, c = int(stage.Shape[0]), int(stage.Shape[1]), int(stage.Shape[2])
		timeLeading = true
	default:
		return fmt.Errorf("stage layout %q has no declared row order", stage.Layout)
	}
	rowBytes := c * elemSize
	if n <= 0 || t <= 0 || rowBytes <= 0 || stage.Bytes != int64(n)*int64(t)*int64(rowBytes) {
		return fmt.Errorf("stage shape %v and %d bytes do not describe contiguous rows", stage.Shape, stage.Bytes)
	}
	// Visit rows in memory order: batches are contiguous blocks for NTC, and the
	// time axis leads for CUTLASS.
	if timeLeading {
		for ti := 0; ti < t; ti++ {
			if stageRowUnwritten(stage, ti) {
				continue
			}
			for ni := 0; ni < n; ni++ {
				visit((ti*n+ni)*rowBytes, rowBytes)
			}
		}
		return nil
	}
	for ni := 0; ni < n; ni++ {
		for ti := 0; ti < t; ti++ {
			if stageRowUnwritten(stage, ti) {
				continue
			}
			visit((ni*t+ti)*rowBytes, rowBytes)
		}
	}
	return nil
}

func stageRowUnwritten(stage hacgpu.Stage, t int) bool {
	for _, r := range stage.UnwrittenT {
		if t >= r[0] && t < r[0]+r[1] {
			return true
		}
	}
	return false
}

// checkPadding proves the rows outside the crop still hold the declared fill
// pattern. They are declared exclusions from the value comparison, so this is
// the only evidence that the crop and the batch stride were honoured.
func checkPadding(padding []Padding, index int, alloc *tensor.Storage, s *tensor.Stream, params model.SmallConvParams) ([]Padding, error) {
	observed, err := alloc.Download(s)
	if err != nil {
		return padding, err
	}
	rows := params.OutputTPadding
	if rows == 0 {
		return padding, nil
	}
	rowBytes := int(params.COut * 2)
	total := int64(len(observed)) / int64(rowBytes)
	rec := Padding{Index: index, Rows: 2 * rows, Bytes: 2 * rows * int64(rowBytes), AllFill: true, ExampleByte: FillPattern}
	check := func(from, to int64) error {
		for t := from; t < to; t++ {
			chunk := observed[t*int64(rowBytes) : (t+1)*int64(rowBytes)]
			for _, b := range chunk {
				if b != FillPattern {
					rec.AllFill = false
					rec.ExampleByte = b
					return fmt.Errorf("conv%d row %d outside the crop was written (byte %#02x)", index, t, b)
				}
			}
		}
		return nil
	}
	if err = check(0, rows); err != nil {
		return padding, err
	}
	if err = check(total-rows, total); err != nil {
		return padding, err
	}
	return append(padding, rec), nil
}

// stillFill proves the conv2 allocation was never touched, and that the only
// elements of the backing that are not fill lie inside the executed views.
func stillFill(whole []byte, plan *model.HACSmallConvPlan, executed []model.HACSmallConvLayer) (bool, error) {
	if plan.ReservationBytes != int64(len(whole)) {
		return false, fmt.Errorf("backing is %d bytes, the plan reserves %d", len(whole), plan.ReservationBytes)
	}
	// An executed layer's allocation legitimately overlaps a later one, so only
	// the part of a not-yet-run allocation that no executed layer covers is
	// evidence.
	var executedRanges [][2]int64
	for _, l := range executed {
		executedRanges = append(executedRanges, [2]int64{l.OffsetBytes, l.OffsetBytes + l.Geometry.Bytes})
	}
	for i := len(executed); i < len(plan.Layers); i++ {
		l := plan.Layers[i]
		for _, span := range subtract(l.OffsetBytes, l.OffsetBytes+l.Geometry.Bytes, executedRanges) {
			for _, b := range whole[span[0]:span[1]] {
				if b != FillPattern {
					return false, fmt.Errorf("conv%d allocation [%d,%d) was written before it ran", i, span[0], span[1])
				}
			}
		}
	}
	return true, nil
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hashString(s string) string { return hashBytes([]byte(s)) }

// runtimePins hashes the CUDA and Torch objects this process actually mapped, so
// the receipt names the files that ran rather than a link line.
func runtimePins() (map[string]string, error) {
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	families := map[string]bool{}
	for _, prefix := range []string{"libtorch_cpu.so", "libc10.so", "libtorch_cuda.so", "libc10_cuda.so",
		"libcudart.so", "libcublas.so", "libcuda.so", "libkoi"} {
		families[prefix] = true
	}
	seen := map[string]bool{}
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !filepath.IsAbs(path) || seen[path] {
			continue
		}
		base := filepath.Base(path)
		matched := false
		for prefix := range families {
			if strings.HasPrefix(base, prefix) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		seen[path] = true
		sum, herr := fileSHA(path)
		if herr != nil {
			return nil, herr
		}
		out[path] = sum
	}
	if len(out) < 6 {
		names := make([]string, 0, len(out))
		for k := range out {
			names = append(names, k)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("only %d pinned runtime objects were mapped: %v", len(out), names)
	}
	return out, nil
}

// subtract returns the parts of [start,end) not covered by any of the given
// half-open ranges.
func subtract(start, end int64, covered [][2]int64) [][2]int64 {
	spans := [][2]int64{{start, end}}
	for _, c := range covered {
		var next [][2]int64
		for _, s := range spans {
			if c[1] <= s[0] || s[1] <= c[0] {
				next = append(next, s)
				continue
			}
			if s[0] < c[0] {
				next = append(next, [2]int64{s[0], c[0]})
			}
			if c[1] < s[1] {
				next = append(next, [2]int64{c[1], s[1]})
			}
		}
		spans = next
	}
	return spans
}
