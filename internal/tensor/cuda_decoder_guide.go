package tensor

import "math"

// DecoderGuideProgram returns the exact driver-linked cubin retained by this
// device. Initialization compiles pinned PTX but performs no decoder timestep.
func (d *Device) DecoderGuideProgram() ([]byte, error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "decoder_guide_program", device: d.state.id})
	if err != nil {
		return nil, err
	}
	if r.count <= 0 || r.count > 16*1024*1024 {
		return nil, &LayoutError{Op: "decoder_guide_program", Reason: "invalid linked cubin extent"}
	}
	full, err := e.backend.cuda(cudaRequest{op: "decoder_guide_program", device: d.state.id, count: r.count})
	if err != nil {
		return nil, err
	}
	if full.count != r.count || len(full.bytes) != r.count {
		return nil, &LayoutError{Op: "decoder_guide_program", Reason: "linked cubin changed while copying"}
	}
	return full.bytes, nil
}

func decoderGuideMetadata(m []StorageMeta, clamp, blank float32) error {
	const op = "decoder_guide_frame"
	fail := func(why string) error { return &LayoutError{Op: op, Reason: why} }
	if math.IsNaN(float64(clamp)) || math.IsInf(float64(clamp), 0) || clamp < 0 || math.IsNaN(float64(blank)) || math.IsInf(float64(blank), 0) {
		return fail("finite nonnegative clamp and finite blank required")
	}
	if len(m) != 3 {
		return fail("three guide operands required")
	}
	for i, v := range m {
		want := DTypeF32
		if i == 0 {
			want = DTypeF16
		}
		if v.DType != want {
			return &DTypeError{Op: op, Reason: "half scores and float guide rows required"}
		}
		if len(v.Shape) != 2 || len(v.Strides) != 2 || len(v.Tags) != 2 || v.Tags[0] != TagN || v.Tags[1] != TagC || v.ScalePresent || v.Strides[1] != 1 || v.Strides[0] < v.Shape[1] {
			return fail("rank-two unscaled row-major NC views required")
		}
	}
	n, states := m[1].Shape[0], m[1].Shape[1]
	if !((states == 256 && n > 0 && n%64 == 0) || (states == 1024 && n > 0 && n%32 == 0)) || n > math.MaxInt32 {
		return fail("fixed HAC/SUP batch tiles required")
	}
	if !equalShape(m[0].Shape, []int64{n, 4 * states}) || !equalShape(m[2].Shape, m[1].Shape) || m[1].Strides[0] != states || m[2].Strides[0] != states {
		return fail("guide dimensions or contiguous guide row layout differ")
	}
	starts, ends := [3]uint64{}, [3]uint64{}
	for i, v := range m {
		end, err := storageSpan(v.Shape, v.Strides, v.OffsetElements)
		if err != nil {
			return err
		}
		size := uint64(v.DType.Size())
		if end > math.MaxUint64/size {
			return fail("guide byte interval overflows")
		}
		if end > v.StorageElements {
			return fail("guide view exceeds parent storage")
		}
		starts[i], ends[i] = v.OffsetElements*size, end*size
	}
	for i := 0; i < 2; i++ {
		if m[i].Allocation == m[2].Allocation && starts[i] < ends[2] && starts[2] < ends[i] {
			return fail("guide output byte interval overlaps input")
		}
	}
	return nil
}

// DecoderGuideFrame performs one reverse-scan timestep. The caller chooses
// score and next/output rows; this boundary never runs a time loop.
func (d *Device) DecoderGuideFrame(s *Stream, scores, next, out *Storage, clamp, blank float32) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	handles := []*Storage{scores, next, out}
	metas := make([]StorageMeta, 3)
	for i, h := range handles {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "decoder_guide_frame", Reason: "foreign device operand"}
		}
		if err := h.state.check(s, "decoder_guide_frame"); err != nil {
			return err
		}
		r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return err
		}
		metas[i] = r.meta
	}
	if err := decoderGuideMetadata(metas, clamp, blank); err != nil {
		return err
	}
	_, err := e.backend.cuda(cudaRequest{op: "decoder_guide_frame", stream: s.state.id, a: scores.state.id, b: next.state.id, out: out.state.id, guideClamp: clamp, guideBlank: blank})
	return err
}
