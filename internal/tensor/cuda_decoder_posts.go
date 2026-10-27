package tensor

import "math"

type decoderPostOperation uint32

const (
	decoderForwardAlpha decoderPostOperation = iota + 1
	decoderPosteriorNormalize
	decoderQualityProbability
	decoderQualityGroup
	decoderQualitySummary
)

func decoderPostMetadata(op decoderPostOperation, m []StorageMeta, clamp, blank float32) error {
	if op >= decoderQualityProbability && op <= decoderQualitySummary {
		return decoderQualityMetadata(op, m, clamp, blank)
	}
	const name = "decoder_post_numbers"
	fail := func(why string) error { return &LayoutError{Op: name, Reason: why} }
	if (op != decoderForwardAlpha && op != decoderPosteriorNormalize) || len(m) != 3 {
		return fail("checked posterior operation and three operands required")
	}
	if math.IsNaN(float64(clamp)) || math.IsInf(float64(clamp), 0) || clamp < 0 || math.IsNaN(float64(blank)) || math.IsInf(float64(blank), 0) {
		return fail("finite nonnegative clamp and finite blank required")
	}
	if op == decoderPosteriorNormalize && (math.Float32bits(clamp) != 0 || math.Float32bits(blank) != 0) {
		return fail("normalization has no scalar options")
	}
	for i, v := range m {
		want := DTypeF32
		if (op == decoderForwardAlpha && i == 0) || (op == decoderPosteriorNormalize && i == 2) {
			want = DTypeF16
		}
		if v.DType != want {
			return &DTypeError{Op: name, Reason: "posterior operand dtype differs"}
		}
		if len(v.Shape) != 2 || len(v.Strides) != 2 || len(v.Tags) != 2 || v.Tags[0] != TagN || v.Tags[1] != TagC || v.ScalePresent || v.Strides[1] != 1 || v.Strides[0] < v.Shape[1] {
			return fail("unscaled row-major NC matrices required")
		}
	}
	n, states := m[1].Shape[0], m[1].Shape[1]
	if !((states == 256 && n > 0 && n%64 == 0) || (states == 1024 && n > 0 && n%32 == 0)) || n > math.MaxInt32 {
		return fail("fixed HAC/SUP batch tiles required")
	}
	width := states
	if op == decoderForwardAlpha {
		width *= 4
	}
	if !equalShape(m[0].Shape, []int64{n, width}) || !equalShape(m[2].Shape, m[1].Shape) || m[1].Strides[0] != states {
		return fail("posterior dimensions or second input layout differ")
	}
	if op == decoderForwardAlpha && m[2].Strides[0] != states || op == decoderPosteriorNormalize && m[0].Strides[0] != states {
		return fail("alpha rows must be contiguous")
	}
	starts, ends := [3]uint64{}, [3]uint64{}
	for i, v := range m {
		end, err := storageSpan(v.Shape, v.Strides, v.OffsetElements)
		if err != nil {
			return err
		}
		size := uint64(v.DType.Size())
		if end > math.MaxUint64/size || end > v.StorageElements {
			return fail("posterior view exceeds parent or byte interval overflows")
		}
		starts[i], ends[i] = v.OffsetElements*size, end*size
	}
	for i := 0; i < 2; i++ {
		if m[i].Allocation == m[2].Allocation && starts[i] < ends[2] && starts[2] < ends[i] {
			return fail("posterior writable byte interval overlaps input")
		}
	}
	return nil
}

func (d *Device) decoderPostNumbers(s *Stream, op decoderPostOperation, a, b, out, extra *Storage, clamp, blank float32) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	if s == nil || s.state == nil {
		return ErrClosed
	}
	if err := s.state.valid("decoder_post_numbers"); err != nil {
		return err
	}
	if s.state.device != d.state {
		return &DeviceError{Op: "decoder_post_numbers", Reason: "foreign stream owner"}
	}
	handles := []*Storage{a, b, out}
	if extra != nil {
		handles = append(handles, extra)
	}
	metas := make([]StorageMeta, len(handles))
	for i, h := range handles {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "decoder_post_numbers", Reason: "foreign device operand"}
		}
		if err := h.state.check(s, "decoder_post_numbers"); err != nil {
			return err
		}
		r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return err
		}
		metas[i] = r.meta
	}
	if err := decoderPostMetadata(op, metas, clamp, blank); err != nil {
		return err
	}
	var aux uint64
	if extra != nil {
		aux = extra.state.id
	}
	_, err := e.backend.cuda(cudaRequest{decoderPostAux: aux, op: "decoder_post_numbers", decoderPostOp: uint32(op), stream: s.state.id, a: a.state.id, b: b.state.id, out: out.state.id, guideClamp: clamp, guideBlank: blank})
	return err
}

// DecoderForwardFrame evaluates one chosen forward-alpha frame. The caller
// authenticates DecoderPosteriorProgram first and chooses time/alpha owners.
func (d *Device) DecoderForwardFrame(s *Stream, scores, previous, out *Storage, clamp, blank float32) error {
	return d.decoderPostNumbers(s, decoderForwardAlpha, scores, previous, out, nil, clamp, blank)
}

// DecoderPosteriorRow evaluates one chosen alpha/beta row, including source
// softmax reduction and half rounding. Output may use a larger chunk stride.
func (d *Device) DecoderPosteriorRow(s *Stream, alpha, beta, out *Storage) error {
	return d.decoderPostNumbers(s, decoderPosteriorNormalize, alpha, beta, out, nil, 0, 0)
}

// DecoderPosteriorProgram initializes and returns the retained linked cubin
// containing guide, forward-alpha and posterior-row entries on a live stream's
// device. It executes no scan timestep.
func (d *Device) DecoderPosteriorProgram(s *Stream) ([]byte, error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	if s == nil || s.state == nil {
		return nil, ErrClosed
	}
	if err := s.state.valid("decoder_post_program"); err != nil {
		return nil, err
	}
	if s.state.device != d.state {
		return nil, &DeviceError{Op: "decoder_post_program", Reason: "foreign stream owner"}
	}
	r, err := e.backend.cuda(cudaRequest{op: "decoder_guide_program", device: d.state.id})
	if err != nil {
		return nil, err
	}
	if r.count <= 0 || r.count > 16*1024*1024 {
		return nil, &LayoutError{Op: "decoder_post_program", Reason: "invalid linked cubin extent"}
	}
	full, err := e.backend.cuda(cudaRequest{op: "decoder_guide_program", device: d.state.id, count: r.count})
	if err != nil {
		return nil, err
	}
	if full.count != r.count || len(full.bytes) != r.count {
		return nil, &LayoutError{Op: "decoder_post_program", Reason: "linked cubin changed while copying"}
	}
	return full.bytes, nil
}
