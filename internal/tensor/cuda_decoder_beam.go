package tensor

import "math"

type decoderBeamOperation uint32

const (
	decoderBeamLogCut decoderBeamOperation = iota + 1
	decoderBeamExpand
	decoderBeamLogAdd
	decoderBeamSub
	decoderBeamMidpoint
	decoderBeamMax
	decoderBeamMaxRows = 1 << 20
)

type decoderBeamLayout struct {
	dtype DType
	width int64
}

func decoderBeamLayouts(op decoderBeamOperation) ([]decoderBeamLayout, int64) {
	switch op {
	case decoderBeamLogCut:
		return []decoderBeamLayout{{DTypeF32, 1}}, 1
	case decoderBeamExpand:
		return []decoderBeamLayout{{DTypeF32, 1}, {DTypeF16, 4}, {DTypeF32, 5}}, 6
	case decoderBeamLogAdd, decoderBeamSub, decoderBeamMidpoint, decoderBeamMax:
		return []decoderBeamLayout{{DTypeF32, 2}}, 1
	default:
		return nil, 0
	}
}

// DecoderBeamProgram returns the retained bytes actually used by module load.
// The caller authenticates these bytes before selecting numeric operations.
func (d *Device) DecoderBeamProgram(s *Stream) ([]byte, error) {
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
		return nil, &StreamError{Op: "decoder_beam_program", Reason: "live stream required", Closed: true}
	}
	if err := s.state.valid("decoder_beam_program"); err != nil {
		return nil, err
	}
	if s.state.device != d.state {
		return nil, &DeviceError{Op: "decoder_beam_program", Reason: "foreign stream owner"}
	}
	r, err := e.backend.cuda(cudaRequest{op: "decoder_beam_program", device: d.state.id})
	if err != nil {
		return nil, err
	}
	if r.count <= 0 || r.count > 16*1024*1024 {
		return nil, &LayoutError{Op: "decoder_beam_program", Reason: "invalid linked cubin extent"}
	}
	full, err := e.backend.cuda(cudaRequest{op: "decoder_beam_program", device: d.state.id, count: r.count})
	if err != nil {
		return nil, err
	}
	if full.count != r.count || len(full.bytes) != r.count {
		return nil, &LayoutError{Op: "decoder_beam_program", Reason: "linked cubin changed while copying"}
	}
	return full.bytes, nil
}

func decoderBeamMetadata(op decoderBeamOperation, m []StorageMeta, clamp, blank float32) error {
	const name = "decoder_beam_numbers"
	fail := func(why string) error { return &LayoutError{Op: name, Reason: why} }
	layouts, outputWidth := decoderBeamLayouts(op)
	if len(layouts) == 0 || len(m) != len(layouts)+1 {
		return fail("checked beam operation and exact operand count required")
	}
	if math.IsNaN(float64(clamp)) || math.IsInf(float64(clamp), 0) || clamp < 0 || math.IsNaN(float64(blank)) || math.IsInf(float64(blank), 0) || (op != decoderBeamExpand && (math.Float32bits(clamp) != 0 || math.Float32bits(blank) != 0)) {
		return fail("finite beam scalars required; unused scalar bits must be positive zero")
	}
	layouts = append(layouts, decoderBeamLayout{DTypeF32, outputWidth})
	var rows int64
	starts, ends := make([]uint64, len(m)), make([]uint64, len(m))
	for i, v := range m {
		if v.DType != layouts[i].dtype {
			return &DTypeError{Op: name, Reason: "beam operand dtype differs"}
		}
		if len(v.Shape) != 2 || len(v.Strides) != 2 || len(v.Tags) != 2 || v.Tags[0] != TagN || v.Tags[1] != TagC || v.ScalePresent || v.Shape[1] != layouts[i].width || v.Strides[1] != 1 || v.Strides[0] != layouts[i].width {
			return fail("contiguous unscaled rank-two NC beam operands required")
		}
		if i == 0 {
			rows = v.Shape[0]
		}
		if rows <= 0 || rows > decoderBeamMaxRows || v.Shape[0] != rows {
			return fail("positive shared beam row count exceeds workspace or differs")
		}
		end, err := storageSpan(v.Shape, v.Strides, v.OffsetElements)
		if err != nil {
			return err
		}
		size := uint64(v.DType.Size())
		if end > math.MaxUint64/size || end > v.StorageElements {
			return fail("beam byte interval exceeds parent storage or overflows")
		}
		starts[i], ends[i] = v.OffsetElements*size, end*size
	}
	last := len(m) - 1
	for i := 0; i < last; i++ {
		if m[i].Allocation == m[last].Allocation && starts[i] < ends[last] && starts[last] < ends[i] {
			return fail("beam output byte interval overlaps input")
		}
	}
	return nil
}

func (d *Device) decoderBeamNumbers(s *Stream, op decoderBeamOperation, inputs []*Storage, out *Storage, clamp, blank float32) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	handles := append(append([]*Storage(nil), inputs...), out)
	metas := make([]StorageMeta, len(handles))
	for i, h := range handles {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "decoder_beam_numbers", Reason: "foreign device operand"}
		}
		if err := h.state.check(s, "decoder_beam_numbers"); err != nil {
			return err
		}
		r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return err
		}
		metas[i] = r.meta
	}
	if err := decoderBeamMetadata(op, metas, clamp, blank); err != nil {
		return err
	}
	q := cudaRequest{op: "decoder_beam_numbers", stream: s.state.id, beamOp: uint32(op), out: out.state.id, guideClamp: clamp, guideBlank: blank}
	for i, h := range inputs {
		q.beam[i] = h.state.id
	}
	_, err := e.backend.cuda(q)
	return err
}

func (d *Device) DecoderBeamLogCuts(s *Stream, in, out *Storage) error {
	return d.decoderBeamNumbers(s, decoderBeamLogCut, []*Storage{in}, out, 0, 0)
}
func (d *Device) DecoderBeamExpand5(s *Stream, previous, transitions, guides, out *Storage, clamp, blank float32) error {
	return d.decoderBeamNumbers(s, decoderBeamExpand, []*Storage{previous, transitions, guides}, out, clamp, blank)
}
func (d *Device) DecoderBeamLogAddPairs(s *Stream, in, out *Storage) error {
	return d.decoderBeamNumbers(s, decoderBeamLogAdd, []*Storage{in}, out, 0, 0)
}
func (d *Device) DecoderBeamSubPairs(s *Stream, in, out *Storage) error {
	return d.decoderBeamNumbers(s, decoderBeamSub, []*Storage{in}, out, 0, 0)
}
func (d *Device) DecoderBeamMidpoints(s *Stream, in, out *Storage) error {
	return d.decoderBeamNumbers(s, decoderBeamMidpoint, []*Storage{in}, out, 0, 0)
}
func (d *Device) DecoderBeamMaxPairs(s *Stream, in, out *Storage) error {
	return d.decoderBeamNumbers(s, decoderBeamMax, []*Storage{in}, out, 0, 0)
}
