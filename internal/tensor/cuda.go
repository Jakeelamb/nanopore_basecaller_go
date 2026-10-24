package tensor

import (
	"fmt"
	"math"
)

type Device struct{ state *deviceState }
type Stream struct{ state *streamState }
type HalfBuffer struct{ state *halfState }
type deviceState struct {
	engine  *engineState
	id      uint64
	ordinal int32
	closed  bool
}
type streamState struct {
	device *deviceState
	id     uint64
	closed bool
}
type halfState struct {
	device *deviceState
	id     uint64
	closed bool
}
type HalfLayout struct {
	Shape, Strides                  []int64
	OffsetElements, StorageElements uint64
}
type cudaRequest struct {
	op                                      string
	device, stream, half, a, b, out, offset uint64
	storage, src, scale                     uint64
	in, weights, bias                       uint64
	ordinal                                 int32
	dtype                                   uint32
	activation                              uint32
	n, cIn, cOut, tIn                       int32
	window, stride, padding, outNStride     int32
	inNStride, inStride1, outTStride        int32
	tOut, k                                 int32
	inType, outType, accumType              uint32
	count                                   int
	shape, strides                          []int64
	tags                                    []int32
	bits                                    []uint16
	bytes                                   []byte
}
type cudaResult struct {
	id        uint64
	layout    HalfLayout
	meta      StorageMeta
	bits      []uint16
	bytes     []byte
	koiResult int32
}

func halfCount(shape []int64) (int, error) {
	if len(shape) == 0 || len(shape) > 8 {
		return 0, &LayoutError{Op: "half shape", Reason: "rank must be 1 through 8"}
	}
	n := uint64(1)
	for _, d := range shape {
		if d <= 0 || n > uint64(math.MaxInt/2)/uint64(d) {
			return 0, &LayoutError{Op: "half shape", Reason: "nonpositive dimension or byte count overflow"}
		}
		n *= uint64(d)
	}
	return int(n), nil
}
func halfSpan(shape, strides []int64, offset, limit uint64) error {
	if _, err := halfCount(shape); err != nil {
		return err
	}
	if len(shape) != len(strides) {
		return &LayoutError{Op: "half view", Reason: "shape and stride ranks differ"}
	}
	end := offset
	for i, d := range shape {
		s := strides[i]
		if s < 0 || (d > 1 && uint64(s) > (math.MaxUint64-end)/uint64(d-1)) {
			return &LayoutError{Op: "half view", Reason: "negative stride or span overflow"}
		}
		end += uint64(d-1) * uint64(s)
	}
	if end >= limit {
		return &LayoutError{Op: "half view", Reason: "view exceeds parent reachable interval"}
	}
	return nil
}
func (d *deviceState) valid() error {
	if d == nil || d.engine == nil || d.closed || d.engine.closed || d.engine.backend == nil {
		return ErrClosed
	}
	return nil
}
func (s *streamState) valid(op string) error {
	if s == nil || s.device == nil {
		return &StreamError{Op: op, Reason: "nil stream", Closed: true}
	}
	if err := s.device.valid(); err != nil {
		return err
	}
	if s.closed {
		return &StreamError{Op: op, Reason: "closed stream", Detail: CUDADetail{Handle: s.id}, Closed: true}
	}
	return nil
}
func (h *halfState) valid() error {
	if h == nil || h.device == nil || h.closed {
		return ErrClosed
	}
	return h.device.valid()
}
func (s *streamState) check(h *HalfBuffer, role string) error {
	if h == nil || h.state == nil || h.state.device == nil {
		return fmt.Errorf("%s: %w", role, ErrClosed)
	}
	if h.state.device != s.device {
		return &DeviceError{Op: role, Reason: "half belongs to a different engine or device owner", Detail: CUDADetail{Handle: h.state.id, ExpectedDevice: s.device.id, ActualDevice: h.state.device.id, ExpectedOrdinal: s.device.ordinal, ActualOrdinal: h.state.device.ordinal}}
	}
	return h.state.valid()
}
func (e *Engine) NewCUDADevice(ordinal int) (*Device, error) {
	if e == nil || e.engineState == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	if ordinal < 0 || int64(ordinal) > math.MaxInt32 {
		return nil, &DeviceError{Op: "device_open", Reason: "ordinal outside nonnegative int32 range"}
	}
	r, err := e.backend.cuda(cudaRequest{op: "device_open", ordinal: int32(ordinal)})
	if err != nil {
		return nil, err
	}
	return &Device{&deviceState{engine: e.engineState, id: r.id, ordinal: int32(ordinal)}}, nil
}
func (d *Device) NewStream() (*Stream, error) {
	if d == nil || d.state == nil || d.state.engine == nil {
		return nil, ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "stream_open", device: d.state.id})
	if err != nil {
		return nil, err
	}
	return &Stream{&streamState{device: d.state, id: r.id}}, nil
}
func (d *Device) Close() error {
	if d == nil {
		return nil
	}
	if d.state == nil || d.state.engine == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if d.state.closed || e.closed {
		d.state.closed = true
		return nil
	}
	if _, err := e.backend.cuda(cudaRequest{op: "device_close", device: d.state.id}); err != nil {
		return err
	}
	d.state.closed = true
	return nil
}
func (s *Stream) Close() error {
	if s == nil {
		return nil
	}
	if s.state == nil || s.state.device == nil || s.state.device.engine == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if s.state.closed || s.state.device.closed || e.closed {
		s.state.closed = true
		return nil
	}
	if _, err := e.backend.cuda(cudaRequest{op: "stream_close", stream: s.state.id}); err != nil {
		return err
	}
	s.state.closed = true
	return nil
}
func (s *Stream) NewHalf(shape []int64, bits []uint16) (*HalfBuffer, error) {
	if s == nil || s.state == nil || s.state.device == nil {
		return nil, ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid("half_new"); err != nil {
		return nil, err
	}
	n, err := halfCount(shape)
	if err != nil {
		return nil, err
	}
	if n != len(bits) {
		return nil, &LayoutError{Op: "half_new", Reason: "shape and half-bit count differ"}
	}
	r, err := e.backend.cuda(cudaRequest{op: "half_new", device: s.state.device.id, stream: s.state.id, shape: shape, bits: bits})
	if err != nil {
		return nil, err
	}
	return &HalfBuffer{&halfState{device: s.state.device, id: r.id}}, nil
}
func (h *HalfBuffer) View(shape, strides []int64, offset uint64) (*HalfBuffer, error) {
	if h == nil || h.state == nil || h.state.device == nil {
		return nil, ErrClosed
	}
	e := h.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := h.state.valid(); err != nil {
		return nil, err
	}
	meta, err := e.backend.cuda(cudaRequest{op: "half_meta", half: h.state.id})
	if err != nil {
		return nil, err
	}
	limit := uint64(1)
	for i, d := range meta.layout.Shape {
		limit += uint64(d-1) * uint64(meta.layout.Strides[i])
	}
	if err := halfSpan(shape, strides, offset, limit); err != nil {
		return nil, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "half_view", half: h.state.id, offset: offset, shape: shape, strides: strides})
	if err != nil {
		return nil, err
	}
	return &HalfBuffer{&halfState{device: h.state.device, id: r.id}}, nil
}
func (h *HalfBuffer) Layout() (HalfLayout, error) {
	if h == nil || h.state == nil || h.state.device == nil {
		return HalfLayout{}, ErrClosed
	}
	e := h.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := h.state.valid(); err != nil {
		return HalfLayout{}, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "half_meta", half: h.state.id})
	return r.layout, err
}
func (s *Stream) HalfBits(h *HalfBuffer) ([]uint16, error) {
	if s == nil || s.state == nil || s.state.device == nil {
		return nil, ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid("half_bits"); err != nil {
		return nil, err
	}
	if err := s.state.check(h, "half_bits"); err != nil {
		return nil, err
	}
	meta, err := e.backend.cuda(cudaRequest{op: "half_meta", half: h.state.id})
	if err != nil {
		return nil, err
	}
	n, err := halfCount(meta.layout.Shape)
	if err != nil {
		return nil, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "half_bits", stream: s.state.id, half: h.state.id, bits: make([]uint16, n)})
	return r.bits, err
}
func (s *Stream) GEMMHalf(a, b, out *HalfBuffer) error {
	if s == nil || s.state == nil || s.state.device == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid("gemm_f16"); err != nil {
		return err
	}
	for i, h := range []*HalfBuffer{a, b, out} {
		if err := s.state.check(h, []string{"A", "B", "out"}[i]); err != nil {
			return err
		}
	}
	_, err := e.backend.cuda(cudaRequest{op: "gemm_f16", stream: s.state.id, a: a.state.id, b: b.state.id, out: out.state.id})
	return err
}
func (h *HalfBuffer) Close() error {
	if h == nil {
		return nil
	}
	if h.state == nil || h.state.device == nil || h.state.device.engine == nil {
		return ErrClosed
	}
	e := h.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if h.state.closed || h.state.device.closed || e.closed {
		h.state.closed = true
		return nil
	}
	if _, err := e.backend.cuda(cudaRequest{op: "half_release", half: h.state.id}); err != nil {
		return err
	}
	h.state.closed = true
	return nil
}
