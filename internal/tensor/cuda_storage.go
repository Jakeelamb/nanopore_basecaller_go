package tensor

import (
	"fmt"
	"math"
)

// DType is a typed device storage element type. The values mirror the pinned
// Koi type ids so a Koi descriptor can carry them without translation.
type DType uint32

const (
	DTypeNone DType = 0
	DTypeF16  DType = 1
	DTypeI8   DType = 2
	DTypeU8   DType = 3
	DTypeI32  DType = 4
	DTypeF32  DType = 5
	DTypeE4M3 DType = 6
)

func (d DType) Size() int {
	switch d {
	case DTypeF16:
		return 2
	case DTypeI8, DTypeU8, DTypeE4M3:
		return 1
	case DTypeI32, DTypeF32:
		return 4
	}
	return 0
}

func (d DType) String() string {
	switch d {
	case DTypeF16:
		return "f16"
	case DTypeI8:
		return "i8"
	case DTypeU8:
		return "u8"
	case DTypeI32:
		return "i32"
	case DTypeF32:
		return "f32"
	case DTypeE4M3:
		return "e4m3"
	}
	return fmt.Sprintf("dtype(%d)", uint32(d))
}

// KoiTag is a pinned KoiTensor dimension tag: either a named KoiDimTag value or
// the single character the source passes literally (TxModules.cpp:534-556).
type KoiTag int32

const (
	TagQKV      KoiTag = 0 // KOI_DIM_MAT_Q_K_V
	TagGateIFGO KoiTag = 1 // KOI_DIM_GATE_IFGO
	TagWU       KoiTag = 2 // KOI_DIM_MAT_W_U
	TagN        KoiTag = 'N'
	TagT        KoiTag = 'T'
	TagC        KoiTag = 'C'
	TagM        KoiTag = 'M'
	TagK        KoiTag = 'K'
	TagH        KoiTag = 'H'
	TagD        KoiTag = 'D'
	TagW        KoiTag = 'W'
	Tagn        KoiTag = 'n'
	Tagt        KoiTag = 't'
	Tagc        KoiTag = 'c'
	Tagm        KoiTag = 'm'
	Tagk        KoiTag = 'k'
	Tagd        KoiTag = 'd'
)

func (t KoiTag) valid() bool {
	switch t {
	case TagQKV, TagGateIFGO, TagWU, TagN, TagT, TagC, TagM, TagK, TagH, TagD, TagW, Tagn, Tagt, Tagc, Tagm, Tagk, Tagd:
		return true
	}
	return false
}

// Storage is one typed device allocation or a strided view into one. Views
// share the parent allocation; closing a view never frees the parent.
type Storage struct{ state *storageState }

type storageState struct {
	device *deviceState
	id     uint64
	dtype  DType
	closed bool
}

// StorageMeta is the checked native descriptor of a live storage handle.
type StorageMeta struct {
	Shape, Strides             []int64
	Tags                       []KoiTag
	DType                      DType
	Device                     uint64
	Ordinal                    int32
	Allocation, OffsetElements uint64
	StorageElements            uint64
	ScaleDType                 DType
	ScaleTag                   KoiTag
	ScalePresent               bool
}

func storageCount(shape []int64) (int, error) {
	if len(shape) == 0 || len(shape) > 8 {
		return 0, &LayoutError{Op: "storage shape", Reason: "rank must be 1 through 8"}
	}
	n := uint64(1)
	for _, d := range shape {
		if d <= 0 || n > uint64(math.MaxInt)/uint64(d) {
			return 0, &LayoutError{Op: "storage shape", Reason: "nonpositive dimension or element count overflow"}
		}
		n *= uint64(d)
	}
	return int(n), nil
}

func storageTags(tags []KoiTag, rank int) error {
	if len(tags) != rank {
		return &LayoutError{Op: "storage tags", Reason: "tag rank differs from shape rank"}
	}
	for _, t := range tags {
		if !t.valid() {
			return &LayoutError{Op: "storage tags", Reason: fmt.Sprintf("unknown Koi dimension tag %d", int32(t))}
		}
	}
	return nil
}

func storageSpan(shape, strides []int64, offset uint64) (uint64, error) {
	if _, err := storageCount(shape); err != nil {
		return 0, err
	}
	if len(shape) != len(strides) {
		return 0, &LayoutError{Op: "storage view", Reason: "shape and stride ranks differ"}
	}
	end := offset
	for i, d := range shape {
		s := strides[i]
		if s < 0 || (d > 1 && uint64(s) > (math.MaxUint64-end)/uint64(d-1)) {
			return 0, &LayoutError{Op: "storage view", Reason: "negative stride or span overflow"}
		}
		end += uint64(d-1) * uint64(s)
	}
	if end == math.MaxUint64 {
		return 0, &LayoutError{Op: "storage view", Reason: "span overflow"}
	}
	return end + 1, nil
}

func (s *storageState) valid() error {
	if s == nil || s.device == nil || s.closed {
		return ErrClosed
	}
	return s.device.valid()
}

func (s *storageState) check(stream *Stream, role string) error {
	if stream == nil || stream.state == nil {
		return fmt.Errorf("%s: %w", role, ErrClosed)
	}
	if err := stream.state.valid(role); err != nil {
		return err
	}
	if err := s.valid(); err != nil {
		return err
	}
	if s.device != stream.state.device {
		return &DeviceError{Op: role, Reason: "storage belongs to a different engine or device owner", Detail: CUDADetail{Handle: s.id, ExpectedDevice: stream.state.device.id, ActualDevice: s.device.id, ExpectedOrdinal: stream.state.device.ordinal, ActualOrdinal: s.device.ordinal}}
	}
	return nil
}

func newStorageState(d *Device, id uint64, dtype DType) (*Storage, error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	return &Storage{state: &storageState{device: d.state, id: id, dtype: dtype}}, nil
}

// NewStorage allocates typed device storage and uploads data. data must be nil
// only when the shape has no elements.
func (d *Device) NewStorage(stream *Stream, dtype DType, shape []int64, tags []KoiTag, data []byte) (*Storage, error) {
	if d == nil || d.state == nil {
		return nil, ErrClosed
	}
	e := d.state.engine
	if e == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	if stream == nil || stream.state == nil || stream.state.device != d.state {
		return nil, &DeviceError{Op: "storage_new", Reason: "stream does not belong to this device"}
	}
	if err := stream.state.valid("storage_new"); err != nil {
		return nil, err
	}
	size := dtype.Size()
	if size == 0 {
		return nil, &DTypeError{Op: "storage_new", Reason: "unknown storage dtype", Detail: CUDADetail{ExpectedDType: ScalarType(DTypeF16)}}
	}
	n, err := storageCount(shape)
	if err != nil {
		return nil, err
	}
	if err := storageTags(tags, len(shape)); err != nil {
		return nil, err
	}
	if len(data) != n*size {
		return nil, &LayoutError{Op: "storage_new", Reason: "byte count differs from shape and dtype"}
	}
	tagValues := make([]int32, len(tags))
	for i, t := range tags {
		tagValues[i] = int32(t)
	}
	r, err := e.backend.cuda(cudaRequest{op: "storage_new", device: d.state.id, stream: stream.state.id, dtype: uint32(dtype), shape: shape, tags: tagValues, bytes: data})
	if err != nil {
		return nil, err
	}
	return newStorageState(d, r.id, dtype)
}

// View creates a checked strided view sharing the parent allocation.
func (s *Storage) View(offset uint64, shape, strides []int64, tags []KoiTag) (*Storage, error) {
	if s == nil || s.state == nil {
		return nil, ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid(); err != nil {
		return nil, err
	}
	if _, err := storageSpan(shape, strides, offset); err != nil {
		return nil, err
	}
	if err := storageTags(tags, len(shape)); err != nil {
		return nil, err
	}
	tagValues := make([]int32, len(tags))
	for i, t := range tags {
		tagValues[i] = int32(t)
	}
	r, err := e.backend.cuda(cudaRequest{op: "storage_view", storage: s.state.id, dtype: uint32(s.state.dtype), offset: offset, shape: shape, strides: strides, tags: tagValues})
	if err != nil {
		return nil, err
	}
	return &Storage{state: &storageState{device: s.state.device, id: r.id, dtype: s.state.dtype}}, nil
}

func (s *Storage) meta(op string) (StorageMeta, error) {
	var m StorageMeta
	if s == nil || s.state == nil {
		return m, ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid(); err != nil {
		return m, err
	}
	r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: s.state.id})
	if err != nil {
		return m, err
	}
	return r.meta, nil
}

// Meta returns the checked native descriptor.
func (s *Storage) Meta() (StorageMeta, error) { return s.meta("storage_meta") }

// DType returns the element type fixed at allocation.
func (s *Storage) DType() DType {
	if s == nil || s.state == nil {
		return DTypeNone
	}
	return s.state.dtype
}

// Upload copies contiguous host bytes into a contiguous view.
func (s *Storage) Upload(stream *Stream, data []byte) error {
	if s == nil || s.state == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.check(stream, "storage_upload"); err != nil {
		return err
	}
	if s.state.dtype.Size() == 0 {
		return &DTypeError{Op: "storage_upload"}
	}
	_, err := e.backend.cuda(cudaRequest{op: "storage_upload", stream: stream.state.id, storage: s.state.id, bytes: data})
	return err
}

// Download copies a contiguous view into host bytes.
func (s *Storage) Download(stream *Stream) ([]byte, error) {
	if s == nil || s.state == nil {
		return nil, ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.check(stream, "storage_download"); err != nil {
		return nil, err
	}
	m, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: s.state.id})
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range m.meta.Shape {
		n *= int(d)
	}
	r, err := e.backend.cuda(cudaRequest{op: "storage_download", stream: stream.state.id, storage: s.state.id, count: n * s.state.dtype.Size()})
	if err != nil {
		return nil, err
	}
	return r.bytes, nil
}

// CopyTo copies a contiguous view into another contiguous view of the same
// dtype. Overlapping intervals in one allocation are rejected.
func (s *Storage) CopyTo(stream *Stream, dst *Storage) error {
	if s == nil || s.state == nil || dst == nil || dst.state == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.check(stream, "storage_copy"); err != nil {
		return err
	}
	if err := dst.state.check(stream, "storage_copy"); err != nil {
		return err
	}
	if s.state.dtype != dst.state.dtype {
		return &DTypeError{Op: "storage_copy", Reason: "storage copy dtype differs", Detail: CUDADetail{Handle: dst.state.id, ActualDType: ScalarType(s.state.dtype), ExpectedDType: ScalarType(dst.state.dtype)}}
	}
	_, err := e.backend.cuda(cudaRequest{op: "storage_copy", stream: stream.state.id, storage: dst.state.id, src: s.state.id})
	return err
}

// Fill writes one element pattern across a contiguous view.
func (s *Storage) Fill(stream *Stream, pattern []byte) error {
	if s == nil || s.state == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.check(stream, "storage_fill"); err != nil {
		return err
	}
	if len(pattern) != s.state.dtype.Size() {
		return &LayoutError{Op: "storage_fill", Reason: "pattern length differs from element size"}
	}
	_, err := e.backend.cuda(cudaRequest{op: "storage_fill", stream: stream.state.id, storage: s.state.id, bytes: pattern})
	return err
}

// BindScale attaches a live F32 vector as this I8 tensor's typed Koi scale
// owner and records the tag the scale applies to. A nil scale clears it.
func (s *Storage) BindScale(scale *Storage, tag KoiTag) error {
	if s == nil || s.state == nil {
		return ErrClosed
	}
	e := s.state.device.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := s.state.valid(); err != nil {
		return err
	}
	var id uint64
	if scale != nil {
		if scale.state == nil {
			return ErrClosed
		}
		if err := scale.state.valid(); err != nil {
			return err
		}
		if scale.state.device != s.state.device {
			return &DeviceError{Op: "storage_scale", Reason: "scale owner belongs to a different device"}
		}
		if scale.state.id == s.state.id {
			return &LayoutError{Op: "storage_scale", Reason: "storage cannot own itself as its Koi scale"}
		}
		if s.state.dtype != DTypeI8 {
			return &DTypeError{Op: "storage_scale", Reason: "Koi scale owner requires an I8 tensor", Detail: CUDADetail{Handle: s.state.id, ActualDType: ScalarType(s.state.dtype), ExpectedDType: ScalarType(DTypeI8)}}
		}
		if scale.state.dtype != DTypeF32 {
			return &DTypeError{Op: "storage_scale", Reason: "Koi scale owner must be F32", Detail: CUDADetail{Handle: scale.state.id, ActualDType: ScalarType(scale.state.dtype), ExpectedDType: ScalarType(DTypeF32)}}
		}
		if !tag.valid() {
			return &LayoutError{Op: "storage_scale", Reason: "unknown Koi scale tag"}
		}
		id = scale.state.id
	}
	_, err := e.backend.cuda(cudaRequest{op: "storage_scale", storage: s.state.id, scale: id, tags: []int32{int32(tag)}})
	return err
}

// Close releases this handle. Closing a view never frees the parent allocation.
func (s *Storage) Close() error {
	if s == nil || s.state == nil {
		return nil
	}
	e := s.state.device.engine
	if e == nil {
		return ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if s.state.closed || s.state.device.closed || e.closed {
		s.state.closed = true
		return nil
	}
	if _, err := e.backend.cuda(cudaRequest{op: "storage_release", storage: s.state.id}); err != nil {
		return err
	}
	s.state.closed = true
	return nil
}
