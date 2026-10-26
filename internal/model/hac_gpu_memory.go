package model

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// This file is the host-side port of dorado::nn::WorkingMemory
// (nn/include/nn/WorkingMemory.h:55-82, nn/WorkingMemory.cpp:30-114). It plans
// the single backing extent the HAC GPU caller reserves, separates that maximum
// reservation from the extents a use actually occupies, and gives every view a
// checked offset, orientation and source-declared row range.
//
// It allocates no device memory, imports no CUDA and does not execute a model.

// GPUWorkingMemoryAlignment is WorkingMemory::ALIGNMENT
// (WorkingMemory.h:56-58). Every extent is padded to it because CUDA
// allocations are guaranteed to be 256-byte aligned.
const GPUWorkingMemoryAlignment int64 = 256

// GPUStorageLayout mirrors dorado::nn::TensorLayout (WorkingMemory.h:9). The
// storage each layout reserves is fixed by WorkingMemory::next_TC
// (WorkingMemory.cpp:83-102). LSTM layouts carry extra rows that are storage,
// not time steps.
type GPUStorageLayout uint8

const (
	// GPUStorageNTC reserves [N,T,C] half (WorkingMemory.cpp:87-88).
	GPUStorageNTC GPUStorageLayout = iota
	// GPUStorageTNC reserves [T,N,C] half (WorkingMemory.cpp:89-90).
	GPUStorageTNC
	// GPUStorageCUTLASSF16 reserves [T+5,N,C] half (WorkingMemory.cpp:91-92).
	// Rows 0 and T+2 are the zeroed LSTM state sentinels; rows T+3 and T+4 are
	// never initialised by the source.
	GPUStorageCUTLASSF16
	// GPUStorageCUTLASSI8 reserves [T+5,N,C] int8 (WorkingMemory.cpp:93-94).
	GPUStorageCUTLASSI8
	// GPUStorageCUBLASExt2C reserves [T+1,N,2,C] half
	// (WorkingMemory.cpp:95-96).
	GPUStorageCUBLASExt2C
	// GPUStorageCUBLASTNC reserves [T+3,N,C] half (WorkingMemory.cpp:97-98).
	GPUStorageCUBLASTNC
	// GPUStorageRaw is the flat temp() extent (WorkingMemory.cpp:104-106). It
	// has no shape: N is the element count and T, C are 1.
	GPUStorageRaw
)

func (l GPUStorageLayout) String() string {
	switch l {
	case GPUStorageNTC:
		return "NTC"
	case GPUStorageTNC:
		return "TNC"
	case GPUStorageCUTLASSF16:
		return "CUTLASS_TNC_F16"
	case GPUStorageCUTLASSI8:
		return "CUTLASS_TNC_I8"
	case GPUStorageCUBLASExt2C:
		return "CUBLAS_TN2C"
	case GPUStorageCUBLASTNC:
		return "CUBLAS_TNC"
	case GPUStorageRaw:
		return "temp"
	default:
		return fmt.Sprintf("GPUStorageLayout(%d)", uint8(l))
	}
}

// dtype is the element type next_TC allocates for the layout
// (WorkingMemory.cpp:87-98).
func (l GPUStorageLayout) dtype() (GPUDtype, error) {
	switch l {
	case GPUStorageNTC, GPUStorageTNC, GPUStorageCUTLASSF16, GPUStorageCUBLASExt2C, GPUStorageCUBLASTNC:
		return GPUDtypeHalf, nil
	case GPUStorageCUTLASSI8:
		return GPUDtypeInt8, nil
	default:
		return 0, fmt.Errorf("%w: %s has no next_TC element type", ErrGPUPlanLayout, l)
	}
}

// rows is the time-row count next_TC reserves, including the LSTM slack.
func (l GPUStorageLayout) rows(t int64) (int64, error) {
	switch l {
	case GPUStorageNTC, GPUStorageTNC:
		return t, nil
	case GPUStorageCUTLASSF16, GPUStorageCUTLASSI8:
		return gpuAdd(t, 5)
	case GPUStorageCUBLASExt2C:
		return gpuAdd(t, 1)
	case GPUStorageCUBLASTNC:
		return gpuAdd(t, 3)
	default:
		return 0, fmt.Errorf("%w: %s has no next_TC storage shape", ErrGPUPlanLayout, l)
	}
}

// GPUDtype is the source element type of one extent.
type GPUDtype uint8

const (
	// GPUDtypeHalf is at::kHalf.
	GPUDtypeHalf GPUDtype = iota
	// GPUDtypeInt8 is at::kChar for CUTLASS_TNC_I8. Scratch is at::kU8, which
	// has the same element size.
	GPUDtypeInt8
)

func (d GPUDtype) elementSize() (int64, error) {
	switch d {
	case GPUDtypeHalf:
		return 2, nil
	case GPUDtypeInt8:
		return 1, nil
	default:
		return 0, fmt.Errorf("%w: element type %d", ErrGPUPlanLayout, uint8(d))
	}
}

func (d GPUDtype) String() string {
	switch d {
	case GPUDtypeHalf:
		return "half"
	case GPUDtypeInt8:
		return "int8"
	default:
		return fmt.Sprintf("GPUDtype(%d)", uint8(d))
	}
}

// GPUOrientation is WorkingMemory::is_input_to_rev_lstm (WorkingMemory.h:78).
// The source defaults the flag to reverse; this plan requires the direction to
// be stated so a view cannot silently inherit the wrong end rows
// (WorkingMemory.cpp:67-75).
type GPUOrientation uint8

const (
	// GPUOrientationForward binds the rows a forward LSTM reads.
	GPUOrientationForward GPUOrientation = iota
	// GPUOrientationReverse binds the rows a reverse LSTM reads.
	GPUOrientationReverse
)

func (o GPUOrientation) String() string {
	switch o {
	case GPUOrientationForward:
		return "forward"
	case GPUOrientationReverse:
		return "reverse"
	default:
		return fmt.Sprintf("GPUOrientation(%d)", uint8(o))
	}
}

// narrowStart is the row offset get_current_NTC_view applies to CUTLASS and
// CUBLAS_TNC storage: reverse reads rows [1,1+T), forward reads [2,2+T)
// (WorkingMemory.cpp:69,75).
func (o GPUOrientation) narrowStart() (int64, error) {
	switch o {
	case GPUOrientationReverse:
		return 1, nil
	case GPUOrientationForward:
		return 2, nil
	default:
		return 0, fmt.Errorf("%w: %s", ErrGPUViewOrientation, o)
	}
}

// interleavedStart is the row offset for CUBLAS_TN2C: reverse reads rows
// [1,1+T), forward reads [0,T) (WorkingMemory.cpp:71-73).
func (o GPUOrientation) interleavedStart() (int64, error) {
	switch o {
	case GPUOrientationReverse:
		return 1, nil
	case GPUOrientationForward:
		return 0, nil
	default:
		return 0, fmt.Errorf("%w: %s", ErrGPUViewOrientation, o)
	}
}

// subPlane is the interleaved plane CUBLAS_TN2C reads: reverse reads plane 1,
// forward reads plane 0 (WorkingMemory.cpp:73).
func (o GPUOrientation) subPlane() (int64, error) {
	switch o {
	case GPUOrientationReverse:
		return 1, nil
	case GPUOrientationForward:
		return 0, nil
	default:
		return 0, fmt.Errorf("%w: %s", ErrGPUViewOrientation, o)
	}
}

var (
	// ErrGPUPlanOverflow reports plan arithmetic that would exceed int64.
	ErrGPUPlanOverflow = errors.New("model: GPU working memory arithmetic overflows int64")
	// ErrGPUPlanOverlap is the source's "overlap detected" guard
	// (WorkingMemory.cpp:42-44): the live pair would not fit the reservation.
	ErrGPUPlanOverlap = errors.New("model: GPU working memory plan overlap detected")
	// ErrGPUPlanUndersized reports a plan, use or native extent smaller than the
	// reservation needs.
	ErrGPUPlanUndersized = errors.New("model: GPU working memory plan is undersized")
	// ErrGPUPlanMismatch reports a use that substitutes a different storage
	// layout or view kind for the reserved one.
	ErrGPUPlanMismatch = errors.New("model: GPU working memory use does not match the reservation")
	// ErrGPUPlanLayout reports an unknown layout, element type or shape.
	ErrGPUPlanLayout = errors.New("model: unsupported GPU storage layout")
	// ErrGPUViewOrientation reports a view bound with the wrong LSTM direction.
	ErrGPUViewOrientation = errors.New("model: GPU working memory view orientation mismatch")
	// ErrGPUScratchExpired reports a temp() view read after a later view reused
	// its memory (WorkingMemory.h:22-27).
	ErrGPUScratchExpired = errors.New("model: GPU working memory scratch view was overwritten")
	// ErrGPUPlanState reports a plan or use used out of order.
	ErrGPUPlanState = errors.New("model: GPU working memory plan state")
)

func gpuAdd(a, b int64) (int64, error) {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return 0, fmt.Errorf("%w: %d + %d", ErrGPUPlanOverflow, a, b)
	}
	return a + b, nil
}

func gpuMul(a, b int64) (int64, error) {
	if a < 0 || b < 0 || (a != 0 && b > math.MaxInt64/a) {
		return 0, fmt.Errorf("%w: %d * %d", ErrGPUPlanOverflow, a, b)
	}
	return a * b, nil
}

// gpuPadTo is utils::pad_to (utils/math_utils.h:23-26), the padding
// WorkingMemory::tensor_bytes applies.
func gpuPadTo(a, b int64) (int64, error) {
	if b <= 0 {
		return 0, fmt.Errorf("%w: padding %d", ErrGPUPlanLayout, b)
	}
	rounded, err := gpuAdd(a, b-1)
	if err != nil {
		return 0, err
	}
	return (rounded / b) * b, nil
}

// GPUStorageGeometry is one source-declared storage extent. Bytes is the
// 256-byte-padded extent WorkingMemory::tensor_bytes returns
// (WorkingMemory.cpp:30-33); LiveBytes is the exact element bytes the view can
// hold, so a reservation can be compared with the bytes a use actually touches.
type GPUStorageGeometry struct {
	Layout    GPUStorageLayout
	Dtype     GPUDtype
	N         int64
	T         int64
	C         int64
	Rows      int64
	Elems     int64
	LiveBytes int64
	Bytes     int64
}

// NewGPUStorageGeometry derives the storage for one next_TC call. Dims are
// checked before any multiplication so an overflowing plan is rejected before a
// device allocation can be requested.
func NewGPUStorageGeometry(layout GPUStorageLayout, n, t, c int64) (GPUStorageGeometry, error) {
	if layout == GPUStorageRaw {
		return GPUStorageGeometry{}, fmt.Errorf("%w: temp() needs NewGPUScratchGeometry", ErrGPUPlanLayout)
	}
	if n < 1 || t < 1 || c < 1 {
		return GPUStorageGeometry{}, fmt.Errorf("%w: %s shape N=%d T=%d C=%d", ErrGPUPlanLayout, layout, n, t, c)
	}
	dtype, err := layout.dtype()
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	rows, err := layout.rows(t)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	planes := int64(1)
	if layout == GPUStorageCUBLASExt2C {
		planes = 2
	}
	elems, err := gpuMul(rows, n)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	elems, err = gpuMul(elems, planes)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	elems, err = gpuMul(elems, c)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	return newGPUGeometry(layout, dtype, n, t, c, rows, elems)
}

// NewGPUScratchGeometry derives the flat extent of one temp() call
// (WorkingMemory.cpp:104-106). The sizes are multiplied exactly as
// c10::multiply_integers does, then padded.
func NewGPUScratchGeometry(dtype GPUDtype, sizes ...int64) (GPUStorageGeometry, error) {
	if len(sizes) == 0 {
		return GPUStorageGeometry{}, fmt.Errorf("%w: temp() needs at least one size", ErrGPUPlanLayout)
	}
	elems := int64(1)
	for _, size := range sizes {
		if size < 1 {
			return GPUStorageGeometry{}, fmt.Errorf("%w: temp() size %d", ErrGPUPlanLayout, size)
		}
		var err error
		elems, err = gpuMul(elems, size)
		if err != nil {
			return GPUStorageGeometry{}, err
		}
	}
	return newGPUGeometry(GPUStorageRaw, dtype, elems, 1, 1, 1, elems)
}

func newGPUGeometry(layout GPUStorageLayout, dtype GPUDtype, n, t, c, rows, elems int64) (GPUStorageGeometry, error) {
	elemSize, err := dtype.elementSize()
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	live, err := gpuMul(elems, elemSize)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	padded, err := gpuPadTo(live, GPUWorkingMemoryAlignment)
	if err != nil {
		return GPUStorageGeometry{}, err
	}
	return GPUStorageGeometry{
		Layout:    layout,
		Dtype:     dtype,
		N:         n,
		T:         t,
		C:         c,
		Rows:      rows,
		Elems:     elems,
		LiveBytes: live,
		Bytes:     padded,
	}, nil
}

// gpuMemoryStep is one entry of the reservation traversal. The use phase
// compares its layout, kind and orientation, but not its byte size: the
// source's use-phase guard is a budget check, so a use that grows an extent is
// rejected by the reservation total rather than by replaying a recorded size.
type gpuMemoryStep struct {
	layout      GPUStorageLayout
	orientation GPUOrientation
	scratch     bool
	geometry    GPUStorageGeometry
}

// GPUMemoryPlan records the reservation traversal: the next_TC and temp calls a
// model makes before WorkingMemory::allocate_backing_tensor
// (WorkingMemory.h:33-54).
type GPUMemoryPlan struct {
	batch int64
	steps []gpuMemoryStep
}

// NewGPUMemoryPlan fixes the batch size N, which is constant for the lifetime
// of a source WorkingMemory (WorkingMemory.h:79).
func NewGPUMemoryPlan(batch int64) (*GPUMemoryPlan, error) {
	if batch < 1 {
		return nil, fmt.Errorf("%w: batch size %d", ErrGPUPlanLayout, batch)
	}
	return &GPUMemoryPlan{batch: batch}, nil
}

// NextTC appends one next_TC reservation (WorkingMemory.cpp:83-102).
func (p *GPUMemoryPlan) NextTC(t, c int64, layout GPUStorageLayout, orientation GPUOrientation) error {
	geometry, err := NewGPUStorageGeometry(layout, p.batch, t, c)
	if err != nil {
		return err
	}
	if _, err := orientation.narrowStart(); err != nil {
		return err
	}
	p.steps = append(p.steps, gpuMemoryStep{layout: layout, orientation: orientation, geometry: geometry})
	return nil
}

// Temp appends one temp reservation (WorkingMemory.cpp:104-106). Scratch is
// invocation-owned: it is never made current and never extends the live pair.
func (p *GPUMemoryPlan) Temp(sizes []int64, dtype GPUDtype) error {
	geometry, err := NewGPUScratchGeometry(dtype, sizes...)
	if err != nil {
		return err
	}
	p.steps = append(p.steps, gpuMemoryStep{layout: GPUStorageRaw, scratch: true, geometry: geometry})
	return nil
}

// GPUMemoryReservation is the frozen result of a reservation traversal.
// Bytes is WorkingMemory::reservation_bytes: the maximum of current + new over
// the traversal, always 256-byte aligned. PeakLiveBytes is the same maximum
// without alignment padding, which is what a capture can actually compare.
type GPUMemoryReservation struct {
	Bytes         int64
	PeakLiveBytes int64
	Steps         int

	batch int64
	steps []gpuMemoryStep
}

// Reserve computes the reservation. It rejects an empty traversal and any
// arithmetic overflow before a device allocation is requested.
func (p *GPUMemoryPlan) Reserve() (GPUMemoryReservation, error) {
	if len(p.steps) == 0 {
		return GPUMemoryReservation{}, fmt.Errorf("%w: reservation traversal is empty", ErrGPUPlanUndersized)
	}
	var current, currentLive, reservation, peak int64
	for _, step := range p.steps {
		pair, err := gpuAdd(current, step.geometry.Bytes)
		if err != nil {
			return GPUMemoryReservation{}, err
		}
		if pair > reservation {
			reservation = pair
		}
		live, err := gpuAdd(currentLive, step.geometry.LiveBytes)
		if err != nil {
			return GPUMemoryReservation{}, err
		}
		if live > peak {
			peak = live
		}
		if !step.scratch {
			current, currentLive = step.geometry.Bytes, step.geometry.LiveBytes
		}
	}
	return GPUMemoryReservation{
		Bytes:         reservation,
		PeakLiveBytes: peak,
		Steps:         len(p.steps),
		batch:         p.batch,
		steps:         slices.Clone(p.steps),
	}, nil
}

// GPUNativeAllocation is the typed handoff a native allocator offers for the
// reservation. The plan checks it before any view is bound, so an undersized or
// under-aligned device extent is rejected before it can be used.
type GPUNativeAllocation struct {
	Bytes     int64
	Alignment int64
}

// AdmitNative checks a native extent against the reservation.
func (r GPUMemoryReservation) AdmitNative(a GPUNativeAllocation) error {
	if a.Alignment < GPUWorkingMemoryAlignment || a.Alignment%GPUWorkingMemoryAlignment != 0 {
		return fmt.Errorf("%w: native alignment %d is not a positive multiple of %d",
			ErrGPUPlanUndersized, a.Alignment, GPUWorkingMemoryAlignment)
	}
	if a.Bytes < r.Bytes {
		return fmt.Errorf("%w: native extent %d bytes, plan reserves %d",
			ErrGPUPlanUndersized, a.Bytes, r.Bytes)
	}
	return nil
}

// Use starts the use phase. CRFModel.cpp:88-106 builds one WorkingMemory per
// invocation and allocates it after the reservation traversal, so every use
// starts from an undefined current and binds the first view at the front. It
// binds views from alternating ends of the reservation and never allocates.
func (r GPUMemoryReservation) Use() *GPUMemoryUse {
	return &GPUMemoryUse{reservation: r, currentOffset: -1}
}

// GPUMemoryUse is the checked use phase: the same traversal replayed after
// allocate_backing_tensor (WorkingMemory.cpp:42-58).
type GPUMemoryUse struct {
	reservation GPUMemoryReservation

	currentBytes   int64
	currentLive    int64
	currentOffset  int64
	currentDefined bool

	step      int
	views     []GPUView
	expiredBy []int
	peakLive  int64
}

// NextTC binds one current view and makes it the live current.
func (u *GPUMemoryUse) NextTC(t, c int64, layout GPUStorageLayout, orientation GPUOrientation) (GPUView, error) {
	geometry, err := NewGPUStorageGeometry(layout, u.reservation.batch, t, c)
	if err != nil {
		return GPUView{}, err
	}
	view, err := u.bind(geometry, orientation, false)
	if err != nil {
		return GPUView{}, err
	}
	u.currentBytes, u.currentLive = geometry.Bytes, geometry.LiveBytes
	u.currentOffset, u.currentDefined = view.Offset, true
	return view, nil
}

// Temp binds one invocation-owned scratch view without making it current.
func (u *GPUMemoryUse) Temp(sizes []int64, dtype GPUDtype) (GPUView, error) {
	geometry, err := NewGPUScratchGeometry(dtype, sizes...)
	if err != nil {
		return GPUView{}, err
	}
	return u.bind(geometry, GPUOrientationForward, true)
}

func (u *GPUMemoryUse) bind(geometry GPUStorageGeometry, orientation GPUOrientation, scratch bool) (GPUView, error) {
	if u.step >= u.reservation.Steps {
		return GPUView{}, fmt.Errorf("%w: use step %d has no reservation", ErrGPUPlanUndersized, u.step)
	}
	want := u.reservation.steps[u.step]
	if want.scratch != scratch || want.layout != geometry.Layout || want.orientation != orientation {
		return GPUView{}, fmt.Errorf("%w: use step %d is %s %s/%s, reservation holds %s %s/%s",
			ErrGPUPlanMismatch, u.step, scratchLabel(scratch), geometry.Layout, orientation,
			scratchLabel(want.scratch), want.layout, want.orientation)
	}

	// WorkingMemory.cpp:42-44 rejects a live pair larger than the reservation.
	// With the alternating-end placement below this is exactly the condition
	// that the new extent does not intersect the live current extent.
	pair, err := gpuAdd(u.currentBytes, geometry.Bytes)
	if err != nil {
		return GPUView{}, err
	}
	if pair > u.reservation.Bytes {
		return GPUView{}, fmt.Errorf("%w: step %d needs %d reserved bytes, reservation is %d",
			ErrGPUPlanOverlap, u.step, pair, u.reservation.Bytes)
	}

	offset := int64(0)
	if u.currentDefined && u.currentOffset == 0 {
		offset = u.reservation.Bytes - geometry.Bytes
	}
	view := GPUView{
		Step:        u.step,
		Scratch:     scratch,
		Orientation: orientation,
		Geometry:    geometry,
		Offset:      offset,
		End:         offset + geometry.Bytes,
	}

	// A new view reuses the memory of every live scratch extent it intersects.
	// WorkingMemory.h:22-27 documents that a temp() view is only valid until the
	// next next_TC() hands the same end of the backing store to a new tensor.
	for i := range u.views {
		if u.expiredBy[i] >= 0 || !u.views[i].Scratch {
			continue
		}
		if offset < u.views[i].End && u.views[i].Offset < view.End {
			u.expiredBy[i] = u.step
		}
	}

	pair, err = gpuAdd(u.currentLive, geometry.LiveBytes)
	if err != nil {
		return GPUView{}, err
	}
	if pair > u.peakLive {
		u.peakLive = pair
	}
	u.views = append(u.views, view)
	u.expiredBy = append(u.expiredBy, -1)
	u.step++
	return view, nil
}

func scratchLabel(scratch bool) string {
	if scratch {
		return "scratch"
	}
	return "current"
}

// GPUView is one checked extent inside the reservation.
type GPUView struct {
	Step        int
	Scratch     bool
	Orientation GPUOrientation
	Geometry    GPUStorageGeometry
	Offset      int64
	End         int64
}

// Views returns a copy of the bound views in use order.
func (u *GPUMemoryUse) Views() []GPUView { return slices.Clone(u.views) }

// PeakLiveBytes is the maximum exact, unpadded bytes of one live current plus
// one new extent. It is the "used" figure to compare with
// GPUMemoryReservation.Bytes, the aligned maximum.
func (u *GPUMemoryUse) PeakLiveBytes() int64 { return u.peakLive }

// Alive rejects a scratch view whose memory a later view has reused. Current
// views stay alive: the source keeps the previous current as the input of the
// step that produced the new one.
func (u *GPUMemoryUse) Alive(view GPUView) error {
	if view.Step < 0 || view.Step >= len(u.views) {
		return fmt.Errorf("%w: view %d is not bound", ErrGPUPlanState, view.Step)
	}
	if by := u.expiredBy[view.Step]; by >= 0 {
		return fmt.Errorf("%w: scratch step %d was overwritten by step %d",
			ErrGPUScratchExpired, view.Step, by)
	}
	return nil
}

// Finish closes the use phase. A use that replayed fewer steps than the
// reservation covers is undersized and must be rejected before the extent is
// handed to a device.
func (u *GPUMemoryUse) Finish() error {
	if len(u.views) != u.reservation.Steps {
		return fmt.Errorf("%w: use replayed %d of %d reserved steps",
			ErrGPUPlanUndersized, len(u.views), u.reservation.Steps)
	}
	return nil
}

// GPUNTCView is the logical [N,T,C] view WorkingMemory::get_current_NTC_view
// returns (WorkingMemory.cpp:61-79). Strides are in elements, so a non
// contiguous TNC or interleaved view keeps the source layout instead of being
// copied.
type GPUNTCView struct {
	Offset                    int64
	N, T, C                   int64
	StrideN, StrideT, StrideC int64
}

// NTCView binds the [N,T,C] view for one orientation. A view bound with the
// other LSTM direction is rejected instead of reading the wrong end rows.
func (v GPUView) NTCView(orientation GPUOrientation) (GPUNTCView, error) {
	if orientation != v.Orientation {
		return GPUNTCView{}, fmt.Errorf("%w: view %d was bound %s, requested %s",
			ErrGPUViewOrientation, v.Step, v.Orientation, orientation)
	}
	g := v.Geometry
	elemSize, err := g.Dtype.elementSize()
	if err != nil {
		return GPUNTCView{}, err
	}
	switch g.Layout {
	case GPUStorageNTC:
		return GPUNTCView{Offset: v.Offset, N: g.N, T: g.T, C: g.C,
			StrideN: g.T * g.C, StrideT: g.C, StrideC: 1}, nil
	case GPUStorageTNC:
		return GPUNTCView{Offset: v.Offset, N: g.N, T: g.T, C: g.C,
			StrideN: g.C, StrideT: g.N * g.C, StrideC: 1}, nil
	case GPUStorageCUTLASSF16, GPUStorageCUTLASSI8, GPUStorageCUBLASTNC:
		start, err := orientation.narrowStart()
		if err != nil {
			return GPUNTCView{}, err
		}
		delta, err := gpuMul(start, g.N)
		if err != nil {
			return GPUNTCView{}, err
		}
		delta, err = gpuMul(delta, g.C)
		if err != nil {
			return GPUNTCView{}, err
		}
		delta, err = gpuMul(delta, elemSize)
		if err != nil {
			return GPUNTCView{}, err
		}
		offset, err := gpuAdd(v.Offset, delta)
		if err != nil {
			return GPUNTCView{}, err
		}
		return GPUNTCView{Offset: offset, N: g.N, T: g.T, C: g.C,
			StrideN: g.C, StrideT: g.N * g.C, StrideC: 1}, nil
	case GPUStorageCUBLASExt2C:
		start, err := orientation.interleavedStart()
		if err != nil {
			return GPUNTCView{}, err
		}
		plane, err := orientation.subPlane()
		if err != nil {
			return GPUNTCView{}, err
		}
		row, err := gpuMul(start, g.N)
		if err != nil {
			return GPUNTCView{}, err
		}
		row, err = gpuMul(row, 2)
		if err != nil {
			return GPUNTCView{}, err
		}
		row, err = gpuAdd(row, plane)
		if err != nil {
			return GPUNTCView{}, err
		}
		delta, err := gpuMul(row, g.C)
		if err != nil {
			return GPUNTCView{}, err
		}
		delta, err = gpuMul(delta, elemSize)
		if err != nil {
			return GPUNTCView{}, err
		}
		offset, err := gpuAdd(v.Offset, delta)
		if err != nil {
			return GPUNTCView{}, err
		}
		return GPUNTCView{Offset: offset, N: g.N, T: g.T, C: g.C,
			StrideN: 2 * g.C, StrideT: g.N * 2 * g.C, StrideC: 1}, nil
	default:
		return GPUNTCView{}, fmt.Errorf("%w: %s has no NTC view", ErrGPUPlanLayout, g.Layout)
	}
}

// GPURowLayout declares, from source, which storage rows are valid time steps,
// which rows are zeroed sentinels and which rows the source never initialises.
// A capture comparison uses this before execution instead of inferring a mask
// from observed bytes.
type GPURowLayout struct {
	ValidStart int64
	ValidRows  int64
	Sentinels  []int64
	Unused     []int64
}

// RowLayout returns the source-declared row ranges for one view.
func (v GPUView) RowLayout() (GPURowLayout, error) {
	g := v.Geometry
	switch g.Layout {
	case GPUStorageNTC, GPUStorageTNC:
		return GPURowLayout{ValidStart: 0, ValidRows: g.T}, nil
	case GPUStorageCUTLASSF16, GPUStorageCUTLASSI8:
		start, err := v.Orientation.narrowStart()
		if err != nil {
			return GPURowLayout{}, err
		}
		// LSTMStack.cpp:135-140 zeroes current[0] and current[T+2]. With T+5
		// rows (WorkingMemory.cpp:91-94) rows T+3 and T+4 are storage slack the
		// source never writes.
		sentinel, err := gpuAdd(g.T, 2)
		if err != nil {
			return GPURowLayout{}, err
		}
		firstUnused, err := gpuAdd(g.T, 3)
		if err != nil {
			return GPURowLayout{}, err
		}
		lastUnused, err := gpuAdd(firstUnused, 1)
		if err != nil {
			return GPURowLayout{}, err
		}
		return GPURowLayout{ValidStart: start, ValidRows: g.T,
			Sentinels: []int64{0, sentinel},
			Unused:    []int64{firstUnused, lastUnused}}, nil
	case GPUStorageCUBLASTNC:
		start, err := v.Orientation.narrowStart()
		if err != nil {
			return GPURowLayout{}, err
		}
		// FLSTMStack.cpp:236-237 zeroes inout[0] and inout[T+2] in the T+3
		// storage, so no row is left uninitialised.
		sentinel, err := gpuAdd(g.T, 2)
		if err != nil {
			return GPURowLayout{}, err
		}
		return GPURowLayout{ValidStart: start, ValidRows: g.T,
			Sentinels: []int64{0, sentinel}}, nil
	case GPUStorageCUBLASExt2C:
		start, err := v.Orientation.interleavedStart()
		if err != nil {
			return GPURowLayout{}, err
		}
		// LSTMStack.cpp:74-78 zeroes row 0 plane 1 and row T plane 0. The
		// sentinel rows are listed at row granularity.
		return GPURowLayout{ValidStart: start, ValidRows: g.T,
			Sentinels: []int64{0, g.T}}, nil
	default:
		return GPURowLayout{}, fmt.Errorf("%w: %s has no row layout", ErrGPUPlanLayout, g.Layout)
	}
}
