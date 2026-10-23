//go:build cgo && linux

package tensor

/*
#include "loader.h"
*/
import "C"
import "unsafe"

func (b *nativeBackend) cuda(q cudaRequest) (cudaResult, error) {
	var r cudaResult
	var id C.uint64_t
	var meta C.nt_half_meta
	var d C.nt_cuda_detail
	var buf [C.NT_ERROR_CAP]C.char
	var code C.int
	tail := func(code C.int) error {
		detail := CUDADetail{Stage: uint32(d.stage), NativeCode: int32(d.native_code), Handle: uint64(d.handle), ActualEngine: uint64(d.actual_engine), ExpectedDevice: uint64(d.expected_device), ActualDevice: uint64(d.actual_device), ExpectedOrdinal: int32(d.expected_ordinal), ActualOrdinal: int32(d.actual_ordinal), ExpectedDType: ScalarType(d.expected_dtype), ActualDType: ScalarType(d.actual_dtype)}
		err := cudaStatus(q.op, int(code), detail, C.GoString(&buf[0]))
		if v, ok := err.(*DeviceError); ok {
			v.RequiredEngine = uint64(b.id)
		}
		return err
	}
	switch q.op {
	case "device_open":
		code = C.nt_call_cuda_device_open(b.library, b.id, C.int32_t(q.ordinal), &id, &d, &buf[0], C.NT_ERROR_CAP)
	case "device_close":
		code = C.nt_call_cuda_device_close(b.library, b.id, C.uint64_t(q.device), &d, &buf[0], C.NT_ERROR_CAP)
	case "stream_open":
		code = C.nt_call_cuda_stream_open(b.library, b.id, C.uint64_t(q.device), &id, &d, &buf[0], C.NT_ERROR_CAP)
	case "stream_close":
		code = C.nt_call_cuda_stream_close(b.library, b.id, C.uint64_t(q.stream), &d, &buf[0], C.NT_ERROR_CAP)
	case "half_new":
		code = C.nt_call_cuda_half_new(b.library, b.id, C.uint64_t(q.device), C.uint64_t(q.stream), (*C.int64_t)(unsafe.Pointer(pointer(q.shape))), C.size_t(len(q.shape)), (*C.uint16_t)(unsafe.Pointer(pointer(q.bits))), C.size_t(len(q.bits)), &id, &d, &buf[0], C.NT_ERROR_CAP)
	case "half_view":
		code = C.nt_call_cuda_half_view(b.library, b.id, C.uint64_t(q.half), C.uint64_t(q.offset), (*C.int64_t)(unsafe.Pointer(pointer(q.shape))), (*C.int64_t)(unsafe.Pointer(pointer(q.strides))), C.size_t(len(q.shape)), &id, &d, &buf[0], C.NT_ERROR_CAP)
	case "half_meta":
		code = C.nt_call_cuda_half_meta(b.library, b.id, C.uint64_t(q.half), &meta, &d, &buf[0], C.NT_ERROR_CAP)
	case "half_bits":
		code = C.nt_call_cuda_half_bits(b.library, b.id, C.uint64_t(q.stream), C.uint64_t(q.half), (*C.uint16_t)(unsafe.Pointer(pointer(q.bits))), C.size_t(len(q.bits)), &d, &buf[0], C.NT_ERROR_CAP)
	case "half_release":
		code = C.nt_call_cuda_half_release(b.library, b.id, C.uint64_t(q.half), &d, &buf[0], C.NT_ERROR_CAP)
	case "gemm_f16":
		code = C.nt_call_cuda_gemm_f16(b.library, b.id, C.uint64_t(q.stream), C.uint64_t(q.a), C.uint64_t(q.b), C.uint64_t(q.out), &d, &buf[0], C.NT_ERROR_CAP)
	default:
		return r, &CUDAError{Op: q.op, Subsystem: "boundary", Reason: "unknown CUDA operation"}
	}
	if err := tail(code); err != nil {
		return r, err
	}
	r.id = uint64(id)
	if q.op == "half_meta" {
		if meta.rank == 0 || meta.rank > C.NT_CUDA_MAX_RANK || meta.dtype != C.NT_SCALAR_F16 {
			return r, &LayoutError{Op: q.op, Reason: "invalid native half descriptor"}
		}
		r.layout = HalfLayout{OffsetElements: uint64(meta.offset_elements), StorageElements: uint64(meta.storage_elements), Shape: make([]int64, int(meta.rank)), Strides: make([]int64, int(meta.rank))}
		for i := range r.layout.Shape {
			r.layout.Shape[i] = int64(meta.shape[i])
			r.layout.Strides[i] = int64(meta.strides[i])
		}
	}
	r.bits = q.bits
	return r, nil
}
