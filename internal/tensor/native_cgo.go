//go:build cgo && linux

package tensor

/*
#cgo LDFLAGS: -ldl
#include "loader.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

type nativeBackend struct {
	library *C.nt_library
	id      C.uint64_t
}

func pointer[T any](v []T) *T {
	if len(v) == 0 {
		return nil
	}
	return &v[0]
}
func nativeError(code C.int, buf *[C.NT_ERROR_CAP]C.char) error {
	if code == 0 {
		return nil
	}
	return errors.New(C.GoString(&buf[0]))
}
func openNative(path string) (*nativeBackend, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var errbuf [C.NT_ERROR_CAP]C.char
	l := C.nt_library_open(p, &errbuf[0], C.NT_ERROR_CAP)
	if l == nil {
		return nil, errors.New(C.GoString(&errbuf[0]))
	}
	b := &nativeBackend{library: l}
	if err := nativeError(C.nt_call_open(l, &b.id, &errbuf[0], C.NT_ERROR_CAP), &errbuf); err != nil {
		C.nt_library_close(l)
		return nil, err
	}
	return b, nil
}
func (b *nativeBackend) close() error {
	var buf [C.NT_ERROR_CAP]C.char
	if err := nativeError(C.nt_call_close(b.library, b.id, &buf[0], C.NT_ERROR_CAP), &buf); err != nil {
		return err
	}
	C.nt_library_close(b.library)
	b.library = nil
	return nil
}
func (b *nativeBackend) newTensor(shape []int64, data []float32) (uint64, error) {
	var id C.uint64_t
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_new(b.library, b.id, (*C.int64_t)(unsafe.Pointer(pointer(shape))), C.size_t(len(shape)), (*C.float)(unsafe.Pointer(pointer(data))), C.size_t(len(data)), &id, &buf[0], C.NT_ERROR_CAP)
	return uint64(id), nativeError(code, &buf)
}
func (b *nativeBackend) load(path string) (uint64, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var id C.uint64_t
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_load(b.library, b.id, p, &id, &buf[0], C.NT_ERROR_CAP)
	return uint64(id), nativeError(code, &buf)
}
func (b *nativeBackend) release(id uint64) error {
	var buf [C.NT_ERROR_CAP]C.char
	return nativeError(C.nt_call_release(b.library, b.id, C.uint64_t(id), &buf[0], C.NT_ERROR_CAP), &buf)
}
func (b *nativeBackend) shape(id uint64) ([]int64, error) {
	dims := make([]int64, 32)
	var rank C.size_t
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_shape(b.library, b.id, C.uint64_t(id), (*C.int64_t)(unsafe.Pointer(&dims[0])), 32, &rank, &buf[0], C.NT_ERROR_CAP)
	if err := nativeError(code, &buf); err != nil {
		return nil, err
	}
	if rank > 32 {
		return nil, errors.New("backend shape rank exceeds ABI limit")
	}
	return dims[:int(rank)], nil
}
func (b *nativeBackend) data(id uint64, n int) ([]float32, error) {
	data := make([]float32, n)
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_data(b.library, b.id, C.uint64_t(id), (*C.float)(unsafe.Pointer(pointer(data))), C.size_t(n), &buf[0], C.NT_ERROR_CAP)
	return data, nativeError(code, &buf)
}
func (b *nativeBackend) run(op int, ids []uint64, p []int64, f []float64) (uint64, error) {
	var id C.uint64_t
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_run(b.library, b.id, C.int(op), (*C.uint64_t)(unsafe.Pointer(pointer(ids))), C.size_t(len(ids)), (*C.int64_t)(unsafe.Pointer(pointer(p))), C.size_t(len(p)), (*C.double)(unsafe.Pointer(pointer(f))), C.size_t(len(f)), &id, &buf[0], C.NT_ERROR_CAP)
	return uint64(id), nativeError(code, &buf)
}

func (b *nativeBackend) newInt64(shape, data []int64) (uint64, error) {
	var id C.uint64_t
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_new_int64(b.library, b.id, (*C.int64_t)(unsafe.Pointer(pointer(shape))), C.size_t(len(shape)), (*C.int64_t)(unsafe.Pointer(pointer(data))), C.size_t(len(data)), &id, &buf[0], C.NT_ERROR_CAP)
	return uint64(id), nativeError(code, &buf)
}
func (b *nativeBackend) dataInt64(id uint64, n int) ([]int64, error) {
	data := make([]int64, n)
	var buf [C.NT_ERROR_CAP]C.char
	code := C.nt_call_data_int64(b.library, b.id, C.uint64_t(id), (*C.int64_t)(unsafe.Pointer(pointer(data))), C.size_t(n), &buf[0], C.NT_ERROR_CAP)
	return data, nativeError(code, &buf)
}
