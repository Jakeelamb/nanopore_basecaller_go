//go:build cgo && linux

package tensor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type storageCheck struct {
	Name           string `json:"name"`
	ExpectedSHA256 string `json:"expected_sha256"`
	ObservedSHA256 string `json:"observed_sha256"`
	Bytes          int    `json:"bytes"`
}
type storageRejection struct {
	Name  string `json:"name"`
	Error string `json:"error"`
}
type storageCapture struct {
	Schema     string             `json:"schema"`
	Status     string             `json:"status"`
	Device     uint64             `json:"device"`
	Ordinal    int32              `json:"ordinal"`
	Runtime    map[string]string  `json:"runtime_sha256"`
	Checks     []storageCheck     `json:"checks"`
	Rejections []storageRejection `json:"rejections"`
}

// storageRuntimePins records the CUDA and Torch shared objects this process
// actually mapped, so the orchestrator can re-hash them outside the child.
func storageRuntimePins(t *testing.T, capture *storageCapture) {
	t.Helper()
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	capture.Runtime = map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := strings.Join(fields[5:], " ")
		name := filepath.Base(path)
		if !(strings.HasPrefix(name, "libtorch") || strings.HasPrefix(name, "libc10") || strings.HasPrefix(name, "libcu") || strings.HasPrefix(name, "libnv")) {
			continue
		}
		if strings.Contains(path, "CUDA-12") || strings.Contains(path, "cuda-12") {
			t.Fatalf("foreign CUDA12 library mapped: %s", path)
		}
		if _, ok := capture.Runtime[path]; ok {
			continue
		}
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		capture.Runtime[path] = digest(body)
	}
	for _, want := range []string{"libcuda.so", "libcudart.so", "libcublas.so", "libc10_cuda.so", "libtorch_cuda.so"} {
		found := false
		for path := range capture.Runtime {
			if strings.HasPrefix(filepath.Base(path), want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("runtime pin set lacks %s", want)
		}
	}
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// TestNativeCUDAStorageDevice exercises the typed storage ABI on a real device.
// Every payload assertion is exact-byte, and every rejection is recorded so the
// orchestrator can require the complete control set from the native receipt.
func TestNativeCUDAStorageDevice(t *testing.T) {
	_, d, s := cudaEngine(t)
	var capture storageCapture
	capture.Schema = "nanopore-cuda-storage-koi/v1"
	capture.Status = "failed"
	capture.Device = d.state.id
	capture.Ordinal = d.state.ordinal
	storageRuntimePins(t, &capture)
	check := func(name string, want, got []byte) {
		t.Helper()
		capture.Checks = append(capture.Checks, storageCheck{Name: name, ExpectedSHA256: digest(want), ObservedSHA256: digest(got), Bytes: len(got)})
		if digest(want) != digest(got) {
			t.Fatalf("%s payload differs", name)
		}
	}
	reject := func(name string, err error) {
		t.Helper()
		if err == nil {
			t.Fatalf("%s accepted", name)
		}
		capture.Rejections = append(capture.Rejections, storageRejection{Name: name, Error: err.Error()})
	}

	// Exact roundtrip for every supported dtype.
	f16Bits := []byte{0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x46, 0x00, 0x48}
	f16, err := d.NewStorage(s, DTypeF16, []int64{2, 3}, []KoiTag{TagN, TagT}, f16Bits)
	if err != nil {
		t.Fatal(err)
	}
	got, err := f16.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("f16-roundtrip", f16Bits, got)

	i8Data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	scaleData := []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40}
	scale, err := d.NewStorage(s, DTypeF32, []int64{4}, []KoiTag{TagC}, scaleData)
	if err != nil {
		t.Fatal(err)
	}
	i8, err := d.NewStorage(s, DTypeI8, []int64{2, 4}, []KoiTag{TagN, TagK}, i8Data)
	if err != nil {
		t.Fatal(err)
	}
	if err := i8.BindScale(scale, TagK); err != nil {
		t.Fatal(err)
	}
	koi, err := i8.KoiTensor()
	if err != nil {
		t.Fatal(err)
	}
	if koi.DType != KoiI8 || koi.Scale == nil || koi.Scale.DType != KoiF32 || koi.Scale.Tag != TagK || len(koi.Dims) != 2 || koi.Dims[1].Tag != TagK {
		t.Fatalf("koi descriptor %+v", koi)
	}
	got, err = i8.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("i8-roundtrip", i8Data, got)
	got, err = scale.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("f32-scale-roundtrip", scaleData, got)

	e4m3 := []byte{0x00, 0x38, 0x40, 0x48, 0xb8, 0x7e, 0x7f, 0x01}
	em, err := d.NewStorage(s, DTypeE4M3, []int64{8}, []KoiTag{TagC}, e4m3)
	if err != nil {
		t.Fatal(err)
	}
	got, err = em.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("e4m3-roundtrip", e4m3, got)

	i32Data := []byte{1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0x78, 0x56, 0x34, 0x12}
	i32, err := d.NewStorage(s, DTypeI32, []int64{3}, []KoiTag{TagC}, i32Data)
	if err != nil {
		t.Fatal(err)
	}
	got, err = i32.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("i32-roundtrip", i32Data, got)

	u8Data := []byte{0, 1, 254, 255}
	u8, err := d.NewStorage(s, DTypeU8, []int64{4}, []KoiTag{TagC}, u8Data)
	if err != nil {
		t.Fatal(err)
	}
	got, err = u8.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("u8-roundtrip", u8Data, got)
	_, err = u8.KoiTensor()
	reject("u8-has-no-koi-type", err)

	// Strided view writes stay inside the declared interval. `parent` is a view
	// into a larger allocation so a child that leaves the parent interval is still
	// inside the allocation: only the native interval guard can reject it, not
	// torch's own storage bounds check.
	rootData := make([]byte, 100*2)
	for i := 0; i < 100; i++ {
		rootData[2*i] = byte(i)
	}
	root, err := d.NewStorage(s, DTypeF16, []int64{10, 10}, []KoiTag{TagN, TagT}, rootData)
	if err != nil {
		t.Fatal(err)
	}
	parentData := append([]byte{}, rootData[4*2:40*2]...)
	parent, err := root.View(4, []int64{6, 6}, []int64{6, 1}, []KoiTag{TagN, TagT})
	if err != nil {
		t.Fatal(err)
	}
	view, err := parent.View(1, []int64{2, 2}, []int64{2, 1}, []KoiTag{TagN, TagC})
	if err != nil {
		t.Fatal(err)
	}
	strided, err := parent.View(1, []int64{2, 2}, []int64{6, 1}, []KoiTag{TagN, TagC})
	if err != nil {
		t.Fatal(err)
	}
	patch := []byte{0xaa, 0xaa, 0xbb, 0xbb, 0xcc, 0xcc, 0xdd, 0xdd}
	if err := view.Upload(s, patch); err != nil {
		t.Fatal(err)
	}
	want := append([]byte{}, parentData...)
	for i, at := range []int{1, 2, 3, 4} {
		copy(want[2*at:], patch[2*i:2*i+2])
	}
	got, err = parent.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("strided-view-guard", want, got)

	// Fill on the same view writes only the view elements.
	if err := view.Fill(s, []byte{0x11, 0x11}); err != nil {
		t.Fatal(err)
	}
	for _, at := range []int{1, 2, 3, 4} {
		copy(want[2*at:], []byte{0x11, 0x11})
	}
	got, err = parent.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("strided-view-fill", want, got)

	// Device-to-device copy of a contiguous view.
	dst, err := d.NewStorage(s, DTypeF16, []int64{6, 6}, []KoiTag{TagN, TagT}, make([]byte, 72))
	if err != nil {
		t.Fatal(err)
	}
	if err := parent.CopyTo(s, dst); err != nil {
		t.Fatal(err)
	}
	got, err = dst.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	check("device-copy", want, got)

	// Rejection controls.
	overlap, err := parent.View(1, []int64{5, 5}, []int64{6, 1}, []KoiTag{TagN, TagC})
	if err != nil {
		t.Fatal(err)
	}
	same, err := parent.View(2, []int64{2, 2}, []int64{2, 1}, []KoiTag{TagN, TagC})
	if err != nil {
		t.Fatal(err)
	}
	reject("overlapping-copy", view.CopyTo(s, same))
	reject("dtype-mismatch-copy", f16.CopyTo(s, i32))
	reject("non-contiguous-upload", strided.Upload(s, make([]byte, 8)))
	reject("non-contiguous-download", func() error { _, err := strided.Download(s); return err }())
	reject("scale-requires-i8", f16.BindScale(scale, TagC))
	reject("scale-requires-f32", i8.BindScale(f16, TagK))
	reject("native-scale-requires-i8", func() error {
		_, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_scale", storage: f16.state.id, scale: scale.state.id, tags: []int32{int32(TagC)}})
		return err
	}())
	reject("native-scale-requires-f32", func() error {
		_, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_scale", storage: i8.state.id, scale: f16.state.id, tags: []int32{int32(TagK)}})
		return err
	}())
	reject("scale-self", i8.BindScale(i8, TagK))
	reject("scale-unknown-tag", i8.BindScale(scale, KoiTag(999)))
	reject("scale-owner-release", scale.Close())
	reject("fill-pattern-width", parent.Fill(s, []byte{1}))
	// Same element width, different dtype: only the native dtype guard can reject
	// this pair, so removing that guard is observable.
	f32 := mustStorage(t, d, s, DTypeF32, []int64{3}, []KoiTag{TagC}, i32Data)
	reject("native-copy-dtype-mismatch", func() error {
		_, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_copy", stream: s.state.id, storage: f32.state.id, src: i32.state.id})
		return err
	}())
	reject("native-fill-pattern-width", func() error {
		_, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_fill", stream: s.state.id, storage: parent.state.id, bytes: []byte{1, 1, 1, 1}})
		return err
	}())
	reject("view-exceeds-parent", func() error {
		_, err := parent.View(0, []int64{7, 6}, []int64{6, 1}, []KoiTag{TagN, TagT})
		return err
	}())
	reject("view-negative-stride", func() error {
		_, err := parent.View(0, []int64{2, 2}, []int64{6, -1}, []KoiTag{TagN, TagT})
		return err
	}())
	other, _, _ := cudaEngine(t)
	_, err = other.backend.cuda(cudaRequest{op: "storage_meta", storage: parent.state.id})
	reject("foreign-engine-handle", err)
	reject("unknown-dtype", func() error { _, err := d.NewStorage(s, DType(99), []int64{1}, []KoiTag{TagC}, []byte{0}); return err }())

	// Closing order: views then parents, and a released handle stays closed.
	for _, h := range []*Storage{overlap, same, strided, view, dst, parent, root, em, i32, f32, u8} {
		if err := h.Close(); err != nil {
			t.Fatal(err)
		}
	}
	reject("closed-download", func() error { _, err := parent.Download(s); return err }())
	if err := i8.BindScale(nil, 0); err != nil {
		t.Fatal(err)
	}
	for _, h := range []*Storage{scale, i8, f16} {
		if err := h.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range capture.Checks {
		if c.ExpectedSHA256 != c.ObservedSHA256 {
			t.Fatalf("capture check %s mismatch", c.Name)
		}
	}
	capture.Status = "passed"
	if path := os.Getenv("NANOPORE_TENSOR_STORAGE_CAPTURE"); path != "" {
		raw, err := json.MarshalIndent(capture, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestNativeCUDAStorageProbe loads the backend and opens a device without the
// runtime-pin manifest check, recording the mapped CUDA/Torch objects so the
// orchestrator can pin them before the qualified run.
func TestNativeCUDAStorageProbe(t *testing.T) {
	path := os.Getenv("NANOPORE_TENSOR_STORAGE_PROBE")
	if path == "" {
		t.Skip("storage probe not configured")
	}
	e := nativeEngine(t)
	ordinal := 0
	if raw := os.Getenv("NANOPORE_TENSOR_CUDA_ORDINAL"); raw != "" {
		var err error
		ordinal, err = strconv.Atoi(raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	d, err := e.NewCUDADevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	var capture storageCapture
	storageRuntimePins(t, &capture)
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := json.MarshalIndent(map[string]any{"runtime_sha256": capture.Runtime}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

// mustStorage allocates a device storage or fails the test.
func mustStorage(t *testing.T, d *Device, s *Stream, dtype DType, shape []int64, tags []KoiTag, data []byte) *Storage {
	t.Helper()
	h, err := d.NewStorage(s, dtype, shape, tags, data)
	if err != nil {
		t.Fatal(err)
	}
	return h
}
