//go:build !cgo || !linux

package tensor

type nativeBackend struct{}

func openNative(string) (*nativeBackend, error)                     { return nil, ErrUnsupported }
func (*nativeBackend) close() error                                 { return ErrUnsupported }
func (*nativeBackend) newTensor([]int64, []float32) (uint64, error) { return 0, ErrUnsupported }
func (*nativeBackend) load(string) (uint64, error)                  { return 0, ErrUnsupported }
func (*nativeBackend) release(uint64) error                         { return ErrUnsupported }
func (*nativeBackend) shape(uint64) ([]int64, error)                { return nil, ErrUnsupported }
func (*nativeBackend) data(uint64, int) ([]float32, error)          { return nil, ErrUnsupported }
func (*nativeBackend) run(int, []uint64, []int64, []float64) (uint64, error) {
	return 0, ErrUnsupported
}
func (*nativeBackend) newInt64([]int64, []int64) (uint64, error) { return 0, ErrUnsupported }
func (*nativeBackend) dataInt64(uint64, int) ([]int64, error)    { return nil, ErrUnsupported }
func (*nativeBackend) sdpa(uint64, uint64, uint64, []int64, []byte) (uint64, error) {
	return 0, ErrUnsupported
}
