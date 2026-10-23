//go:build !cgo || !linux

package pod5

import "context"

type nativeReader struct{}

func (*nativeReader) metadata(context.Context) (FileMetadata, error) {
	return FileMetadata{}, ErrUnsupported
}

func openNative(string, string, uint64) (*nativeReader, uint64, uint64, error) {
	return nil, 0, 0, ErrUnsupported
}
func (*nativeReader) close() error                        { return ErrUnsupported }
func (*nativeReader) batchRows(uint64) (uint64, error)    { return 0, ErrUnsupported }
func (*nativeReader) rowID(uint64, uint64) (ID, error)    { return ID{}, ErrUnsupported }
func (*nativeReader) read(location, uint64) (Read, error) { return Read{}, ErrUnsupported }
