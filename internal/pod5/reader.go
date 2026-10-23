// Package pod5 reads POD5 files through the pinned native C API.
package pod5

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"
)

var ErrClosed = errors.New("POD5 reader is closed")
var ErrNotFound = errors.New("POD5 read ID not found")
var ErrUnsupported = errors.New("POD5 requires Linux with cgo enabled")

type ID [16]byte

func ParseID(s string) (ID, error) {
	var id ID
	if len(s) != 36 || s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
		return id, fmt.Errorf("invalid POD5 UUID %q", s)
	}
	raw, err := hex.DecodeString(s[:8] + s[9:13] + s[14:18] + s[19:23] + s[24:])
	if err != nil {
		return id, fmt.Errorf("invalid POD5 UUID: %w", err)
	}
	copy(id[:], raw)
	return id, nil
}
func (id ID) String() string {
	h := hex.EncodeToString(id[:])
	return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

type Options struct {
	// MaxReads bounds the metadata index. A larger file is rejected, not truncated.
	MaxReads uint64
	// MaxSamples bounds each decoded read. Zero selects the default of 100 million.
	MaxSamples uint64
}

type Read struct {
	RunInfo
	ID                                                 ID
	Signal                                             []int16
	NumSamples                                         uint64
	CalibrationScale, CalibrationOffset, OpenPoreLevel float32
	MedianBefore                                       float32
	ReadNumber                                         uint32
	StartSample, EndSample                             uint64
	Channel                                            uint16
	Well                                               uint8
	NumMinknowEvents                                   uint64
	EndReason                                          uint8
	EndReasonForced                                    bool
	EndReasonName, PoreType                            string
	ReadTableVersion                                   uint16
	Filename                                           string
}

type RunInfo struct {
	SampleRate uint64
	// RunID follows Dorado and denotes protocol_run_id, not acquisition_id.
	RunID, AcquisitionID                             string
	AcquisitionStartTimeMS, ProtocolStartTimeMS      int64
	FlowCellID, FlowCellProductCode, SequencingKit   string
	PositionID, PositionType, SampleID, ExperimentID string
	ProtocolName, Software, SystemName, SystemType   string
	ADCMin, ADCMax                                   int16
	ContextTags, TrackingID                          map[string]string
}

type FileMetadata struct {
	ID              ID
	OriginalVersion [3]uint16
	NumReads        uint64
	Filename        string
	Runs            []RunInfo
}

type location struct{ batch, row uint64 }
type Reader struct {
	mu       sync.Mutex
	native   *nativeReader
	options  Options
	filename string
	ids      []ID
	index    map[ID]location
}

func Open(libraryPath, filePath string, options ...Options) (*Reader, error) {
	return OpenContext(context.Background(), libraryPath, filePath, options...)
}
func OpenContext(ctx context.Context, libraryPath, filePath string, options ...Options) (*Reader, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if libraryPath == "" || filePath == "" || strings.ContainsRune(libraryPath, 0) || strings.ContainsRune(filePath, 0) {
		return nil, errors.New("POD5 library and file paths must be nonempty without NUL bytes")
	}
	if len(options) > 1 {
		return nil, errors.New("POD5 accepts at most one Options value")
	}
	o := Options{}
	if len(options) == 1 {
		o = options[0]
	}
	if o.MaxReads == 0 {
		o.MaxReads = 1_000_000
	}
	if o.MaxSamples == 0 {
		o.MaxSamples = 100_000_000
	}
	if o.MaxReads > uint64(math.MaxInt) || o.MaxSamples > uint64(math.MaxInt/2) {
		return nil, errors.New("POD5 bounds exceed platform address space")
	}
	n, count, batches, err := openNative(libraryPath, filePath, o.MaxReads)
	if err != nil {
		return nil, err
	}
	r := &Reader{native: n, options: o, filename: filepath.Base(filePath), index: make(map[ID]location), ids: make([]ID, 0, int(count))}
	fail := func(err error) (*Reader, error) { return nil, errors.Join(err, r.Close()) }
	if count > o.MaxReads || batches > o.MaxReads+1 {
		return fail(errors.New("POD5 metadata exceeds MaxReads"))
	}
	for batch := uint64(0); batch < batches; batch++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		rows, err := n.batchRows(batch)
		if err != nil {
			return fail(err)
		}
		if rows > count-uint64(len(r.ids)) {
			return fail(errors.New("POD5 batch row count exceeds declared read count"))
		}
		for row := uint64(0); row < rows; row++ {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
			id, err := n.rowID(batch, row)
			if err != nil {
				return fail(err)
			}
			if _, duplicate := r.index[id]; duplicate {
				return fail(fmt.Errorf("duplicate POD5 UUID %s", id))
			}
			r.index[id] = location{batch, row}
			r.ids = append(r.ids, id)
		}
	}
	if uint64(len(r.ids)) != count {
		return fail(errors.New("POD5 read count disagrees with batches"))
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	return r, nil
}
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.native == nil {
		return nil
	}
	err := r.native.close()
	r.native = nil
	r.ids = nil
	r.index = nil
	return err
}
func (r *Reader) IDs(ctx context.Context) ([]ID, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.native == nil {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return append([]ID(nil), r.ids...), nil
}

// Metadata copies file and run information without decoding any signal.
func (r *Reader) Metadata(ctx context.Context) (FileMetadata, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.native == nil {
		return FileMetadata{}, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return FileMetadata{}, err
	}
	metadata, err := r.native.metadata(ctx)
	if err != nil {
		return FileMetadata{}, err
	}
	metadata.NumReads = uint64(len(r.ids))
	metadata.Filename = r.filename
	return metadata, nil
}
func (r *Reader) Read(ctx context.Context, id ID) (Read, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.native == nil {
		return Read{}, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return Read{}, err
	}
	loc, ok := r.index[id]
	if !ok {
		return Read{}, fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	read, err := r.native.read(loc, r.options.MaxSamples)
	if err != nil {
		return Read{}, err
	}
	if err := ctx.Err(); err != nil {
		return Read{}, err
	}
	if read.ID != id || read.NumSamples != uint64(len(read.Signal)) {
		return Read{}, errors.New("POD5 read changed identity or signal length")
	}
	if read.StartSample > math.MaxUint64-read.NumSamples {
		return Read{}, errors.New("POD5 end sample overflows uint64")
	}
	read.EndSample = read.StartSample + read.NumSamples
	read.Filename = r.filename
	return read, nil
}

// Iterate preserves selection order and duplicates. Nil selects all reads in file order.
// A missing selection fails before any callback. Callbacks may close the reader.
func (r *Reader) Iterate(ctx context.Context, selection []ID, visit func(Read) error) error {
	if visit == nil {
		return errors.New("POD5 visit callback is nil")
	}
	ids, err := r.IDs(ctx)
	if err != nil {
		return err
	}
	if selection != nil {
		known := make(map[ID]struct{}, len(ids))
		for _, id := range ids {
			known[id] = struct{}{}
		}
		ids = append([]ID(nil), selection...)
		for _, id := range ids {
			if _, ok := known[id]; !ok {
				return fmt.Errorf("%w: %s", ErrNotFound, id)
			}
		}
	}
	for _, id := range ids {
		read, err := r.Read(ctx, id)
		if err != nil {
			return err
		}
		if err := visit(read); err != nil {
			return err
		}
	}
	return ctx.Err()
}
