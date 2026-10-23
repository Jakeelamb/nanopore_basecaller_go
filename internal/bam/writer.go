// Package bam writes the pinned unaligned BAM record and header contract.
// Compression and serialization are Go-owned; no htslib or cgo is used.
package bam

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"strings"
)

// BGZF limits. A block is at most 64 KiB including the 18-byte header and the
// 8-byte trailer, matching the pinned htslib writer.
const (
	bgzfBlockData   = 0xff00
	bgzfMaxBlock    = 1 << 16
	bgzfHeaderBytes = 18
	bgzfFooterBytes = 8
	unmappedBin     = 4680
	unmappedFlag    = 4
)

var bgzfEOF = []byte{
	0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
	0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
	0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00,
}

// Header is the deterministic BAM header contract. Reference records are
// optional because the unaligned caller emits none.
type Header struct {
	Version    string
	SortOrder  string
	Program    Program
	ReadGroup  *ReadGroup
	References []Reference
}

type Program struct {
	ID, Name, Version, CommandLine, Description string
}

type Reference struct {
	Name   string
	Length int64
}

type ReadGroup struct {
	ID, FlowCellID, DeviceID, ExperimentStartTime string
	RunID, BasecallModel, ExperimentID            string
	AcquisitionStartTime, SampleID                string
	ModelStride                                   int
	TrimFlags                                     string
}

func (h Header) validate() error {
	if h.Version != "1.6" || h.SortOrder != "unknown" {
		return fmt.Errorf("unaligned header must be VN:1.6 SO:unknown, got VN:%s SO:%s", h.Version, h.SortOrder)
	}
	if h.Program.ID == "" || h.Program.Name == "" || h.Program.Version == "" {
		return errors.New("incomplete BAM header")
	}
	if strings.ContainsAny(h.Version+h.SortOrder+h.Program.ID+h.Program.Name+h.Program.Version+h.Program.CommandLine+h.Program.Description, "\n\r\x00") {
		return errors.New("BAM header contains a line break or NUL")
	}
	if h.ReadGroup != nil {
		rg := h.ReadGroup
		if rg.ID == "" || strings.ContainsAny(rg.ID+rg.FlowCellID+rg.DeviceID+rg.ExperimentStartTime+rg.RunID+rg.BasecallModel+rg.ExperimentID+rg.AcquisitionStartTime+rg.SampleID+rg.TrimFlags, "\n\r\x00\t") {
			return errors.New("BAM read group contains a separator")
		}
		if rg.ModelStride <= 0 {
			return errors.New("BAM read group has no model stride")
		}
	}
	for _, ref := range h.References {
		if ref.Name == "" || ref.Length <= 0 || strings.ContainsAny(ref.Name, "\n\r\x00\t") {
			return errors.New("invalid BAM reference")
		}
	}
	return nil
}

// Text renders the SAM header text in the pinned line order.
func (h Header) Text() (string, error) {
	if err := h.validate(); err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "@HD\tVN:%s\tSO:%s\n", h.Version, h.SortOrder)
	fmt.Fprintf(&b, "@PG\tID:%s\tPN:%s\tVN:%s", h.Program.ID, h.Program.Name, h.Program.Version)
	if h.Program.CommandLine != "" {
		fmt.Fprintf(&b, "\tCL:%s", h.Program.CommandLine)
	}
	if h.Program.Description != "" {
		fmt.Fprintf(&b, "\tDS:%s", h.Program.Description)
	}
	b.WriteByte('\n')
	if rg := h.ReadGroup; rg != nil {
		fmt.Fprintf(&b, "@RG\tID:%s\tPU:%s\tPM:%s\tDT:%s\tPL:ONT\tDS:runid=%s basecall_model=%s",
			rg.ID, unknown(rg.FlowCellID), unknown(rg.DeviceID), unknown(rg.ExperimentStartTime), unknown(rg.RunID), unknown(rg.BasecallModel))
		if rg.ExperimentID != "" {
			fmt.Fprintf(&b, " experiment_id=%s", rg.ExperimentID)
		}
		fmt.Fprintf(&b, " acquisition_start_time=%s model_stride=%d", unknown(rg.AcquisitionStartTime), rg.ModelStride)
		fmt.Fprintf(&b, "\tLB:%s\ttm:%s\n", unknown(rg.SampleID), rg.TrimFlags)
	}
	for _, ref := range h.References {
		fmt.Fprintf(&b, "@SQ\tSN:%s\tLN:%d\n", ref.Name, ref.Length)
	}
	return b.String(), nil
}

func unknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

// Writer emits BGZF-compressed BAM. It records the first error and never reports
// partial success after one.
type Writer struct {
	w      io.Writer
	buf    []byte
	err    error
	closed bool
}

func NewWriter(w io.Writer) *Writer { return &Writer{w: w} }

// WriteHeader writes the BAM magic, text header and reference dictionary.
func (w *Writer) WriteHeader(h Header) error {
	text, err := h.Text()
	if err != nil {
		return w.fail(err)
	}
	if len(text) > 1<<31-1 {
		return w.fail(errors.New("BAM header text exceeds int32"))
	}
	var head bytes.Buffer
	head.WriteString("BAM\x01")
	var word [4]byte
	binary.LittleEndian.PutUint32(word[:], uint32(len(text)))
	head.Write(word[:])
	head.WriteString(text)
	binary.LittleEndian.PutUint32(word[:], uint32(len(h.References)))
	head.Write(word[:])
	for _, ref := range h.References {
		head.WriteString(ref.Name)
		head.WriteByte(0)
		var length [4]byte
		binary.LittleEndian.PutUint32(length[:], uint32(ref.Length))
		head.Write(length[:])
	}
	if _, err := w.Write(head.Bytes()); err != nil {
		return err
	}
	return nil
}

// WriteRecord appends one unmapped record.
func (w *Writer) WriteRecord(r Record) error {
	block, err := r.encode()
	if err != nil {
		return w.fail(err)
	}
	_, err = w.Write(block)
	return err
}

// Close flushes pending data and writes the BGZF EOF marker. It is idempotent.
func (w *Writer) Close() error {
	if w.closed {
		return w.err
	}
	w.closed = true
	if err := w.flush(); err != nil {
		return w.fail(err)
	}
	if _, err := w.w.Write(bgzfEOF); err != nil {
		return w.fail(err)
	}
	return w.err
}

func (w *Writer) fail(err error) error {
	if w.err == nil {
		w.err = err
	}
	return w.err
}

func (w *Writer) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	if w.closed {
		return 0, w.fail(errors.New("BAM writer is closed"))
	}
	written := 0
	for len(p) > 0 {
		n := min(len(p), bgzfBlockData-len(w.buf))
		w.buf = append(w.buf, p[:n]...)
		p = p[n:]
		written += n
		if len(w.buf) == bgzfBlockData {
			if err := w.flush(); err != nil {
				return written, w.fail(err)
			}
		}
	}
	return written, nil
}

func (w *Writer) flush() error {
	if len(w.buf) == 0 {
		return nil
	}
	block, err := bgzfBlock(w.buf)
	if err != nil {
		return err
	}
	n, err := w.w.Write(block)
	if err == nil && n != len(block) {
		err = io.ErrShortWrite
	}
	w.buf = w.buf[:0]
	return err
}

// bgzfBlock wraps one uncompressed chunk in a BGZF member. A stored deflate
// block is used when compression would exceed the 64 KiB member limit.
func bgzfBlock(data []byte) ([]byte, error) {
	if len(data) > bgzfBlockData {
		return nil, errors.New("BGZF block input too large")
	}
	var compressed bytes.Buffer
	fw, err := flate.NewWriter(&compressed, flate.DefaultCompression)
	if err != nil {
		return nil, err
	}
	if _, err := fw.Write(data); err != nil {
		return nil, err
	}
	if err := fw.Close(); err != nil {
		return nil, err
	}
	payload := compressed.Bytes()
	if bgzfHeaderBytes+len(payload)+bgzfFooterBytes > bgzfMaxBlock {
		payload = storedDeflate(data)
	}
	if bgzfHeaderBytes+len(payload)+bgzfFooterBytes > bgzfMaxBlock {
		return nil, errors.New("BGZF member exceeds 64 KiB")
	}
	total := bgzfHeaderBytes + len(payload) + bgzfFooterBytes
	out := make([]byte, 0, total)
	out = append(out,
		0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
		0x06, 0x00, 0x42, 0x43, 0x02, 0x00)
	out = append(out, byte(total-1), byte((total-1)>>8))
	out = append(out, payload...)
	var footer [8]byte
	binary.LittleEndian.PutUint32(footer[:4], crc32.ChecksumIEEE(data))
	binary.LittleEndian.PutUint32(footer[4:], uint32(len(data)))
	out = append(out, footer[:]...)
	return out, nil
}

// storedDeflate emits a raw deflate stored block.
func storedDeflate(data []byte) []byte {
	out := make([]byte, 0, len(data)+5)
	out = append(out, 0x01)
	n := uint16(len(data))
	out = append(out, byte(n), byte(n>>8), byte(^n), byte(^n>>8))
	return append(out, data...)
}

// Record is one unmapped read. Quality holds phred scores, not FASTQ bytes.
type Record struct {
	Name     string
	Flag     uint16
	Sequence string
	Quality  []byte
	Aux      []Aux
}

func (r Record) encode() ([]byte, error) {
	if err := r.validate(); err != nil {
		return nil, err
	}
	name := append([]byte(r.Name), 0)
	seq := make([]byte, (len(r.Sequence)+1)/2)
	for i := range r.Sequence {
		code, ok := baseCodes[r.Sequence[i]]
		if !ok {
			return nil, fmt.Errorf("unsupported base %q", r.Sequence[i])
		}
		if i%2 == 0 {
			seq[i/2] = code << 4
		} else {
			seq[i/2] |= code
		}
	}
	var aux bytes.Buffer
	for _, a := range r.Aux {
		if err := a.encode(&aux); err != nil {
			return nil, err
		}
	}
	block := 32 + len(name) + len(seq) + len(r.Quality) + aux.Len()
	if block > 1<<31-1 {
		return nil, errors.New("BAM record exceeds int32")
	}
	out := make([]byte, 0, 4+block)
	var word [4]byte
	binary.LittleEndian.PutUint32(word[:], uint32(block))
	out = append(out, word[:]...)
	put32 := func(v int32) {
		binary.LittleEndian.PutUint32(word[:], uint32(v))
		out = append(out, word[:]...)
	}
	put32(-1) // refID
	put32(-1) // pos
	out = append(out, byte(len(name)), 0)
	out = append(out, byte(unmappedBin&0xff), byte(unmappedBin>>8))
	var cigar [2]byte
	binary.LittleEndian.PutUint16(cigar[:], 0)
	out = append(out, cigar[:]...)
	var flag [2]byte
	binary.LittleEndian.PutUint16(flag[:], r.Flag)
	out = append(out, flag[:]...)
	put32(int32(len(r.Sequence)))
	put32(-1) // next refID
	put32(-1) // next pos
	put32(0)  // tlen
	out = append(out, name...)
	out = append(out, seq...)
	out = append(out, r.Quality...)
	out = append(out, aux.Bytes()...)
	return out, nil
}

func (r Record) validate() error {
	if r.Name == "" || len(r.Name) > 254 || strings.ContainsRune(r.Name, 0) {
		return errors.New("invalid BAM read name")
	}
	if r.Sequence == "" {
		return errors.New("empty BAM sequence")
	}
	if len(r.Quality) != len(r.Sequence) {
		return errors.New("BAM quality and sequence lengths differ")
	}
	for _, q := range r.Quality {
		if q > 93 {
			return errors.New("BAM quality exceeds 93")
		}
	}
	if r.Flag != unmappedFlag {
		return fmt.Errorf("unaligned writer requires flag 4, got %d", r.Flag)
	}
	return nil
}

var baseCodes = map[byte]byte{
	'=': 0, 'A': 1, 'C': 2, 'M': 3, 'G': 4, 'R': 5, 'S': 6, 'V': 7,
	'T': 8, 'W': 9, 'Y': 10, 'H': 11, 'K': 12, 'D': 13, 'B': 14, 'N': 15,
}

// Aux is one BAM auxiliary field. Name is exactly two characters and Type is
// the BAM type byte. Values are checked at encode time.
type Aux struct {
	Name  string
	Type  byte
	Value any
}

func Float(name string, v float32) Aux    { return Aux{Name: name, Type: 'f', Value: v} }
func Int32(name string, v int32) Aux      { return Aux{Name: name, Type: 'i', Value: v} }
func Uint32(name string, v uint32) Aux    { return Aux{Name: name, Type: 'I', Value: v} }
func String(name string, v string) Aux    { return Aux{Name: name, Type: 'Z', Value: v} }
func Char(name string, v byte) Aux        { return Aux{Name: name, Type: 'A', Value: v} }
func ByteArray(name string, v []byte) Aux { return Aux{Name: name, Type: 'B', Value: v} }

// ArrayInt32 is a B:i payload. Source poly-tail and barcode tags use it.
type ArrayInt32 []int32

// ArrayFloat32 is a B:f payload.
type ArrayFloat32 []float32

func Int32Array(name string, v []int32) Aux { return Aux{Name: name, Type: 'B', Value: ArrayInt32(v)} }
func Float32Array(name string, v []float32) Aux {
	return Aux{Name: name, Type: 'B', Value: ArrayFloat32(v)}
}

func (a Aux) encode(b *bytes.Buffer) error {
	if len(a.Name) != 2 {
		return fmt.Errorf("BAM tag name %q is not two characters", a.Name)
	}
	b.WriteString(a.Name)
	b.WriteByte(a.Type)
	switch a.Type {
	case 'f':
		v, ok := a.Value.(float32)
		if !ok {
			return fmt.Errorf("BAM tag %s is not float32", a.Name)
		}
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(v))
		b.Write(word[:])
	case 'i':
		v, ok := a.Value.(int32)
		if !ok {
			return fmt.Errorf("BAM tag %s is not int32", a.Name)
		}
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], uint32(v))
		b.Write(word[:])
	case 'I':
		v, ok := a.Value.(uint32)
		if !ok {
			return fmt.Errorf("BAM tag %s is not uint32", a.Name)
		}
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], v)
		b.Write(word[:])
	case 'Z':
		v, ok := a.Value.(string)
		if !ok || bytes.IndexByte([]byte(v), 0) >= 0 {
			return fmt.Errorf("BAM tag %s is not a NUL-free string", a.Name)
		}
		b.WriteString(v)
		b.WriteByte(0)
	case 'A':
		v, ok := a.Value.(byte)
		if !ok || v < 33 || v > 126 {
			return fmt.Errorf("BAM tag %s is not a printable character", a.Name)
		}
		b.WriteByte(v)
	case 'B':
		var word [4]byte
		switch v := a.Value.(type) {
		case []byte:
			if len(v) > math.MaxInt32 {
				return fmt.Errorf("BAM tag %s array is too large", a.Name)
			}
			b.WriteByte('c')
			binary.LittleEndian.PutUint32(word[:], uint32(len(v)))
			b.Write(word[:])
			b.Write(v)
		case ArrayInt32:
			if len(v) > math.MaxInt32 {
				return fmt.Errorf("BAM tag %s array is too large", a.Name)
			}
			b.WriteByte('i')
			binary.LittleEndian.PutUint32(word[:], uint32(len(v)))
			b.Write(word[:])
			for _, x := range v {
				binary.LittleEndian.PutUint32(word[:], uint32(x))
				b.Write(word[:])
			}
		case ArrayFloat32:
			if len(v) > math.MaxInt32 {
				return fmt.Errorf("BAM tag %s array is too large", a.Name)
			}
			b.WriteByte('f')
			binary.LittleEndian.PutUint32(word[:], uint32(len(v)))
			b.Write(word[:])
			for _, x := range v {
				binary.LittleEndian.PutUint32(word[:], math.Float32bits(x))
				b.Write(word[:])
			}
		default:
			return fmt.Errorf("BAM tag %s has unsupported array type %T", a.Name, a.Value)
		}
	default:
		return fmt.Errorf("BAM tag %s has unsupported type %q", a.Name, a.Type)
	}
	return nil
}

// Metadata is the typed input to the pinned source tag order. The caller
// computes MeanQ from the accepted quality rule and Duration from the sample
// clock; this package does not reinterpret either.
type Metadata struct {
	MeanQ         float32
	Duration      float32
	NumSamples    int32
	Trimmed       int32
	Mux           int32
	Channel       int32
	StartTime     string
	Strand        byte
	UMI           string
	ReadNumber    int32
	Filename      string
	Shift         float32
	Scale         float32
	ScalingMethod string
	DuplexParent  bool
	ReadGroup     string
	ParentID      string
	SplitPoint    int32
	PoreType      string
	EndReason     string
	MinknowEvents uint32
	// Moves and ModelStride emit the optional mv array. A nil Moves slice means
	// moves were not requested, matching emit_moves=false.
	Moves       []uint8
	ModelStride uint8
	// SampleRate checks the duration invariant when non-zero.
	SampleRate uint64
	// IsDuplex marks a duplex read. The duplex tag set is not ported, so Tags
	// rejects it instead of silently emitting the simplex set.
	IsDuplex bool
}

// Tags returns the pinned simplex tag order from
// read_pipeline/base/messages.cpp:108-212. RNA poly-tail tags (pt/pa) are not
// part of the pinned plain-DNA path and are rejected by omission.
func Tags(m Metadata) ([]Aux, error) {
	if m.IsDuplex {
		return nil, errors.New("duplex BAM tags are not ported")
	}
	if m.Trimmed < 0 || m.NumSamples < 0 || uint64(m.Trimmed) > uint64(m.NumSamples) {
		return nil, fmt.Errorf("trimmed samples %d are outside %d total samples", m.Trimmed, m.NumSamples)
	}
	if m.SampleRate > 0 {
		want := float32(m.NumSamples) / float32(m.SampleRate)
		if m.Duration != want {
			return nil, fmt.Errorf("duration %v differs from %d/%d=%v", m.Duration, m.NumSamples, m.SampleRate, want)
		}
	}
	dx := int32(0)
	if m.DuplexParent {
		dx = -1
	}
	tags := []Aux{
		Float("qs", m.MeanQ),
		Float("du", m.Duration),
		Int32("ns", m.NumSamples),
		Int32("ts", m.Trimmed),
		Int32("mx", m.Mux),
		Int32("ch", m.Channel),
		String("st", m.StartTime),
	}
	if m.Strand != 0 {
		tags = append(tags, Char("TS", m.Strand))
	}
	if m.UMI != "" {
		tags = append(tags, String("RX", m.UMI))
	}
	tags = append(tags,
		Int32("rn", m.ReadNumber),
		String("fn", m.Filename),
		Float("sm", m.Shift),
		Float("sd", m.Scale),
		String("sv", m.ScalingMethod),
		Int32("dx", dx),
	)
	if m.ReadGroup != "" {
		tags = append(tags, String("RG", m.ReadGroup))
	}
	if m.ParentID != "" {
		tags = append(tags, String("pi", m.ParentID), Int32("sp", m.SplitPoint))
	}
	if m.Moves != nil {
		mv, err := MovesTag(m.ModelStride, m.Moves)
		if err != nil {
			return nil, err
		}
		tags = append(tags, mv)
	}
	if m.PoreType != "" {
		tags = append(tags, String("po", m.PoreType))
	}
	if m.EndReason != "" {
		tags = append(tags, String("er", m.EndReason))
	}
	return append(tags, Uint32("me", m.MinknowEvents)), nil
}

// MovesTag encodes the mv array: the model stride followed by binary moves.
func MovesTag(stride uint8, moves []uint8) (Aux, error) {
	if stride == 0 {
		return Aux{}, errors.New("moves tag requires a model stride")
	}
	for _, m := range moves {
		if m > 1 {
			return Aux{}, errors.New("moves tag requires binary moves")
		}
	}
	payload := make([]byte, len(moves)+1)
	payload[0] = stride
	copy(payload[1:], moves)
	return ByteArray("mv", payload), nil
}

// Unclassified is the source barcode sentinel that never joins a read group.
const Unclassified = "unclassified"

// ReadGroupID reproduces ReadCommon::generate_read_group.
func ReadGroupID(runID, model, barcode string) string {
	if runID == "" {
		return ""
	}
	if model == "" {
		model = "unknown"
	}
	id := runID + "_" + model
	if barcode != "" && barcode != Unclassified {
		id += "_" + barcode
	}
	return id
}
