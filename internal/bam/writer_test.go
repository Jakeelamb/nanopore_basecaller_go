package bam

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"reflect"
	"strings"
	"syscall"
	"testing"
)

func testHeader() Header {
	return Header{
		Version: "1.6", SortOrder: "unknown",
		Program: Program{ID: "basecaller", Name: "dorado", Version: "2.1.2+8b8fc5d",
			CommandLine: `dorado "basecaller" "model" "reads.pod5"`, Description: "gpu:NVIDIA GeForce RTX 5070 Laptop GPU"},
		ReadGroup: &ReadGroup{ID: "run_model", FlowCellID: "PAW70337", DeviceID: "PC48LILITH",
			ExperimentStartTime: "2024-06-19T12:36:05.543000+00:00", RunID: "run", BasecallModel: "model",
			ExperimentID: "exp", AcquisitionStartTime: "2024-06-19T12:42:14.753000+00:00", SampleID: "sample",
			ModelStride: 6, TrimFlags: "adapter,primer"},
	}
}

func TestHeaderTextPinnedOrder(t *testing.T) {
	text, err := testHeader().Text()
	if err != nil {
		t.Fatal(err)
	}
	want := "@HD\tVN:1.6\tSO:unknown\n" +
		"@PG\tID:basecaller\tPN:dorado\tVN:2.1.2+8b8fc5d\tCL:dorado \"basecaller\" \"model\" \"reads.pod5\"\tDS:gpu:NVIDIA GeForce RTX 5070 Laptop GPU\n" +
		"@RG\tID:run_model\tPU:PAW70337\tPM:PC48LILITH\tDT:2024-06-19T12:36:05.543000+00:00\tPL:ONT\tDS:runid=run basecall_model=model experiment_id=exp acquisition_start_time=2024-06-19T12:42:14.753000+00:00 model_stride=6\tLB:sample\ttm:adapter,primer\n"
	if text != want {
		t.Fatalf("header text:\n%q\nwant\n%q", text, want)
	}
	h := testHeader()
	h.ReadGroup = nil
	h.Program.CommandLine, h.Program.Description = "", ""
	text, err = h.Text()
	if err != nil || strings.Count(text, "\n") != 2 || strings.Contains(text, "@RG") || strings.Contains(text, "\tCL:") {
		t.Fatalf("minimal header %q %v", text, err)
	}
	h.References = []Reference{{Name: "chr1", Length: 100}}
	text, _ = h.Text()
	if !strings.HasSuffix(text, "@SQ\tSN:chr1\tLN:100\n") {
		t.Fatalf("reference line %q", text)
	}
	for name, mutate := range map[string]func(*Header){
		"version": func(h *Header) { h.Version = "1.5" },
		"sort":    func(h *Header) { h.SortOrder = "coordinate" },
		"program": func(h *Header) { h.Program.Name = "" },
		"newline": func(h *Header) { h.Program.CommandLine = "a\nb" },
		"group":   func(h *Header) { h.ReadGroup.ID = "" },
		"stride":  func(h *Header) { h.ReadGroup.ModelStride = 0 },
		"ref":     func(h *Header) { h.References = []Reference{{Name: "", Length: 1}} },
	} {
		h := testHeader()
		mutate(&h)
		if _, err := h.Text(); err == nil {
			t.Fatalf("%s header accepted", name)
		}
	}
}

func TestReadGroupID(t *testing.T) {
	cases := []struct{ run, model, barcode, want string }{
		{"run", "model", "", "run_model"},
		{"run", "", "", "run_unknown"},
		{"run", "model", Unclassified, "run_model"},
		{"run", "model", "barcode01", "run_model_barcode01"},
		{"", "model", "barcode01", ""},
	}
	for _, c := range cases {
		if got := ReadGroupID(c.run, c.model, c.barcode); got != c.want {
			t.Fatalf("ReadGroupID(%q,%q,%q)=%q want %q", c.run, c.model, c.barcode, got, c.want)
		}
	}
}

func metadata() Metadata {
	return Metadata{MeanQ: 20.4353, Duration: 5.8532, NumSamples: 29266, Trimmed: 682, Mux: 1, Channel: 2372,
		SampleRate: 5000,
		StartTime:  "2024-06-19T21:36:39.995000+00:00", Strand: '+', UMI: "ACGT", ReadNumber: 4484,
		Filename: "PAW.pod5", Shift: 94, Scale: 24, ScalingMethod: "pa", ReadGroup: "run_model",
		PoreType: "not_set", EndReason: "signal_positive", MinknowEvents: 3653}
}

func TestTagsPinnedOrderAndTypes(t *testing.T) {
	tags, err := Tags(metadata())
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	var types []byte
	for _, tag := range tags {
		names = append(names, tag.Name)
		types = append(types, tag.Type)
	}
	wantNames := []string{"qs", "du", "ns", "ts", "mx", "ch", "st", "TS", "RX", "rn", "fn", "sm", "sd", "sv", "dx", "RG", "po", "er", "me"}
	wantTypes := []byte{'f', 'f', 'i', 'i', 'i', 'i', 'Z', 'A', 'Z', 'i', 'Z', 'f', 'f', 'Z', 'i', 'Z', 'Z', 'Z', 'I'}
	if !reflect.DeepEqual(names, wantNames) || !reflect.DeepEqual(types, wantTypes) {
		t.Fatalf("names %v types %v", names, types)
	}
	m := metadata()
	m.Strand, m.UMI, m.ReadGroup, m.PoreType, m.EndReason = 0, "", "", "", ""
	m.ParentID, m.SplitPoint = "parent", 42
	tags, err = Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	names = names[:0]
	for _, tag := range tags {
		names = append(names, tag.Name)
	}
	wantNames = []string{"qs", "du", "ns", "ts", "mx", "ch", "st", "rn", "fn", "sm", "sd", "sv", "dx", "pi", "sp", "me"}
	if !reflect.DeepEqual(names, wantNames) {
		t.Fatalf("split tag order %v", names)
	}
	m.DuplexParent = true
	moves, err := Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	var dx Aux
	for _, tag := range moves {
		if tag.Name == "dx" {
			dx = tag
		}
	}
	if dx.Value != int32(-1) {
		t.Fatalf("duplex parent dx %+v", dx)
	}
}

func TestTagsEmitMovesInPinnedPosition(t *testing.T) {
	m := metadata()
	m.ParentID, m.SplitPoint = "parent", 42
	m.Moves, m.ModelStride = []byte{0, 1, 1}, 6
	tags, err := Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tag := range tags {
		names = append(names, tag.Name)
	}
	want := []string{"qs", "du", "ns", "ts", "mx", "ch", "st", "TS", "RX", "rn", "fn", "sm", "sd", "sv", "dx", "RG", "pi", "sp", "mv", "po", "er", "me"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("moves tag order %v", names)
	}
	m.ModelStride = 0
	if _, err := Tags(m); err == nil {
		t.Fatal("moves without a stride accepted")
	}
}

func TestTagsRejectsUnportedOrInconsistentRequests(t *testing.T) {
	for name, mutate := range map[string]func(*Metadata){
		"trim":      func(m *Metadata) { m.Trimmed = m.NumSamples + 1 },
		"negative":  func(m *Metadata) { m.Trimmed = -1 },
		"duration":  func(m *Metadata) { m.Duration = 1 },
		"bad-moves": func(m *Metadata) { m.Moves, m.ModelStride = []byte{1}, 0 },
	} {
		m := metadata()
		mutate(&m)
		if _, err := Tags(m); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	m := metadata()
	m.Moves, m.ModelStride = []byte{0, 1}, 6
	m.SampleRate = 0
	m.Duration = 1
	if _, err := Tags(m); err != nil {
		t.Fatalf("duration check ran without a sample rate: %v", err)
	}
}

func TestTagsDuplexOrderAndTypes(t *testing.T) {
	m := metadata()
	m.IsDuplex = true
	m.ParentID = "parent"
	tags, err := Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tag := range tags {
		names = append(names, tag.Name)
	}
	want := []string{"qs", "dx", "mx", "ch", "st", "RG", "pi", "po"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("duplex tag order %v", names)
	}
	if tags[1].Type != 'i' || tags[1].Value != int32(1) {
		t.Fatalf("duplex dx %+v", tags[1])
	}
	m.ParentID, m.ReadGroup, m.PoreType = "", "", ""
	tags, err = Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	names = names[:0]
	for _, tag := range tags {
		names = append(names, tag.Name)
	}
	if !reflect.DeepEqual(names, []string{"qs", "dx", "mx", "ch", "st"}) {
		t.Fatalf("minimal duplex order %v", names)
	}
}

func TestTagsBarcodeAndPolyTailOrder(t *testing.T) {
	m := metadata()
	m.Barcode = "barcode01"
	m.BarcodeVariant = "barcode01"
	m.BarcodeScore = &BarcodeScore{Score: 12.5, TopStart: 10, TopEnd: 42, TopScore: 9.25,
		BottomStart: 480, BottomEnd: 512, BottomScore: 8.75}
	m.PolyTail = &PolyTail{Length: 18, Anchor: 300, Boundaries: [2][2]int32{{290, 340}, {-1, -1}}}
	tags, err := Tags(m)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	types := make(map[string]byte, len(tags))
	for _, tag := range tags {
		names = append(names, tag.Name)
		types[tag.Name] = tag.Type
	}
	want := []string{"BC", "bv", "bi", "qs", "du", "ns", "ts", "mx", "ch", "st", "TS", "RX",
		"rn", "fn", "sm", "sd", "sv", "dx", "RG", "po", "er", "pt", "pa", "me"}
	// pt/pa sit after mv and before po; with no moves they follow the read-group
	// and split block directly.
	want = []string{"BC", "bv", "bi", "qs", "du", "ns", "ts", "mx", "ch", "st", "TS", "RX",
		"rn", "fn", "sm", "sd", "sv", "dx", "RG", "pt", "pa", "po", "er", "me"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("barcode/poly-tail order %v", names)
	}
	if types["bi"] != 'B' || types["pa"] != 'B' || types["pt"] != 'i' {
		t.Fatalf("types %v", types)
	}
}

func TestTagsRejectsBarcodeAndPolyTailMisuse(t *testing.T) {
	for name, mutate := range map[string]func(*Metadata){
		"variant-without-barcode": func(m *Metadata) { m.BarcodeVariant = "v" },
		"score-without-barcode":   func(m *Metadata) { m.BarcodeScore = &BarcodeScore{} },
		"variant-without-score":   func(m *Metadata) { m.Barcode, m.BarcodeVariant = "barcode01", "barcode01" },
		"score-unclassified":      func(m *Metadata) { m.Barcode, m.BarcodeScore = Unclassified, &BarcodeScore{} },
		"score-nan":               func(m *Metadata) { m.Barcode, m.BarcodeScore = "barcode01", &BarcodeScore{Score: float32(math.NaN())} },
		"polytail-sentinel":       func(m *Metadata) { m.PolyTail = &PolyTail{Length: -2} },
		"duplex-polytail":         func(m *Metadata) { m.IsDuplex, m.PolyTail = true, &PolyTail{Length: 1} },
	} {
		m := metadata()
		mutate(&m)
		if _, err := Tags(m); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestMovesTag(t *testing.T) {
	tag, err := MovesTag(6, []byte{0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if tag.Name != "mv" || tag.Type != 'B' || !bytes.Equal(tag.Value.([]byte), []byte{6, 0, 1, 1}) {
		t.Fatalf("moves tag %+v", tag)
	}
	if _, err := MovesTag(0, nil); err == nil {
		t.Fatal("zero stride accepted")
	}
	if _, err := MovesTag(6, []byte{2}); err == nil {
		t.Fatal("non-binary move accepted")
	}
}

func TestAuxEncoding(t *testing.T) {
	cases := []struct {
		aux  Aux
		want []byte
	}{
		{Float("qs", 1.5), []byte{'q', 's', 'f', 0, 0, 0xc0, 0x3f}},
		{Int32("ns", -2), []byte{'n', 's', 'i', 0xfe, 0xff, 0xff, 0xff}},
		{Uint32("me", 3653), []byte{'m', 'e', 'I', 0x45, 0x0e, 0, 0}},
		{String("st", "ab"), []byte{'s', 't', 'Z', 'a', 'b', 0}},
		{Char("TS", '+'), []byte{'T', 'S', 'A', '+'}},
		{ByteArray("mv", []byte{6, 1}), []byte{'m', 'v', 'B', 'c', 2, 0, 0, 0, 6, 1}},
		{Int32Array("pa", []int32{1, -2}), []byte{'p', 'a', 'B', 'i', 2, 0, 0, 0, 1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff}},
		{Float32Array("bi", []float32{1.5}), []byte{'b', 'i', 'B', 'f', 1, 0, 0, 0, 0, 0, 0xc0, 0x3f}},
	}
	for _, c := range cases {
		var b bytes.Buffer
		if err := c.aux.encode(&b); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b.Bytes(), c.want) {
			t.Fatalf("%s encoded %x want %x", c.aux.Name, b.Bytes(), c.want)
		}
	}
	bad := []Aux{
		{Name: "x", Type: 'i', Value: int32(1)},
		{Name: "ns", Type: 'i', Value: int64(1)},
		{Name: "qs", Type: 'f', Value: float64(1)},
		{Name: "st", Type: 'Z', Value: "a\x00b"},
		{Name: "TS", Type: 'A', Value: byte(7)},
		{Name: "mv", Type: 'B', Value: "not bytes"},
		{Name: "pa", Type: 'B', Value: []int32{1}},
		{Name: "zz", Type: 'Q', Value: 1},
	}
	for i, aux := range bad {
		var b bytes.Buffer
		if err := aux.encode(&b); err == nil {
			t.Fatalf("bad aux %d accepted", i)
		}
	}
}

func testRecord() Record {
	tags, err := Tags(metadata())
	if err != nil {
		panic(err)
	}
	return Record{Name: "bb76c31a-36f0-48a0-abd2-107a10e1ade0", Flag: unmappedFlag, Sequence: "ACGTN", Quality: []byte{3, 5, 6, 8, 12}, Aux: tags}
}

func TestRecordEncoding(t *testing.T) {
	block, err := testRecord().encode()
	if err != nil {
		t.Fatal(err)
	}
	body := block[4:]
	if got := binary.LittleEndian.Uint32(block[:4]); int(got) != len(body) {
		t.Fatalf("block size %d body %d", got, len(body))
	}
	core := body[:32]
	if refID := int32(binary.LittleEndian.Uint32(core[0:4])); refID != -1 {
		t.Fatalf("refID %d", refID)
	}
	if pos := int32(binary.LittleEndian.Uint32(core[4:8])); pos != -1 {
		t.Fatalf("pos %d", pos)
	}
	if core[8] != 37 || core[9] != 0 {
		t.Fatalf("read name length %d mapq %d", core[8], core[9])
	}
	if bin := binary.LittleEndian.Uint16(core[10:12]); bin != 4680 {
		t.Fatalf("bin %d", bin)
	}
	if flag := binary.LittleEndian.Uint16(core[14:16]); flag != 4 {
		t.Fatalf("flag %d", flag)
	}
	if lseq := int32(binary.LittleEndian.Uint32(core[16:20])); lseq != 5 {
		t.Fatalf("l_seq %d", lseq)
	}
	name := body[32 : 32+37]
	if string(name[:36]) != "bb76c31a-36f0-48a0-abd2-107a10e1ade0" || name[36] != 0 {
		t.Fatalf("name %q", name)
	}
	seq := body[32+37 : 32+37+3]
	if !bytes.Equal(seq, []byte{0x12, 0x48, 0xf0}) {
		t.Fatalf("packed sequence %x", seq)
	}
	if !bytes.Equal(body[32+37+3:32+37+3+5], []byte{3, 5, 6, 8, 12}) {
		t.Fatal("quality bytes differ")
	}
	if !bytes.HasPrefix(body[32+37+3+5:], []byte("qsf")) {
		t.Fatal("aux does not start with qs")
	}
}

func TestRecordRejectsBadInput(t *testing.T) {
	cases := map[string]func(*Record){
		"name":     func(r *Record) { r.Name = "" },
		"quality":  func(r *Record) { r.Quality = r.Quality[:2] },
		"base":     func(r *Record) { r.Sequence = "ACXTN" },
		"flag":     func(r *Record) { r.Flag = 0 },
		"flag-16":  func(r *Record) { r.Flag = 4 | 16 },
		"flag-256": func(r *Record) { r.Flag = 4 | 256 },
		"seq":      func(r *Record) { r.Sequence, r.Quality = "", nil },
		"aux":      func(r *Record) { r.Aux = []Aux{{Name: "bad", Type: 'i', Value: int32(1)}} },
	}
	for name, mutate := range cases {
		r := testRecord()
		mutate(&r)
		if _, err := r.encode(); err == nil {
			t.Fatalf("%s record accepted", name)
		}
	}
}

func TestWriterRoundTrip(t *testing.T) {
	var out bytes.Buffer
	w := NewWriter(&out)
	if err := w.WriteHeader(testHeader()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := w.WriteRecord(testRecord()); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if _, err := w.Write([]byte("x")); err == nil {
		t.Fatal("write after close accepted")
	}
	raw := out.Bytes()
	if !bytes.HasSuffix(raw, bgzfEOF) {
		t.Fatal("BGZF EOF marker missing")
	}
	zr, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(data, []byte("BAM\x01")) {
		t.Fatal("BAM magic missing")
	}
	lText := binary.LittleEndian.Uint32(data[4:8])
	text := string(data[8 : 8+lText])
	if !strings.HasPrefix(text, "@HD\tVN:1.6\tSO:unknown\n") || !strings.Contains(text, "@RG\tID:run_model") {
		t.Fatalf("header text %q", text)
	}
	if nref := binary.LittleEndian.Uint32(data[8+lText : 12+lText]); nref != 0 {
		t.Fatalf("n_ref %d", nref)
	}
	off := 12 + int(lText)
	records := 0
	for off < len(data) {
		size := int(binary.LittleEndian.Uint32(data[off : off+4]))
		off += 4 + size
		records++
	}
	if records != 3 || off != len(data) {
		t.Fatalf("records %d offset %d len %d", records, off, len(data))
	}
}

func TestWriterShortWrite(t *testing.T) {
	w := NewWriter(shortWriter{limit: 10})
	if err := w.WriteHeader(testHeader()); err != nil {
		t.Fatalf("buffered header: %v", err)
	}
	// A full block forces the first flush, which must report the short write.
	if _, err := w.Write(make([]byte, bgzfBlockData)); err == nil {
		t.Fatal("short block write accepted")
	}
	if err := w.Close(); err == nil {
		t.Fatal("close after failure reported success")
	}
}

type shortWriter struct{ limit int }

func (s shortWriter) Write(p []byte) (int, error) {
	if len(p) > s.limit {
		return s.limit, nil
	}
	return len(p), nil
}

func TestWriterCloseRejectsShortEOF(t *testing.T) {
	dst := &shortEOFWriter{}
	w := NewWriter(dst)
	if err := w.WriteHeader(testHeader()); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("close: got %v, want %v", err, io.ErrShortWrite)
	}
	if err := w.Close(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("second close: got %v, want %v", err, io.ErrShortWrite)
	}
	if dst.eofWrites != 1 {
		t.Fatalf("EOF writes: got %d, want 1", dst.eofWrites)
	}
}

type shortEOFWriter struct{ eofWrites int }

func (w *shortEOFWriter) Write(p []byte) (int, error) {
	if bytes.Equal(p, bgzfEOF) {
		w.eofWrites++
		return len(p) - 1, nil
	}
	return len(p), nil
}

func TestStoredBlockFallback(t *testing.T) {
	// Incompressible data forces the stored-block path.
	data := make([]byte, bgzfBlockData)
	for i := range data {
		data[i] = byte((i*2654435761 + i*i) % 251)
	}
	block, err := bgzfBlock(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(block) > bgzfMaxBlock {
		t.Fatalf("block %d exceeds 64 KiB", len(block))
	}
	if int(binary.LittleEndian.Uint16(block[16:18]))+1 != len(block) {
		t.Fatalf("BSIZE %d block %d", binary.LittleEndian.Uint16(block[16:18])+1, len(block))
	}
	zr, err := gzip.NewReader(bytes.NewReader(block))
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(zr)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("stored block round trip %v", err)
	}
}

func TestFloatPrecisionPreserved(t *testing.T) {
	var b bytes.Buffer
	if err := Float("qs", 20.4353).encode(&b); err != nil {
		t.Fatal(err)
	}
	bits := binary.LittleEndian.Uint32(b.Bytes()[3:])
	if math.Float32frombits(bits) != float32(20.4353) {
		t.Fatal("float32 tag not preserved")
	}
}

func TestWriterENOSPC(t *testing.T) {
	f, err := os.OpenFile("/dev/full", os.O_WRONLY, 0)
	if err != nil {
		t.Skip("no /dev/full on this host")
	}
	defer f.Close()
	w := NewWriter(f)
	if err := w.WriteHeader(testHeader()); err != nil {
		t.Fatalf("buffered header: %v", err)
	}
	if err := w.WriteRecord(testRecord()); err != nil {
		t.Fatalf("buffered record: %v", err)
	}
	if err := w.Close(); !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("close error %v, want ENOSPC", err)
	}
}
