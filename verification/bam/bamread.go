package bamverify

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
)

const maxBAMHeader = 16 << 20
const maxBAMRecord = 256 << 20

var bgzfEOFMarker = []byte{
	0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
	0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
	0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00,
}

// BGZFMember is one physical BGZF member. Member boundaries depend on the
// producer's flush schedule, so this table is reported separately from payload
// identity and is never a byte-identity claim.
type BGZFMember struct {
	Offset          int64  `json:"offset"`
	CompressedBytes int    `json:"compressed_bytes"`
	PayloadBytes    int    `json:"payload_bytes"`
	PayloadSHA256   string `json:"payload_sha256"`
}

// ParseOptions binds a parsed manifest to its retained source.
type ParseOptions struct {
	SourceBAM  string
	SourceHash string
}

// ParseResult carries the manifest plus the native facts a reviewer needs to
// reproduce the pin without trusting the verifier.
type ParseResult struct {
	Manifest      Manifest
	HeaderText    string
	HeaderSHA256  string
	Members       []BGZFMember
	PayloadSHA256 string
	PayloadBytes  int
}

// ParseBAM decodes a retained native BGZF BAM, parses its header and records
// with exact auxiliary bits, and returns the writer manifest that must
// reproduce its uncompressed payload.
func ParseBAM(data []byte, opts ParseOptions) (ParseResult, error) {
	if opts.SourceBAM == "" || !isSHA256(opts.SourceHash) {
		return ParseResult{}, errors.New("parse options are not pinned")
	}
	payload, members, err := decodeBGZF(data)
	if err != nil {
		return ParseResult{}, err
	}
	header, text, records, err := parseBAMPayload(payload)
	if err != nil {
		return ParseResult{}, err
	}
	rendered, err := header.Text()
	if err != nil {
		return ParseResult{}, err
	}
	if rendered != text {
		return ParseResult{}, errors.New("parsed header does not round-trip through the writer")
	}
	manifest := Manifest{
		Schema:        ManifestSchema,
		SourceBAM:     opts.SourceBAM,
		SourceBAMHash: opts.SourceHash,
		Header:        header,
		Records:       records,
	}
	if err := manifest.Validate(); err != nil {
		return ParseResult{}, err
	}
	return ParseResult{
		Manifest:      manifest,
		HeaderText:    text,
		HeaderSHA256:  HashBytes([]byte(text)),
		Members:       members,
		PayloadSHA256: HashBytes(payload),
		PayloadBytes:  len(payload),
	}, nil
}

// decodeBGZF inflates every member and records the member table. It rejects
// missing EOF markers, malformed BC fields, CRC mismatches and trailing bytes.
func decodeBGZF(data []byte) ([]byte, []BGZFMember, error) {
	if !bytes.HasSuffix(data, bgzfEOFMarker) {
		return nil, nil, errors.New("BGZF EOF marker is missing")
	}
	var payload bytes.Buffer
	var members []BGZFMember
	for off := 0; off < len(data); {
		if len(data)-off < 18 {
			return nil, nil, fmt.Errorf("BGZF member at %d is truncated", off)
		}
		if data[off] != 0x1f || data[off+1] != 0x8b || data[off+2] != 8 || data[off+3]&4 == 0 {
			return nil, nil, fmt.Errorf("BGZF member at %d has no FEXTRA header", off)
		}
		xlen := int(binary.LittleEndian.Uint16(data[off+10:]))
		if xlen < 6 || len(data)-off < 12+xlen+8 {
			return nil, nil, fmt.Errorf("BGZF member at %d has a bad XLEN", off)
		}
		extra := data[off+12 : off+12+xlen]
		bsize := -1
		for i := 0; i+4 <= len(extra); {
			si1, si2 := extra[i], extra[i+1]
			slen := int(binary.LittleEndian.Uint16(extra[i+2:]))
			if i+4+slen > len(extra) {
				return nil, nil, fmt.Errorf("BGZF member at %d has a bad extra field", off)
			}
			if si1 == 66 && si2 == 67 {
				if slen != 2 {
					return nil, nil, fmt.Errorf("BGZF member at %d has a bad BC length", off)
				}
				bsize = int(binary.LittleEndian.Uint16(extra[i+4:])) + 1
			}
			i += 4 + slen
		}
		if bsize < 26 {
			return nil, nil, fmt.Errorf("BGZF member at %d has no BC size", off)
		}
		if off+bsize > len(data) {
			return nil, nil, fmt.Errorf("BGZF member at %d overruns the file", off)
		}
		member := data[off : off+bsize]
		compressed := member[12+xlen : bsize-8]
		wantCRC := binary.LittleEndian.Uint32(member[bsize-8:])
		wantSize := int(binary.LittleEndian.Uint32(member[bsize-4:]))
		fr := flate.NewReader(bytes.NewReader(compressed))
		body, err := io.ReadAll(fr)
		closeErr := fr.Close()
		if err != nil {
			return nil, nil, fmt.Errorf("BGZF member at %d: %w", off, err)
		}
		if closeErr != nil {
			return nil, nil, fmt.Errorf("BGZF member at %d: %w", off, closeErr)
		}
		if len(body) != wantSize {
			return nil, nil, fmt.Errorf("BGZF member at %d inflates to %d bytes, footer says %d", off, len(body), wantSize)
		}
		if crc32.ChecksumIEEE(body) != wantCRC {
			return nil, nil, fmt.Errorf("BGZF member at %d fails its CRC", off)
		}
		members = append(members, BGZFMember{
			Offset:          int64(off),
			CompressedBytes: bsize,
			PayloadBytes:    len(body),
			PayloadSHA256:   HashBytes(body),
		})
		payload.Write(body)
		off += bsize
	}
	if len(members) == 0 {
		return nil, nil, errors.New("BAM has no BGZF members")
	}
	return payload.Bytes(), members, nil
}

func parseBAMPayload(payload []byte) (bam.Header, string, []Record, error) {
	if len(payload) < 12 || string(payload[:4]) != "BAM\x01" {
		return bam.Header{}, "", nil, errors.New("BAM magic is missing")
	}
	lText := int(int32(binary.LittleEndian.Uint32(payload[4:])))
	if lText < 0 || lText > maxBAMHeader || 8+lText+4 > len(payload) {
		return bam.Header{}, "", nil, fmt.Errorf("BAM l_text %d is invalid", lText)
	}
	text := string(payload[8 : 8+lText])
	off := 8 + lText
	nRef := int(int32(binary.LittleEndian.Uint32(payload[off:])))
	off += 4
	if nRef < 0 || nRef > 1<<20 {
		return bam.Header{}, "", nil, fmt.Errorf("BAM n_ref %d is invalid", nRef)
	}
	var refs []bam.Reference
	for i := 0; i < nRef; i++ {
		if off+4 > len(payload) {
			return bam.Header{}, "", nil, errors.New("BAM reference table is truncated")
		}
		lName := int(int32(binary.LittleEndian.Uint32(payload[off:])))
		off += 4
		if lName <= 1 || off+lName+4 > len(payload) {
			return bam.Header{}, "", nil, fmt.Errorf("BAM reference %d has a bad name length", i)
		}
		name := string(payload[off : off+lName-1])
		if payload[off+lName-1] != 0 {
			return bam.Header{}, "", nil, fmt.Errorf("BAM reference %d name is not NUL terminated", i)
		}
		off += lName
		length := int64(int32(binary.LittleEndian.Uint32(payload[off:])))
		off += 4
		refs = append(refs, bam.Reference{Name: name, Length: length})
	}
	header, err := parseHeaderText(text)
	if err != nil {
		return bam.Header{}, "", nil, err
	}
	header.References = refs
	records := []Record{}
	for off < len(payload) {
		record, next, err := parseBAMRecord(payload, off)
		if err != nil {
			return bam.Header{}, "", nil, err
		}
		records = append(records, record)
		off = next
	}
	return header, text, records, nil
}

func parseHeaderText(text string) (bam.Header, error) {
	var hd map[string]string
	var program map[string]string
	var readGroup map[string]string
	lineNo := 0
	for _, line := range splitLines(text) {
		lineNo++
		fields := strings.Split(line, "\t")
		if len(fields) == 0 || !strings.HasPrefix(fields[0], "@") {
			return bam.Header{}, fmt.Errorf("header line %d is not a SAM header record", lineNo)
		}
		switch fields[0] {
		case "@HD":
			if hd != nil {
				return bam.Header{}, fmt.Errorf("header line %d: duplicate @HD", lineNo)
			}
			m, err := headerFields(fields, []string{"VN", "SO"})
			if err != nil {
				return bam.Header{}, fmt.Errorf("header line %d: %w", lineNo, err)
			}
			hd = m
		case "@PG":
			if program != nil {
				return bam.Header{}, fmt.Errorf("header line %d: multiple @PG lines are not pinned", lineNo)
			}
			m, err := headerFields(fields, []string{"ID", "PN", "VN", "CL", "DS"})
			if err != nil {
				return bam.Header{}, fmt.Errorf("header line %d: %w", lineNo, err)
			}
			if m["ID"] == "" || m["PN"] == "" || m["VN"] == "" {
				return bam.Header{}, fmt.Errorf("header line %d: @PG lacks ID, PN or VN", lineNo)
			}
			program = m
		case "@RG":
			if readGroup != nil {
				return bam.Header{}, fmt.Errorf("header line %d: duplicate @RG", lineNo)
			}
			m, err := headerFields(fields, []string{"ID", "PU", "PM", "DT", "PL", "DS", "LB", "tm"})
			if err != nil {
				return bam.Header{}, fmt.Errorf("header line %d: %w", lineNo, err)
			}
			readGroup = m
		default:
			return bam.Header{}, fmt.Errorf("header line %d: unsupported record %q", lineNo, fields[0])
		}
	}
	if hd == nil || hd["VN"] != "1.6" || hd["SO"] != "unknown" {
		return bam.Header{}, errors.New("BAM header is not @HD VN:1.6 SO:unknown")
	}
	if program == nil {
		return bam.Header{}, errors.New("BAM header has no caller @PG")
	}
	header := bam.Header{
		Version: hd["VN"], SortOrder: hd["SO"],
		Program: bam.Program{ID: program["ID"], Name: program["PN"], Version: program["VN"],
			CommandLine: program["CL"], Description: program["DS"]},
	}
	if readGroup != nil {
		rg, err := parseReadGroup(readGroup)
		if err != nil {
			return bam.Header{}, err
		}
		header.ReadGroup = rg
	}
	return header, nil
}

func headerFields(fields []string, allowed []string) (map[string]string, error) {
	ok := make(map[string]bool, len(allowed))
	for _, a := range allowed {
		ok[a] = true
	}
	out := make(map[string]string, len(fields)-1)
	for _, field := range fields[1:] {
		if len(field) < 3 || field[2] != ':' {
			return nil, fmt.Errorf("malformed header field %q", field)
		}
		key, value := field[:2], field[3:]
		if !ok[key] {
			return nil, fmt.Errorf("unsupported header field %q", key)
		}
		if _, dup := out[key]; dup {
			return nil, fmt.Errorf("duplicate header field %q", key)
		}
		out[key] = value
	}
	return out, nil
}

func parseReadGroup(m map[string]string) (*bam.ReadGroup, error) {
	if m["ID"] == "" {
		return nil, errors.New("@RG lacks ID")
	}
	if m["PL"] != "ONT" {
		return nil, fmt.Errorf("@RG platform %q is not ONT", m["PL"])
	}
	ds := make(map[string]string)
	for _, field := range strings.Fields(m["DS"]) {
		k, v, found := strings.Cut(field, "=")
		if !found {
			return nil, fmt.Errorf("@RG DS field %q is not key=value", field)
		}
		switch k {
		case "runid", "basecall_model", "experiment_id", "acquisition_start_time", "model_stride":
		default:
			return nil, fmt.Errorf("@RG DS key %q is unsupported", k)
		}
		if _, dup := ds[k]; dup {
			return nil, fmt.Errorf("@RG DS key %q is duplicated", k)
		}
		ds[k] = v
	}
	for _, required := range []string{"runid", "basecall_model", "acquisition_start_time", "model_stride"} {
		if ds[required] == "" {
			return nil, fmt.Errorf("@RG DS lacks %s", required)
		}
	}
	stride, err := strconv.Atoi(ds["model_stride"])
	if err != nil || stride <= 0 {
		return nil, errors.New("@RG model_stride is not a positive integer")
	}
	un := func(s string) string {
		if s == "unknown" {
			return ""
		}
		return s
	}
	return &bam.ReadGroup{
		ID: m["ID"], FlowCellID: un(m["PU"]), DeviceID: un(m["PM"]), ExperimentStartTime: un(m["DT"]),
		RunID: un(ds["runid"]), BasecallModel: un(ds["basecall_model"]), ExperimentID: un(ds["experiment_id"]),
		AcquisitionStartTime: un(ds["acquisition_start_time"]), SampleID: un(m["LB"]),
		ModelStride: stride, TrimFlags: m["tm"],
	}, nil
}

func parseBAMRecord(payload []byte, off int) (Record, int, error) {
	if off+4 > len(payload) {
		return Record{}, 0, errors.New("BAM record size is truncated")
	}
	blockSize := int(int32(binary.LittleEndian.Uint32(payload[off:])))
	if blockSize < 32 || blockSize > maxBAMRecord || off+4+blockSize > len(payload) {
		return Record{}, 0, fmt.Errorf("BAM record at %d has block size %d", off, blockSize)
	}
	rec := payload[off+4 : off+4+blockSize]
	next := off + 4 + blockSize
	refID := int32(binary.LittleEndian.Uint32(rec[0:]))
	pos := int32(binary.LittleEndian.Uint32(rec[4:]))
	lReadName := int(rec[8])
	mapq := rec[9]
	bin := binary.LittleEndian.Uint16(rec[10:])
	nCigar := binary.LittleEndian.Uint16(rec[12:])
	flag := binary.LittleEndian.Uint16(rec[14:])
	lSeq := int(int32(binary.LittleEndian.Uint32(rec[16:])))
	nextRef := int32(binary.LittleEndian.Uint32(rec[20:]))
	nextPos := int32(binary.LittleEndian.Uint32(rec[24:]))
	tlen := int32(binary.LittleEndian.Uint32(rec[28:]))
	if refID != -1 || pos != -1 || mapq != 0 || bin != 4680 || nCigar != 0 || flag != 4 || nextRef != -1 || nextPos != -1 || tlen != 0 {
		return Record{}, 0, fmt.Errorf("BAM record at %d is not the pinned unmapped shape", off)
	}
	if lSeq <= 0 || lReadName < 2 {
		return Record{}, 0, fmt.Errorf("BAM record at %d has invalid lengths", off)
	}
	nameEnd := 32 + lReadName
	cigarEnd := nameEnd + int(nCigar)*4
	seqEnd := cigarEnd + (lSeq+1)/2
	qualEnd := seqEnd + lSeq
	if qualEnd > len(rec) {
		return Record{}, 0, fmt.Errorf("BAM record at %d is truncated", off)
	}
	if rec[nameEnd-1] != 0 {
		return Record{}, 0, fmt.Errorf("BAM record at %d name is not NUL terminated", off)
	}
	name := string(rec[32 : nameEnd-1])
	if name == "" || len(name) > 254 || strings.ContainsRune(name, 0) || strings.ContainsRune(name, '@') || !utf8.ValidString(name) {
		return Record{}, 0, fmt.Errorf("BAM record at %d has invalid QNAME", off)
	}
	seqBytes := rec[cigarEnd:seqEnd]
	var seq strings.Builder
	seq.Grow(lSeq)
	for i := 0; i < lSeq; i++ {
		var code byte
		if i%2 == 0 {
			code = seqBytes[i/2] >> 4
		} else {
			code = seqBytes[i/2] & 0x0f
		}
		if int(code) >= len(baseAlphabet) {
			return Record{}, 0, fmt.Errorf("BAM record at %d has invalid base code %d", off, code)
		}
		seq.WriteByte(baseAlphabet[code])
	}
	quality := make(Phred, lSeq)
	copy(quality, rec[seqEnd:qualEnd])
	for i, q := range quality {
		if q > 93 {
			return Record{}, 0, fmt.Errorf("BAM record at %d quality %d at %d is invalid", off, q, i)
		}
	}
	aux, err := parseBAMAux(rec[qualEnd:])
	if err != nil {
		return Record{}, 0, fmt.Errorf("BAM record at %d: %w", off, err)
	}
	return Record{Name: name, Flag: flag, Sequence: seq.String(), Quality: quality, Aux: aux}, next, nil
}

var baseAlphabet = [...]byte{'=', 'A', 'C', 'M', 'G', 'R', 'S', 'V', 'T', 'W', 'Y', 'H', 'K', 'D', 'B', 'N'}

func parseBAMAux(data []byte) ([]Aux, error) {
	var out []Aux
	seen := make(map[string]bool)
	for len(data) > 0 {
		if len(data) < 3 {
			return nil, errors.New("aux field is truncated")
		}
		name := string(data[:2])
		typ := data[2]
		data = data[3:]
		if !validTagName(name) {
			return nil, fmt.Errorf("aux tag %q is malformed", name)
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate aux tag %s", name)
		}
		seen[name] = true
		a := Aux{Name: name, Type: typ}
		switch typ {
		case 'A':
			if len(data) < 1 {
				return nil, fmt.Errorf("aux tag %s is truncated", name)
			}
			a.Value = data[0]
			data = data[1:]
		case 'i':
			if len(data) < 4 {
				return nil, fmt.Errorf("aux tag %s is truncated", name)
			}
			a.Value = int32(binary.LittleEndian.Uint32(data))
			data = data[4:]
		case 'I':
			if len(data) < 4 {
				return nil, fmt.Errorf("aux tag %s is truncated", name)
			}
			a.Value = binary.LittleEndian.Uint32(data)
			data = data[4:]
		case 'f':
			if len(data) < 4 {
				return nil, fmt.Errorf("aux tag %s is truncated", name)
			}
			v := math.Float32frombits(binary.LittleEndian.Uint32(data))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("aux tag %s is not finite", name)
			}
			a.Value = v
			data = data[4:]
		case 'Z':
			end := bytes.IndexByte(data, 0)
			if end < 0 {
				return nil, fmt.Errorf("aux tag %s is not NUL terminated", name)
			}
			v := string(data[:end])
			if !utf8.ValidString(v) {
				return nil, fmt.Errorf("aux tag %s is not valid UTF-8", name)
			}
			a.Value = v
			data = data[end+1:]
		case 'B':
			if len(data) < 5 {
				return nil, fmt.Errorf("aux tag %s is truncated", name)
			}
			sub := data[0]
			count := int(int32(binary.LittleEndian.Uint32(data[1:])))
			data = data[5:]
			if count < 0 || count > maxBAMRecord {
				return nil, fmt.Errorf("aux tag %s has array count %d", name, count)
			}
			switch sub {
			case 'c':
				if len(data) < count {
					return nil, fmt.Errorf("aux tag %s is truncated", name)
				}
				out := make(Array, count)
				copy(out, data[:count])
				for _, x := range out {
					if x > 127 {
						return nil, fmt.Errorf("aux tag %s array value %d is outside a signed byte", name, x)
					}
				}
				a.Value = out
				data = data[count:]
			case 'i':
				if len(data) < count*4 {
					return nil, fmt.Errorf("aux tag %s is truncated", name)
				}
				out := make(IntArray, count)
				for i := range out {
					out[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
				}
				a.Value = out
				data = data[count*4:]
			case 'f':
				if len(data) < count*4 {
					return nil, fmt.Errorf("aux tag %s is truncated", name)
				}
				out := make(FloatArray, count)
				for i := range out {
					v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						return nil, fmt.Errorf("aux tag %s array value is not finite", name)
					}
					out[i] = v
				}
				a.Value = out
				data = data[count*4:]
			default:
				return nil, fmt.Errorf("aux tag %s array subtype %q is not pinned", name, sub)
			}
		default:
			return nil, fmt.Errorf("aux tag %s type %q is not representable by the pinned writer", name, typ)
		}
		out = append(out, a)
	}
	return out, nil
}
