// Package bamverify derives and checks the unaligned BAM writer contract. It
// never runs a native producer: inputs are retained SAM text and a declared
// attempt manifest, and every comparison goes through htslib via samtools.
package bamverify

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
)

const ManifestSchema = "bam-manifest/1"
const maxSAMLine = 64 << 20

// Manifest is the declared writer input: the caller-authored header plus the
// final records in emission order. It is derived from a retained SAM and bound
// to that SAM by hash, so a reviewer can rebuild it.
type Manifest struct {
	Schema        string     `json:"schema"`
	SourceSAM     string     `json:"source_sam"`
	SourceSAMHash string     `json:"source_sam_sha256"`
	Header        bam.Header `json:"header"`
	Records       []Record   `json:"records"`
}

// Phred is a phred-score byte slice rendered as a JSON number array, not
// base64. Quality values are 0..93.
type Phred []byte

func (p Phred) MarshalJSON() ([]byte, error) {
	out := make([]int, len(p))
	for i, v := range p {
		out[i] = int(v)
	}
	return json.Marshal(out)
}

func (p *Phred) UnmarshalJSON(data []byte) error {
	var v []int
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	out := make(Phred, len(v))
	for i, n := range v {
		if n < 0 || n > 255 {
			return fmt.Errorf("phred value %d is outside a byte", n)
		}
		out[i] = byte(n)
	}
	*p = out
	return nil
}

// Array is a BAM B-array payload rendered as a JSON number array.
type Array []byte

func (a Array) MarshalJSON() ([]byte, error) {
	out := make([]int, len(a))
	for i, v := range a {
		out[i] = int(v)
	}
	return json.Marshal(out)
}

func (a *Array) UnmarshalJSON(data []byte) error {
	var v []int
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	out := make(Array, len(v))
	for i, n := range v {
		if n < 0 || n > 127 {
			return fmt.Errorf("array value %d is outside a signed byte", n)
		}
		out[i] = byte(n)
	}
	*a = out
	return nil
}

// Record mirrors bam.Record with JSON-friendly typed tags. Quality holds phred
// scores, not FASTQ bytes.
type Record struct {
	Name     string `json:"name"`
	Flag     uint16 `json:"flag"`
	Sequence string `json:"sequence"`
	Quality  Phred  `json:"quality"`
	Aux      []Aux  `json:"aux"`
}

// Aux keeps the BAM type byte explicit. Value is float32, int32, uint32, string,
// byte or []byte depending on Type.
type Aux struct {
	Name  string
	Type  byte
	Value any
}

func (a Aux) MarshalJSON() ([]byte, error) {
	type wire struct {
		Name  string `json:"name"`
		Type  string `json:"type"`
		Value any    `json:"value"`
	}
	w := wire{Name: a.Name, Type: string(a.Type), Value: a.Value}
	if a.Type == 'A' {
		v, ok := a.Value.(byte)
		if !ok {
			return nil, errors.New("BAM A tag is not a byte")
		}
		w.Value = string([]byte{v})
	}
	return json.Marshal(w)
}

func (a *Aux) UnmarshalJSON(data []byte) error {
	var raw struct {
		Name  string          `json:"name"`
		Type  string          `json:"type"`
		Value json.RawMessage `json:"value"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	if len(raw.Name) != 2 || len(raw.Type) != 1 {
		return fmt.Errorf("BAM tag %q type %q is not pinned", raw.Name, raw.Type)
	}
	a.Name, a.Type = raw.Name, raw.Type[0]
	var err error
	switch a.Type {
	case 'f':
		var v float32
		err = json.Unmarshal(raw.Value, &v)
		a.Value = v
	case 'i':
		var v int32
		err = json.Unmarshal(raw.Value, &v)
		a.Value = v
	case 'I':
		var v uint32
		err = json.Unmarshal(raw.Value, &v)
		a.Value = v
	case 'Z':
		var v string
		err = json.Unmarshal(raw.Value, &v)
		a.Value = v
	case 'A':
		var v string
		err = json.Unmarshal(raw.Value, &v)
		if err == nil && len(v) != 1 {
			err = errors.New("BAM A tag is not one character")
		}
		if err == nil {
			a.Value = v[0]
		}
	case 'B':
		var v Array
		err = json.Unmarshal(raw.Value, &v)
		a.Value = v
	default:
		err = fmt.Errorf("BAM tag %s has unsupported type %q", a.Name, a.Type)
	}
	return err
}

// Validate checks the declared manifest before any file is written.
func (m Manifest) Validate() error {
	if m.Schema != ManifestSchema {
		return fmt.Errorf("manifest schema %q is not %q", m.Schema, ManifestSchema)
	}
	if m.SourceSAM == "" || !isSHA256(m.SourceSAMHash) {
		return errors.New("manifest source SAM pin is incomplete")
	}
	for i, r := range m.Records {
		if r.Flag&4 == 0 {
			return fmt.Errorf("record %d is not unmapped", i)
		}
		if len(r.Quality) != len(r.Sequence) {
			return fmt.Errorf("record %d quality length differs", i)
		}
		seen := make(map[string]bool, len(r.Aux))
		for _, a := range r.Aux {
			if len(a.Name) != 2 {
				return fmt.Errorf("record %d has tag name %q", i, a.Name)
			}
			if seen[a.Name] {
				return fmt.Errorf("record %d has duplicate tag %s", i, a.Name)
			}
			seen[a.Name] = true
		}
	}
	return nil
}

func isSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// Encode renders the canonical JSON form used on disk.
func (m Manifest) Encode() ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// DecodeManifest rejects unknown fields, trailing JSON and invalid content.
func DecodeManifest(data []byte) (Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Manifest{}, err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return Manifest{}, errors.New("trailing manifest JSON")
	}
	if err := m.Validate(); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// WriteBAM serializes the manifest as an unaligned BAM.
func (m Manifest) WriteBAM(w io.Writer) error {
	if err := m.Validate(); err != nil {
		return err
	}
	bw := bam.NewWriter(w)
	if err := bw.WriteHeader(m.Header); err != nil {
		return err
	}
	for i, r := range m.Records {
		if err := bw.WriteRecord(r.bamRecord()); err != nil {
			return fmt.Errorf("record %d: %w", i, err)
		}
	}
	return bw.Close()
}

func (r Record) bamRecord() bam.Record {
	out := bam.Record{Name: r.Name, Flag: r.Flag, Sequence: r.Sequence,
		Quality: append([]byte(nil), r.Quality...), Aux: make([]bam.Aux, len(r.Aux))}
	for i, a := range r.Aux {
		value := a.Value
		if b, ok := value.(Array); ok {
			value = []byte(b)
		}
		out.Aux[i] = bam.Aux{Name: a.Name, Type: a.Type, Value: value}
	}
	return out
}

// AuxNames returns the tag order of one record.
func (r Record) AuxNames() []string {
	out := make([]string, len(r.Aux))
	for i, a := range r.Aux {
		out[i] = a.Name
	}
	return out
}

// ParseOptions binds a parsed manifest to its retained source.
type ParseOptions struct {
	SourceSAM           string
	SourceHash          string
	DropProgramPrefixes []string
}

// ParseResult carries the manifest plus the exact header lines that were
// excluded, so the receipt can show what normalization happened.
type ParseResult struct {
	Manifest        Manifest
	DroppedPrograms []string
}

// ParseSAM accepts the pinned simplex unaligned SAM shape. Unknown header
// fields, unknown tag types and non-unmapped records are rejected rather than
// silently dropped.
func ParseSAM(r io.Reader, opts ParseOptions) (ParseResult, error) {
	if opts.SourceSAM == "" || !isSHA256(opts.SourceHash) {
		return ParseResult{}, errors.New("parse options are not pinned")
	}
	var hd map[string]string
	var programs []map[string]string
	var readGroup map[string]string
	var refs []bam.Reference
	var dropped []string
	records := []Record{}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64<<10), maxSAMLine)
	lineNo := 0
	seenRecord := false
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if strings.HasPrefix(line, "@") {
			if seenRecord {
				return ParseResult{}, fmt.Errorf("line %d: header after records", lineNo)
			}
			fields := strings.Split(line, "\t")
			switch fields[0] {
			case "@HD":
				if hd != nil {
					return ParseResult{}, fmt.Errorf("line %d: duplicate @HD", lineNo)
				}
				m, err := headerFields(fields, []string{"VN", "SO"})
				if err != nil {
					return ParseResult{}, fmt.Errorf("line %d: %w", lineNo, err)
				}
				hd = m
			case "@PG":
				m, err := headerFields(fields, []string{"ID", "PN", "VN", "CL", "DS", "PP"})
				if err != nil {
					return ParseResult{}, fmt.Errorf("line %d: %w", lineNo, err)
				}
				if dropProgram(m["ID"], opts.DropProgramPrefixes) {
					dropped = append(dropped, line)
					continue
				}
				programs = append(programs, m)
			case "@RG":
				if readGroup != nil {
					return ParseResult{}, fmt.Errorf("line %d: duplicate @RG", lineNo)
				}
				m, err := headerFields(fields, []string{"ID", "PU", "PM", "DT", "PL", "DS", "LB", "tm"})
				if err != nil {
					return ParseResult{}, fmt.Errorf("line %d: %w", lineNo, err)
				}
				readGroup = m
			case "@SQ":
				m, err := headerFields(fields, []string{"SN", "LN"})
				if err != nil {
					return ParseResult{}, fmt.Errorf("line %d: %w", lineNo, err)
				}
				length, err := strconv.ParseInt(m["LN"], 10, 64)
				if err != nil || length <= 0 {
					return ParseResult{}, fmt.Errorf("line %d: invalid @SQ length", lineNo)
				}
				refs = append(refs, bam.Reference{Name: m["SN"], Length: length})
			default:
				return ParseResult{}, fmt.Errorf("line %d: unsupported header line %q", lineNo, fields[0])
			}
			continue
		}
		seenRecord = true
		record, err := parseRecord(line)
		if err != nil {
			return ParseResult{}, fmt.Errorf("line %d: %w", lineNo, err)
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		return ParseResult{}, err
	}
	if hd == nil || hd["VN"] == "" || hd["SO"] == "" {
		return ParseResult{}, errors.New("SAM header lacks @HD VN or SO")
	}
	if len(programs) != 1 {
		return ParseResult{}, fmt.Errorf("SAM has %d caller @PG lines, need exactly one", len(programs))
	}
	program := programs[0]
	if program["ID"] == "" || program["PN"] == "" || program["VN"] == "" {
		return ParseResult{}, errors.New("SAM @PG lacks ID, PN or VN")
	}
	if program["PP"] != "" {
		return ParseResult{}, errors.New("SAM @PG has an unsupported PP link")
	}
	header := bam.Header{
		Version: hd["VN"], SortOrder: hd["SO"], References: refs,
		Program: bam.Program{ID: program["ID"], Name: program["PN"], Version: program["VN"],
			CommandLine: program["CL"], Description: program["DS"]},
	}
	if readGroup != nil {
		rg, err := parseReadGroup(readGroup)
		if err != nil {
			return ParseResult{}, err
		}
		header.ReadGroup = rg
	}
	if len(records) == 0 {
		return ParseResult{}, errors.New("SAM has no records")
	}
	manifest := Manifest{Schema: ManifestSchema, SourceSAM: opts.SourceSAM, SourceSAMHash: opts.SourceHash,
		Header: header, Records: records}
	if err := manifest.Validate(); err != nil {
		return ParseResult{}, err
	}
	return ParseResult{Manifest: manifest, DroppedPrograms: dropped}, nil
}

func dropProgram(id string, prefixes []string) bool {
	for _, p := range prefixes {
		if p != "" && strings.HasPrefix(id, p) {
			return true
		}
	}
	return false
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

func parseRecord(line string) (Record, error) {
	fields := strings.Split(line, "\t")
	if len(fields) < 11 {
		return Record{}, fmt.Errorf("record has %d fields", len(fields))
	}
	flag, err := strconv.ParseUint(fields[1], 10, 16)
	if err != nil || flag&4 == 0 {
		return Record{}, fmt.Errorf("record %q is not unmapped", fields[0])
	}
	if fields[2] != "*" || fields[3] != "0" || fields[4] != "0" || fields[5] != "*" || fields[6] != "*" || fields[7] != "0" || fields[8] != "0" {
		return Record{}, errors.New("record has mapped or non-canonical mandatory fields")
	}
	if fields[0] == "" || len(fields[0]) > 254 || strings.Contains(fields[0], "@") {
		return Record{}, fmt.Errorf("invalid QNAME %q", fields[0])
	}
	seq := fields[9]
	if seq == "" || seq == "*" {
		return Record{}, errors.New("record has no sequence")
	}
	for i := 0; i < len(seq); i++ {
		if !strings.ContainsRune("=ACMGRSVTWYHKDBN", rune(seq[i])) {
			return Record{}, fmt.Errorf("unsupported base %q at %d", seq[i], i)
		}
	}
	qual := fields[10]
	if len(qual) != len(seq) {
		return Record{}, errors.New("record quality length differs from sequence")
	}
	quality := make(Phred, len(qual))
	for i := 0; i < len(qual); i++ {
		if qual[i] < 33 || qual[i] > 126 {
			return Record{}, fmt.Errorf("invalid quality byte at %d", i)
		}
		quality[i] = qual[i] - 33
	}
	aux := make([]Aux, 0, len(fields)-11)
	seen := make(map[string]bool, len(fields)-11)
	for _, field := range fields[11:] {
		a, err := parseAux(field)
		if err != nil {
			return Record{}, err
		}
		if seen[a.Name] {
			return Record{}, fmt.Errorf("duplicate tag %s", a.Name)
		}
		seen[a.Name] = true
		aux = append(aux, a)
	}
	return Record{Name: fields[0], Flag: uint16(flag), Sequence: seq, Quality: quality, Aux: aux}, nil
}

func parseAux(field string) (Aux, error) {
	parts := strings.SplitN(field, ":", 3)
	if len(parts) != 3 || len(parts[0]) != 2 || len(parts[1]) != 1 {
		return Aux{}, fmt.Errorf("malformed tag %q", field)
	}
	a := Aux{Name: parts[0], Type: parts[1][0]}
	value := parts[2]
	switch a.Type {
	case 'f':
		v, err := strconv.ParseFloat(value, 32)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return Aux{}, fmt.Errorf("tag %s is not a finite float", a.Name)
		}
		a.Value = float32(v)
	case 'i':
		v, err := strconv.ParseInt(value, 10, 32)
		if err != nil {
			return Aux{}, fmt.Errorf("tag %s is not int32", a.Name)
		}
		a.Value = int32(v)
	case 'I':
		v, err := strconv.ParseUint(value, 10, 32)
		if err != nil {
			return Aux{}, fmt.Errorf("tag %s is not uint32", a.Name)
		}
		a.Value = uint32(v)
	case 'Z':
		if value == "" || strings.ContainsAny(value, "\x00\n\r") {
			return Aux{}, fmt.Errorf("tag %s is not a non-empty printable string", a.Name)
		}
		a.Value = value
	case 'A':
		if len(value) != 1 || value[0] < 33 || value[0] > 126 {
			return Aux{}, fmt.Errorf("tag %s is not one printable character", a.Name)
		}
		a.Value = value[0]
	case 'B':
		sub, rest, found := strings.Cut(value, ",")
		if !found || sub != "c" {
			return Aux{}, fmt.Errorf("tag %s array subtype %q is unsupported", a.Name, sub)
		}
		var out []uint8
		for _, item := range strings.Split(rest, ",") {
			v, err := strconv.ParseInt(item, 10, 8)
			if err != nil || v < 0 {
				return Aux{}, fmt.Errorf("tag %s array value %q is not a non-negative int8", a.Name, item)
			}
			out = append(out, uint8(v))
		}
		a.Value = Array(out)
	default:
		return Aux{}, fmt.Errorf("tag %s has unsupported type %q", a.Name, a.Type)
	}
	return a, nil
}

// Normalize removes exactly the declared program lines from retained SAM bytes.
func Normalize(data []byte, dropped []string) ([]byte, error) {
	remaining := make(map[string]int, len(dropped))
	for _, line := range dropped {
		remaining[line]++
	}
	lines := strings.SplitAfter(string(data), "\n")
	var out bytes.Buffer
	for _, line := range lines {
		trimmed := strings.TrimSuffix(line, "\n")
		if remaining[trimmed] > 0 {
			remaining[trimmed]--
			continue
		}
		out.WriteString(line)
	}
	for line, count := range remaining {
		if count != 0 {
			return nil, fmt.Errorf("declared program line was not found: %q", line)
		}
	}
	return out.Bytes(), nil
}

// HashBytes is the receipt hash helper.
func HashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
