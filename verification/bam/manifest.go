// Package bamverify derives and checks the unaligned BAM writer contract. It
// never runs a native producer: the oracle is a retained native BAM and every
// comparison is made against its exact bytes or its htslib decode.
package bamverify

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
)

const ManifestSchema = "bam-manifest/1"

// Manifest is the declared writer input: the caller-authored header plus the
// final records in emission order. It is derived from a retained native BAM and
// bound to that file by hash, so a reviewer can rebuild it.
type Manifest struct {
	Schema        string     `json:"schema"`
	SourceBAM     string     `json:"source_bam"`
	SourceBAMHash string     `json:"source_bam_sha256"`
	Header        bam.Header `json:"header"`
	Records       []Record   `json:"records"`
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

// Array is a B:c payload rendered as a JSON number array.
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

// IntArray and FloatArray carry B:i and B:f payloads. float32 JSON encoding is
// round-trip exact, so float bits survive the manifest.
type IntArray []int32
type FloatArray []float32

// Aux keeps the BAM type byte explicit. Value is float32, int32, uint32, string,
// byte, Array, IntArray or FloatArray depending on Type.
type Aux struct {
	Name  string
	Type  byte
	Value any
}

func (a Aux) MarshalJSON() ([]byte, error) {
	type wire struct {
		Name    string `json:"name"`
		Type    string `json:"type"`
		Subtype string `json:"subtype,omitempty"`
		Value   any    `json:"value"`
	}
	w := wire{Name: a.Name, Type: string(a.Type), Value: a.Value}
	switch a.Type {
	case 'A':
		v, ok := a.Value.(byte)
		if !ok {
			return nil, errors.New("BAM A tag is not a byte")
		}
		w.Value = string([]byte{v})
	case 'B':
		switch a.Value.(type) {
		case Array:
			w.Subtype = "c"
		case IntArray:
			w.Subtype = "i"
		case FloatArray:
			w.Subtype = "f"
		default:
			return nil, fmt.Errorf("BAM tag %s has unsupported array type %T", a.Name, a.Value)
		}
	}
	return json.Marshal(w)
}

func (a *Aux) UnmarshalJSON(data []byte) error {
	var raw struct {
		Name    string          `json:"name"`
		Type    string          `json:"type"`
		Subtype string          `json:"subtype"`
		Value   json.RawMessage `json:"value"`
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
		switch raw.Subtype {
		case "c":
			var v []int
			if err = json.Unmarshal(raw.Value, &v); err == nil {
				out := make(Array, len(v))
				for i, n := range v {
					if n < 0 || n > 127 {
						return fmt.Errorf("BAM tag %s array value %d is outside a signed byte", a.Name, n)
					}
					out[i] = byte(n)
				}
				a.Value = out
			}
		case "i":
			var v []int32
			err = json.Unmarshal(raw.Value, &v)
			a.Value = IntArray(v)
		case "f":
			var v []float32
			err = json.Unmarshal(raw.Value, &v)
			a.Value = FloatArray(v)
		default:
			err = fmt.Errorf("BAM tag %s array subtype %q is not pinned", a.Name, raw.Subtype)
		}
	default:
		err = fmt.Errorf("BAM tag %s has unsupported type %q", a.Name, a.Type)
	}
	return err
}

func (a Aux) validate() error {
	if len(a.Name) != 2 || !validTagName(a.Name) {
		return fmt.Errorf("BAM tag name %q is not two letters/digits", a.Name)
	}
	switch a.Type {
	case 'f':
		v, ok := a.Value.(float32)
		if !ok {
			return fmt.Errorf("BAM tag %s is not float32", a.Name)
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("BAM tag %s is not finite", a.Name)
		}
	case 'i':
		if _, ok := a.Value.(int32); !ok {
			return fmt.Errorf("BAM tag %s is not int32", a.Name)
		}
	case 'I':
		if _, ok := a.Value.(uint32); !ok {
			return fmt.Errorf("BAM tag %s is not uint32", a.Name)
		}
	case 'Z':
		v, ok := a.Value.(string)
		if !ok || strings.ContainsRune(v, 0) {
			return fmt.Errorf("BAM tag %s is not a NUL-free string", a.Name)
		}
	case 'A':
		v, ok := a.Value.(byte)
		if !ok || v < 33 || v > 126 {
			return fmt.Errorf("BAM tag %s is not a printable character", a.Name)
		}
	case 'B':
		switch v := a.Value.(type) {
		case Array:
			for _, x := range v {
				if x > 127 {
					return fmt.Errorf("BAM tag %s array value %d is outside a signed byte", a.Name, x)
				}
			}
		case IntArray, FloatArray:
		default:
			return fmt.Errorf("BAM tag %s has unsupported array type %T", a.Name, a.Value)
		}
	default:
		return fmt.Errorf("BAM tag %s has unsupported type %q", a.Name, a.Type)
	}
	return nil
}

func validTagName(s string) bool {
	alpha := func(c byte) bool { return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' }
	digit := func(c byte) bool { return c >= '0' && c <= '9' }
	return alpha(s[0]) && (alpha(s[1]) || digit(s[1]))
}

// Validate checks the declared manifest before any file is written.
func (m Manifest) Validate() error {
	if m.Schema != ManifestSchema {
		return fmt.Errorf("manifest schema %q is not %q", m.Schema, ManifestSchema)
	}
	if m.SourceBAM == "" || !isSHA256(m.SourceBAMHash) {
		return errors.New("manifest source BAM pin is incomplete")
	}
	if _, err := m.Header.Text(); err != nil {
		return fmt.Errorf("header: %w", err)
	}
	for i, r := range m.Records {
		if r.Flag != 4 {
			return fmt.Errorf("record %d is not flag 4", i)
		}
		if r.Sequence == "" {
			return fmt.Errorf("record %d has no sequence", i)
		}
		if len(r.Quality) != len(r.Sequence) {
			return fmt.Errorf("record %d quality length differs", i)
		}
		seen := make(map[string]bool, len(r.Aux))
		for _, a := range r.Aux {
			if err := a.validate(); err != nil {
				return fmt.Errorf("record %d: %w", i, err)
			}
			if seen[a.Name] {
				return fmt.Errorf("record %d has duplicate tag %s", i, a.Name)
			}
			seen[a.Name] = true
		}
	}
	if m.Header.ReadGroup != nil {
		for i, r := range m.Records {
			found := false
			for _, a := range r.Aux {
				if a.Name != "RG" {
					continue
				}
				found = true
				id, _ := a.Value.(string)
				if id != m.Header.ReadGroup.ID {
					return fmt.Errorf("record %d RG %q is not the declared read group %q", i, id, m.Header.ReadGroup.ID)
				}
			}
			if !found {
				return fmt.Errorf("record %d has no RG tag for the declared read group", i)
			}
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
		switch v := value.(type) {
		case Array:
			value = []byte(v)
		case IntArray:
			value = bam.ArrayInt32(v)
		case FloatArray:
			value = bam.ArrayFloat32(v)
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

// HashBytes is the receipt hash helper.
func HashBytes(b []byte) string {
	sum := sha256Sum(b)
	return hex.EncodeToString(sum[:])
}

func sha256Sum(b []byte) [32]byte { return sha256.Sum256(b) }

// splitLines keeps empty trailing lines out of header parsing.
func splitLines(text string) []string {
	text = strings.TrimSuffix(text, "\n")
	if text == "" {
		return nil
	}
	return strings.Split(text, "\n")
}
