// Package recordcompare compares literal SAM records as canonical multisets.
package recordcompare

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	MaxLineBytes      = 64 << 20
	MaxCanonicalBytes = 256 << 20
	MaxRecords        = 1_000_000
	MaxExamples       = 5
	MaxExampleBytes   = 512
)

type InputSummary struct {
	Path            string `json:"path"`
	Format          string `json:"format"`
	RawSHA256       string `json:"raw_sha256"`
	CanonicalSHA256 string `json:"canonical_sha256"`
	HeaderSHA256    string `json:"header_text_sha256"`
	Records         uint64 `json:"records"`
	Headers         uint64 `json:"headers"`
}
type Example struct {
	RecordSHA256 string `json:"record_sha256"`
	Prefix       string `json:"record_prefix"`
	Truncated    bool   `json:"truncated"`
	LeftCount    uint64 `json:"left_count"`
	RightCount   uint64 `json:"right_count"`
}
type Report struct {
	Left             InputSummary `json:"left"`
	Right            InputSummary `json:"right"`
	ByteIdentical    bool         `json:"byte_identical"`
	CanonicalEqual   bool         `json:"canonical_equal"`
	HeadersExcluded  bool         `json:"headers_excluded"`
	DifferentRecords uint64       `json:"different_canonical_records"`
	LeftOnly         uint64       `json:"left_excess_records"`
	RightOnly        uint64       `json:"right_excess_records"`
	Examples         []Example    `json:"examples"`
}
type Records struct {
	Summary InputSummary
	counts  map[string]uint64
}

func hashText(s string) string { x := sha256.Sum256([]byte(s)); return hex.EncodeToString(x[:]) }
func uintField(s string, bits int) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	_, e := strconv.ParseUint(s, 10, bits)
	return e == nil
}
func intField(s string, bits int) bool { _, e := strconv.ParseInt(s, 10, bits); return e == nil }
func printable(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < 33 || c > 126 {
			return false
		}
	}
	return true
}
func alpha(c byte) bool { return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' }
func digit(c byte) bool { return c >= '0' && c <= '9' }
func finite(s string, bits int) bool {
	v, e := strconv.ParseFloat(s, bits)
	return e == nil && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func validateTag(s string) error {
	if len(s) < 5 || s[2] != ':' || s[4] != ':' || !alpha(s[0]) || !(alpha(s[1]) || digit(s[1])) {
		return fmt.Errorf("malformed optional tag %q", s)
	}
	v := s[5:]
	valid := false
	switch s[3] {
	case 'A':
		valid = len(v) == 1 && printable(v)
	case 'i':
		n, e := strconv.ParseInt(v, 10, 64)
		valid = e == nil && n >= -2147483648 && n <= 4294967295
	case 'f':
		valid = finite(v, 32)
	case 'Z':
		valid = true
		for _, c := range v {
			if c < 32 || c > 126 {
				valid = false
			}
		}
	case 'H':
		_, e := hex.DecodeString(v)
		valid = e == nil
	case 'B':
		if len(v) >= 3 && v[1] == ',' {
			valid = true
			for _, x := range strings.Split(v[2:], ",") {
				switch v[0] {
				case 'c':
					valid = intField(x, 8)
				case 'C':
					valid = uintField(x, 8)
				case 's':
					valid = intField(x, 16)
				case 'S':
					valid = uintField(x, 16)
				case 'i':
					valid = intField(x, 32)
				case 'I':
					valid = uintField(x, 32)
				case 'f':
					valid = finite(x, 32)
				default:
					valid = false
				}
				if !valid {
					break
				}
			}
		}
	}
	if !valid {
		return fmt.Errorf("invalid optional tag value %q", s)
	}
	return nil
}
func validCigar(s string) bool {
	if s == "" {
		return false
	}
	if s == "*" {
		return true
	}
	for len(s) > 0 {
		i := 0
		for i < len(s) && digit(s[i]) {
			i++
		}
		if i == 0 || i == len(s) || !uintField(s[:i], 32) || !strings.ContainsRune("MIDNSHP=X", rune(s[i])) {
			return false
		}
		s = s[i+1:]
	}
	return true
}

// Canonical preserves each mandatory field literally and sorts complete optional fields.
func Canonical(line string) (string, error) {
	fields := strings.Split(line, "\t")
	if len(fields) < 11 {
		return "", fmt.Errorf("record has %d fields, need at least 11", len(fields))
	}
	if !printable(fields[0]) || len(fields[0]) > 254 || strings.Contains(fields[0], "@") {
		return "", fmt.Errorf("invalid QNAME")
	}
	if !uintField(fields[1], 16) || !uintField(fields[3], 31) || !uintField(fields[4], 8) || !uintField(fields[7], 31) || !intField(fields[8], 32) {
		return "", fmt.Errorf("invalid FLAG, POS, MAPQ, PNEXT or TLEN")
	}
	if !printable(fields[2]) || !printable(fields[6]) || !validCigar(fields[5]) {
		return "", fmt.Errorf("invalid reference or CIGAR")
	}
	seq, qual := fields[9], fields[10]
	if seq != "*" {
		if seq == "" {
			return "", fmt.Errorf("empty SEQ")
		}
		for _, c := range []byte(seq) {
			if !alpha(c) && c != '=' && c != '.' {
				return "", fmt.Errorf("invalid SEQ")
			}
		}
	}
	if qual != "*" {
		if !printable(qual) || (seq != "*" && len(seq) != len(qual)) {
			return "", fmt.Errorf("invalid QUAL or SEQ/QUAL length")
		}
	}
	seen := make(map[string]bool)
	for _, tag := range fields[11:] {
		if err := validateTag(tag); err != nil {
			return "", err
		}
		if seen[tag[:2]] {
			return "", fmt.Errorf("duplicate optional tag %s", tag[:2])
		}
		seen[tag[:2]] = true
	}
	sort.Strings(fields[11:])
	return strings.Join(fields, "\t"), nil
}

// Parse excludes SAM headers. Header SHA hashes their decoded text with LF endings.
func Parse(r io.Reader) (*Records, error) {
	out := &Records{counts: make(map[string]uint64)}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64<<10), MaxLineBytes)
	headerHash := sha256.New()
	lineNo := 0
	total := 0
	seenRecord := false
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if strings.HasPrefix(line, "@") {
			if seenRecord {
				return nil, fmt.Errorf("line %d: header after records", lineNo)
			}
			if len(line) < 3 || !strings.Contains(" HD SQ RG PG CO ", " "+line[1:3]+" ") || (len(line) > 3 && line[3] != '\t') {
				return nil, fmt.Errorf("line %d: malformed header", lineNo)
			}
			io.WriteString(headerHash, line+"\n")
			out.Summary.Headers++
			continue
		}
		seenRecord = true
		canonical, err := Canonical(line)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		total += len(canonical) + 1
		if total > MaxCanonicalBytes || out.Summary.Records >= MaxRecords {
			return nil, fmt.Errorf("input exceeds MaxCanonicalBytes or MaxRecords")
		}
		out.counts[canonical]++
		out.Summary.Records++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	out.Summary.HeaderSHA256 = hex.EncodeToString(headerHash.Sum(nil))
	keys := sortedKeys(out.counts)
	h := sha256.New()
	for _, k := range keys {
		for i := uint64(0); i < out.counts[k]; i++ {
			io.WriteString(h, k+"\n")
		}
	}
	out.Summary.CanonicalSHA256 = hex.EncodeToString(h.Sum(nil))
	return out, nil
}
func sortedKeys(counts map[string]uint64) []string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func Compare(left, right *Records) Report {
	report := Report{Left: left.Summary, Right: right.Summary, HeadersExcluded: true, Examples: []Example{}}
	report.ByteIdentical = left.Summary.RawSHA256 != "" && left.Summary.RawSHA256 == right.Summary.RawSHA256
	union := make(map[string]uint64, len(left.counts)+len(right.counts))
	for k := range left.counts {
		union[k] = 0
	}
	for k := range right.counts {
		union[k] = 0
	}
	for _, k := range sortedKeys(union) {
		l, r := left.counts[k], right.counts[k]
		if l == r {
			continue
		}
		report.DifferentRecords++
		if l > r {
			report.LeftOnly += l - r
		} else {
			report.RightOnly += r - l
		}
		if len(report.Examples) < MaxExamples {
			prefix := k
			if len(prefix) > MaxExampleBytes {
				prefix = prefix[:MaxExampleBytes]
			}
			report.Examples = append(report.Examples, Example{hashText(k), prefix, len(k) > MaxExampleBytes, l, r})
		}
	}
	report.CanonicalEqual = report.DifferentRecords == 0
	return report
}

func Load(path string) (*Records, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return nil, err
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	magic := make([]byte, 4)
	n, readErr := f.Read(magic)
	if readErr != nil && readErr != io.EOF {
		return nil, readErr
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	format := ".sam"
	if strings.EqualFold(filepath.Ext(path), ".bam") || (n >= 2 && magic[0] == 0x1f && magic[1] == 0x8b) || (n == 4 && string(magic) == "BAM\x01") {
		format = ".bam"
	}
	var records *Records
	switch format {
	case ".sam":
		records, err = Parse(f)
	case ".bam":
		command := exec.Command("samtools", "view", "--no-PG", "-h", "--", path)
		var stderr bytes.Buffer
		command.Stderr = &stderr
		var stdout io.ReadCloser
		stdout, err = command.StdoutPipe()
		if err != nil {
			return nil, err
		}
		if err = command.Start(); err != nil {
			return nil, err
		}
		records, err = Parse(stdout)
		if err != nil {
			command.Process.Kill()
			command.Wait()
			return nil, err
		}
		if err = command.Wait(); err != nil {
			return nil, fmt.Errorf("samtools decoding failed: %w: %.2048s", err, stderr.String())
		}
	}
	if err != nil {
		return nil, err
	}
	records.Summary.Path = path
	records.Summary.Format = strings.TrimPrefix(format, ".")
	records.Summary.RawSHA256 = hex.EncodeToString(h.Sum(nil))
	return records, nil
}
func ComparePaths(left, right string) (Report, error) {
	l, e := Load(left)
	if e != nil {
		return Report{}, fmt.Errorf("left input: %w", e)
	}
	r, e := Load(right)
	if e != nil {
		return Report{}, fmt.Errorf("right input: %w", e)
	}
	return Compare(l, r), nil
}
