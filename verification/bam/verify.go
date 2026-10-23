package bamverify

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/recordcompare"
)

const PlanSchema = "bam-verify-plan/1"
const ReceiptSchema = "bam-verify-receipt/1"

// Plan pins the retained oracle and the exact normalizations and expectations
// the verifier may apply. Raw BAM identity cannot be required until a native
// unaligned BAM is retained.
type Plan struct {
	Schema              string   `json:"schema"`
	OracleSAM           string   `json:"oracle_sam"`
	OracleSAMHash       string   `json:"oracle_sam_sha256"`
	DropProgramPrefixes []string `json:"drop_program_prefixes"`
	ExpectedRecords     uint64   `json:"expected_records"`
	ExpectedAuxOrder    []string `json:"expected_aux_order"`
	RequireRawIdentity  bool     `json:"require_raw_identity"`
}

func (p Plan) Validate() error {
	if p.Schema != PlanSchema {
		return fmt.Errorf("plan schema %q is not %q", p.Schema, PlanSchema)
	}
	if p.OracleSAM == "" || !isSHA256(p.OracleSAMHash) {
		return errors.New("plan oracle pin is incomplete")
	}
	if p.ExpectedRecords == 0 {
		return errors.New("plan declares no expected records")
	}
	if len(p.ExpectedAuxOrder) == 0 {
		return errors.New("plan declares no expected tag order")
	}
	seen := make(map[string]bool, len(p.ExpectedAuxOrder))
	for _, name := range p.ExpectedAuxOrder {
		if len(name) != 2 || seen[name] {
			return fmt.Errorf("plan tag order entry %q is not a unique two-character tag", name)
		}
		seen[name] = true
	}
	if p.RequireRawIdentity {
		return errors.New("raw BAM identity cannot be required: no retained native unaligned BAM exists")
	}
	return nil
}

type Check struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail,omitempty"`
}

type Control struct {
	Name     string `json:"name"`
	Expected string `json:"expected"`
	Observed string `json:"observed"`
	Passed   bool   `json:"passed"`
	Detail   string `json:"detail,omitempty"`
}

// RawVerdict is deliberately separate from the canonical and header verdicts.
type RawVerdict struct {
	Claimed       bool   `json:"claimed"`
	ByteIdentical bool   `json:"byte_identical"`
	Reason        string `json:"reason"`
}

type Receipt struct {
	Schema                 string               `json:"schema"`
	Started                string               `json:"started"`
	Finished               string               `json:"finished"`
	PlanSHA256             string               `json:"plan_sha256"`
	OracleSHA256           string               `json:"oracle_sha256"`
	NormalizedOracleSHA256 string               `json:"normalized_oracle_sha256"`
	ManifestSHA256         string               `json:"manifest_sha256"`
	BAMSAH256              string               `json:"bam_sha256"`
	DecodedSAMSAH256       string               `json:"decoded_sam_sha256"`
	Samtools               string               `json:"samtools"`
	Records                uint64               `json:"records"`
	Checks                 []Check              `json:"checks"`
	Canonical              recordcompare.Report `json:"canonical"`
	AuxOrder               []string             `json:"aux_order"`
	RawBAM                 RawVerdict           `json:"raw_bam"`
	DroppedPrograms        []string             `json:"dropped_programs"`
	FaultControls          []string             `json:"fault_controls"`
	Controls               []Control            `json:"controls"`
}

// Verify rebuilds a BAM from the declared attempt manifest, decodes it with
// htslib and checks the canonical, header and literal record verdicts against
// the pinned oracle. Every control must behave as declared.
func Verify(planPath, attemptDir, outDir string) (Receipt, error) {
	started := time.Now().UTC().Format(time.RFC3339)
	if err := os.Mkdir(outDir, 0o755); err != nil {
		return Receipt{}, fmt.Errorf("fresh output required: %w", err)
	}
	receipt := Receipt{Schema: ReceiptSchema, Started: started, FaultControls: []string{
		"internal/bam TestWriterShortWrite", "internal/bam TestWriterENOSPC"}}
	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		return Receipt{}, err
	}
	receipt.PlanSHA256 = HashBytes(planBytes)
	var plan Plan
	if err := strictJSON(planBytes, &plan); err != nil {
		return Receipt{}, fmt.Errorf("plan: %w", err)
	}
	if err := plan.Validate(); err != nil {
		return Receipt{}, err
	}
	oracleBytes, err := os.ReadFile(plan.OracleSAM)
	if err != nil {
		return Receipt{}, err
	}
	receipt.OracleSHA256 = HashBytes(oracleBytes)
	if receipt.OracleSHA256 != plan.OracleSAMHash {
		return Receipt{}, fmt.Errorf("oracle hash %s differs from plan pin %s", receipt.OracleSHA256, plan.OracleSAMHash)
	}
	parsed, err := ParseSAM(bytes.NewReader(oracleBytes), ParseOptions{SourceSAM: plan.OracleSAM,
		SourceHash: plan.OracleSAMHash, DropProgramPrefixes: plan.DropProgramPrefixes})
	if err != nil {
		return Receipt{}, fmt.Errorf("oracle parse: %w", err)
	}
	receipt.DroppedPrograms = parsed.DroppedPrograms
	if uint64(len(parsed.Manifest.Records)) != plan.ExpectedRecords {
		return Receipt{}, fmt.Errorf("oracle has %d records, plan expects %d", len(parsed.Manifest.Records), plan.ExpectedRecords)
	}
	if got := parsed.Manifest.Records[0].AuxNames(); !equalStrings(got, plan.ExpectedAuxOrder) {
		return Receipt{}, fmt.Errorf("oracle tag order %v differs from plan %v", got, plan.ExpectedAuxOrder)
	}
	normalized, err := Normalize(oracleBytes, parsed.DroppedPrograms)
	if err != nil {
		return Receipt{}, err
	}
	receipt.NormalizedOracleSHA256 = HashBytes(normalized)
	if err := os.WriteFile(filepath.Join(outDir, "oracle.normalized.sam"), normalized, 0o644); err != nil {
		return Receipt{}, err
	}

	manifestBytes, err := os.ReadFile(filepath.Join(attemptDir, "manifest.json"))
	if err != nil {
		return Receipt{}, err
	}
	receipt.ManifestSHA256 = HashBytes(manifestBytes)
	manifest, err := DecodeManifest(manifestBytes)
	if err != nil {
		return Receipt{}, fmt.Errorf("manifest: %w", err)
	}
	if manifest.SourceSAMHash != plan.OracleSAMHash {
		return Receipt{}, fmt.Errorf("manifest is bound to %s, plan pins %s", manifest.SourceSAMHash, plan.OracleSAMHash)
	}
	derived, err := parsed.Manifest.Encode()
	if err != nil {
		return Receipt{}, err
	}
	if !bytes.Equal(derived, manifestBytes) {
		return Receipt{}, errors.New("attempt manifest differs from the oracle-derived manifest")
	}

	bamPath := filepath.Join(outDir, "unaligned.bam")
	bamBytes, err := writeBAM(manifest, bamPath)
	if err != nil {
		return Receipt{}, err
	}
	receipt.BAMSAH256 = HashBytes(bamBytes)
	receipt.Records = uint64(len(manifest.Records))
	var again bytes.Buffer
	if err := manifest.WriteBAM(&again); err != nil {
		return Receipt{}, err
	}
	deterministic := bytes.Equal(bamBytes, again.Bytes())

	version, err := samtoolsVersion()
	if err != nil {
		return Receipt{}, err
	}
	receipt.Samtools = version
	quick, quickDetail := quickcheck(bamPath)

	decoded, err := samtoolsView(bamPath)
	if err != nil {
		return Receipt{}, err
	}
	if err := os.WriteFile(filepath.Join(outDir, "decoded.sam"), decoded, 0o644); err != nil {
		return Receipt{}, err
	}
	receipt.DecodedSAMSAH256 = HashBytes(decoded)
	oracleDecoded, err := samtoolsView(filepath.Join(outDir, "oracle.normalized.sam"))
	if err != nil {
		return Receipt{}, err
	}
	if err := os.WriteFile(filepath.Join(outDir, "oracle.decoded.sam"), oracleDecoded, 0o644); err != nil {
		return Receipt{}, err
	}

	report, err := recordcompare.ComparePaths(bamPath, filepath.Join(outDir, "oracle.normalized.sam"))
	if err != nil {
		return Receipt{}, err
	}
	receipt.Canonical = report
	decodedHeader, decodedRecords := splitSAM(decoded)
	oracleHeader, oracleRecords := splitSAM(oracleDecoded)
	headerEqual := HashBytes(decodedHeader) == HashBytes(oracleHeader)
	recordsLiteral := bytes.Equal(decodedRecords, oracleRecords)
	auxOrder := recordAuxOrder(decodedRecords, 0)
	receipt.AuxOrder = auxOrder
	receipt.RawBAM = RawVerdict{Claimed: false, ByteIdentical: report.ByteIdentical,
		Reason: "no retained native unaligned BAM exists; the oracle is SAM text transcoded by samtools"}

	checks := []Check{
		{Name: "plan-pin", Passed: true, Detail: plan.OracleSAMHash},
		{Name: "oracle-hash", Passed: true, Detail: receipt.OracleSHA256},
		{Name: "manifest-binding", Passed: true, Detail: manifest.SourceSAMHash},
		{Name: "manifest-oracle-identity", Passed: true, Detail: "attempt equals the derived manifest"},
		{Name: "quickcheck-unmapped", Passed: quick == 0, Detail: quickDetail},
		{Name: "canonical-records", Passed: report.CanonicalEqual, Detail: canonicalDetail(report)},
		{Name: "decoded-header", Passed: headerEqual, Detail: HashBytes(decodedHeader)},
		{Name: "literal-records", Passed: recordsLiteral, Detail: literalDetail(decodedRecords, oracleRecords)},
		{Name: "aux-order", Passed: equalStrings(auxOrder, plan.ExpectedAuxOrder), Detail: strings.Join(auxOrder, ",")},
		{Name: "bam-deterministic", Passed: deterministic, Detail: receipt.BAMSAH256},
	}
	receipt.Checks = checks

	controls, err := runControls(outDir, manifest)
	if err != nil {
		return Receipt{}, err
	}
	receipt.Controls = controls

	receipt.Finished = time.Now().UTC().Format(time.RFC3339)
	if err := publishReceipt(filepath.Join(outDir, "receipt.json"), receipt); err != nil {
		return Receipt{}, err
	}
	var failed []string
	for _, c := range checks {
		if !c.Passed {
			failed = append(failed, c.Name)
		}
	}
	for _, c := range controls {
		if !c.Passed {
			failed = append(failed, "control:"+c.Name)
		}
	}
	if len(failed) > 0 {
		return receipt, fmt.Errorf("verification failed: %s", strings.Join(failed, ", "))
	}
	return receipt, nil
}

func writeBAM(m Manifest, path string) ([]byte, error) {
	var buf bytes.Buffer
	if err := m.WriteBAM(&buf); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func strictJSON(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return errors.New("trailing JSON")
	}
	return nil
}

func samtoolsVersion() (string, error) {
	out, err := exec.Command("samtools", "--version").Output()
	if err != nil {
		return "", fmt.Errorf("samtools: %w", err)
	}
	line, _, _ := strings.Cut(string(out), "\n")
	return strings.TrimSpace(line), nil
}

func quickcheck(path string) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "samtools", "quickcheck", "-u", "--", path)
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
		} else {
			code = -1
		}
	}
	return code, strings.TrimSpace(string(out))
}

func samtoolsView(path string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "samtools", "view", "--no-PG", "-h", "--", path)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("samtools view %s: %w: %.2048s", path, err, stderr.String())
	}
	return out, nil
}

func splitSAM(data []byte) (header, records []byte) {
	var h, r bytes.Buffer
	for _, line := range strings.SplitAfter(string(data), "\n") {
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "@") {
			h.WriteString(line)
		} else {
			r.WriteString(line)
		}
	}
	return h.Bytes(), r.Bytes()
}

func recordAuxOrder(records []byte, index int) []string {
	lines := strings.Split(strings.TrimSuffix(string(records), "\n"), "\n")
	if index >= len(lines) || lines[index] == "" {
		return nil
	}
	fields := strings.Split(lines[index], "\t")
	if len(fields) < 12 {
		return nil
	}
	out := make([]string, 0, len(fields)-11)
	for _, field := range fields[11:] {
		if len(field) < 2 {
			return nil
		}
		out = append(out, field[:2])
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func canonicalDetail(r recordcompare.Report) string {
	return fmt.Sprintf("records %d/%d, differing %d, left-only %d, right-only %d",
		r.Left.Records, r.Right.Records, r.DifferentRecords, r.LeftOnly, r.RightOnly)
}

func literalDetail(got, want []byte) string {
	if bytes.Equal(got, want) {
		return "record text identical"
	}
	return fmt.Sprintf("record text differs: got %d bytes sha %s, want %d bytes sha %s",
		len(got), HashBytes(got), len(want), HashBytes(want))
}

func publishReceipt(path string, r Receipt) (err error) {
	body, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	f, err := os.CreateTemp(filepath.Dir(path), ".receipt-*.tmp")
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, os.Remove(f.Name())) }()
	n, writeErr := f.Write(body)
	if writeErr == nil && n != len(body) {
		writeErr = errors.New("short receipt write")
	}
	if writeErr == nil {
		writeErr = f.Sync()
	}
	if err := errors.Join(writeErr, f.Close()); err != nil {
		return err
	}
	return os.Link(f.Name(), path)
}

// runControls proves the comparator is sensitive to the mutations the card
// names. A control that does not behave as declared fails verification.
func runControls(outDir string, manifest Manifest) ([]Control, error) {
	dir := filepath.Join(outDir, "controls")
	if err := os.Mkdir(dir, 0o755); err != nil {
		return nil, err
	}
	samPath := filepath.Join(outDir, "oracle.normalized.sam")
	write := func(name string, m Manifest) (string, error) {
		if err := os.Mkdir(filepath.Join(dir, name), 0o755); err != nil {
			return "", err
		}
		path := filepath.Join(dir, name, "unaligned.bam")
		_, err := writeBAM(m, path)
		return path, err
	}
	compare := func(path string) (recordcompare.Report, error) {
		return recordcompare.ComparePaths(path, samPath)
	}
	var controls []Control
	add := func(name, expected string, observed bool, detail string) {
		controls = append(controls, Control{Name: name, Expected: expected,
			Observed: map[bool]string{true: "as-declared", false: "unexpected"}[observed],
			Passed:   observed, Detail: detail})
	}

	// Comparator self-check: self-comparison is not evidence of a Go writer.
	self, err := recordcompare.ComparePaths(samPath, samPath)
	if err != nil {
		return nil, err
	}
	add("comparator-self", "canonical and raw equal", self.CanonicalEqual && self.ByteIdentical, canonicalDetail(self))

	// Changed tag: canonical must differ.
	changed := cloneManifest(manifest)
	for i := range changed.Records[0].Aux {
		if changed.Records[0].Aux[i].Type == 'f' {
			changed.Records[0].Aux[i].Value = changed.Records[0].Aux[i].Value.(float32) + 1
			break
		}
	}
	path, err := write("changed-tag", changed)
	if err != nil {
		return nil, err
	}
	rep, err := compare(path)
	if err != nil {
		return nil, err
	}
	add("changed-tag", "canonical differs", !rep.CanonicalEqual, canonicalDetail(rep))

	// Tag reorder: canonical must stay equal while raw bytes differ.
	reordered := cloneManifest(manifest)
	if len(reordered.Records[0].Aux) < 2 {
		return nil, errors.New("oracle record has fewer than two tags")
	}
	reordered.Records[0].Aux[0], reordered.Records[0].Aux[1] = reordered.Records[0].Aux[1], reordered.Records[0].Aux[0]
	path, err = write("tag-reorder", reordered)
	if err != nil {
		return nil, err
	}
	rep, err = compare(path)
	if err != nil {
		return nil, err
	}
	add("tag-reorder", "canonical equal, raw differs", rep.CanonicalEqual && !rep.ByteIdentical, canonicalDetail(rep))

	// Duplicate record: canonical must differ by one excess record.
	duplicated := cloneManifest(manifest)
	duplicated.Records = append(duplicated.Records, duplicated.Records[len(duplicated.Records)-1])
	path, err = write("duplicate-record", duplicated)
	if err != nil {
		return nil, err
	}
	rep, err = compare(path)
	if err != nil {
		return nil, err
	}
	add("duplicate-record", "canonical differs with one excess", !rep.CanonicalEqual && rep.LeftOnly == 1, canonicalDetail(rep))

	// Missing record: canonical must differ by one absent record.
	missing := cloneManifest(manifest)
	missing.Records = missing.Records[:len(missing.Records)-1]
	path, err = write("missing-record", missing)
	if err != nil {
		return nil, err
	}
	rep, err = compare(path)
	if err != nil {
		return nil, err
	}
	add("missing-record", "canonical differs with one absent", !rep.CanonicalEqual && rep.RightOnly == 1, canonicalDetail(rep))

	// Header-only mutation: records stay canonical, header must differ.
	headerMutated := cloneManifest(manifest)
	headerMutated.Header.SortOrder = "coordinate"
	path, err = write("header-mutation", headerMutated)
	if err != nil {
		return nil, err
	}
	rep, err = compare(path)
	if err != nil {
		return nil, err
	}
	headerDiffers := rep.Left.HeaderSHA256 != rep.Right.HeaderSHA256
	add("header-mutation", "records equal, header differs", rep.CanonicalEqual && headerDiffers, canonicalDetail(rep))

	// Truncated BGZF: quickcheck must reject the missing EOF marker.
	truncated := filepath.Join(dir, "truncated-bgzf")
	if err := os.Mkdir(truncated, 0o755); err != nil {
		return nil, err
	}
	valid := filepath.Join(truncated, "unaligned.bam")
	data, err := writeBAM(manifest, valid)
	if err != nil {
		return nil, err
	}
	if len(data) < 28 {
		return nil, errors.New("BAM is too small to truncate")
	}
	cut := filepath.Join(truncated, "truncated.bam")
	if err := os.WriteFile(cut, data[:len(data)-28], 0o644); err != nil {
		return nil, err
	}
	code, detail := quickcheck(cut)
	add("truncated-bgzf", "quickcheck rejects", code != 0, fmt.Sprintf("exit %d %s", code, detail))

	// Truncated record: quickcheck must reject a file cut inside the payload.
	cutRecord := filepath.Join(truncated, "truncated-record.bam")
	if err := os.WriteFile(cutRecord, data[:len(data)/2], 0o644); err != nil {
		return nil, err
	}
	code, detail = quickcheck(cutRecord)
	add("truncated-record", "quickcheck rejects", code != 0, fmt.Sprintf("exit %d %s", code, detail))

	return controls, nil
}

func cloneManifest(m Manifest) Manifest {
	out := m
	out.Header = m.Header
	if m.Header.ReadGroup != nil {
		rg := *m.Header.ReadGroup
		out.Header.ReadGroup = &rg
	}
	out.Records = make([]Record, len(m.Records))
	for i, r := range m.Records {
		out.Records[i] = r
		out.Records[i].Quality = append(Phred(nil), r.Quality...)
		out.Records[i].Aux = make([]Aux, len(r.Aux))
		for j, a := range r.Aux {
			out.Records[i].Aux[j] = a
			if b, ok := a.Value.(Array); ok {
				out.Records[i].Aux[j].Value = append(Array(nil), b...)
			}
		}
	}
	return out
}
