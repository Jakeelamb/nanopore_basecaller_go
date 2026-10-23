package bamverify

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/recordcompare"
)

const PlanSchema = "bam-verify-plan/1"
const ReceiptSchema = "bam-verify-receipt/1"

// Plan pins the retained native BAM and the expectations the verifier may
// apply. Payload identity is the claimed byte verdict; compressed identity and
// BGZF member topology are explicitly not claimable.
type Plan struct {
	Schema                 string   `json:"schema"`
	OracleBAM              string   `json:"oracle_bam"`
	OracleBAMHash          string   `json:"oracle_bam_sha256"`
	ExpectedRecords        uint64   `json:"expected_records"`
	ExpectedAuxOrder       []string `json:"expected_aux_order"`
	RequirePayloadIdentity bool     `json:"require_payload_identity"`
}

func (p Plan) Validate() error {
	if p.Schema != PlanSchema {
		return fmt.Errorf("plan schema %q is not %q", p.Schema, PlanSchema)
	}
	if p.OracleBAM == "" || !isSHA256(p.OracleBAMHash) {
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
	if !p.RequirePayloadIdentity {
		return errors.New("plan must require uncompressed payload identity")
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

type Fault struct {
	Name     string `json:"name"`
	Injected string `json:"injected"`
	Observed string `json:"observed"`
	Passed   bool   `json:"passed"`
}

// Verdict separates a claim from an observation. An unclaimed verdict may
// differ without failing verification; a claimed one may not.
type Verdict struct {
	Claimed   bool   `json:"claimed"`
	Identical bool   `json:"identical"`
	Reason    string `json:"reason"`
}

type BGZFVerdict struct {
	Claimed      bool   `json:"claimed"`
	MembersMatch bool   `json:"members_match"`
	OracleCount  int    `json:"oracle_members"`
	GoCount      int    `json:"go_members"`
	Reason       string `json:"reason"`
}

type Receipt struct {
	Schema             string               `json:"schema"`
	Started            string               `json:"started"`
	Finished           string               `json:"finished"`
	PlanSHA256         string               `json:"plan_sha256"`
	OracleBAMPath      string               `json:"oracle_bam"`
	OracleBAMHash      string               `json:"oracle_bam_sha256"`
	OraclePayloadHash  string               `json:"oracle_payload_sha256"`
	OraclePayloadBytes int                  `json:"oracle_payload_bytes"`
	ManifestSHA256     string               `json:"manifest_sha256"`
	GoBAMHash          string               `json:"go_bam_sha256"`
	GoPayloadHash      string               `json:"go_payload_sha256"`
	GoPayloadBytes     int                  `json:"go_payload_bytes"`
	DecodedSAMHash     string               `json:"decoded_sam_sha256"`
	Samtools           string               `json:"samtools"`
	Records            uint64               `json:"records"`
	HeaderTextHash     string               `json:"header_text_sha256"`
	Checks             []Check              `json:"checks"`
	Canonical          recordcompare.Report `json:"canonical"`
	AuxOrder           []string             `json:"aux_order"`
	Payload            Verdict              `json:"payload"`
	Compressed         Verdict              `json:"compressed"`
	BGZF               BGZFVerdict          `json:"bgzf"`
	Controls           []Control            `json:"controls"`
	Faults             []Fault              `json:"faults"`
}

// Verify rebuilds a BAM from the declared attempt manifest and checks its
// uncompressed payload, header text, record bits and htslib decode against the
// pinned native BAM. Every control must behave as declared.
func Verify(planPath, attemptDir, outDir string) (Receipt, error) {
	started := time.Now().UTC().Format(time.RFC3339)
	if err := os.Mkdir(outDir, 0o755); err != nil {
		return Receipt{}, fmt.Errorf("fresh output required: %w", err)
	}
	receipt := Receipt{Schema: ReceiptSchema, Started: started}
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
	oracleBytes, err := os.ReadFile(plan.OracleBAM)
	if err != nil {
		return Receipt{}, err
	}
	receipt.OracleBAMPath = plan.OracleBAM
	receipt.OracleBAMHash = HashBytes(oracleBytes)
	if receipt.OracleBAMHash != plan.OracleBAMHash {
		return Receipt{}, fmt.Errorf("oracle hash %s differs from plan pin %s", receipt.OracleBAMHash, plan.OracleBAMHash)
	}
	oracle, err := ParseBAM(oracleBytes, ParseOptions{SourceBAM: plan.OracleBAM, SourceHash: plan.OracleBAMHash})
	if err != nil {
		return Receipt{}, fmt.Errorf("oracle parse: %w", err)
	}
	receipt.OraclePayloadHash = oracle.PayloadSHA256
	receipt.OraclePayloadBytes = oracle.PayloadBytes
	receipt.HeaderTextHash = oracle.HeaderSHA256
	if uint64(len(oracle.Manifest.Records)) != plan.ExpectedRecords {
		return Receipt{}, fmt.Errorf("oracle has %d records, plan expects %d", len(oracle.Manifest.Records), plan.ExpectedRecords)
	}
	if got := oracle.Manifest.Records[0].AuxNames(); !equalStrings(got, plan.ExpectedAuxOrder) {
		return Receipt{}, fmt.Errorf("oracle tag order %v differs from plan %v", got, plan.ExpectedAuxOrder)
	}
	oraclePayload, _, err := decodeBGZF(oracleBytes)
	if err != nil {
		return Receipt{}, fmt.Errorf("oracle payload: %w", err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "oracle.payload"), oraclePayload, 0o644); err != nil {
		return Receipt{}, err
	}
	if err := os.WriteFile(filepath.Join(outDir, "oracle.header.sam"), []byte(oracle.HeaderText), 0o644); err != nil {
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
	if manifest.SourceBAMHash != plan.OracleBAMHash {
		return Receipt{}, fmt.Errorf("manifest is bound to %s, plan pins %s", manifest.SourceBAMHash, plan.OracleBAMHash)
	}
	derived, err := oracle.Manifest.Encode()
	if err != nil {
		return Receipt{}, err
	}
	manifestIdentity := bytes.Equal(derived, manifestBytes)
	if !manifestIdentity {
		return Receipt{}, errors.New("attempt manifest differs from the oracle-derived manifest")
	}

	bamPath := filepath.Join(outDir, "unaligned.bam")
	goBytes, err := writeBAMFile(manifest, bamPath)
	if err != nil {
		return Receipt{}, err
	}
	receipt.GoBAMHash = HashBytes(goBytes)
	receipt.Records = uint64(len(manifest.Records))
	goPayload, goMembers, err := decodeBGZF(goBytes)
	if err != nil {
		return Receipt{}, fmt.Errorf("go BAM decode: %w", err)
	}
	receipt.GoPayloadHash = HashBytes(goPayload)
	receipt.GoPayloadBytes = len(goPayload)
	if err := os.WriteFile(filepath.Join(outDir, "go.payload"), goPayload, 0o644); err != nil {
		return Receipt{}, err
	}

	var again bytes.Buffer
	if err := manifest.WriteBAM(&again); err != nil {
		return Receipt{}, err
	}
	deterministic := bytes.Equal(goBytes, again.Bytes())

	goHeader, goHeaderText, goRecords, err := parseBAMPayload(goPayload)
	if err != nil {
		return Receipt{}, fmt.Errorf("go payload parse: %w", err)
	}
	recordsBits := reflect.DeepEqual(goRecords, oracle.Manifest.Records)
	headerBits := reflect.DeepEqual(goHeader, oracle.Manifest.Header)
	headerTextIdentity := goHeaderText == oracle.HeaderText
	payloadIdentity := bytes.Equal(goPayload, oraclePayload)

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
	receipt.DecodedSAMHash = HashBytes(decoded)
	oracleDecoded, err := samtoolsView(plan.OracleBAM)
	if err != nil {
		return Receipt{}, err
	}
	if err := os.WriteFile(filepath.Join(outDir, "oracle.decoded.sam"), oracleDecoded, 0o644); err != nil {
		return Receipt{}, err
	}
	decodedHeader, decodedRecords := splitSAM(decoded)
	oracleHeader, oracleRecords := splitSAM(oracleDecoded)
	headerDecoded := bytes.Equal(decodedHeader, oracleHeader)
	recordsLiteral := bytes.Equal(decodedRecords, oracleRecords)
	auxOrder := recordAuxOrder(decodedRecords, 0)

	report, err := recordcompare.ComparePaths(bamPath, plan.OracleBAM)
	if err != nil {
		return Receipt{}, err
	}
	receipt.Canonical = report
	receipt.AuxOrder = auxOrder
	receipt.Payload = Verdict{Claimed: true, Identical: payloadIdentity,
		Reason: "uncompressed BAM payload compared byte for byte"}
	receipt.Compressed = Verdict{Claimed: false, Identical: bytes.Equal(goBytes, oracleBytes),
		Reason: "Go flate is not zlib; compressed bytes are not a claimable verdict"}
	receipt.BGZF = BGZFVerdict{Claimed: false, MembersMatch: membersMatch(oracle.Members, goMembers),
		OracleCount: len(oracle.Members), GoCount: len(goMembers),
		Reason: "native member boundaries follow the producer flush schedule; only the inflated payload is claimed"}

	qnamesEqual, qnameDetail := qnameSetEqual(oracle.Manifest.Records, goRecords)
	rgErr := checkReadGroups(manifest)
	checks := []Check{
		{Name: "plan-pin", Passed: true, Detail: plan.OracleBAMHash},
		{Name: "oracle-hash", Passed: receipt.OracleBAMHash == plan.OracleBAMHash, Detail: receipt.OracleBAMHash},
		{Name: "manifest-binding", Passed: manifest.SourceBAMHash == plan.OracleBAMHash, Detail: manifest.SourceBAMHash},
		{Name: "manifest-oracle-identity", Passed: manifestIdentity, Detail: "attempt equals the oracle-derived manifest"},
		{Name: "quickcheck-unmapped", Passed: quick == 0, Detail: quickDetail},
		{Name: "header-text-identity", Passed: headerTextIdentity, Detail: oracle.HeaderSHA256},
		{Name: "header-bits", Passed: headerBits, Detail: "parsed header structs match"},
		{Name: "payload-identity", Passed: payloadIdentity, Detail: fmt.Sprintf("%d bytes sha %s", len(goPayload), receipt.GoPayloadHash)},
		{Name: "record-bits", Passed: recordsBits, Detail: fmt.Sprintf("%d records, exact tag types and float bits", len(goRecords))},
		{Name: "qname-set", Passed: qnamesEqual, Detail: qnameDetail},
		{Name: "rg-consistency", Passed: rgErr == nil, Detail: errorDetail(rgErr)},
		{Name: "canonical-records", Passed: report.CanonicalEqual, Detail: canonicalDetail(report)},
		{Name: "decoded-header", Passed: headerDecoded, Detail: HashBytes(decodedHeader)},
		{Name: "literal-records", Passed: recordsLiteral, Detail: literalDetail(decodedRecords, oracleRecords)},
		{Name: "aux-order", Passed: equalStrings(auxOrder, plan.ExpectedAuxOrder), Detail: strings.Join(auxOrder, ",")},
		{Name: "bam-deterministic", Passed: deterministic, Detail: receipt.GoBAMHash},
		{Name: "bgzf-eof", Passed: bytes.HasSuffix(goBytes, bgzfEOFMarker), Detail: "28-byte EOF marker"},
	}
	receipt.Checks = checks

	controls, faults, err := runControls(outDir, plan.OracleBAM, oracle)
	if err != nil {
		return Receipt{}, err
	}
	receipt.Controls = controls
	receipt.Faults = faults

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
	for _, f := range faults {
		if !f.Passed {
			failed = append(failed, "fault:"+f.Name)
		}
	}
	if len(failed) > 0 {
		return receipt, fmt.Errorf("verification failed: %s", strings.Join(failed, ", "))
	}
	return receipt, nil
}

func writeBAMFile(m Manifest, path string) ([]byte, error) {
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

func errorDetail(err error) string {
	if err == nil {
		return "every RG tag names the declared read group"
	}
	return err.Error()
}

func membersMatch(a, b []BGZFMember) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].CompressedBytes != b[i].CompressedBytes || a[i].PayloadSHA256 != b[i].PayloadSHA256 {
			return false
		}
	}
	return true
}

func qnameSetEqual(a, b []Record) (bool, string) {
	left := make([]string, len(a))
	for i, r := range a {
		left[i] = r.Name
	}
	right := make([]string, len(b))
	for i, r := range b {
		right[i] = r.Name
	}
	sort.Strings(left)
	sort.Strings(right)
	if equalStrings(left, right) {
		return true, fmt.Sprintf("%d unique QNAMEs", len(left))
	}
	return false, fmt.Sprintf("QNAME sets differ: %d vs %d", len(left), len(right))
}

func checkReadGroups(m Manifest) error {
	if m.Header.ReadGroup == nil {
		for i, r := range m.Records {
			for _, a := range r.Aux {
				if a.Name == "RG" {
					return fmt.Errorf("record %d has RG but the header declares none", i)
				}
			}
		}
		return nil
	}
	for i, r := range m.Records {
		found := false
		for _, a := range r.Aux {
			if a.Name != "RG" {
				continue
			}
			found = true
			id, _ := a.Value.(string)
			if id != m.Header.ReadGroup.ID {
				return fmt.Errorf("record %d RG %q is not %q", i, id, m.Header.ReadGroup.ID)
			}
		}
		if !found {
			return fmt.Errorf("record %d has no RG tag", i)
		}
	}
	return nil
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

type faultWriter struct {
	err   error
	short bool
}

func (f faultWriter) Write(p []byte) (int, error) {
	if f.err != nil {
		return 0, f.err
	}
	if f.short && len(p) > 0 {
		return len(p) - 1, nil
	}
	return len(p), nil
}

// runControls proves the verifier and writer are sensitive to the mutations the
// card names. Every control executes; none may skip.
func runControls(outDir, oraclePath string, oracle ParseResult) ([]Control, []Fault, error) {
	dir := filepath.Join(outDir, "controls")
	if err := os.Mkdir(dir, 0o755); err != nil {
		return nil, nil, err
	}
	oracleBytes, err := os.ReadFile(oraclePath)
	if err != nil {
		return nil, nil, err
	}
	oraclePayload, _, err := decodeBGZF(oracleBytes)
	if err != nil {
		return nil, nil, err
	}
	base := oracle.Manifest
	var controls []Control
	add := func(name, expected string, passed bool, detail string) {
		observed := "as-declared"
		if !passed {
			observed = "unexpected"
		}
		controls = append(controls, Control{Name: name, Expected: expected, Observed: observed, Passed: passed, Detail: detail})
	}
	writeVariant := func(name string, m Manifest) (string, []byte, error) {
		sub := filepath.Join(dir, name)
		if err := os.Mkdir(sub, 0o755); err != nil {
			return "", nil, err
		}
		path := filepath.Join(sub, "unaligned.bam")
		data, err := writeBAMFile(m, path)
		if err != nil {
			return "", nil, err
		}
		payload, _, err := decodeBGZF(data)
		return path, payload, err
	}
	compare := func(path string) (recordcompare.Report, error) {
		return recordcompare.ComparePaths(path, oraclePath)
	}

	// Comparator self-check: self-comparison is not evidence of a Go writer.
	self, err := recordcompare.ComparePaths(oraclePath, oraclePath)
	if err != nil {
		return nil, nil, err
	}
	add("comparator-self", "canonical and raw equal", self.CanonicalEqual && self.ByteIdentical, canonicalDetail(self))

	// Changed float tag: payload and parsed bits must differ.
	changed := cloneManifest(base)
	changedOK := false
	for i := range changed.Records[0].Aux {
		if v, ok := changed.Records[0].Aux[i].Value.(float32); ok && changed.Records[0].Aux[i].Type == 'f' {
			changed.Records[0].Aux[i].Value = math.Nextafter32(v, float32(math.Inf(1)))
			changedOK = true
			break
		}
	}
	if !changedOK {
		return nil, nil, errors.New("oracle record has no float tag for the 1-ULP control")
	}
	_, changedPayload, err := writeVariant("changed-tag", changed)
	if err != nil {
		return nil, nil, err
	}
	changedBits := !recordsEqual(base.Records, changed.Records)
	add("changed-tag", "payload and record bits differ", !bytes.Equal(changedPayload, oraclePayload) && changedBits, "1 ULP float change")

	// Tag reorder: payload differs while the canonical multiset stays equal.
	reordered := cloneManifest(base)
	if len(reordered.Records[0].Aux) < 2 {
		return nil, nil, errors.New("oracle record has fewer than two tags")
	}
	reordered.Records[0].Aux[0], reordered.Records[0].Aux[1] = reordered.Records[0].Aux[1], reordered.Records[0].Aux[0]
	reorderPath, reorderPayload, err := writeVariant("tag-reorder", reordered)
	if err != nil {
		return nil, nil, err
	}
	rep, err := compare(reorderPath)
	if err != nil {
		return nil, nil, err
	}
	add("tag-reorder", "payload differs, canonical equal", !bytes.Equal(reorderPayload, oraclePayload) && rep.CanonicalEqual, canonicalDetail(rep))

	// Duplicate and missing records: canonical counts must move by one.
	duplicated := cloneManifest(base)
	duplicated.Records = append(duplicated.Records, duplicated.Records[len(duplicated.Records)-1])
	dupPath, dupPayload, err := writeVariant("duplicate-record", duplicated)
	if err != nil {
		return nil, nil, err
	}
	rep, err = compare(dupPath)
	if err != nil {
		return nil, nil, err
	}
	add("duplicate-record", "payload differs, one excess record", !bytes.Equal(dupPayload, oraclePayload) && rep.LeftOnly == 1, canonicalDetail(rep))

	missing := cloneManifest(base)
	missing.Records = missing.Records[:len(missing.Records)-1]
	missingPath, missingPayload, err := writeVariant("missing-record", missing)
	if err != nil {
		return nil, nil, err
	}
	rep, err = compare(missingPath)
	if err != nil {
		return nil, nil, err
	}
	add("missing-record", "payload differs, one absent record", !bytes.Equal(missingPayload, oraclePayload) && rep.RightOnly == 1, canonicalDetail(rep))

	// Record reorder: payload differs, canonical multiset stays equal.
	if len(base.Records) < 2 {
		return nil, nil, errors.New("oracle has fewer than two records for the reorder control")
	}
	recordOrder := cloneManifest(base)
	recordOrder.Records[0], recordOrder.Records[1] = recordOrder.Records[1], recordOrder.Records[0]
	orderPath, orderPayload, err := writeVariant("record-reorder", recordOrder)
	if err != nil {
		return nil, nil, err
	}
	rep, err = compare(orderPath)
	if err != nil {
		return nil, nil, err
	}
	add("record-reorder", "payload differs, canonical equal", !bytes.Equal(orderPayload, oraclePayload) && rep.CanonicalEqual, canonicalDetail(rep))

	// Header-only mutation: payload and header text differ, records stay canonical.
	headerMutated := cloneManifest(base)
	headerMutated.Header.Program.Description += " (control)"
	headerPath, headerPayload, err := writeVariant("header-mutation", headerMutated)
	if err != nil {
		return nil, nil, err
	}
	rep, err = compare(headerPath)
	if err != nil {
		return nil, nil, err
	}
	mutatedText, err := headerMutated.Header.Text()
	if err != nil {
		return nil, nil, err
	}
	add("header-mutation", "payload and header text differ, records equal",
		!bytes.Equal(headerPayload, oraclePayload) && mutatedText != oracle.HeaderText && rep.CanonicalEqual, canonicalDetail(rep))

	// Dangling read group: validation must reject before a file is written.
	dangling := cloneManifest(base)
	if len(dangling.Records[0].Aux) > 0 {
		for i := range dangling.Records[0].Aux {
			if dangling.Records[0].Aux[i].Name == "RG" {
				dangling.Records[0].Aux[i].Value = "not-the-declared-rg"
				break
			}
		}
	}
	add("dangling-rg", "manifest validation rejects", dangling.Validate() != nil, "RG tag changed to an undeclared ID")

	// Unported requests must be refused, not silently mis-encoded.
	_, duplexErr := bam.Tags(bam.Metadata{IsDuplex: true})
	add("duplex-reject", "duplex tags refused", duplexErr != nil, "bam.Tags(IsDuplex)")
	flagErr := bam.NewWriter(io.Discard).WriteRecord(bam.Record{Name: "read", Flag: 4 | 16, Sequence: "A", Quality: []byte{0}})
	add("flag-reject", "flag 20 refused", flagErr != nil, "unaligned writer requires flag 4")

	// Truncation controls: htslib must reject both a missing EOF and a cut payload.
	truncated := filepath.Join(dir, "truncated")
	if err := os.Mkdir(truncated, 0o755); err != nil {
		return nil, nil, err
	}
	valid := filepath.Join(truncated, "unaligned.bam")
	data, err := writeBAMFile(base, valid)
	if err != nil {
		return nil, nil, err
	}
	if len(data) < 28 {
		return nil, nil, errors.New("BAM is too small to truncate")
	}
	noEOF := filepath.Join(truncated, "no-eof.bam")
	if err := os.WriteFile(noEOF, data[:len(data)-28], 0o644); err != nil {
		return nil, nil, err
	}
	code, detail := quickcheck(noEOF)
	add("truncated-bgzf", "quickcheck rejects", code != 0, fmt.Sprintf("exit %d %s", code, detail))
	cutRecord := filepath.Join(truncated, "cut-record.bam")
	if err := os.WriteFile(cutRecord, data[:len(data)/2], 0o644); err != nil {
		return nil, nil, err
	}
	code, detail = quickcheck(cutRecord)
	add("truncated-record", "quickcheck rejects", code != 0, fmt.Sprintf("exit %d %s", code, detail))

	faults, err := runFaults(base)
	if err != nil {
		return nil, nil, err
	}
	return controls, faults, nil
}

func runFaults(m Manifest) ([]Fault, error) {
	if len(m.Records) == 0 {
		return nil, errors.New("fault controls need one record")
	}
	record := m.Records[0].bamRecord()
	run := func(w io.Writer) error {
		bw := bam.NewWriter(w)
		if err := bw.WriteHeader(m.Header); err != nil {
			return err
		}
		if err := bw.WriteRecord(record); err != nil {
			return err
		}
		return bw.Close()
	}
	var faults []Fault
	check := func(name, injected string, err error) {
		observed := "no error"
		if err != nil {
			observed = err.Error()
		}
		faults = append(faults, Fault{Name: name, Injected: injected,
			Observed: observed, Passed: err != nil})
	}
	check("short-write", "underlying writer returns n-1, nil", run(faultWriter{short: true}))
	check("enospc", "underlying writer returns ENOSPC", run(faultWriter{err: syscall.ENOSPC}))
	check("broken-pipe", "underlying writer returns EPIPE", run(faultWriter{err: syscall.EPIPE}))

	var buf bytes.Buffer
	bw := bam.NewWriter(&buf)
	if err := bw.WriteHeader(m.Header); err != nil {
		return nil, err
	}
	if err := bw.WriteRecord(record); err != nil {
		return nil, err
	}
	if err := bw.Close(); err != nil {
		return nil, err
	}
	second := bw.Close()
	secondObserved := "no error"
	if second != nil {
		secondObserved = second.Error()
	}
	faults = append(faults, Fault{Name: "double-close", Injected: "Close called twice",
		Observed: secondObserved, Passed: second == nil})
	after := bw.WriteRecord(record)
	faults = append(faults, Fault{Name: "write-after-close", Injected: "WriteRecord after Close",
		Observed: errorDetail(after), Passed: after != nil})
	return faults, nil
}

func recordsEqual(a, b []Record) bool { return reflect.DeepEqual(a, b) }

func cloneManifest(m Manifest) Manifest {
	out := m
	if m.Header.ReadGroup != nil {
		rg := *m.Header.ReadGroup
		out.Header.ReadGroup = &rg
	}
	out.Header.References = append([]bam.Reference(nil), m.Header.References...)
	out.Records = make([]Record, len(m.Records))
	for i, r := range m.Records {
		out.Records[i] = r
		out.Records[i].Quality = append(Phred(nil), r.Quality...)
		out.Records[i].Aux = make([]Aux, len(r.Aux))
		for j, a := range r.Aux {
			out.Records[i].Aux[j] = a
			switch v := a.Value.(type) {
			case Array:
				out.Records[i].Aux[j].Value = append(Array(nil), v...)
			case IntArray:
				out.Records[i].Aux[j].Value = append(IntArray(nil), v...)
			case FloatArray:
				out.Records[i].Aux[j].Value = append(FloatArray(nil), v...)
			}
		}
	}
	return out
}
