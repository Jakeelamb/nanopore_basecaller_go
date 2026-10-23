package bamverify

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const testSAM = `@HD	VN:1.6	SO:unknown
@PG	ID:basecaller	PN:dorado	VN:2.1.2+8b8fc5d	CL:dorado "basecaller" "model" "reads.pod5"	DS:gpu:GPU
@PG	ID:samtools	PN:samtools	PP:basecaller	VN:1.24	CL:samtools view -h out
@RG	ID:run_model	PU:PAW70337	PM:PC48LILITH	DT:2024-06-19T12:36:05.543000+00:00	PL:ONT	DS:runid=run basecall_model=model experiment_id=exp acquisition_start_time=2024-06-19T12:42:14.753000+00:00 model_stride=6	LB:sample	tm:adapter,primer
bb76c31a-36f0-48a0-abd2-107a10e1ade0	4	*	0	0	*	*	0	0	ACGTN	IIIII	qs:f:20.4353	du:f:5.8532	ns:i:29266	ts:i:682	mx:i:1	ch:i:2372	st:Z:2024-06-19T21:36:39.995000+00:00	rn:i:4484	fn:Z:PAW.pod5	sm:f:94	sd:f:24	sv:Z:pa	dx:i:0	RG:Z:run_model	po:Z:not_set	er:Z:signal_positive	me:i:3653
`

func testParse(t *testing.T) ParseResult {
	t.Helper()
	parsed, err := ParseSAM(strings.NewReader(testSAM), ParseOptions{SourceSAM: "oracle.sam",
		SourceHash: HashBytes([]byte(testSAM)), DropProgramPrefixes: []string{"samtools"}})
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}

func TestParseSAMDerivesPinnedShape(t *testing.T) {
	parsed := testParse(t)
	if len(parsed.DroppedPrograms) != 1 || !strings.HasPrefix(parsed.DroppedPrograms[0], "@PG\tID:samtools") {
		t.Fatalf("dropped %v", parsed.DroppedPrograms)
	}
	m := parsed.Manifest
	if m.Header.Version != "1.6" || m.Header.SortOrder != "unknown" || m.Header.Program.ID != "basecaller" ||
		m.Header.Program.Description != "gpu:GPU" || m.Header.ReadGroup == nil || m.Header.ReadGroup.ModelStride != 6 {
		t.Fatalf("header %+v", m.Header)
	}
	if len(m.Records) != 1 || m.Records[0].Flag != 4 || m.Records[0].Sequence != "ACGTN" {
		t.Fatalf("record %+v", m.Records)
	}
	want := []string{"qs", "du", "ns", "ts", "mx", "ch", "st", "rn", "fn", "sm", "sd", "sv", "dx", "RG", "po", "er", "me"}
	if got := m.Records[0].AuxNames(); !equalStrings(got, want) {
		t.Fatalf("tag order %v", got)
	}
	if m.Records[0].Aux[0].Value != float32(20.4353) {
		t.Fatalf("qs %v", m.Records[0].Aux[0].Value)
	}
	if q := m.Records[0].Quality; len(q) != 5 || q[0] != 40 {
		t.Fatalf("quality %v", q)
	}
}

func TestManifestRoundTripAndBAM(t *testing.T) {
	parsed := testParse(t)
	body, err := parsed.Manifest.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"quality": [`)) {
		t.Fatal("quality is not a number array")
	}
	decoded, err := DecodeManifest(body)
	if err != nil {
		t.Fatal(err)
	}
	again, err := decoded.Encode()
	if err != nil || !bytes.Equal(body, again) {
		t.Fatalf("manifest is not stable: %v", err)
	}
	var out bytes.Buffer
	if err := decoded.WriteBAM(&out); err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(out.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(zr)
	if err != nil || !bytes.HasPrefix(data, []byte("BAM\x01")) {
		t.Fatalf("bam payload %v", err)
	}
	if !bytes.HasSuffix(out.Bytes(), []byte{0x1f, 0x8b, 0x08, 0x04}) {
		// The EOF marker is a complete empty member; check its full suffix.
		t.Log("checking full EOF marker below")
	}
	if !bytes.HasSuffix(out.Bytes(), []byte{0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}) {
		t.Fatal("BGZF EOF marker missing")
	}
}

func TestParseSAMRejectsLooseInput(t *testing.T) {
	base := testSAM
	cases := map[string]string{
		"unknown-header":    strings.Replace(base, "@HD\tVN:1.6\tSO:unknown", "@HD\tVN:1.6\tSO:unknown\tGO:none", 1),
		"unknown-tag":       strings.Replace(base, "er:Z:signal_positive", "er:Z:signal_positive\tzz:H:00", 1),
		"duplicate-tag":     strings.Replace(base, "er:Z:signal_positive", "er:Z:signal_positive\ter:Z:again", 1),
		"mapped":            strings.Replace(base, "\t4\t*\t0\t0\t*", "\t0\tchr1\t1\t0\t*", 1),
		"no-quality":        strings.Replace(base, "\tACGTN\tIIIII\t", "\tACGTN\t*\t", 1),
		"two-callers":       strings.Replace(base, "@PG\tID:samtools", "@PG\tID:other\tPN:dorado\tVN:1\tCL:x\n@PG\tID:samtools", 1),
		"pp-link":           strings.Replace(base, "DS:gpu:GPU", "DS:gpu:GPU\tPP:other", 1),
		"bad-rg":            strings.Replace(base, "PL:ONT", "PL:ILLUMINA", 1),
		"bad-ds":            strings.Replace(base, " experiment_id=exp", " unknown_key=1", 1),
		"header-after":      base + "@HD\tVN:1.6\tSO:unknown\n",
		"bad-array-subtype": strings.Replace(base, "er:Z:signal_positive", "er:B:f,1,2", 1),
		"bad-float":         strings.Replace(base, "qs:f:20.4353", "qs:f:nan", 1),
	}
	for name, sam := range cases {
		if _, err := ParseSAM(strings.NewReader(sam), ParseOptions{SourceSAM: "oracle.sam",
			SourceHash: HashBytes([]byte(sam)), DropProgramPrefixes: []string{"samtools"}}); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := ParseSAM(strings.NewReader(testSAM), ParseOptions{SourceSAM: "oracle.sam",
		SourceHash: "short", DropProgramPrefixes: []string{"samtools"}}); err == nil {
		t.Fatal("unpinned options accepted")
	}
}

func TestDecodeManifestRejectsLooseJSON(t *testing.T) {
	body, err := testParse(t).Manifest.Encode()
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{
		"unknown":  append([]byte(`{"extra":1,`), body[1:]...),
		"trailing": append(append([]byte(nil), body...), '{', '}'),
		"schema":   bytes.Replace(body, []byte(`"bam-manifest/1"`), []byte(`"bam-manifest/2"`), 1),
		"hash":     bytes.Replace(body, []byte(`"source_sam_sha256": "`), []byte(`"source_sam_sha256": "zz`), 1),
	} {
		if _, err := DecodeManifest(data); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestNormalizeRequiresDeclaredLines(t *testing.T) {
	out, err := Normalize([]byte(testSAM), []string{"@PG\tID:samtools\tPN:samtools\tPP:basecaller\tVN:1.24\tCL:samtools view -h out"})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(out, []byte("ID:samtools")) || !bytes.Contains(out, []byte("ID:basecaller")) {
		t.Fatalf("normalized %q", out)
	}
	if _, err := Normalize([]byte(testSAM), []string{"@PG\tID:missing"}); err == nil {
		t.Fatal("missing declared line accepted")
	}
}

func TestPlanValidation(t *testing.T) {
	good := Plan{Schema: PlanSchema, OracleSAM: "x.sam", OracleSAMHash: strings.Repeat("a", 64),
		ExpectedRecords: 1, ExpectedAuxOrder: []string{"qs"}}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Plan){
		"schema":    func(p *Plan) { p.Schema = "x" },
		"oracle":    func(p *Plan) { p.OracleSAM = "" },
		"hash":      func(p *Plan) { p.OracleSAMHash = "abc" },
		"records":   func(p *Plan) { p.ExpectedRecords = 0 },
		"order":     func(p *Plan) { p.ExpectedAuxOrder = nil },
		"tag":       func(p *Plan) { p.ExpectedAuxOrder = []string{"q"} },
		"duplicate": func(p *Plan) { p.ExpectedAuxOrder = []string{"qs", "qs"} },
		"raw-claim": func(p *Plan) { p.RequireRawIdentity = true },
	} {
		p := good
		mutate(&p)
		if err := p.Validate(); err == nil {
			t.Fatalf("%s plan accepted", name)
		}
	}
}

func TestVerifySyntheticOracle(t *testing.T) {
	if _, err := exec.LookPath("samtools"); err != nil {
		t.Skip("samtools is not installed")
	}
	dir := t.TempDir()
	oracle := filepath.Join(dir, "oracle.sam")
	if err := os.WriteFile(oracle, []byte(testSAM), 0o644); err != nil {
		t.Fatal(err)
	}
	plan := Plan{Schema: PlanSchema, OracleSAM: oracle, OracleSAMHash: HashBytes([]byte(testSAM)),
		DropProgramPrefixes: []string{"samtools"}, ExpectedRecords: 1,
		ExpectedAuxOrder: []string{"qs", "du", "ns", "ts", "mx", "ch", "st", "rn", "fn", "sm", "sd", "sv", "dx", "RG", "po", "er", "me"}}
	planBody, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(dir, "plan.json")
	if err := os.WriteFile(planPath, planBody, 0o644); err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseSAM(strings.NewReader(testSAM), ParseOptions{SourceSAM: oracle,
		SourceHash: plan.OracleSAMHash, DropProgramPrefixes: plan.DropProgramPrefixes})
	if err != nil {
		t.Fatal(err)
	}
	attempt := filepath.Join(dir, "attempt")
	if err := os.Mkdir(attempt, 0o755); err != nil {
		t.Fatal(err)
	}
	body, err := parsed.Manifest.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(attempt, "manifest.json"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	receipt, err := Verify(planPath, attempt, out)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Canonical.CanonicalEqual || receipt.RawBAM.Claimed || len(receipt.Controls) != 8 {
		t.Fatalf("receipt %+v", receipt)
	}
	for _, c := range receipt.Controls {
		if !c.Passed {
			t.Fatalf("control %s failed: %+v", c.Name, c)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "receipt.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(planPath, attempt, out); err == nil {
		t.Fatal("reused output directory accepted")
	}
	mutated := bytes.Replace(body, []byte(`"value": 2372`), []byte(`"value": 2373`), 1)
	if bytes.Equal(mutated, body) {
		t.Fatal("test mutation did not apply")
	}
	if err := os.WriteFile(filepath.Join(attempt, "manifest.json"), mutated, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(planPath, attempt, filepath.Join(dir, "out2")); err == nil {
		t.Fatal("mutated attempt manifest accepted")
	}
}
