package bamverify

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
)

const testRG = "run_model"

func syntheticManifest() Manifest {
	rg := &bam.ReadGroup{
		ID: testRG, FlowCellID: "PAW70337", DeviceID: "PC48LILITH",
		ExperimentStartTime: "2024-06-19T12:36:05.543000+00:00",
		RunID:               "run", BasecallModel: "model", ExperimentID: "exp",
		AcquisitionStartTime: "2024-06-19T12:42:14.753000+00:00",
		SampleID:             "sample", ModelStride: 6, TrimFlags: "adapter,primer",
	}
	header := bam.Header{
		Version: "1.6", SortOrder: "unknown",
		Program: bam.Program{ID: "basecaller", Name: "dorado", Version: "2.1.2+8b8fc5d",
			CommandLine: `dorado "basecaller" "model" "reads.pod5"`, Description: "gpu:GPU"},
		ReadGroup: rg,
	}
	base := func(name, seq string, q byte) []Aux {
		return []Aux{
			{Name: "qs", Type: 'f', Value: float32(20.435345)},
			{Name: "du", Type: 'f', Value: float32(5.8532)},
			{Name: "ns", Type: 'i', Value: int32(29266)},
			{Name: "ts", Type: 'i', Value: int32(682)},
			{Name: "mx", Type: 'i', Value: int32(1)},
			{Name: "ch", Type: 'i', Value: int32(2372)},
			{Name: "st", Type: 'Z', Value: "2024-06-19T21:36:39.995000+00:00"},
			{Name: "rn", Type: 'i', Value: int32(4484)},
			{Name: "fn", Type: 'Z', Value: "PAW.pod5"},
			{Name: "sm", Type: 'f', Value: float32(93.99999)},
			{Name: "sd", Type: 'f', Value: float32(24)},
			{Name: "sv", Type: 'Z', Value: "pa"},
			{Name: "dx", Type: 'i', Value: int32(0)},
			{Name: "RG", Type: 'Z', Value: testRG},
			{Name: "po", Type: 'Z', Value: "not_set"},
			{Name: "er", Type: 'Z', Value: "signal_positive"},
			{Name: "me", Type: 'I', Value: uint32(3653)},
		}
	}
	first := base("read-one", "ACGTN", 40)
	first = append(first,
		Aux{Name: "pa", Type: 'B', Value: IntArray{1, -2, 3}},
		Aux{Name: "bi", Type: 'B', Value: FloatArray{1.5, -2.25}},
		Aux{Name: "mv", Type: 'B', Value: Array{6, 0, 1, 1, 0}},
	)
	records := []Record{
		{Name: "read-one", Flag: 4, Sequence: "ACGTN", Quality: Phred{40, 40, 40, 40, 40}, Aux: first},
		{Name: "read-two", Flag: 4, Sequence: "TTTT", Quality: Phred{1, 2, 3, 4}, Aux: base("read-two", "TTTT", 0)},
		{Name: "read-three", Flag: 4, Sequence: "GATTACA", Quality: Phred{30, 31, 32, 33, 34, 35, 36}, Aux: base("read-three", "GATTACA", 0)},
	}
	return Manifest{Schema: ManifestSchema, SourceBAM: "synthetic.bam",
		SourceBAMHash: strings.Repeat("0", 64), Header: header, Records: records}
}

// writeSyntheticOracle writes the seed with the Go writer, then re-binds the
// manifest to the file's hash. The source pin is metadata only, so the bytes do
// not change.
func writeSyntheticOracle(t *testing.T, dir string) (path, hash string, manifestBytes []byte) {
	t.Helper()
	seed := syntheticManifest()
	path = filepath.Join(dir, "oracle.bam")
	data, err := writeBAMFile(seed, path)
	if err != nil {
		t.Fatal(err)
	}
	hash = HashBytes(data)
	bound := seed
	bound.SourceBAM = path
	bound.SourceBAMHash = hash
	manifestBytes, err = bound.Encode()
	if err != nil {
		t.Fatal(err)
	}
	return path, hash, manifestBytes
}

func TestParseBAMRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path, hash, _ := writeSyntheticOracle(t, dir)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseBAM(data, ParseOptions{SourceBAM: path, SourceHash: hash})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.HeaderSHA256 != HashBytes([]byte(parsed.HeaderText)) {
		t.Fatal("header hash is not the header text hash")
	}
	if parsed.PayloadSHA256 != HashBytes(mustDecode(t, data)) {
		t.Fatal("payload hash is not the inflated payload hash")
	}
	if len(parsed.Members) != 2 {
		t.Fatalf("members %d", len(parsed.Members))
	}
	if len(parsed.Manifest.Records) != 3 {
		t.Fatalf("records %d", len(parsed.Manifest.Records))
	}
	seed := syntheticManifest()
	if !reflect.DeepEqual(parsed.Manifest.Records, seed.Records) {
		t.Fatalf("records did not round-trip\n got %#v\nwant %#v", parsed.Manifest.Records, seed.Records)
	}
	if !reflect.DeepEqual(parsed.Manifest.Header, seed.Header) {
		t.Fatalf("header did not round-trip: %#v", parsed.Manifest.Header)
	}
	// Exact float bits, not decimal text.
	if got := math.Float32bits(parsed.Manifest.Records[0].Aux[0].Value.(float32)); got != math.Float32bits(float32(20.435345)) {
		t.Fatalf("qs bits 0x%08x", got)
	}
	if got := parsed.Manifest.Records[0].Aux[17].Value.(IntArray); !reflect.DeepEqual(got, IntArray{1, -2, 3}) {
		t.Fatalf("pa %v", got)
	}
	if got := parsed.Manifest.Records[0].Aux[18].Value.(FloatArray); !reflect.DeepEqual(got, FloatArray{1.5, -2.25}) {
		t.Fatalf("bi %v", got)
	}
}

func mustDecode(t *testing.T, data []byte) []byte {
	t.Helper()
	payload, _, err := decodeBGZF(data)
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func TestParseBAMRejectsLooseInput(t *testing.T) {
	dir := t.TempDir()
	path, hash, _ := writeSyntheticOracle(t, dir)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	opts := ParseOptions{SourceBAM: path, SourceHash: hash}
	if _, err := ParseBAM(data, ParseOptions{SourceBAM: path, SourceHash: "short"}); err == nil {
		t.Fatal("unpinned options accepted")
	}
	if _, err := ParseBAM(data[:len(data)-28], opts); err == nil {
		t.Fatal("missing EOF marker accepted")
	}
	if _, err := ParseBAM(data[:len(data)/2], opts); err == nil {
		t.Fatal("truncated BAM accepted")
	}
	badMagic := append([]byte(nil), data...)
	badMagic[0] = 0
	if _, err := ParseBAM(badMagic, opts); err == nil {
		t.Fatal("bad magic accepted")
	}
	// Patch a record flag in the inflated payload, then re-frame with the writer.
	payload := mustDecode(t, data)
	flagAt := headerEnd(t, payload) + 4 + 14
	payload[flagAt] = 4 | 16
	patched := reframe(t, payload)
	if _, err := ParseBAM(patched, opts); err == nil {
		t.Fatal("mapped flag accepted")
	}
}

func headerEnd(t *testing.T, payload []byte) int {
	t.Helper()
	lText := int(int32(binaryLE(payload[4:8])))
	nRef := int(int32(binaryLE(payload[8+lText:])))
	off := 8 + lText + 4
	for i := 0; i < nRef; i++ {
		lName := int(int32(binaryLE(payload[off:])))
		off += 4 + lName + 4
	}
	return off
}

func binaryLE(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func bamBlock(data []byte) ([]byte, error) {
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
	total := 18 + len(payload) + 8
	out := make([]byte, 0, total)
	out = append(out, 0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00)
	out = append(out, byte(total-1), byte((total-1)>>8))
	out = append(out, payload...)
	var footer [8]byte
	binary.LittleEndian.PutUint32(footer[:4], crc32.ChecksumIEEE(data))
	binary.LittleEndian.PutUint32(footer[4:], uint32(len(data)))
	out = append(out, footer[:]...)
	return out, nil
}

// reframe wraps an uncompressed payload in a single BGZF member plus EOF.
func reframe(t *testing.T, payload []byte) []byte {
	t.Helper()
	// Reuse the writer by writing a header-only BAM is not possible here; use
	// the package's member builder through a tiny manifest-free path.
	var out bytes.Buffer
	for len(payload) > 0 {
		n := min(len(payload), 0xff00)
		block, err := bamBlock(payload[:n])
		if err != nil {
			t.Fatal(err)
		}
		out.Write(block)
		payload = payload[n:]
	}
	out.Write(bgzfEOFMarker)
	return out.Bytes()
}

func TestManifestJSONRoundTrip(t *testing.T) {
	seed := syntheticManifest()
	body, err := seed.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"quality": [`)) || !bytes.Contains(body, []byte(`"subtype": "c"`)) ||
		!bytes.Contains(body, []byte(`"subtype": "i"`)) || !bytes.Contains(body, []byte(`"subtype": "f"`)) {
		t.Fatalf("manifest JSON lost a typed value:\n%s", body)
	}
	decoded, err := DecodeManifest(body)
	if err != nil {
		t.Fatal(err)
	}
	again, err := decoded.Encode()
	if err != nil || !bytes.Equal(body, again) {
		t.Fatalf("manifest is not stable: %v", err)
	}
	if !reflect.DeepEqual(decoded.Records, seed.Records) {
		t.Fatal("records changed through JSON")
	}
	for name, data := range map[string][]byte{
		"unknown":  append([]byte(`{"extra":1,`), body[1:]...),
		"trailing": append(append([]byte(nil), body...), '{', '}'),
		"schema":   bytes.Replace(body, []byte(`"bam-manifest/1"`), []byte(`"bam-manifest/2"`), 1),
		"hash":     bytes.Replace(body, []byte(`"source_bam_sha256": "`), []byte(`"source_bam_sha256": "zz`), 1),
		"subtype":  bytes.Replace(body, []byte(`"subtype": "c"`), []byte(`"subtype": "C"`), 1),
	} {
		if _, err := DecodeManifest(data); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestManifestValidation(t *testing.T) {
	seed := syntheticManifest()
	if err := seed.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Manifest){
		"schema":       func(m *Manifest) { m.Schema = "x" },
		"pin":          func(m *Manifest) { m.SourceBAM = "" },
		"hash":         func(m *Manifest) { m.SourceBAMHash = "abc" },
		"flag":         func(m *Manifest) { m.Records[0].Flag = 0 },
		"quality":      func(m *Manifest) { m.Records[0].Quality = m.Records[0].Quality[:2] },
		"sequence":     func(m *Manifest) { m.Records[0].Sequence = "" },
		"duplicate":    func(m *Manifest) { m.Records[0].Aux[1] = m.Records[0].Aux[0] },
		"bad-tag":      func(m *Manifest) { m.Records[0].Aux[0].Name = "q" },
		"float-type":   func(m *Manifest) { m.Records[0].Aux[0].Value = int32(1) },
		"array-type":   func(m *Manifest) { m.Records[0].Aux[17].Value = []int32{1} },
		"dangling-rg":  func(m *Manifest) { m.Records[0].Aux[13].Value = "other" },
		"missing-rg":   func(m *Manifest) { m.Records[0].Aux = append(m.Records[0].Aux[:13], m.Records[0].Aux[14:]...) },
		"header-sort":  func(m *Manifest) { m.Header.SortOrder = "coordinate" },
		"header-ver":   func(m *Manifest) { m.Header.Version = "1.5" },
		"header-empty": func(m *Manifest) { m.Header.Program.ID = "" },
	} {
		m := cloneManifest(seed)
		mutate(&m)
		if err := m.Validate(); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestPlanValidation(t *testing.T) {
	good := Plan{Schema: PlanSchema, OracleBAM: "x.bam", OracleBAMHash: strings.Repeat("a", 64),
		ExpectedRecords: 1, ExpectedAuxOrder: []string{"qs"}, RequirePayloadIdentity: true}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Plan){
		"schema":    func(p *Plan) { p.Schema = "x" },
		"oracle":    func(p *Plan) { p.OracleBAM = "" },
		"hash":      func(p *Plan) { p.OracleBAMHash = "abc" },
		"records":   func(p *Plan) { p.ExpectedRecords = 0 },
		"order":     func(p *Plan) { p.ExpectedAuxOrder = nil },
		"tag":       func(p *Plan) { p.ExpectedAuxOrder = []string{"q"} },
		"duplicate": func(p *Plan) { p.ExpectedAuxOrder = []string{"qs", "qs"} },
		"no-payload": func(p *Plan) {
			p.RequirePayloadIdentity = false
		},
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
		t.Fatal("samtools is required for the BAM verifier")
	}
	dir := t.TempDir()
	oracle, hash, manifestBytes := writeSyntheticOracle(t, dir)
	attempt := filepath.Join(dir, "attempt")
	if err := os.Mkdir(attempt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(attempt, "manifest.json"), manifestBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	plan := Plan{Schema: PlanSchema, OracleBAM: oracle, OracleBAMHash: hash,
		ExpectedRecords: 3, ExpectedAuxOrder: syntheticManifest().Records[0].AuxNames(),
		RequirePayloadIdentity: true}
	planBody, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(dir, "plan.json")
	if err := os.WriteFile(planPath, planBody, 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	receipt, err := Verify(planPath, attempt, out)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Payload.Claimed || !receipt.Payload.Identical {
		t.Fatalf("payload verdict %+v", receipt.Payload)
	}
	if !receipt.Canonical.CanonicalEqual || len(receipt.Controls) != 12 || len(receipt.Faults) != 5 {
		t.Fatalf("receipt controls %d faults %d canonical %+v", len(receipt.Controls), len(receipt.Faults), receipt.Canonical)
	}
	for _, c := range receipt.Checks {
		if !c.Passed {
			t.Fatalf("check %s failed: %+v", c.Name, c)
		}
	}
	for _, c := range receipt.Controls {
		if !c.Passed {
			t.Fatalf("control %s failed: %+v", c.Name, c)
		}
	}
	for _, f := range receipt.Faults {
		if !f.Passed {
			t.Fatalf("fault %s failed: %+v", f.Name, f)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "receipt.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(planPath, attempt, out); err == nil {
		t.Fatal("reused output directory accepted")
	}
	mutated := bytes.Replace(manifestBytes, []byte(`"value": 29266`), []byte(`"value": 29267`), 1)
	if bytes.Equal(mutated, manifestBytes) {
		t.Fatal("test mutation did not apply")
	}
	if err := os.WriteFile(filepath.Join(attempt, "manifest.json"), mutated, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(planPath, attempt, filepath.Join(dir, "out2")); err == nil {
		t.Fatal("mutated attempt manifest accepted")
	}
}
