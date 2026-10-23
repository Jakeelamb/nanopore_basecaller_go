package gpuprotocol

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
)

type Pin struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

type Cohort struct {
	Sample     string `json:"sample"`
	Role       string `json:"role"`
	Flowcell   string `json:"flowcell"`
	Parents    int    `json:"parents"`
	Membership Pin    `json:"membership"`
}

type Endpoint struct {
	ID   string `json:"id"`
	Rule string `json:"rule"`
}

type Plan struct {
	Schema       int               `json:"schema"`
	State        string            `json:"state"`
	SourceCommit string            `json:"source_commit"`
	References   map[string]Pin    `json:"references"`
	Cohorts      []Cohort          `json:"cohorts"`
	Endpoints    []Endpoint        `json:"endpoints"`
	Policies     map[string]string `json:"policies"`
	Deferred     []string          `json:"deferred"`
}

func Endpoints() []Endpoint {
	return []Endpoint{
		{"engineering", "Go owns caller computation; named native kernels only; required tests execute without skips"},
		{"source13_bytes", "exact source-declared live and initialized bytes, guards and sentinels; source exclusions frozen before capture"},
		{"official12_bytes", "separate official CUDA12 oracle; never substitute source CUDA13 equality"},
		{"canonical_records", "all mandatory fields and typed tags; only record/tag order and validated provenance excluded"},
		{"headers", "separate exact header comparison and independently validated provenance differences"},
		{"raw_fastq_bam", "separate literal byte equality; canonical equality cannot replace it"},
		{"accuracy_artifacts", "literal artifact equality; historical six alternate-alignment row-order failures remain FAIL"},
		{"biological", "HG002 GIAB v4.2.1 reference-identical error (X+I+D)/(M+X+I+D); MAPQ 30..254; 100bp primary and 500bp sensitivity"},
		{"alt", "separate truth-ALT retention counts and paired parent uncertainty; any lost control-supported ALT observation fails; reference-only accuracy is no surrogate"},
		{"yield", "fixed parent UUID and signal-sample denominators; no additional absent parents or reduced evaluable parents/eligible-query exposure; report gained/lost sets"},
		{"native_process_timing", "fresh equal-work paired complete-process clocks; failed/partial attempts excluded from successful times and retained in failure rate"},
		{"workflow_timing", "separate outer basecall+fixed-map+sort clock; unavailable until qualified; never sum audit stage clocks"},
	}
}

func Policies() map[string]string {
	return map[string]string{
		"selection":     "HG002 development only; freeze candidate before single validation decision; HG005 forbidden",
		"order":         "membership file byte order and original parent sample counts fixed; children remain with parent",
		"holdout":       "validation PAW71238 used once after registration; no HG005 selection or tuning",
		"denominators":  "fixed original parents including missing and C=0; no invented perfect accuracy; report exposure and common support",
		"uncertainty":   "paired parent bootstrap 20000 seed 20261023; zero denominators reported; paired timing blocks within session; no donor-generalization claim",
		"timing":        "unarmed pending frozen arm/block/session schedule, device, precision, batch, cache, affinity, profile, cooldown, mapper and output policy; no invented speed target",
		"receipts":      "exact source/input/model/library/consumer pins, argv, UTC start/end, exit, raw logs and payload hashes; preserve failed attempts",
		"reuse":         "reuse only if every declared input pin matches; repair observers from retained bytes; no recapture for reports",
		"storage":       "audit exact live bytes and source-declared initialized ranges; no mismatch-derived masks or posthoc tolerance",
		"qualification": "genuine retained controls plus corrupted-input rejection and independent review before expensive cohorts; source-only cannot close runtime gates",
		"historical":    "source13 and official12 distinct; eight numerical score equalities do not erase six literal artifact FAILs; stock clocks never fresh matched timing",
	}
}

// Decode rejects ambiguous or incomplete representations before typed decoding.
func Decode(r io.Reader) (Plan, error) {
	var p Plan
	b, err := io.ReadAll(io.LimitReader(r, (1<<20)+1))
	if err != nil {
		return p, err
	}
	if len(b) > 1<<20 {
		return p, fmt.Errorf("protocol exceeds 1MiB")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := strictValue(d, reflect.TypeOf(p), ""); err != nil {
		return p, err
	}
	if _, err := d.Token(); err != io.EOF {
		return p, fmt.Errorf("expected exactly one JSON object")
	}
	err = json.Unmarshal(b, &p)
	return p, err
}

func strictValue(d *json.Decoder, typ reflect.Type, path string) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	if token == nil {
		return fmt.Errorf("null at %s", path)
	}
	switch typ.Kind() {
	case reflect.Struct, reflect.Map:
		if token != json.Delim('{') {
			return fmt.Errorf("object required at %s", path)
		}
		fields := make(map[string]reflect.Type)
		if typ.Kind() == reflect.Struct {
			for i := 0; i < typ.NumField(); i++ {
				f := typ.Field(i)
				fields[f.Tag.Get("json")] = f.Type
			}
		} else {
			switch path {
			case "/references":
				for key := range approvedReferences() {
					fields[key] = typ.Elem()
				}
			case "/policies":
				for key := range Policies() {
					fields[key] = typ.Elem()
				}
			default:
				return fmt.Errorf("unknown map at %s", path)
			}
		}
		seen := make(map[string]bool)
		for d.More() {
			keyToken, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return fmt.Errorf("invalid object key")
			}
			fieldType, known := fields[key]
			if !known || seen[key] {
				return fmt.Errorf("unknown, case-variant or duplicate field %s/%s", path, key)
			}
			seen[key] = true
			if err := strictValue(d, fieldType, path+"/"+key); err != nil {
				return err
			}
		}
		if _, err := d.Token(); err != nil {
			return err
		}
		if len(seen) != len(fields) {
			return fmt.Errorf("missing field at %s", path)
		}
	case reflect.Slice:
		if token != json.Delim('[') {
			return fmt.Errorf("array required at %s", path)
		}
		for d.More() {
			if err := strictValue(d, typ.Elem(), path+"/[]"); err != nil {
				return err
			}
		}
		if _, err := d.Token(); err != nil {
			return err
		}
	case reflect.String:
		if _, ok := token.(string); !ok {
			return fmt.Errorf("string required at %s", path)
		}
	case reflect.Int, reflect.Int64:
		n, ok := token.(json.Number)
		if !ok {
			return fmt.Errorf("integer required at %s", path)
		}
		if _, err := n.Int64(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported field type at %s", path)
	}
	return nil
}

// These identities freeze the seven retained controls reviewed for A00.
// They are independent of the supplied plan and cannot be replaced by its hashes.
func approvedReferences() map[string]Pin {
	return map[string]Pin{
		"stock_plan":           {"/home/jake/Projects/nanopore_basecaller/results/stock/controlled-20261022/plan.json", "b1a90f50b18dbf51b3f2d7ffa328749d79b62a5acc4db18a7b275585175a5cb1", 4045008},
		"accuracy_inputs":      {"/home/jake/Projects/nanopore_basecaller/results/accuracy/controlled-20261022/inputs.json", "15b477dbdf098187b33fcbe383f8e5b5cefa7646a805d256b476783aabcb9f3a", 80822},
		"source_qualification": {"/home/jake/Projects/nanopore_basecaller/provenance/verification/source-qualification-full-20261023.json", "844f3b46a4eede9eebbda871c2a4ffa51aa8ef11e7830ca6f41c5b01d5a743ae", 40131},
		"source_build":         {"/home/jake/Projects/nanopore_basecaller/provenance/source-build-20261023.json", "cc69dea84a6ab16ed68bc9f32586735e4c67f96c0b29370a7484d8b26cd47dac", 50085},
		"accuracy_comparison":  {"/home/jake/Projects/nanopore_basecaller/results/source-qualification/20261023-performance/accuracy-comparison-full.json", "c007f516d9c1055b52d07fa24f06b35f634cc356b6cdfdf1d86ba623de9382bc", 298433},
	}
}

func approvedMemberships() []Pin {
	return []Pin{
		{"/home/jake/Projects/nanopore_basecaller/data/splits/development.txt", "b809632eb09feb6074789754a4d9dddc3121937f926e2790f8129fd2482a7fdf", 96496},
		{"/home/jake/Projects/nanopore_basecaller/data/splits/validation.txt", "9fd091ec82a876738bdbe3d3f0f55c755979d999912c81a1d547e16be2b93beb", 74999},
	}
}

func Check(p Plan) error {
	if !reflect.DeepEqual(p.References, approvedReferences()) {
		return fmt.Errorf("unapproved retained reference identity")
	}
	if len(p.Cohorts) == 2 {
		for i, pin := range approvedMemberships() {
			if p.Cohorts[i].Membership != pin {
				return fmt.Errorf("unapproved membership identity")
			}
		}
	}
	if p.Schema != 1 || p.State != "prospective-unarmed" || p.SourceCommit != "8b8fc5d36a9c0baab262a743cb175b5878e38ca6" {
		return fmt.Errorf("schema, state or source commit changed")
	}
	if !reflect.DeepEqual(p.Endpoints, Endpoints()) || !reflect.DeepEqual(p.Policies, Policies()) {
		return fmt.Errorf("frozen endpoint or policy changed")
	}
	if len(p.Cohorts) != 2 {
		return fmt.Errorf("require development and validation only")
	}
	for i, want := range []struct {
		role, flowcell string
		parents        int
	}{{"development", "PAW70337", 2608}, {"validation", "PAW71238", 2027}} {
		c := p.Cohorts[i]
		if c.Sample != "HG002" || c.Role != want.role || c.Flowcell != want.flowcell || c.Parents != want.parents {
			return fmt.Errorf("cohort %d changed", i)
		}
		if err := checkPin(c.Membership); err != nil {
			return err
		}
	}
	for _, key := range []string{"stock_plan", "accuracy_inputs", "source_qualification", "source_build", "accuracy_comparison"} {
		pin, ok := p.References[key]
		if !ok {
			return fmt.Errorf("missing reference %s", key)
		}
		if err := checkPin(pin); err != nil {
			return err
		}
	}
	if len(p.References) != 5 {
		return fmt.Errorf("unexpected reference")
	}
	if !reflect.DeepEqual(p.Deferred, []string{"required_tests", "retained_control_ingestion", "corruption_rejection", "independent_review", "native_parity", "complete_input", "biological_alt", "matched_timing"}) {
		return fmt.Errorf("runtime gates must remain deferred")
	}
	return nil
}

func checkPin(p Pin) error {
	b, err := hex.DecodeString(p.SHA256)
	if err != nil || len(b) != sha256.Size || p.SHA256 != hex.EncodeToString(b) || p.Size <= 0 || !filepath.IsAbs(p.Path) {
		return fmt.Errorf("invalid pin %q", p.Path)
	}
	return nil
}

// VerifyPins validates approved identities before reading their retained bytes.
func VerifyPins(p Plan) error {
	if err := Check(p); err != nil {
		return err
	}
	return verifyPinSet(p.References, p.Cohorts)
}

// verifyPinSet is a generic reader, not production protocol authorization.
func verifyPinSet(references map[string]Pin, cohorts []Cohort) error {
	for _, pin := range references {
		b, err := readPin(pin)
		if err != nil {
			return err
		}
		if !json.Valid(b) {
			return fmt.Errorf("invalid retained JSON: %s", pin.Path)
		}
	}
	all := make(map[string]bool)
	for _, c := range cohorts {
		b, err := readPin(c.Membership)
		if err != nil {
			return err
		}
		ids := bytes.Fields(b)
		if len(ids) != c.Parents {
			return fmt.Errorf("membership count changed: %s", c.Membership.Path)
		}
		for _, id := range ids {
			key := string(id)
			if !validUUID(key) || all[key] {
				return fmt.Errorf("invalid, duplicate or overlapping parent in %s", c.Membership.Path)
			}
			all[key] = true
		}
	}
	return nil
}

func validUUID(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if s[i] != '-' {
				return false
			}
			continue
		}
		if !(s[i] >= '0' && s[i] <= '9' || s[i] >= 'a' && s[i] <= 'f') {
			return false
		}
	}
	return true
}

func readPin(pin Pin) ([]byte, error) {
	if err := checkPin(pin); err != nil {
		return nil, err
	}
	if pin.Size > 8<<20 {
		return nil, fmt.Errorf("manifest exceeds 8MiB limit: %s", pin.Path)
	}
	f, err := os.Open(pin.Path)
	if err != nil {
		return nil, err
	}
	b, err := io.ReadAll(io.LimitReader(f, pin.Size+1))
	closeErr := f.Close()
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	h := sha256.Sum256(b)
	if int64(len(b)) != pin.Size || hex.EncodeToString(h[:]) != pin.SHA256 {
		return nil, fmt.Errorf("pin mismatch: %s", pin.Path)
	}
	return b, nil
}
