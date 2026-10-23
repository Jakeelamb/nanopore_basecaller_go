package gpuprotocol

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func frozenPlan(t *testing.T) Plan {
	t.Helper()
	f, err := os.Open("../../provenance/gpu-acceptance-plan.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestFrozenProtocol(t *testing.T) {
	if err := Check(frozenPlan(t)); err != nil {
		t.Fatal(err)
	}
}

func TestRejectWeakenedProtocol(t *testing.T) {
	for name, change := range map[string]func(*Plan){
		"HG005_selection":        func(p *Plan) { p.Cohorts[0].Sample = "HG005" },
		"membership_denominator": func(p *Plan) { p.Cohorts[0].Parents-- },
		"reference_only_ALT":     func(p *Plan) { p.Endpoints[8].Rule = p.Endpoints[7].Rule },
		"relaxed_literal_gate":   func(p *Plan) { p.Endpoints[6].Rule = "row multiset equality" },
		"posthoc_tolerance":      func(p *Plan) { p.Policies["storage"] = "tolerance after mismatches" },
		"historical_clocks":      func(p *Plan) { p.Policies["timing"] = "historical stock clocks are fresh matched data" },
		"armed":                  func(p *Plan) { p.State = "accepted" },
		"missing_pin":            func(p *Plan) { delete(p.References, "accuracy_inputs") },
		"missing_gate":           func(p *Plan) { p.Deferred = p.Deferred[1:] },
	} {
		t.Run(name, func(t *testing.T) {
			p := frozenPlan(t)
			change(&p)
			if Check(p) == nil {
				t.Fatal("accepted corrupted protocol")
			}
		})
	}
}

func TestRejectInvalidJSON(t *testing.T) {
	for _, input := range []string{`{"unknown":1}`, `{} {}`, `null false`, `{`} {
		if _, err := Decode(strings.NewReader(input)); err == nil {
			t.Fatalf("accepted %q", input)
		}
	}
}

func TestGenericPinReaderAndCorruption(t *testing.T) {
	p := frozenPlan(t)
	dir := t.TempDir()
	write := func(name, data string) Pin {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256([]byte(data))
		return Pin{path, hex.EncodeToString(h[:]), int64(len(data))}
	}
	for name := range p.References {
		p.References[name] = write(name, "{}\n")
	}
	for i, c := range p.Cohorts {
		var ids strings.Builder
		for j := 0; j < c.Parents; j++ {
			fmt.Fprintf(&ids, "%08x-0000-0000-0000-%012x\n", j, i+1)
		}
		p.Cohorts[i].Membership = write(c.Role, ids.String())
	}
	if err := verifyPinSet(p.References, p.Cohorts); err != nil {
		t.Fatal(err)
	}
	pin := p.References["source_build"]
	if err := os.WriteFile(pin.Path, []byte("[]\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if verifyPinSet(p.References, p.Cohorts) == nil {
		t.Fatal("accepted changed bytes of equal size")
	}
}

func testPin(t *testing.T, name, content string) Pin {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256([]byte(content))
	return Pin{path, hex.EncodeToString(h[:]), int64(len(content))}
}

func TestRejectReplacementWithRecomputedPin(t *testing.T) {
	for key := range approvedReferences() {
		t.Run(key, func(t *testing.T) {
			p := frozenPlan(t)
			p.References[key] = testPin(t, key, "{}\n")
			if Check(p) == nil || VerifyPins(p) == nil {
				t.Fatal("replacement reference authorized")
			}
		})
	}
	for i := 0; i < 2; i++ {
		t.Run(fmt.Sprintf("membership-%d", i), func(t *testing.T) {
			p := frozenPlan(t)
			var ids strings.Builder
			for j := 0; j < p.Cohorts[i].Parents; j++ {
				fmt.Fprintf(&ids, "%08x-0000-0000-0000-%012x\n", j, i+1)
			}
			p.Cohorts[i].Membership = testPin(t, "members", ids.String())
			if Check(p) == nil || VerifyPins(p) == nil {
				t.Fatal("replacement membership authorized")
			}
		})
	}
}

func TestStrictUUIDAndDisjointMemberships(t *testing.T) {
	const a = "01234567-89ab-cdef-0123-456789abcdef"
	const b = "11234567-89ab-cdef-0123-456789abcdef"
	for name, pair := range map[string][2]string{
		"positive":        {a, b},
		"overlap":         {a, a},
		"nonhex":          {a, "g1234567-89ab-cdef-0123-456789abcdef"},
		"hyphen_position": {a, "1123456-789ab-cdef-0123-456789abcdef"},
		"uppercase":       {a, strings.ToUpper(b)},
		"short":           {a, "11234567"},
	} {
		t.Run(name, func(t *testing.T) {
			cohorts := []Cohort{
				{Parents: 1, Membership: testPin(t, "dev", pair[0]+"\n")},
				{Parents: 1, Membership: testPin(t, "val", pair[1]+"\n")},
			}
			err := verifyPinSet(nil, cohorts)
			if (err == nil) != (name == "positive") {
				t.Fatalf("unexpected result %v", err)
			}
		})
	}
	p := testPin(t, "duplicates", a+"\n"+a+"\n")
	if verifyPinSet(nil, []Cohort{{Parents: 2, Membership: p}}) == nil {
		t.Fatal("duplicate UUID accepted")
	}
}

func TestRejectAmbiguousAndIncompleteJSONAtEveryLevel(t *testing.T) {
	b, err := json.Marshal(frozenPlan(t))
	if err != nil {
		t.Fatal(err)
	}
	var corruptions func(json.RawMessage) []json.RawMessage
	marshal := func(v any) json.RawMessage {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	corruptions = func(raw json.RawMessage) []json.RawMessage {
		out := []json.RawMessage{json.RawMessage("null")}
		switch raw[0] {
		case '{':
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(raw, &obj); err != nil {
				t.Fatal(err)
			}
			for key, value := range obj {
				delete(obj, key)
				out = append(out, marshal(obj))
				obj[key] = value
				delete(obj, key)
				obj[strings.ToUpper(key)] = value
				out = append(out, marshal(obj))
				delete(obj, strings.ToUpper(key))
				obj[key] = value
				encodedKey := marshal(key)
				duplicate := append(json.RawMessage{'{'}, encodedKey...)
				duplicate = append(duplicate, ':')
				duplicate = append(duplicate, value...)
				duplicate = append(duplicate, ',')
				duplicate = append(duplicate, raw[1:]...)
				out = append(out, duplicate)
				for _, child := range corruptions(value) {
					obj[key] = child
					out = append(out, marshal(obj))
				}
				obj[key] = value
			}
		case '[':
			var items []json.RawMessage
			if err := json.Unmarshal(raw, &items); err != nil {
				t.Fatal(err)
			}
			for i, value := range items {
				for _, child := range corruptions(value) {
					items[i] = child
					out = append(out, marshal(items))
				}
				items[i] = value
			}
		}
		return out
	}
	for i, bad := range corruptions(b) {
		if _, err := Decode(strings.NewReader(string(bad))); err == nil {
			t.Fatalf("accepted corruption %d: %s", i, bad)
		}
	}
}
