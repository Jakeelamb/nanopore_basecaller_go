package packed

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func scalar(v float64) *float64 { return &v }

// Schema fixtures are synthetic validator inputs, never genuine native evidence.
func schemaRequest() Request {
	r := Request{Schema: RequestSchema, Case: "first", Split: "development", Dataset: "synthetic-schema-test", Geometry: FirstGeometry(), ExpectedOrder: []string{"a", "b", "c"}, Recipe: Recipe{Workers: 1, CPUWorkspacePinned: true, UnusedInputFill: 0x3c00, ObservationFills: []uint8{0x35, 0xa7}, GuardBytes: 64, Assertions: []string{"source-upload-spans"}}}
	for i, length := range []int64{768, 1536, 12288} {
		r.Members = append(r.Members, Member{ID: r.ExpectedOrder[i], ReadID: r.ExpectedOrder[i], Dataset: r.Dataset, Split: r.Split, RawParentSamples: length + 10, RawOffset: 10, RawLength: length, RoundedLength: length, Normalisation: Normalisation{Algorithm: "dorado-pa-standardisation-f32-to-f16/v1", CalibrationScale: scalar(1), CalibrationOffset: scalar(0), Mean: scalar(0), Stdev: scalar(1), TrimStart: 10}, NormalizedSamples: length, Normalized: Pin{Bytes: 2 * length}})
	}
	return r
}

func TestFirstPlanSourceCapacities(t *testing.T) {
	p, err := ValidateRequest(schemaRequest())
	if err != nil {
		t.Fatal(err)
	}
	if p.ActiveInput != 14592 || p.ActiveScores != 2432 || p.TLSTM != 2050 || p.InputCapacity != 786432 || p.RecurrentParentBytes != 2055*64*1024 {
		t.Fatalf("first geometry differs: %+v", p)
	}
	want := []UploadSpan{{"OutLayout", 131200, 131264, 267340}, {"ForwardEncoding", 131200, 131200, 2438}, {"InLayout", 2432, 2432, 133638}, {"BackwardEncoding", 131264, 131264, 136076}}
	for i, s := range want {
		if p.Uploads[i] != s {
			t.Fatalf("upload %d: got %+v want %+v", i, p.Uploads[i], s)
		}
	}
	if p.CPUWorkspaceWords != 787584 || p.InitializedWorkspaceWords != 398540 {
		t.Fatalf("workspace differs: %+v", p)
	}
}

func TestSchemaRejections(t *testing.T) {
	controls := map[string]func(*Request){
		"schema":              func(r *Request) { r.Schema = "other" },
		"holdout":             func(r *Request) { r.Split = "HG005" },
		"hidden-holdout":      func(r *Request) { r.Members[0].ReadID = "hg005/read" },
		"split":               func(r *Request) { r.Members[0].Split = "validation" },
		"order":               func(r *Request) { r.Members[0], r.Members[1] = r.Members[1], r.Members[0] },
		"duplicate":           func(r *Request) { r.Members[1].ID = "a" },
		"shape":               func(r *Request) { r.Geometry.Channels = 512 },
		"overflow":            func(r *Request) { r.Geometry.N = 1 << 62 },
		"zero":                func(r *Request) { r.Members[0].RawLength = 0 },
		"non-granular":        func(r *Request) { r.Members[0].RoundedLength = 767 },
		"wrong-bin":           func(r *Request) { r.Members[0].RawLength = 762 },
		"parent-overflow":     func(r *Request) { r.Members[0].RawParentSamples = 1 << 62 },
		"unknown-calibration": func(r *Request) { r.Members[0].Normalisation.CalibrationOffset = nil },
		"trim-alias":          func(r *Request) { r.Members[0].NormalizedOffset = 1 },
		"bad-f16-extent":      func(r *Request) { r.Members[0].Normalized.Bytes-- },
		"fill":                func(r *Request) { r.Recipe.UnusedInputFill = 0x7c00 },
		"two-fills":           func(r *Request) { r.Recipe.ObservationFills[1] = r.Recipe.ObservationFills[0] },
	}
	for name, change := range controls {
		t.Run(name, func(t *testing.T) {
			r := schemaRequest()
			change(&r)
			if _, err := ValidateRequest(r); err == nil {
				t.Fatal("rejection control accepted")
			}
		})
	}
}

func testPin(t *testing.T, path string, b []byte) Pin {
	t.Helper()
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	return Pin{path, digest(b), int64(len(b))}
}

func TestPinAndCanonicalBoundaries(t *testing.T) {
	dir := t.TempDir()
	p := testPin(t, filepath.Join(dir, "fixture"), []byte("independent fixture"))
	if _, err := readPin(p); err != nil {
		t.Fatal(err)
	}
	p.SHA256 = digest([]byte("wrong"))
	if _, err := readPin(p); err == nil {
		t.Fatal("wrong pin accepted")
	}
	p.SHA256 = digest([]byte("independent fixture"))
	alias := filepath.Join(dir, "alias")
	if err := os.Symlink(p.Path, alias); err != nil {
		t.Fatal(err)
	}
	p.Path = alias
	if _, err := readPin(p); err == nil {
		t.Fatal("symlink input accepted")
	}
	if err := freshOutput(filepath.Join(alias, "out"), nil); err == nil {
		t.Fatal("aliased output parent accepted")
	}
	if err := freshOutput(filepath.Join(dir, "fresh"), nil); err != nil {
		t.Fatal(err)
	}
	if err := freshOutput(dir, nil); err == nil {
		t.Fatal("existing output accepted")
	}
	if err := freshOutput(filepath.Join(dir, "fresh"), []Pin{{Path: filepath.Join(dir, "fresh", "input")}}); err == nil {
		t.Fatal("output contains input")
	}
	if err := freshOutput(dir+"/../escape", nil); err == nil {
		t.Fatal("lexical alias accepted")
	}
	if err := strictJSON([]byte(`{"schema":"x","unexpected":1}`), new(Request)); err == nil {
		t.Fatal("unknown schema member accepted")
	}
	if err := strictJSON([]byte(`{} {}`), new(Request)); err == nil {
		t.Fatal("trailing JSON accepted")
	}
	if err := strictJSON([]byte(`{"schema":"a","schema":"b"}`), new(Request)); err == nil {
		t.Fatal("duplicate JSON key accepted")
	}
}

func TestStrictRequestDeclarations(t *testing.T) {
	r := schemaRequest()
	r.Custody.Authorities = []Pin{}
	r.Recipe.UnusedInputFill = 0
	for i := range r.Members {
		m := &r.Members[i]
		m.RawParentSamples, m.RawOffset = m.RawLength, 0
		m.Normalisation.TrimStart, m.NormalizedOffset = 0, 0
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Request
	if err := strictJSON(b, &decoded); err != nil {
		t.Fatalf("explicit zero declarations rejected: %v", err)
	}
	if _, err := ValidateRequest(decoded); err != nil {
		t.Fatalf("explicit zeros invalid: %v", err)
	}
	geometry, err := json.Marshal(r.Geometry)
	if err != nil {
		t.Fatal(err)
	}
	controls := map[string][2]string{
		"root-case":                 {`"schema":`, `"SCHEMA":`},
		"nested-case":               {`"algorithm":`, `"ALGORITHM":`},
		"pin-case":                  {`"path":`, `"PATH":`},
		"schema-alias":              {`"schema":"PackedHACRequest/v1"`, `"schema":"wrong","SCHEMA":"PackedHACRequest/v1"`},
		"split-alias":               {`"split":"development"`, `"split":"HG005","SPLIT":"development"`},
		"nested-alias":              {`"trim_start":0`, `"trim_start":10,"TRIM_START":0`},
		"duplicate":                 {`"trim_start":0`, `"trim_start":10,"trim_start":0`},
		"missing-fill":              {`"unused_input_f16_bits":0,`, ``},
		"null-fill":                 {`"unused_input_f16_bits":0`, `"unused_input_f16_bits":null`},
		"missing-trim":              {`"trim_start":0,`, ``},
		"null-trim":                 {`"trim_start":0`, `"trim_start":null`},
		"missing-raw-offset":        {`"raw_parent_offset":0,`, ``},
		"null-raw-offset":           {`"raw_parent_offset":0`, `"raw_parent_offset":null`},
		"missing-normalized-offset": {`,"normalized_offset":0`, ``},
		"null-normalized-offset":    {`"normalized_offset":0`, `"normalized_offset":null`},
		"null-array":                {`"source_authorities":[]`, `"source_authorities":null`},
		"null-object":               {`"geometry":` + string(geometry), `"geometry":null`},
		"missing-calibration":       {`"calibration_offset":0,`, ``},
	}
	for name, change := range controls {
		t.Run(name, func(t *testing.T) {
			bad := strings.Replace(string(b), change[0], change[1], 1)
			if bad == string(b) {
				t.Fatal("control did not change request")
			}
			if err := strictJSON([]byte(bad), new(Request)); err == nil {
				t.Fatal("invalid declaration accepted")
			}
		})
	}
	for _, field := range []string{"calibration_scale", "calibration_offset", "model_mean", "model_stdev"} {
		t.Run("null-"+field, func(t *testing.T) {
			value := "0"
			if field == "calibration_scale" || field == "model_stdev" {
				value = "1"
			}
			bad := strings.Replace(string(b), `"`+field+`":`+value, `"`+field+`":null`, 1)
			var unknown Request
			if err := strictJSON([]byte(bad), &unknown); err != nil {
				t.Fatal(err)
			}
			if _, err := ValidateRequest(unknown); err == nil {
				t.Fatal("unavailable calibration accepted")
			}
		})
	}
	plan, err := ValidateRequest(r)
	if err != nil {
		t.Fatal(err)
	}
	p := Prepared{Schema: PreparedSchema, Request: r, Plan: plan, Pins: []Pin{}, Limits: []string{}}
	b, err = json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var prepared Prepared
	if err := strictJSON(b, &prepared); err != nil || prepared.NativeClosure != nil {
		t.Fatalf("explicit unavailable native closure lost: %v", err)
	}
	missing := strings.Replace(string(b), `"native_closure":null,`, "", 1)
	if err := strictJSON([]byte(missing), new(Prepared)); err == nil {
		t.Fatal("missing native closure declaration accepted")
	}
}

func preparedTestRequest(t *testing.T, dir string) (Request, Pin) {
	t.Helper()
	g, err := Generate(pinnedSourceRoot, filepath.Join(dir, "generated"))
	if err != nil {
		t.Fatal(err)
	}
	if err = ValidateGeneration(g); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "generated", "generated.json"))
	if err != nil {
		t.Fatal(err)
	}
	r := schemaRequest()
	r.Custody = Custody{Model: testPin(t, filepath.Join(dir, "model.json"), []byte("synthetic model custody")), Config: testPin(t, filepath.Join(dir, "config.toml"), []byte("synthetic config custody")), Weights: testPin(t, filepath.Join(dir, "weights.json"), []byte("synthetic weight custody")), Dataset: testPin(t, filepath.Join(dir, "dataset.json"), []byte("synthetic dataset custody")), CPU21: testPin(t, filepath.Join(dir, "cpu21.json"), []byte("synthetic CPU21 custody")), Generation: Pin{filepath.Join(dir, "generated", "generated.json"), digest(b), int64(len(b))}}
	for _, s := range sourceAuthorities() {
		path := filepath.Join(pinnedSourceRoot, s.Path)
		info, e := os.Stat(path)
		if e != nil {
			t.Fatal(e)
		}
		r.Custody.Authorities = append(r.Custody.Authorities, Pin{path, s.SHA, info.Size()})
	}
	for i := range r.Members {
		m := &r.Members[i]
		m.Source = testPin(t, filepath.Join(dir, m.ID+"-source"), []byte("synthetic raw source "+m.ID))
		m.Normalisation.Provenance = testPin(t, filepath.Join(dir, m.ID+"-normalization.json"), []byte("synthetic provenance "+m.ID))
		fixture := make([]byte, m.NormalizedSamples*2)
		for j := int64(0); j < m.NormalizedSamples; j++ {
			binary.LittleEndian.PutUint16(fixture[j*2:], uint16(0x3000+j%1000))
		}
		m.Normalized = testPin(t, filepath.Join(dir, m.ID+".f16le"), fixture)
	}
	b, err = json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	return r, testPin(t, filepath.Join(dir, "request.json"), b)
}

func TestPreparePinnedFixturesAndRepeatPrefix(t *testing.T) {
	dir := t.TempDir()
	r, request := preparedTestRequest(t, dir)
	r.Members[0].RawLength = 763
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	request = testPin(t, request.Path, b)
	p, err := Prepare(request, filepath.Join(dir, "prepared"))
	if err != nil {
		t.Fatal(err)
	}
	input, err := readPin(p.Input)
	if err != nil {
		t.Fatal(err)
	}
	if p.NativeClosure != nil || p.Status != "source-only-uncompiled" {
		t.Fatal("invented native closure")
	}
	for j := int64(763); j < 768; j++ {
		if binary.LittleEndian.Uint16(input[2*j:]) != uint16(0x3000+j%763) {
			t.Fatal("repeat prefix differs")
		}
	}
	if binary.LittleEndian.Uint16(input[2*p.Plan.ActiveInput:]) != r.Recipe.UnusedInputFill {
		t.Fatal("capacity fill differs")
	}
	if _, err := Prepare(request, filepath.Join(dir, "prepared")); err == nil {
		t.Fatal("existing output accepted")
	}
	// Wrong payload must reject before creating an attempt directory.
	if err := os.WriteFile(r.Members[1].Normalized.Path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	badOut := filepath.Join(dir, "bad")
	if _, err := Prepare(request, badOut); err == nil {
		t.Fatal("changed fixture accepted")
	}
	if _, err := os.Lstat(badOut); !os.IsNotExist(err) {
		t.Fatal("output created before pin rejection")
	}
}

func TestPrepareHardLinkAliasRejection(t *testing.T) {
	dir := t.TempDir()
	r, request := preparedTestRequest(t, dir)
	alias := filepath.Join(dir, "linked-source")
	if err := os.Link(r.Members[0].Source.Path, alias); err != nil {
		t.Fatal(err)
	}
	r.Members[1].Source = r.Members[0].Source
	r.Members[1].Source.Path = alias
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	request = testPin(t, request.Path, b)
	if _, err := Prepare(request, filepath.Join(dir, "prepared")); err == nil {
		t.Fatal("hard-linked input alias accepted")
	}
}

func TestPrepareCompletePinClosure(t *testing.T) {
	for _, control := range []string{"generation-source-original-link", "request-generation-link", "identical-path", "conflicting-path"} {
		t.Run(control, func(t *testing.T) {
			dir := t.TempDir()
			r, request := preparedTestRequest(t, dir)
			b, err := os.ReadFile(r.Custody.Generation.Path)
			if err != nil {
				t.Fatal(err)
			}
			var g Generated
			if err := strictJSON(b, &g); err != nil {
				t.Fatal(err)
			}
			switch control {
			case "generation-source-original-link":
				// Copy genuine pinned bytes into a local source tree so the
				// link control does not require cross-filesystem hard links.
				body, err := readPin(g.Units[0].Source)
				if err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(dir, "source", sourceSpecs()[0].Path)
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				g.Units[0].Source = testPin(t, path, body)
				alias := filepath.Join(dir, "original-link")
				if err := os.Link(path, alias); err != nil {
					t.Fatal(err)
				}
				g.Units[0].Original = g.Units[0].Source
				g.Units[0].Original.Path = alias
				if err := ValidateGeneration(g); err == nil || !strings.Contains(err.Error(), "hard-link") {
					t.Fatalf("generation alias rejection: %v", err)
				}
				b, err = json.Marshal(g)
				if err != nil {
					t.Fatal(err)
				}
				r.Custody.Generation = testPin(t, filepath.Join(dir, "mutated-generation.json"), b)
			case "request-generation-link":
				alias := filepath.Join(dir, "header-link")
				if err := os.Link(g.Header.Path, alias); err != nil {
					t.Fatal(err)
				}
				r.Custody.Model = g.Header
				r.Custody.Model.Path = alias
			case "identical-path":
				r.Custody.Model = g.Header
			case "conflicting-path":
				r.Custody.Model = g.Header
				g.Header.SHA256 = digest([]byte("conflicting declaration"))
				b, err = json.Marshal(g)
				if err != nil {
					t.Fatal(err)
				}
				r.Custody.Generation = testPin(t, filepath.Join(dir, "mutated-generation.json"), b)
			}
			b, err = json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			request = testPin(t, request.Path, b)
			out := filepath.Join(dir, "prepared")
			p, err := Prepare(request, out)
			if control == "identical-path" {
				if err != nil {
					t.Fatal(err)
				}
				seen := map[string]bool{}
				for _, pin := range p.Pins {
					if seen[pin.Path] {
						t.Fatal("duplicate authenticated path")
					}
					seen[pin.Path] = true
				}
				return
			}
			if err == nil {
				t.Fatal("invalid closure accepted")
			}
			if strings.HasSuffix(control, "link") && !strings.Contains(err.Error(), "hard-link") {
				t.Fatalf("wrong rejection: %v", err)
			}
			if control == "conflicting-path" && !strings.Contains(err.Error(), "conflicting pin") {
				t.Fatalf("wrong conflicting-pin rejection: %v", err)
			}
			if _, err := os.Lstat(out); !os.IsNotExist(err) {
				t.Fatal("output created before closure rejection")
			}
		})
	}
}

func TestPrepareNonFiniteFixtureRejection(t *testing.T) {
	dir := t.TempDir()
	r, request := preparedTestRequest(t, dir)
	bad, err := os.ReadFile(r.Members[0].Normalized.Path)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint16(bad, 0x7c00)
	r.Members[0].Normalized = testPin(t, r.Members[0].Normalized.Path, bad)
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	request = testPin(t, request.Path, b)
	if _, err := Prepare(request, filepath.Join(dir, "prepared")); err == nil {
		t.Fatal("non-finite input accepted despite matching pin")
	}
}
