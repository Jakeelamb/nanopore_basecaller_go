package supgpu

import (
	"bytes"
	"debug/elf"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

func TestUnnamedRuntimeInertRejections(t *testing.T) {
	for _, kind := range []string{"malformed", "startup", "reference", "observer", "audit", "alternate"} {
		t.Run(kind, func(t *testing.T) {
			p, _, _ := metadataGuardFixture(t)
			object := runtimeTestFile(t, t.TempDir(), "not-elf", []byte("not ELF"))
			c := RuntimeCode{Object: object, Needed: []string{}, Tier: "test"}
			switch kind {
			case "startup":
				c.Startup = true
			case "reference":
				p.Policy.Reference = object
			case "observer":
				p.Policy.Observer = object
			case "audit":
				p.Policy.Audit = object
			}
			p.Policy.Allowed = []RuntimeCode{c}
			if kind == "alternate" {
				p.Policy.Allowed = []RuntimeCode{}
				p.Policy.Alternates = []RuntimeCode{c}
			}
			selected := metadataGuardJSON(t, filepath.Join(t.TempDir(), "policy.json"), p.Policy)
			if _, err := ReadRuntimePolicy(selected); err == nil {
				t.Fatal("unproven unnamed image accepted")
			}
		})
	}
}

// Opt-in static reads only: never compiles, executes or loads an ELF image.
// All derived policies are rehashed and retained under the selected controls dir.
func TestUnnamedRuntimeGenuine(t *testing.T) {
	keys := []string{"H08_UNNAMED_PROOF", "H08_UNNAMED_PROOF_SHA256", "H08_UNNAMED_BACKEND", "H08_UNNAMED_BACKEND_SHA256", "H08_UNNAMED_POLICY", "H08_UNNAMED_POLICY_SHA256", "H08_UNNAMED_CONTROLS"}
	set := 0
	for _, k := range keys {
		if os.Getenv(k) != "" {
			set++
		}
	}
	if set == 0 {
		t.Skip("independent unnamed backend/policy/proof pins and controls directory absent")
	}
	if set != len(keys) {
		t.Fatal("partial H08_UNNAMED environment")
	}
	for _, k := range []string{keys[0], keys[2], keys[4], keys[6]} {
		p := os.Getenv(k)
		canonical, err := filepath.EvalSymlinks(p)
		if err != nil || !filepath.IsAbs(p) || canonical != p {
			t.Fatalf("canonical absolute %s required: %v", k, err)
		}
	}
	for i := 0; i < 6; i += 2 {
		p, err := pin(os.Getenv(keys[i]))
		if err != nil || p.SHA256 != os.Getenv(keys[i+1]) {
			t.Fatalf("independent %s pin differs: %v", keys[i], err)
		}
	}
	selected, err := pin(os.Getenv("H08_UNNAMED_POLICY"))
	if err != nil {
		t.Fatal(err)
	}
	var p RuntimePolicy
	if err := readPinnedJSON(selected.Path, selected.SHA256, &p); err != nil {
		t.Fatal(err)
	}
	backend, err := pin(os.Getenv("H08_UNNAMED_BACKEND"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := nativeproof.InspectELF(backend.Path)
	if err != nil || m.SONAME != "" || m.Pin.Path != backend.Path || m.Pin.SHA256 != backend.SHA256 || len(m.Needed) == 0 {
		t.Fatalf("selected genuine unnamed metadata differs: %+v: %v", m, err)
	}
	index := -1
	for i, c := range p.Allowed {
		if c.Object == backend {
			index = i
		}
	}
	if index < 0 || p.Allowed[index].SONAME != "" || !slices.Equal(p.Allowed[index].Needed, m.Needed) {
		t.Fatal("policy must contain truthful selected backend metadata")
	}
	dir, err := os.MkdirTemp(os.Getenv("H08_UNNAMED_CONTROLS"), "supgpu-unnamed-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained controls: %s", dir)
	t.Run("required-empty-positive", func(t *testing.T) {
		if _, err := ReadRuntimePolicy(selected); err != nil {
			t.Fatalf("truthful required empty SONAME rejected: %v", err)
		}
	})
	t.Run("embedded-plan-positive", func(t *testing.T) {
		contract := metadataGuardJSON(t, filepath.Join(dir, "contract.json"), PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"}})
		plan := GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Policy: p, Contract: contract, Inputs: []Pin{selected}}
		v := metadataGuardJSON(t, filepath.Join(dir, "embedded-plan.json"), plan)
		if _, _, err := ReadGuardPlan(v, contract); err != nil {
			t.Fatalf("embedded truthful policy rejected: %v", err)
		}
	})
	for _, kind := range []string{"elf-exec", "elf-malformed", "elf-empty-soname"} {
		t.Run(kind, func(t *testing.T) {
			raw, err := os.ReadFile(backend.Path)
			if err != nil {
				t.Fatal(err)
			}
			e, err := elf.NewFile(bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			switch kind {
			case "elf-exec":
				e.ByteOrder.PutUint16(raw[16:18], uint16(elf.ET_EXEC))
			case "elf-malformed":
				raw[0] = 0
			case "elf-empty-soname":
				// Patch one actual dynamic entry using debug/elf's section extent;
				// no synthetic ELF layout or copied parser fixture is constructed.
				s := e.SectionByType(elf.SHT_DYNAMIC)
				if e.Class != elf.ELFCLASS64 || s == nil || s.Entsize != 16 || s.Offset+s.Size > uint64(len(raw)) {
					t.Fatal("selected ELF64 dynamic extent required")
				}
				patched := false
				for at := s.Offset; at+16 <= s.Offset+s.Size; at += 16 {
					if elf.DynTag(e.ByteOrder.Uint64(raw[at:])) == elf.DT_NEEDED {
						e.ByteOrder.PutUint64(raw[at:], uint64(elf.DT_SONAME))
						e.ByteOrder.PutUint64(raw[at+8:], 0)
						patched = true
						break
					}
				}
				if !patched {
					t.Fatal("actual Needed entry required")
				}
			}
			object := runtimeTestFile(t, dir, kind+".so", raw)
			if kind == "elf-empty-soname" {
				if _, err := nativeproof.InspectELF(object.Path); err == nil {
					t.Fatal("present empty SONAME control must fail inspection")
				}
			}
			q := p
			q.Allowed = slices.Clone(p.Allowed)
			q.Allowed[index].Object = object
			v := metadataGuardJSON(t, filepath.Join(dir, kind+".json"), q)
			if _, err := ReadRuntimePolicy(v); err == nil {
				t.Fatalf("accepted rehashed %s", kind)
			}
		})
	}
	for _, kind := range []string{"startup", "hash", "size", "symlink", "noncanonical", "changed-needed", "reordered-needed", "omitted-needed", "missing-provider", "alternate-only", "backend-basename-provider", "duplicate-name", "duplicate-path", "named-as-empty"} {
		t.Run(kind, func(t *testing.T) {
			q := p
			q.Allowed = slices.Clone(p.Allowed)
			q.Allowed[index].Needed = slices.Clone(p.Allowed[index].Needed)
			provider := -1
			for i, c := range q.Allowed {
				if c.SONAME == m.Needed[0] {
					provider = i
				}
			}
			if provider < 0 {
				t.Fatal("genuine first Needed lacks selected named provider")
			}
			switch kind {
			case "startup":
				q.Allowed[index].Startup = true
			case "hash":
				q.Allowed[index].Object.SHA256 = strings.Repeat("0", 64)
			case "size":
				q.Allowed[index].Object.Bytes++
			case "symlink":
				alias := filepath.Join(dir, "backend-alias")
				if err := os.Symlink(backend.Path, alias); err != nil {
					t.Fatal(err)
				}
				q.Allowed[index].Object.Path = alias
			case "noncanonical":
				q.Allowed[index].Object.Path = filepath.Dir(backend.Path) + "/./" + filepath.Base(backend.Path)
			case "changed-needed":
				q.Allowed[index].Needed[0] = "not-selected.so"
			case "reordered-needed":
				if len(q.Allowed[index].Needed) < 2 {
					t.Fatal("two actual Needed entries required")
				}
				q.Allowed[index].Needed[0], q.Allowed[index].Needed[1] = q.Allowed[index].Needed[1], q.Allowed[index].Needed[0]
			case "omitted-needed":
				q.Allowed[index].Needed = []string{}
			case "missing-provider":
				q.Allowed = slices.Delete(q.Allowed, provider, provider+1)
			case "alternate-only":
				q.Alternates = append(slices.Clone(p.Alternates), q.Allowed[provider])
				q.Allowed = slices.Delete(q.Allowed, provider, provider+1)
			case "backend-basename-provider":
				q.Allowed[provider].SONAME = filepath.Base(backend.Path)
			case "duplicate-name":
				c := q.Allowed[provider]
				c.Object = runtimeTestFile(t, dir, "duplicate-name-object", []byte("inert named provider"))
				q.Allowed = append(q.Allowed, c)
			case "duplicate-path":
				q.Allowed = append(q.Allowed, q.Allowed[provider])
			case "named-as-empty":
				q.Allowed[provider].SONAME = ""
			}
			v := metadataGuardJSON(t, filepath.Join(dir, kind+".json"), q)
			if _, err := ReadRuntimePolicy(v); err == nil {
				t.Fatalf("accepted rehashed %s", kind)
			}
		})
	}
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	base := string(raw)
	needed, err := json.Marshal(p.Allowed[index].Needed)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, body string }{
		{"missing-soname", strings.Replace(base, `"soname":"",`, "", 1)},
		{"null-soname", strings.Replace(base, `"soname":""`, `"soname":null`, 1)},
		{"null-needed", strings.Replace(base, `"needed":`+string(needed), `"needed":null`, 1)},
		{"duplicate", strings.Replace(base, `"soname":""`, `"soname":"","soname":""`, 1)},
		{"case", strings.Replace(base, `"soname":""`, `"SONAME":""`, 1)},
		{"unknown", strings.Replace(base, `"soname":""`, `"soname":"","extra":true`, 1)},
		{"trailing", base + " {}"},
	} {
		t.Run("wire-"+tc.name, func(t *testing.T) {
			if tc.body == base {
				t.Fatal("wire mutation unchanged")
			}
			path := filepath.Join(dir, "wire-"+tc.name+".json")
			if err := os.WriteFile(path, []byte(tc.body), 0600); err != nil {
				t.Fatal(err)
			}
			v, err := pin(path)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ReadRuntimePolicy(v); err == nil {
				t.Fatal("invalid rehashed wire accepted")
			}
		})
	}
}
