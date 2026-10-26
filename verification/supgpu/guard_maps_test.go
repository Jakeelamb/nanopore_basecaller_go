package supgpu

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Retained capture001 reference map0034, SHA256
// 41e49c4230cfc94693b2669fa7a33ec57c21396c08d97a0fe6970eaf131b781b.
const cudaDataMappings = `10004800000-10004a00000 rw-s 00000000 00:01 428917                       /dev/zero (deleted)
10004a00000-10004c00000 rw-s 00000000 00:01 428918                       /dev/zero (deleted)
10005800000-10005a00000 rw-s 00000000 00:01 428920                       /dev/zero (deleted)
7ff8d6800000-7ff8d6a00000 rw-s 00000000 00:01 428919                     /dev/zero (deleted)
`

func TestMapClassifierRetainsDeletedCUDADataWithoutCodeApproval(t *testing.T) {
	raw := []byte(cudaDataMappings)
	o, err := parseCodeMaps(raw, RuntimePolicy{})
	if err != nil {
		t.Fatalf("non-code CUDA data blocked code observation: %v", err)
	}
	if !reflect.DeepEqual(o.Raw, raw) || len(o.Names) != 0 || o.VDSO {
		t.Fatal("data bytes were removed or promoted to code")
	}

}

func TestMapClassifierPreservesCodeAndTierRejections(t *testing.T) {
	dir := t.TempDir()
	root := runtimeTestFile(t, dir, "target", []byte("inert selected executable"))
	audit := runtimeTestFile(t, dir, "audit", []byte("inert auditor"))
	provider := runtimeTestFile(t, dir, "qualified.so", []byte("inert selected provider"))
	wrong := runtimeTestFile(t, dir, "wrong.so", []byte("inert known alternate"))
	unknown := runtimeTestFile(t, dir, "unknown.so", []byte("inert unknown executable"))
	p := RuntimePolicy{Reference: root, Observer: root, Audit: audit, Allowed: []RuntimeCode{{Object: provider}}, Alternates: []RuntimeCode{{Object: wrong}}}
	for _, item := range []struct {
		name, perms, path, parseError, gateError string
	}{
		{"deleted-unknown-executable", "r-xp", unknown.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-known-executable", "r-xp", provider.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-known-nonexec", "r--p", provider.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-root-nonexec", "rw-p", root.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-audit-nonexec", "r--p", audit.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-alternate-nonexec", "r--p", wrong.Path + " (deleted)", "deleted mapped object", ""},
		{"deleted-anonymous-executable", "rwxp", "/dev/zero (deleted)", "deleted mapped object", ""},
		{"anonymous-executable", "rwxp", "", "anonymous/unrecognized executable mapping", ""},
		{"unknown-executable", "r-xp", unknown.Path, "", "unknown mapped code"},
		{"known-wrong-nonexec", "r--p", wrong.Path, "", "known wrong runtime tier"},
		{"known-selected-nonexec", "r--p", provider.Path, "", ""},
		{"known-selected-executable", "r-xp", provider.Path, "", ""},
	} {
		t.Run(item.name, func(t *testing.T) {
			raw := []byte(fmt.Sprintf("1000-2000 %s 00000000 00:01 1 %s\n", item.perms, item.path))
			o, err := parseCodeMaps(raw, p)
			if item.parseError != "" {
				if err == nil || !strings.Contains(err.Error(), item.parseError) {
					t.Fatalf("code mapping rejection differs: %v", err)
				}
				return
			}
			if err != nil || len(o.Names) != 1 {
				t.Fatalf("code was silently excluded: %v %+v", err, o)
			}
			q := GateRequest{Schema: "nanopore-supgpu-loader-gate/v1", Kind: "open", Sequence: 1, Cookie: 1, Path: item.path, PID: 1}
			s := gateState{hello: true, opened: map[uint64]Pin{}}
			_, err = decideGate(q, "audit", p, "reference", "attest-only", 1, &s, o)
			if item.gateError != "" {
				if err == nil || !strings.Contains(err.Error(), item.gateError) {
					t.Fatalf("selected provider policy weakened: %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
	t.Run("selected-object-hash-drift", func(t *testing.T) {
		if err := os.WriteFile(provider.Path, []byte("changed selected provider"), 0600); err != nil {
			t.Fatal(err)
		}
		o, err := parseCodeMaps([]byte(fmt.Sprintf("1000-2000 r--p 00000000 00:01 1 %s\n", provider.Path)), p)
		if err != nil || len(o.Names) != 1 {
			t.Fatalf("selected image was not observed: %v", err)
		}
		q := GateRequest{Schema: "nanopore-supgpu-loader-gate/v1", Kind: "open", Sequence: 1, Cookie: 1, Path: provider.Path, PID: 1}
		s := gateState{hello: true, opened: map[uint64]Pin{}}
		if _, err = decideGate(q, "audit", p, "reference", "attest-only", 1, &s, o); err == nil {
			t.Fatal("changed selected object hash was accepted")
		}
	})
}
