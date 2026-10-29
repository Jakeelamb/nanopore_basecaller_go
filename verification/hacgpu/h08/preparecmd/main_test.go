package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
)

func requestFile(t *testing.T, body string) h08.Pin {
	t.Helper()
	path := filepath.Join(t.TempDir(), "request.json")
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256([]byte(body))
	return h08.Pin{Path: path, SHA256: hex.EncodeToString(h[:])}
}

func TestRequestAuthenticationAndWireBoundary(t *testing.T) {
	for _, body := range []string{"", " \n\t", `null`, `[]`, `{} {}`, `{} trailing`, `{"Schema":`, `{"Schema":"x","Unknown":true}`, `{"Schema":3}`} {
		t.Run(body, func(t *testing.T) {
			p := requestFile(t, body)
			var q admissionPreflight
			if err := h08.ReadPreparationJSON(p, &q); err == nil {
				t.Fatalf("unsupported wire accepted: %s", body)
			}
		})
	}
	p := requestFile(t, `{"Schema":"nanopore-h08-cpu13-preflight/v1"}`)
	var q admissionPreflight
	if err := h08.ReadPreparationJSON(p, &q); err != nil || q.Schema != "nanopore-h08-cpu13-preflight/v1" {
		t.Fatalf("valid typed request rejected: %+v: %v", q, err)
	}
	if err := os.WriteFile(p.Path, []byte(`{}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := h08.ReadPreparationJSON(p, &q); err == nil || !strings.Contains(err.Error(), "SHA256") {
		t.Fatalf("changed request accepted: %v", err)
	}
	for _, pin := range []h08.Pin{{}, {Path: p.Path}, {SHA256: p.SHA256}, {Path: "relative", SHA256: p.SHA256}, {Path: p.Path, SHA256: strings.Repeat("z", 64)}} {
		if err := h08.ReadPreparationJSON(pin, &q); err == nil {
			t.Fatal("partial or malformed pin accepted")
		}
	}
}

func TestOperationsRejectInvalidRequestFiles(t *testing.T) {
	for _, operation := range []string{"fixture", "admission", "cpu13-preflight", "fake-preflight"} {
		for _, input := range []string{"directory", "changed-hash", "empty"} {
			t.Run(operation+"/"+input, func(t *testing.T) {
				p := requestFile(t, `{}`)
				want := "SHA256"
				switch input {
				case "directory":
					p.Path = t.TempDir()
					want = "regular"
				case "changed-hash":
					if err := os.WriteFile(p.Path, []byte(`{"Schema":"changed"}`), 0600); err != nil {
						t.Fatal(err)
					}
				case "empty":
					p = requestFile(t, "")
					want = "typed JSON object"
				}
				var out, diagnostics bytes.Buffer
				err := run([]string{"-operation", operation, "-request", p.Path, "-request-sha", p.SHA256}, &out, &diagnostics)
				if err == nil || !strings.Contains(err.Error(), want) {
					t.Fatalf("%s request failed outside shared boundary: %v", input, err)
				}
				if out.Len() != 0 {
					t.Fatalf("invalid request emitted success output: %s", out.String())
				}
			})
		}
	}
}

func TestOperationsRejectWithoutAcceptanceOutput(t *testing.T) {
	for _, tc := range []struct{ op, body string }{
		{"unknown", `{}`},
		{"fixture", `{"Schema":"unsupported"}`},
		{"fixture", `{}`},
		{"admission", `{"Schema":"unsupported"}`},
		{"cpu13-preflight", `{"Schema":"unsupported"}`},
		{"cpu13-preflight", `{"Schema":"nanopore-h08-cpu13-preflight/v1"}`},
		{"fake-preflight", `{"Schema":"unsupported"}`},
		{"fake-preflight", `{"Schema":"nanopore-h08-fake-preflight/v1"}`},
	} {
		t.Run(tc.op+tc.body, func(t *testing.T) {
			p := requestFile(t, tc.body)
			var out, diagnostic bytes.Buffer
			if err := run([]string{"-operation", tc.op, "-request", p.Path, "-request-sha", p.SHA256}, &out, &diagnostic); err == nil {
				t.Fatal("invalid operation/schema/pins accepted")
			}
			if out.Len() != 0 {
				t.Fatalf("acceptance output before success: %s", out.String())
			}
		})
	}
}

func TestFixtureBoundaryRejectsCollisionsBeforeReadingArtifacts(t *testing.T) {
	dir := t.TempDir()
	p := requestFile(t, `{"inert":"not a fixture or native artifact"}`)
	o := h08.FixtureOptions{SourceManifest: p, Archive: p, Backend: p, Weight: p, Directory: dir, Runtime: map[string]string{}}
	if err := fixtureBoundary(o); err == nil || !strings.Contains(err.Error(), "must be absent") {
		t.Fatalf("existing directory collision not rejected: %v", err)
	}
	o.Directory = filepath.Join(dir, "fresh")
	o.Weight.SHA256 = ""
	if err := fixtureBoundary(o); err == nil {
		t.Fatal("partial weight pin accepted")
	}
	body, err := json.Marshal(o)
	if err != nil {
		t.Fatal(err)
	}
	selected := requestFile(t, string(body))
	var out, diagnostic bytes.Buffer
	if err := run([]string{"-operation", "fixture", "-request", selected.Path, "-request-sha", selected.SHA256}, &out, &diagnostic); err == nil {
		t.Fatal("partial fixture options accepted")
	}
	if _, err := os.Stat(o.Directory); !os.IsNotExist(err) {
		t.Fatalf("invalid boundary created directory: %v", err)
	}
}

func TestFixtureBackendCannotAliasGeneratedMetadata(t *testing.T) {
	p := requestFile(t, `{"inert":"metadata"}`)
	for _, name := range []string{"manifest.json", "runtime.json", "view.json"} {
		o := h08.FixtureOptions{SourceManifest: p, Archive: p, Backend: h08.Pin{Path: filepath.Join(filepath.Dir(p.Path), name), SHA256: p.SHA256}, Weight: p, Directory: filepath.Join(t.TempDir(), "fresh"), Runtime: map[string]string{}}
		if err := fixtureBoundary(o); err == nil || !strings.Contains(err.Error(), "symlink collides") {
			t.Fatalf("backend metadata alias accepted: %v", err)
		}
	}
}
