package supgpu

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This test intentionally uses only entry points present at c373bac. Copy this
// file alone onto that revision for the genuine historical-fixture RED check.
func TestHistoricalControllerRetainedProducerRoot(t *testing.T) {
	path, sha := os.Getenv("SUPGPU_HISTORICAL_CONTROLLER"), os.Getenv("SUPGPU_HISTORICAL_CONTROLLER_SHA256")
	if path == "" && sha == "" {
		t.Skip("set SUPGPU_HISTORICAL_CONTROLLER and SUPGPU_HISTORICAL_CONTROLLER_SHA256; skip is not qualification")
	}
	if path == "" || sha == "" {
		t.Fatal("both historical controller fixture variables are required")
	}
	fixture, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.SHA256 != sha {
		t.Fatal("historical controller differs from external selected pin")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var original GoController
	if err = json.Unmarshal(raw, &original); err != nil {
		t.Fatal(err)
	}
	if len(original.Snapshots) != 917 {
		t.Fatalf("selected genuine fixture has %d snapshots, want 917", len(original.Snapshots))
	}
	consumer := t.TempDir()
	for _, s := range original.Snapshots {
		if s.Original == filepath.Join(consumer, "go.mod") {
			t.Fatal("fixture unexpectedly belongs to consumer root")
		}
	}
	t.Chdir(consumer)
	// Newly added consumer source must not change a historical selection.
	if err = os.WriteFile("new-consumer-source.go", []byte("package consumer\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = readController(fixture, false); err != nil {
		t.Fatalf("genuine retained controller rejected from different consumer root: %v", err)
	}
	t.Log("validated all 917 retained snapshots from a different consumer root")
	put := func(name string, body []byte) Pin {
		t.Helper()
		p := filepath.Join(t.TempDir(), name)
		if e := os.WriteFile(p, body, 0600); e != nil {
			t.Fatal(e)
		}
		v, e := pin(p)
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
	for _, name := range []string{"wrongModuleRoot", "sourceBinding", "omittedSnapshot", "changedCopy", "rehashedChangedCopy"} {
		t.Run(name, func(t *testing.T) {
			var c GoController
			if e := json.Unmarshal(raw, &c); e != nil {
				t.Fatal(e)
			}
			switch name {
			case "wrongModuleRoot":
				deps, e := os.ReadFile(c.Dependencies.Path)
				if e != nil {
					t.Fatal(e)
				}
				decoder := json.NewDecoder(bytes.NewReader(deps))
				var changed bytes.Buffer
				encoder := json.NewEncoder(&changed)
				mutated := false
				for {
					var pkg map[string]any
					e = decoder.Decode(&pkg)
					if e == io.EOF {
						break
					}
					if e != nil {
						t.Fatal(e)
					}
					if mod, ok := pkg["Module"].(map[string]any); ok && mod["Main"] == true && !mutated {
						mod["Dir"] = consumer
						mutated = true
					}
					if e = encoder.Encode(pkg); e != nil {
						t.Fatal(e)
					}
				}
				if !mutated {
					t.Fatal("fixture contains no selected main module")
				}
				c.Dependencies = put("go-list.json", changed.Bytes())
				c.Commands[1].Stdout, c.Commands[1].StdoutSHA = c.Dependencies.Path, c.Dependencies.SHA256
			case "sourceBinding":
				c.Snapshots[1].Original += ".wrong-root"
			case "omittedSnapshot":
				c.Snapshots = c.Snapshots[1:]
			case "changedCopy", "rehashedChangedCopy":
				found := false
				for i, s := range c.Snapshots {
					if strings.HasSuffix(s.Original, "/go.mod") {
						v := put("changed.mod", []byte("changed retained source\n"))
						c.Snapshots[i].Copy.Path = v.Path
						if name == "rehashedChangedCopy" {
							c.Snapshots[i].Copy = v
						}
						found = true
						break
					}
				}
				if !found {
					t.Fatal("fixture contains no go.mod snapshot")
				}
			}
			body, e := json.Marshal(c)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = readController(put("controller.json", body), false); e == nil {
				t.Fatal("rehashed invalid historical controller accepted")
			}
		})
	}
}

func TestHistoricalRootFixPreservesCurrentGuards(t *testing.T) {
	c, save, source, _ := syntheticController(t)
	if _, err := readController(save(c), true); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("changed live source"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readController(save(c), true); err == nil || !strings.Contains(err.Error(), "source/tool drift") {
		t.Fatalf("changed live source accepted or wrong rejection: %v", err)
	}
	if err = os.WriteFile(source, before, 0600); err != nil {
		t.Fatal(err)
	}
	binary, err := os.ReadFile(c.Binary.Path)
	if err != nil {
		t.Fatal(err)
	}
	alternate := filepath.Join(t.TempDir(), "alternate-controller")
	if err = os.WriteFile(alternate, binary, 0600); err != nil {
		t.Fatal(err)
	}
	c.Binary, err = pin(alternate)
	if err != nil {
		t.Fatal(err)
	}
	c.Commands[2].Args = []string{c.Commands[0].Args[0], "build", "-buildvcs=false", "-o", c.Binary.Path, "./verification/supgpu/cmd"}
	if _, err = readController(save(c), true); err == nil || !strings.Contains(err.Error(), "executing controller binary differs") {
		t.Fatalf("different actual executable accepted or wrong rejection: %v", err)
	}
}
