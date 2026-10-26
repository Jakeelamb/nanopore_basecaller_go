package hacvcsaux

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func mapFixture(t *testing.T) (map[string]string, closedbuild.Pin, string) {
	t.Helper()
	directory := t.TempDir()
	var pins []closedbuild.Pin
	for _, name := range []string{"approved-elf", "loader-cache", "unexpected-executable"} {
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, []byte(name+" original bytes"), 0600); err != nil {
			t.Fatal(err)
		}
		pin, err := closedbuild.PinFile(path)
		if err != nil {
			t.Fatal(err)
		}
		pins = append(pins, pin)
	}
	return map[string]string{pins[0].Path: pins[0].SHA256}, pins[1], pins[2].Path
}

func mapLine(path, permissions string) string {
	return fmt.Sprintf("1000-2000 %s 00000000 00:01 1 %s\n", permissions, path)
}

func TestMapsKeepELFClosureAndPinnedReadOnlyCache(t *testing.T) {
	for _, permissions := range []string{"r--p", "r--s"} {
		t.Run(permissions, func(t *testing.T) {
			allowed, cache, _ := mapFixture(t)
			var body string
			for path := range allowed {
				body += mapLine(path, "r-xp")
			}
			if err := compareMaps([]byte(body+mapLine(cache.Path, permissions)), allowed, cache); err != nil {
				t.Fatal(err)
			}
			if err := compareMaps([]byte(body), allowed, cache); err != nil {
				t.Fatalf("optional unmapped cache rejected: %v", err)
			}
		})
	}
}

func TestMapsRejectCorruptCacheAndUnexpectedMappings(t *testing.T) {
	for _, name := range []string{"corrupt-cache", "unexpected-executable", "executable-cache", "writable-cache", "missing-required-elf"} {
		t.Run(name, func(t *testing.T) {
			allowed, cache, unexpected := mapFixture(t)
			var body string
			for path := range allowed {
				body += mapLine(path, "r-xp")
			}
			permissions, want := "r--p", ""
			switch name {
			case "corrupt-cache":
				if err := os.WriteFile(cache.Path, []byte("changed cache bytes"), 0600); err != nil {
					t.Fatal(err)
				}
				want = "mapped native file changed"
			case "unexpected-executable":
				body += mapLine(unexpected, "r-xp")
				want = "unapproved native mapping"
			case "executable-cache":
				permissions, want = "r-xp", "read-only non-executable"
			case "writable-cache":
				permissions, want = "rw-p", "read-only non-executable"
			case "missing-required-elf":
				body, want = "", "closed native file not observed"
			}
			err := compareMaps([]byte(body+mapLine(cache.Path, permissions)), allowed, cache)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("%s control failed: %v", name, err)
			}
		})
	}
}

func TestIndependentLoaderCacheIsExactSystemCache(t *testing.T) {
	pin, err := closedbuild.PinFile(CompilerBaselinePath)
	if err != nil || pin.SHA256 != CompilerBaselineSHA256 {
		t.Fatalf("independent compiler baseline differs: %v", err)
	}
	body, err := os.ReadFile(pin.Path)
	if err != nil {
		t.Fatal(err)
	}
	var baseline struct{ Compiler closedbuild.Compiler }
	if err := json.Unmarshal(body, &baseline); err != nil {
		t.Fatal(err)
	}
	cache := baseline.Compiler.LoaderCache
	actual, err := closedbuild.PinFile("/etc/ld.so.cache")
	if err != nil || cache.Path != "/etc/ld.so.cache" || actual != cache {
		t.Fatalf("independent exact loader-cache pin differs: %v", err)
	}
}
