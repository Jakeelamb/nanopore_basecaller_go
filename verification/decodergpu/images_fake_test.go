package decodergpu

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPassiveCUPTIObserverFakeHost(t *testing.T) {
	if os.Getenv("D01_FAKE_HOST_TEST") != "1" {
		t.Skip("requires explicit D01 fake-host compiler release")
	}
	cxx := os.Getenv("D01_FAKE_CXX")
	if !filepath.IsAbs(cxx) {
		t.Fatal("explicit absolute fake C++ compiler required")
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "observer-fake")
	run := func(argv ...string) string {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
		cmd.Env = []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "CUDA_VISIBLE_DEVICES="}
		raw, err := cmd.CombinedOutput()
		t.Logf("fake-only argv=%q output=%q", argv, raw)
		if err != nil {
			t.Fatalf("fake-only command failed: %v", err)
		}
		return string(raw)
	}
	run(cxx, "-std=c++20", "-O0", "-I", "native/fake", "native/fake/images.cpp", "-o", binary)
	b, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("fake observer executable sha256=%x", sha256.Sum256(b))
	for _, mode := range []string{"positive", "malformed", "oversize", "capacity", "buffer-capacity", "callback-exception", "dropped"} {
		t.Run(mode, func(t *testing.T) {
			out := filepath.Join(dir, mode)
			log := run(binary, mode, out)
			status, err := os.ReadFile(filepath.Join(out, "images-final/status.txt"))
			if err != nil {
				t.Fatal(err)
			}
			if mode == "positive" {
				if !strings.Contains(log, "callback-failure-retained=0") || !strings.Contains(string(status), "functions=4\nlaunches=8") {
					t.Fatalf("positive copy lost: %s %s", log, status)
				}
				raw, e := os.ReadFile(filepath.Join(out, "images-final/module-0.cubin"))
				if e != nil || len(raw) != 32 || raw[0] != 'c' {
					t.Fatalf("callback borrowed source bytes: %q %v", raw, e)
				}
				fs, e := os.ReadFile(filepath.Join(out, "images-final/functions.tsv"))
				if e != nil || !strings.Contains(string(fs), "back_guide_step") || strings.Contains(string(fs), "Xack") {
					t.Fatal("callback borrowed function name", e)
				}
			} else if strings.Contains(log, "callback-failure-retained=0") {
				t.Fatal("injected callback failure ignored")
			}
		})
	}
}
