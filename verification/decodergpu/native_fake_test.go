package decodergpu

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// This opt-in test compiles only the producer plus fake headers/implementations.
// A coordinator must release this small host compiler process separately from
// genuine CUDA toolchain queries, producer builds, native loads or GPU work.
func TestNativeProducerFakeHostFailures(t *testing.T) {
	if os.Getenv("D01_FAKE_HOST_TEST") != "1" {
		t.Skip("requires explicit D01 fake-host compiler release")
	}
	cxx := os.Getenv("D01_FAKE_CXX")
	if !filepath.IsAbs(cxx) {
		t.Fatal("D01_FAKE_CXX must identify the explicitly released host compiler")
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "d01-fake-host")
	args := []string{"-std=c++20", "-O0", "-I", "native/fake", "native/fake/ownership.cpp", "-o", binary}
	run := func(t *testing.T, exe string, args ...string) (int, string) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, args...)
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
		cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=", "LD_PRELOAD=", "LD_LIBRARY_PATH=", "CPATH=", "CPLUS_INCLUDE_PATH=", "C_INCLUDE_PATH=", "LIBRARY_PATH=")
		b, err := cmd.CombinedOutput()
		t.Logf("host fake argv=%q output=%q", append([]string{exe}, args...), b)
		if ctx.Err() != nil {
			t.Fatalf("host fake timeout: %v", ctx.Err())
		}
		if err == nil {
			return 0, string(b)
		}
		if e, ok := err.(*exec.ExitError); ok {
			return e.ExitCode(), string(b)
		}
		t.Fatal(err)
		return -1, ""
	}
	if code, _ := run(t, cxx, args...); code != 0 {
		t.Fatalf("fake-header host compilation failed: %d", code)
	}
	b, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("host fake executable sha256=%x", sha256.Sum256(b))
	for name, body := range map[string][]byte{"params": FixedContract().Parameters(), "chunks": FixedContract().DescriptorBytes(), "scores": make([]byte, ScoresBytes)} {
		if err := os.WriteFile(filepath.Join(dir, name), body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, mode := range []string{"h2d-enqueue", "h2d-completion", "d2h-enqueue", "d2h-completion", "final-drain", "raw-capture-close"} {
		t.Run(mode, func(t *testing.T) {
			out := filepath.Join(dir, mode)
			code, log := run(t, binary, mode, "--fixed-hac-staged-oracle", filepath.Join(dir, "params"), filepath.Join(dir, "chunks"), filepath.Join(dir, "scores"), out)
			if code != 1 || strings.Contains(log, "fake-owner-released-before-drain") {
				t.Fatalf("failed operation lost custody: exit=%d log=%s", code, log)
			}
			if mode == "final-drain" {
				if !strings.Contains(log, "fake-drain-failed borrowed=1") || !strings.Contains(log, "host and device owners retained until process teardown") || strings.Contains(log, "fake-device-free") {
					t.Fatalf("failed final drain released owners: %s", log)
				}
			} else if !strings.Contains(log, "fake-drain-ok borrowed=") || strings.Count(log, "fake-device-free") != 8 {
				t.Fatalf("successful final drain/cleanup not observed: %s", log)
			}
			if mode == "raw-capture-close" && !strings.Contains(log, "raw observation write failed") {
				t.Fatalf("buffered raw close failure not observed: %s", log)
			}
			for _, name := range []string{"events.jsonl", "buffers.tsv"} {
				b, err := os.ReadFile(filepath.Join(out, name))
				if err != nil || strings.Contains(string(b), "completed") || strings.Contains(string(b), "captured") || len(b) > 0 && name == "buffers.tsv" {
					t.Fatalf("incomplete copy recorded as successful: %s %q %v", name, b, err)
				}
			}
		})
	}
	for _, control := range []struct {
		mode string
		exit int
		want string
	}{{"owner-control", 91, "fake-owner-released-before-drain"}, {"raw-close", 0, "raw observation write failed"}} {
		t.Run(control.mode, func(t *testing.T) {
			code, log := run(t, binary, control.mode)
			if code != control.exit || !strings.Contains(log, control.want) {
				t.Fatalf("ineffective control: exit=%d log=%s", code, log)
			}
		})
	}
}
