package h08

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// This file uses only the existing public constructor so it can also run
// unchanged against frozen production before the request-reader correction.
func TestCPU13PreparationRejectsFIFO(t *testing.T) {
	const childKey = "H08_PREPARATION_FIFO_CHILD"
	if path := os.Getenv(childKey); path != "" {
		got, err := PrepareCPU13Admission(Pin{Path: path, SHA256: strings.Repeat("0", 64)})
		if err == nil || !strings.Contains(err.Error(), "regular") {
			t.Fatalf("FIFO options must reject as nonregular: %v", err)
		}
		if got != (CPU13Preparation{}) {
			t.Fatalf("FIFO rejection returned preparation output: %+v", got)
		}
		return
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "options.fifo")
	if err := syscall.Mkfifo(path, 0600); err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	// CommandContext owns killing a blocked child; CombinedOutput waits and
	// reaps it. WaitDelay also bounds inherited output pipes after termination.
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestCPU13PreparationRejectsFIFO$", "-test.count=1")
	cmd.Env = append(os.Environ(), childKey+"="+path)
	cmd.WaitDelay = time.Second
	output, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("FIFO options rejection blocked: %v: %s", ctx.Err(), output)
	}
	if err != nil {
		t.Fatalf("FIFO options rejection child failed: %v: %s", err, output)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 || entries[0].Name() != "options.fifo" {
		t.Fatalf("FIFO rejection created attempt/output files: %v: %v", entries, err)
	}
}
