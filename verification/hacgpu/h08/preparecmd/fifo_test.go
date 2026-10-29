package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Use only run and its existing flags, allowing this file to reproduce each
// blocked external request on frozen production without the shared helper.
func TestCPU13PreparationCLIRejectsFIFO(t *testing.T) {
	const pathKey = "H08_PREPARECMD_FIFO_CHILD"
	const operationKey = "H08_PREPARECMD_FIFO_OPERATION"
	if path := os.Getenv(pathKey); path != "" {
		var out, diagnostics bytes.Buffer
		err := run([]string{"-operation", os.Getenv(operationKey), "-request", path, "-request-sha", strings.Repeat("0", 64)}, &out, &diagnostics)
		if err == nil || !strings.Contains(err.Error(), "regular") {
			t.Fatalf("FIFO request must reject as nonregular: %v", err)
		}
		if out.Len() != 0 {
			t.Fatalf("FIFO rejection emitted success output: %s", out.String())
		}
		return
	}
	for _, operation := range []string{"fixture", "admission", "cpu13-preflight", "fake-preflight"} {
		t.Run(operation, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "request.fifo")
			if err := syscall.Mkfifo(path, 0600); err != nil {
				t.Fatal(err)
			}
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			// Run just this named test and operation in an owned, killable child;
			// CombinedOutput reaps it even if the original reader blocks forever.
			cmd := exec.CommandContext(ctx, executable, "-test.run=^TestCPU13PreparationCLIRejectsFIFO$", "-test.count=1")
			cmd.Env = append(os.Environ(), pathKey+"="+path, operationKey+"="+operation)
			cmd.WaitDelay = time.Second
			output, err := cmd.CombinedOutput()
			if ctx.Err() != nil {
				t.Fatalf("%s FIFO request rejection blocked: %v: %s", operation, ctx.Err(), output)
			}
			if err != nil {
				t.Fatalf("%s FIFO rejection child failed: %v: %s", operation, err, output)
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 1 || entries[0].Name() != "request.fifo" {
				t.Fatalf("FIFO rejection created attempt/output files: %v: %v", entries, err)
			}
		})
	}
}
