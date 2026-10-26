package hacgpu

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// These tests run copied scripts with a fake Go command. They never build a
// backend, access CUDA, or mutate the repository's native sources.
func TestControlScriptFailureAndEvidence(t *testing.T) {
	for _, unit := range []string{"h02", "h03"} {
		t.Run(unit, func(t *testing.T) {
			for _, mode := range []string{"held", "wrong-result", "missing-result", "build-failed", "probe-failed", "unknown"} {
				t.Run(mode, func(t *testing.T) {
					root := t.TempDir()
					write := func(name string, data []byte, perm os.FileMode) {
						t.Helper()
						path := filepath.Join(root, name)
						if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
							t.Fatal(err)
						}
						if err := os.WriteFile(path, data, perm); err != nil {
							t.Fatal(err)
						}
					}
					read := func(path string) []byte {
						t.Helper()
						data, err := os.ReadFile(path)
						if err != nil {
							t.Fatal(err)
						}
						return data
					}
					script := filepath.Join("verification/hacgpu", unit+"-controls.sh")
					write(script, read(unit+"-controls.sh"), 0755)
					sourceName := "hac_smallconv.inc"
					control := "omitted-input-padding"
					if unit == "h03" {
						sourceName, control = "hac_lastconv.inc", "padding"
					}
					if mode == "unknown" {
						control = "missing-control"
					}
					sourcePath := filepath.Join("verification/tensor", sourceName)
					original := read(filepath.Join("../tensor", sourceName))
					write(sourcePath, original, 0644)
					write("fakebin/go", []byte(`#!/bin/bash
case "$*" in
  *verification/tensor/build*)
    [ "$CONTROL_TEST_MODE" != build-failed ] || exit 17
    exit 0 ;;
esac
[ "$CONTROL_TEST_MODE" != probe-failed ] || exit 19
[ "$CONTROL_TEST_MODE" != missing-result ] || exit 0
case "$*" in
  *build-baseline*) echo 'padding: original guard'; exit 0 ;;
esac
if [ "$CONTROL_TEST_MODE" = held ]; then
  echo 'padding: output batch stride differs from the view; last convolution input rows are not T_in plus two paddings'
else
  echo 'padding: original guard'
fi
`), 0755)
					out := filepath.Join(root, "attempt")
					run := func() ([]byte, error) {
						cmd := exec.Command("bash", filepath.Join(root, script), control)
						cmd.Env = append(os.Environ(), "PATH="+filepath.Join(root, "fakebin")+":"+os.Getenv("PATH"), "OUT="+out, "CONTROL_TEST_MODE="+mode)
						return cmd.CombinedOutput()
					}
					output, err := run()
					if (err == nil) != (mode == "held") {
						t.Fatalf("mode %s: %v\n%s", mode, err, output)
					}
					if !bytes.Equal(original, read(filepath.Join(root, sourcePath))) {
						t.Fatal("script did not restore its source")
					}
					if mode == "held" {
						transcriptPath := filepath.Join(out, "transcript.txt")
						before := read(transcriptPath)
						if output, err := run(); err == nil {
							t.Fatalf("existing attempt accepted: %s", output)
						}
						if !bytes.Equal(before, read(transcriptPath)) {
							t.Fatal("reused attempt changed retained transcript")
						}
					}
				})
			}
		})
	}
}
