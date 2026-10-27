package h07

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

func TestOutputConsumerAdmissionPrecedesNativeLoader(t *testing.T) {
	if _, err := Consumer(context.Background(), ConsumerOptions{Backend: "absent", Out: "absent"}); err == nil {
		t.Fatal("unadmitted native loader accepted")
	}
	if _, err := Consumer(context.Background(), ConsumerOptions{Backend: "absent", Out: "absent", Admitted: true}); err == nil {
		t.Fatal("unpinned cache receipt accepted")
	}
	if _, err := Consumer(context.Background(), ConsumerOptions{Backend: "absent", Out: "absent", Admitted: true, NativeSHA: sum(nil)}); err == nil {
		t.Fatal("consumer lacking source/tool build custody accepted")
	}
	if _, err := Consumer(context.Background(), ConsumerOptions{Backend: "absent", Out: "absent", Admitted: true, NativeSHA: sum(nil), ControllerBuildSHA: sum(nil)}); err == nil {
		t.Fatal("consumer lacking prospective runtime accepted")
	}
}

func TestOutputConsumerRowAndCacheControls(t *testing.T) {
	dir := t.TempDir()
	const rowBytes = 64 * 1024
	parent := make([]byte, 37*rowBytes)
	for row := 0; row < 37; row++ {
		for i := row * rowBytes; i < (row+1)*rowBytes; i++ {
			parent[i] = byte(row)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "flstm6.bin"), parent, 0600); err != nil {
		t.Fatal(err)
	}
	w := Witness{Input: hacgpu.Stage{DataFile: "flstm6.bin"}, InputParent: parent}
	native := OracleCaptureReceipt{Output: dir}
	for i, role := range []string{"down", "down-scale", "up"} {
		body := bytes.Repeat([]byte{byte(i)}, 16)
		file := role + ".bin"
		if err := os.WriteFile(filepath.Join(dir, file), body, 0600); err != nil {
			t.Fatal(err)
		}
		native.Caches = append(native.Caches, CacheTensor{Role: role, File: file, SHA256: sum(body)})
	}
	r := ConsumerReceipt{Attempt: dir, Outputs: map[string]string{}}
	if err := cpuConsumerControls(ConsumerOptions{Capture: dir}, w, native, &r); err != nil {
		t.Fatal(err)
	}
	if len(r.Comparisons) != 1 || r.Comparisons[0].NativeOffset != 2*rowBytes || len(r.Controls) != 4 {
		t.Fatalf("missing effective controls: %+v", r)
	}
	for _, c := range r.Controls {
		if err := verifyConsumerControl(c); err != nil {
			t.Fatal(err)
		}
	}
	for i := range parent {
		parent[i] = 0
	}
	if err := os.WriteFile(filepath.Join(dir, "flstm6.bin"), parent, 0600); err != nil {
		t.Fatal(err)
	}
	r = ConsumerReceipt{Attempt: dir, Outputs: map[string]string{}}
	if err := cpuConsumerControls(ConsumerOptions{Capture: dir}, w, native, &r); err == nil {
		t.Fatal("indistinguishable wrong input row silently accepted as effective control")
	}
}
func TestOutputConsumerRetainedComparisonRejectsChangedPayload(t *testing.T) {
	dir := t.TempDir()
	goPath := filepath.Join(dir, "go-cache-down.bin")
	nativePath := filepath.Join(dir, "source-parent.bin")
	if err := os.WriteFile(goPath, []byte{1, 2}, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(nativePath, []byte{1, 2, 3, 4}, 0600); err != nil {
		t.Fatal(err)
	}
	c := OutputComparison{Role: "down", GoFile: goPath, GoSHA: sum([]byte{1, 2}), NativeFile: nativePath, NativeSHA: sum([]byte{1, 2}), Bytes: 2, Equal: true}
	if err := verifyOutputComparison(c); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goPath, []byte{1, 3}, 0600); err != nil {
		t.Fatal(err)
	}
	if err := verifyOutputComparison(c); err == nil {
		t.Fatal("overwritten retained Go payload accepted")
	}
	c.GoSHA = sum([]byte{1, 3})
	if err := verifyOutputComparison(c); err == nil {
		t.Fatal("rehashed Go corruption accepted against native bytes")
	}
}
