package koiquant

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestFiniteRangeFixturesExhaustAllPositiveFiniteHalfRanges(t *testing.T) {
	seen := map[uint16]int{}
	for batch := 0; batch < 21; batch++ {
		values, err := fixtureHalf(fmt.Sprintf("finite-ranges-%02d", batch))
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < 1536; row++ {
			maximum := values[row*512]
			if maximum == 0 || maximum > 0x7bff {
				t.Fatalf("nonfinite or zero range %04x", maximum)
			}
			seen[maximum]++
			for col := 0; col < 512; col++ {
				if values[row*512+col]&0x7fff != maximum {
					t.Fatal("row range is not fixed")
				}
			}
		}
	}
	if len(seen) != 31743 || seen[0x7bff] != 514 {
		t.Fatalf("incomplete range coverage %d, last count %d", len(seen), seen[0x7bff])
	}
	for bits := uint16(1); bits <= 0x7bff; bits++ {
		if seen[bits] == 0 {
			t.Fatalf("missing range %04x", bits)
		}
	}
}

func TestClosedCaptureRequiresActualFieldPresence(t *testing.T) {
	m := RawCapture{Environment: map[string]EnvironmentValue{"CUDA_DEVICE_ORDER": {false, ""}}}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var tree map[string]any
	if err = json.Unmarshal(b, &tree); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err = os.WriteFile(path, b, 0644); err != nil {
		t.Fatal(err)
	}
	var got RawCapture
	if err = readClosed(path, &got); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"Ordinal", "Error", "BinaryOrigin", "Commands", "MappedBefore"} {
		value := tree[field]
		delete(tree, field)
		if err = writeJSON(path, tree); err != nil {
			t.Fatal(err)
		}
		if err = readClosed(path, &got); err == nil || !strings.Contains(err.Error(), "missing required field") {
			t.Fatalf("deleted field accepted %s: %v", field, err)
		}
		tree[field] = value
	}
	for _, raw := range []string{
		`{"Ordinal":0,"Ordinal":1}`,
		strings.Replace(string(b), `"Ordinal":0`, `"Ordinal":null`, 1),
		strings.Replace(string(b), `"Present":false,`, ``, 1),
		strings.Replace(string(b), `"Path":"",`, ``, 1),
		strings.TrimSuffix(string(b), "}") + `,"Unexpected":true}`,
		strings.TrimSuffix(string(b), "}") + `,"ordinal":1}`,
		string(b) + ` {}`,
	} {
		if err = os.WriteFile(path, []byte(raw), 0644); err != nil {
			t.Fatal(err)
		}
		if err = readClosed(path, &got); err == nil {
			t.Fatalf("open receipt accepted %s", raw)
		}
	}
}

func TestNativeAdmissionRejectsBeforeExternalCommands(t *testing.T) {
	out := filepath.Join(t.TempDir(), "attempt-blocked")
	for _, admitted := range []bool{false, true} {
		_, err := Capture(CaptureOptions{Admitted: admitted, Basic: "missing-basic", Build: "missing-build", Prepared: "missing-preparation", Out: out})
		if err == nil {
			t.Fatal("missing admission/qualification accepted")
		}
		if !admitted && !strings.Contains(err.Error(), "zero external commands") {
			t.Fatal(err)
		}
		if _, err = os.Stat(out); !os.IsNotExist(err) {
			t.Fatal("rejected admission created an attempt")
		}
	}
}

func TestCaptureUsesExistingNonblockingLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "existing.lock")
	if _, err := runner.Acquire(path); err == nil {
		t.Fatal("missing lock accepted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("missing lock was created")
	}
	const text = "preserve existing host lock bytes"
	if err := os.WriteFile(path, []byte(text), 0644); err != nil {
		t.Fatal(err)
	}
	first, err := runner.Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	if second, err := runner.Acquire(path); err == nil {
		second.Close()
		t.Fatal("held lock accepted")
	}
	b, err := os.ReadFile(path)
	if err != nil || string(b) != text {
		t.Fatal("lock was truncated")
	}
}

func intsText(values []int) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ",")
}

func syntheticMetadata(t *testing.T) (root string, input koiweights.Data) {
	t.Helper()
	stages := zeroStages(t)
	root = t.TempDir()
	dir := filepath.Join(root, "observations/genuine")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	var metadata strings.Builder
	for _, spec := range StageSpecs() {
		d := stages[spec.Name]
		path := filepath.Join(dir, spec.Name+".bin")
		if err := os.Link(d.Path, path); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&metadata, "%s\t%s\tcpu\t%s\t%s\t%d\n", spec.Name, spec.DType, intsText(spec.Shape), intsText(spec.Strides), d.Bytes)
		if spec.Name == "input" {
			input = d
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "stages.tsv"), []byte(metadata.String()), 0644); err != nil {
		t.Fatal(err)
	}
	return root, input
}

func capturedBridgeInput(t *testing.T) (root string, input koiweights.Data) {
	t.Helper()
	path := os.Getenv("NANOPORE_KOIQUANT_REPLAY")
	if path == "" {
		return syntheticMetadata(t)
	}
	var raw RawCapture
	if err := readClosed(path, &raw); err != nil {
		t.Fatal(err)
	}
	if err := validateRaw(path, raw); err != nil {
		t.Fatal(err)
	}
	root = filepath.Dir(path)
	input = raw.Prepared.Fixtures[0].Input
	if input.SHA256 != GenuineSHA256 {
		t.Fatal("replay did not select genuine input")
	}
	t.Logf("retained genuine native capture %s", path)
	return root, input
}

func TestCaptureMetadataBridge(t *testing.T) {
	root, input := capturedBridgeInput(t)
	device, ordinal := "cpu", 0
	if path := os.Getenv("NANOPORE_KOIQUANT_REPLAY"); path != "" {
		var raw RawCapture
		if err := readClosed(path, &raw); err != nil {
			t.Fatal(err)
		}
		device, ordinal = raw.Device, raw.Ordinal
	}
	o, err := importStages(root, "genuine", device, ordinal, input, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(o.Stages) != 14 || o.Classification.Rows != 1536 {
		t.Fatal("bridge lost stages or rows")
	}
	if o.Stages["input"].SHA256 != input.SHA256 {
		t.Fatal("bridge selected a different input")
	}
}

func TestCapturedRawRejection(t *testing.T) {
	root, input := capturedBridgeInput(t)
	copyRoot := t.TempDir()
	source := filepath.Join(root, "observations/genuine")
	target := filepath.Join(copyRoot, "observations/genuine")
	if err := os.MkdirAll(target, 0755); err != nil {
		t.Fatal(err)
	}
	for _, spec := range StageSpecs() {
		if err := os.Link(filepath.Join(source, spec.Name+".bin"), filepath.Join(target, spec.Name+".bin")); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(target, "stages.tsv")
	metadata, err := os.ReadFile(filepath.Join(source, "stages.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	device, ordinal := "cpu", 0
	if rawPath := os.Getenv("NANOPORE_KOIQUANT_REPLAY"); rawPath != "" {
		var raw RawCapture
		if err = readClosed(rawPath, &raw); err != nil {
			t.Fatal(err)
		}
		device, ordinal = raw.Device, raw.Ordinal
	}
	for _, mutant := range []string{
		strings.Replace(string(metadata), "hscale\tbinary16", "hscale\tfloat64", 1),
		strings.Replace(string(metadata), "hscale\t", "missing-stage\t", 1),
		string(metadata) + strings.Split(string(metadata), "\n")[0] + "\n",
		strings.Replace(string(metadata), "\tcpu\t", "\tcuda:99\t", 1),
	} {
		if err = os.WriteFile(path, []byte(mutant), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err = importStages(copyRoot, "genuine", device, ordinal, input, false); err == nil {
			t.Fatal("corrupt genuine native metadata accepted")
		}
	}
	if err = os.WriteFile(path, metadata, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = importStages(copyRoot, "genuine", device, ordinal, input, false); err != nil {
		t.Fatal(err)
	}
}

func TestMappedLibrariesDeduplicateSegmentsAndKeepLoader(t *testing.T) {
	raw := []byte("100-200 r--p 0 00:00 1 /usr/lib/libc.so.6\n200-300 r-xp 0 00:00 1 /usr/lib/libc.so.6\n300-400 r--p 0 00:00 1 /usr/lib/ld-linux-x86-64.so.2\n")
	paths, err := mappedPaths(raw)
	if err != nil || !reflect.DeepEqual(paths, []string{"/usr/lib/libc.so.6", "/usr/lib/ld-linux-x86-64.so.2"}) {
		t.Fatalf("incorrect mapped closure %v %v", paths, err)
	}
	linked, err := lddPaths([]byte("libc.so.6 => /usr/lib/libc.so.6 (0x1)\n\t/usr/lib/ld-linux-x86-64.so.2 (0x2)\n"))
	if err != nil || !reflect.DeepEqual(linked, paths) {
		t.Fatalf("linker lost loader %v %v", linked, err)
	}
	if err = requirePaths(paths, map[string]string{paths[0]: strings.Repeat("a", 64)}); err == nil {
		t.Fatal("deleted mapped pin accepted")
	}
}

func TestStageComparisonPreservesFloatBits(t *testing.T) {
	a := zeroStages(t)["freciprocal"]
	b := a
	data, err := os.ReadFile(a.Path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 1
	b.Path = filepath.Join(t.TempDir(), "different.bin")
	if err = os.WriteFile(b.Path, data, 0644); err != nil {
		t.Fatal(err)
	}
	d, err := dataDifference("synthetic", "freciprocal", a, b)
	if err != nil || d.Different != 1 || d.FirstIndex != 0 || d.FirstCPU != 0 || d.FirstCUDA != 1 {
		t.Fatalf("float bit mismatch was hidden %v %v", d, err)
	}
}

func TestArchiveSymlinkPinsDistinguishLinkTextFromTarget(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "lib")
	if err := os.Mkdir(path, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "libexample.so.1.2"), []byte("target"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("libexample.so.1.2", filepath.Join(path, "libexample.so.1")); err != nil {
		t.Fatal(err)
	}
	members := map[string]string{"libtorch/lib/libexample.so.1": digest([]byte("libexample.so.1.2")), "libtorch/lib/libexample.so.1.2": digest([]byte("target"))}
	if err := archivedPathMatches(root, filepath.Join(path, "libexample.so.1"), digest([]byte("target")), members); err != nil {
		t.Fatal(err)
	}
	delete(members, "libtorch/lib/libexample.so.1")
	if err := archivedPathMatches(root, filepath.Join(path, "libexample.so.1"), digest([]byte("target")), members); err == nil {
		t.Fatal("missing symlink pin accepted")
	}
}
