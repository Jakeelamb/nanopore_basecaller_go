package chunk

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFixed(t *testing.T) {
	c := Config{ChunkSize: 9996, Stride: 6, Overlap: 498}
	for _, tc := range []struct {
		n    uint64
		want []uint64
	}{{4998, []uint64{0}}, {9996, []uint64{0}}, {9997, []uint64{0, 6}}, {14994, []uint64{0, 4998}}, {29988, []uint64{0, 9498, 18996, 19992}}} {
		got, err := Fixed(tc.n, c)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("n=%d got %v, %v want %v", tc.n, got, err, tc.want)
		}
	}
}

func TestRepeatPad(t *testing.T) {
	signal := []uint16{0x8000, 0x7e01, 0x3c00}
	got, err := RepeatPad(signal, 8)
	want := []uint16{0x8000, 0x7e01, 0x3c00, 0x8000, 0x7e01, 0x3c00, 0x8000, 0x7e01}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %x, %v", got, err)
	}
	got[0] = 1
	if signal[0] != 0x8000 {
		t.Fatal("output aliases input")
	}
	for _, tc := range []struct {
		s []uint16
		n uint64
	}{{nil, 2}, {signal, 2}, {signal, MaxPaddedSamples + 1}} {
		if _, err := RepeatPad(tc.s, tc.n); err == nil {
			t.Fatal("expected rejection")
		}
	}
}

func TestLimits(t *testing.T) {
	c := Config{ChunkSize: 2, Stride: 1, Granularity: 1}
	for _, f := range []func(uint64, Config) error{
		func(n uint64, c Config) error { _, e := Fixed(n, c); return e },
		func(n uint64, c Config) error { _, e := Variable(n, c); return e },
		func(n uint64, c Config) error { _, e := Transformer(n, c); return e },
	} {
		for _, n := range []uint64{0, math.MaxUint64, MaxCoordinate, MaxChunks*2 + 3} {
			if err := f(n, c); err == nil {
				t.Fatalf("expected rejection n=%d", n)
			}
		}
	}
}

type oracleCase struct {
	mode string
	n    uint64
	c    Config
}

func (tc oracleCase) result() string {
	var out strings.Builder
	out.WriteString("ok")
	if tc.mode == "fixed" {
		xs, e := Fixed(tc.n, tc.c)
		if e != nil {
			return "error"
		}
		for _, x := range xs {
			fmt.Fprintf(&out, " %d", x)
		}
	} else {
		var xs []Interval
		var e error
		if tc.mode == "variable" {
			xs, e = Variable(tc.n, tc.c)
		} else {
			xs, e = Transformer(tc.n, tc.c)
		}
		if e != nil {
			return "error"
		}
		for _, x := range xs {
			fmt.Fprintf(&out, " %d,%d", x.Start, x.End)
		}
	}
	return out.String()
}

func TestNativeParity(t *testing.T) {
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		t.Skip("native parity not run: set DORADO_SOURCE to pinned Dorado source")
	}
	pin, err := exec.Command("git", "-C", source, "rev-parse", "HEAD").Output()
	if err != nil || strings.TrimSpace(string(pin)) != "8b8fc5d36a9c0baab262a743cb175b5878e38ca6" {
		t.Fatalf("wrong or unreadable Dorado source pin: %s %v", pin, err)
	}
	dirty, err := exec.Command("git", "-C", source, "status", "--porcelain", "--untracked-files=no").Output()
	if err != nil || len(dirty) != 0 {
		t.Fatalf("native source must have clean tracked files: %s %v", dirty, err)
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	artifactDir := os.Getenv("CHUNK_ARTIFACT_DIR")
	if artifactDir == "" {
		artifactDir = t.TempDir()
	} else {
		artifactDir, err = filepath.Abs(artifactDir)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(artifactDir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	retain := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(artifactDir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	binary := filepath.Join(artifactDir, "chunk-oracle")
	cmd := exec.Command("g++", "-std=c++20", "-O2", "-I", filepath.Join(source, "dorado/read_pipeline/base/include"), filepath.Join(root, "verification/chunk/oracle.cpp"), filepath.Join(source, "dorado/read_pipeline/base/chunk.cpp"), "-o", binary)
	compileOutput, compileErr := cmd.CombinedOutput()
	retain("compile.log", compileOutput)
	if compileErr != nil {
		t.Fatalf("compile oracle: %v\n%s", compileErr, compileOutput)
	}
	cases := make([]oracleCase, 0, 2000)
	configs := []Config{{9996, 6, 498, 12}, {12288, 6, 600, 12}, {555, 5, 25, 5}, {83, 1, 13, 1}, {123, 1, 0, 1}, {12, 6, 0, 6}, {12, 6, 3, 6}, {0, 6, 0, 6}, {12, 0, 0, 6}, {12, 6, 12, 6}, {12, 6, 18, 6}, {13, 6, 0, 6}, {12, 6, 6, 0}, {12, 6, 6, 5}, {6, 6, 0, 6}}
	for _, c := range configs {
		for _, n := range []uint64{0, 1, 5, 11, 12, 13, 4998, 9996, 9997, 14994, 29988} {
			for _, mode := range []string{"fixed", "variable", "transformer"} {
				cases = append(cases, oracleCase{mode, n, c})
			}
		}
	}
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1500; i++ {
		stride := uint64(1 + rng.Intn(17))
		size := stride * uint64(2+rng.Intn(400))
		overlap := stride * uint64(rng.Intn(int(size/stride)))
		c := Config{size, stride, overlap, stride}
		n := uint64(1 + rng.Intn(100000))
		for _, mode := range []string{"fixed", "variable", "transformer"} {
			cases = append(cases, oracleCase{mode, n, c})
		}
	}
	var input bytes.Buffer
	for _, tc := range cases {
		fmt.Fprintf(&input, "%s %d %d %d %d %d\n", tc.mode, tc.n, tc.c.ChunkSize, tc.c.Stride, tc.c.Overlap, tc.c.Granularity)
	}
	retain("input.txt", input.Bytes())
	cmd = exec.Command(binary)
	cmd.Stdin = &input
	var nativeStderr bytes.Buffer
	cmd.Stderr = &nativeStderr
	raw, err := cmd.Output()
	retain("native-output.txt", raw)
	retain("native-stderr.txt", nativeStderr.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 4096), 32<<20)
	for i, tc := range cases {
		if !scanner.Scan() {
			t.Fatalf("missing oracle result %d: %v", i, scanner.Err())
		}
		got, want := tc.result(), scanner.Text()
		if got != want {
			t.Fatalf("case %d %+v Go=%s native=%s", i, tc, got, want)
		}
	}
	if scanner.Scan() || scanner.Err() != nil {
		t.Fatalf("extra output or scan error: %v", scanner.Err())
	}
	t.Logf("native parity passed for %d planner cases against %s", len(cases), strings.TrimSpace(string(pin)))
}

func FuzzPlannerCoverage(f *testing.F) {
	f.Add(uint32(9997), uint16(100), uint8(6), uint16(5))
	f.Add(uint32(1), uint16(1), uint8(1), uint16(0))
	f.Fuzz(func(t *testing.T, rawN uint32, rawSize uint16, rawStride uint8, rawOverlap uint16) {
		n := uint64(rawN%100000) + 1
		stride := uint64(rawStride%17) + 1
		size := stride * (uint64(rawSize%400) + 2)
		overlap := stride * (uint64(rawOverlap) % (size / stride))
		c := Config{size, stride, overlap, stride}
		offsets, err := Fixed(n, c)
		if err != nil {
			t.Fatal(err)
		}
		if len(offsets) == 0 || offsets[0] != 0 {
			t.Fatal("fixed planner lost first sample")
		}
		for i, x := range offsets {
			if x%stride != 0 || x >= n {
				t.Fatal("invalid fixed boundary")
			}
			if i > 0 && (x <= offsets[i-1] || x > offsets[i-1]+size) {
				t.Fatal("unordered or uncovered fixed interval")
			}
		}
		if offsets[len(offsets)-1]+size < n {
			t.Fatal("fixed planner lost final sample")
		}
		for _, planner := range []func(uint64, Config) ([]Interval, error){Variable, Transformer} {
			intervals, err := planner(n, c)
			if err != nil {
				if overlap == 0 && stride != 1 {
					continue
				}
				t.Fatal(err)
			}
			if len(intervals) == 0 || intervals[0].Start != 0 || intervals[len(intervals)-1].End != n {
				t.Fatal("planner lost endpoint")
			}
			for i, x := range intervals {
				if x.Start >= x.End || x.End > n || x.End-x.Start > size {
					t.Fatal("invalid interval")
				}
				if i > 0 && (x.Start <= intervals[i-1].Start || x.Start > intervals[i-1].End) {
					t.Fatal("unordered or uncovered interval")
				}
			}
		}
	})
}
