package signal

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	scalingoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/scaling"
)

func TestPinnedNativeScaling(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE to pinned Dorado checkout for native differential")
	}
	output := os.Getenv("SCALING_ARTIFACT_DIR")
	if output == "" {
		output = t.TempDir()
	} else if err := os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	executable, err := scalingoracle.Build(root, output)
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile(filepath.Join(root, scalingoracle.SourcePath))
	if err != nil {
		t.Fatal(err)
	}
	source[len(source)/2] ^= 1
	if _, err := scalingoracle.Extract(source); err == nil {
		t.Fatal("native oracle accepted changed source")
	}
	raw := make([]int16, 65536+10)
	for i := 0; i < 65536; i++ {
		raw[i+10] = int16(i - 32768)
	}
	cases := []struct {
		c Calibration
		s Standardisation
		a float32
	}{
		{Calibration{0.146207, -238}, Standardisation{90, 25}, 0},
		{Calibration{0.25, -12.5}, Standardisation{100, 20}, 0},
		{Calibration{0.146207, -238}, Standardisation{90, 25}, 3.75},
		{Calibration{0.125, 15.5}, Standardisation{85.25, 17.75}, -2.5},
		{Calibration{1, 0}, Standardisation{0, 1}, 0},
		{Calibration{0.2, -200}, Standardisation{95.7, 28.3}, 0.1},
	}
	for index, entry := range cases {
		args := []string{}
		for _, value := range []float32{entry.c.Scale, entry.c.Offset, entry.s.Mean, entry.s.Stdev, entry.a} {
			args = append(args, strconv.FormatFloat(float64(value), 'g', -1, 32))
		}
		data, err := exec.Command(executable, args...).Output()
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 8+65536*2 {
			t.Fatalf("native output size %d", len(data))
		}
		got, err := Normalize(raw, entry.c, entry.s, entry.a)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(got.Shift) != binary.LittleEndian.Uint32(data[:4]) || math.Float32bits(got.Scale) != binary.LittleEndian.Uint32(data[4:8]) {
			t.Fatalf("case %d metadata mismatch", index)
		}
		if got.Trimmed != 10 || len(got.Samples) != 65536 {
			t.Fatal("wrong native comparison input coverage")
		}
		for i, bits := range got.Samples {
			want := binary.LittleEndian.Uint16(data[8+i*2:])
			if bits != want {
				t.Fatalf("case %d ADC %d half got %04x want %04x", index, i-32768, bits, want)
			}
		}
		if os.Getenv("SCALING_ARTIFACT_DIR") != "" {
			if err := os.WriteFile(filepath.Join(output, "case-"+strconv.Itoa(index)+".bin"), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("case %d matched all 65536 ADC inputs and exact float32 pA metadata", index)
	}
	t.Logf("source SHA256 %s; source slices and native executable at %s", scalingoracle.SourceSHA256, output)
}
