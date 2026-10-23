package signal

import (
	"math"
	"testing"
)

func TestNormalizeMetadataAndTrimming(t *testing.T) {
	for _, length := range []int{0, 1, 9, 10, 11, 21} {
		raw := make([]int16, length)
		for i := range raw {
			raw[i] = int16(i)
		}
		result, err := Normalize(raw, Calibration{2, -3}, Standardisation{100, 20}, 1)
		if err != nil {
			t.Fatal(err)
		}
		trim := 0
		if length > 10 {
			trim = 10
		}
		if result.Trimmed != trim || len(result.Samples) != length-trim || result.Shift != 100 || result.Scale != 20 {
			t.Fatalf("length %d result %+v", length, result)
		}
		for i, bits := range result.Samples {
			expected := Float16Bits((float32(i+trim) - 54) / 10)
			if bits != expected {
				t.Fatalf("sample %d got %04x want %04x", i, bits, expected)
			}
		}
	}
}

func TestRejectInvalidParameters(t *testing.T) {
	bad := []float32{0, -1, float32(math.Inf(1)), float32(math.NaN())}
	for _, value := range bad {
		if _, err := Normalize(nil, Calibration{value, 0}, Standardisation{1, 2}, 0); err == nil {
			t.Fatalf("accepted calibration scale %v", value)
		}
		if _, err := Normalize(nil, Calibration{1, 0}, Standardisation{1, value}, 0); err == nil {
			t.Fatalf("accepted stdev %v", value)
		}
	}
	for _, value := range []float32{float32(math.Inf(-1)), float32(math.NaN())} {
		for _, entry := range []struct {
			c Calibration
			s Standardisation
			a float32
		}{
			{Calibration{1, value}, Standardisation{1, 2}, 0}, {Calibration{1, 0}, Standardisation{value, 2}, 0}, {Calibration{1, 0}, Standardisation{1, 2}, value},
		} {
			if _, err := Normalize(nil, entry.c, entry.s, entry.a); err == nil {
				t.Fatal("accepted nonfinite parameter")
			}
		}
	}
	if _, err := Normalize(nil, Calibration{math.SmallestNonzeroFloat32, 0}, Standardisation{1, 2}, 0); err == nil {
		t.Fatal("accepted derived overflow")
	}
}

var benchmarkResult Normalized

func BenchmarkNormalize(b *testing.B) {
	raw := make([]int16, 65536)
	for i := range raw {
		raw[i] = int16(i - 32768)
	}
	b.SetBytes(int64(len(raw) * 2))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var err error
		benchmarkResult, err = Normalize(raw, Calibration{0.146207, -238}, Standardisation{90, 25}, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}
