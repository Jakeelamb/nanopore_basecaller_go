package readtrim_test

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/intervaltrim"
)

func TestPinnedNativeInterval(t *testing.T) {
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		t.Skip("set DORADO_SOURCE for original-source native parity")
	}
	root := os.Getenv("INTERVALTRIM_ARTIFACT_ROOT")
	if root == "" {
		root = "../../artifacts/intervaltrim"
	}
	root, err := filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(root, "attempt-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("native evidence %s", attempt)
	executable, err := oracle.Build(source, filepath.Join(attempt, "native-build"))
	if err != nil {
		t.Fatal(err)
	}
	var trims []oracle.TrimCase
	r := intervalFixture()
	for a := 0; a < 5; a++ {
		for b := a; b <= 5; b++ {
			trims = append(trims, oracle.TrimCase{Read: r, Interval: readtrim.Interval{Start: a, End: b}, Leading: 10})
		}
	}
	trims = append(trims, oracle.TrimCase{Read: readtrim.MuxRead{Stride: 1}})
	rng := rand.New(rand.NewSource(217))
	for k := 0; k < 800; k++ {
		n := 1 + rng.Intn(120)
		moves := make([]uint8, rng.Intn(4))
		q := make([]byte, n)
		seq := make([]byte, n)
		for i := range q {
			q[i] = byte(33 + rng.Intn(94))
			seq[i] = "ACGTN"[rng.Intn(5)]
			moves = append(moves, 1)
			for z := rng.Intn(5); z > 0; z-- {
				moves = append(moves, 0)
			}
		}
		stride := 1 + rng.Intn(9)
		samples := len(moves)*stride + rng.Intn(2*stride-1) - (stride - 1)
		signal := make([]uint16, samples)
		for i := range signal {
			signal[i] = uint16(rng.Intn(0x7c00))
			if rng.Intn(2) == 1 {
				signal[i] |= 0x8000
			}
		}
		a := rng.Intn(n)
		b := a + rng.Intn(n-a+1)
		read := readtrim.MuxRead{Sequence: string(seq), Qstring: string(q), Moves: moves, Signal: signal, Stride: uint32(stride), NumMinknowEvents: uint64(k), PreTrimSequenceLength: uint64(n + 23)}
		trims = append(trims, oracle.TrimCase{Read: read, Interval: readtrim.Interval{Start: a, End: b}, Leading: uint64(k + 10)})
	}
	var filters []oracle.FilterCase
	for _, n := range []int{0, 1, 4, 5, 59, 60, 61, 121} {
		for _, start := range []int{0, 60, n, n + 1} {
			for _, q := range []string{strings.Repeat("I", n), strings.Repeat("!", min(n, 60)) + strings.Repeat("?", max(0, n-60))} {
				selected := q
				if len(q) > start {
					selected = q[start:]
				}
				mean, err := decode.MeanQScore(selected)
				if err != nil {
					t.Fatal(err)
				}
				for _, threshold := range []float32{-1, 0, math.Nextafter32(mean, float32(math.Inf(-1))), mean, math.Nextafter32(mean, float32(math.Inf(1)))} {
					filters = append(filters, oracle.FilterCase{Sequence: strings.Repeat("A", n), Qstring: q, Start: start, Threshold: threshold})
				}
			}
		}
	}
	for k := 0; k < 300; k++ {
		n := rng.Intn(200)
		q := make([]byte, n)
		for i := range q {
			q[i] = byte(33 + rng.Intn(94))
		}
		filters = append(filters, oracle.FilterCase{Sequence: strings.Repeat("N", n), Qstring: string(q), Start: rng.Intn(90), Threshold: float32(rng.Intn(51))})
	}
	if err := oracle.Compare(executable, attempt, trims, filters); err != nil {
		t.Fatal(err)
	}
	t.Logf("exact native bytes for %d interval cases and %d filter cases", len(trims), len(filters))
	manifests := os.Getenv("INTERVALTRIM_CAPTURE_MANIFESTS")
	if manifests == "" {
		t.Run("retained", func(t *testing.T) {
			t.Skip("set INTERVALTRIM_CAPTURE_MANIFESTS for retained HAC and SUP interval comparison")
		})
		return
	}
	var retained []oracle.TrimCase
	var pins []map[string]any
	paths := filepath.SplitList(manifests)
	if len(paths) != 2 {
		t.Fatal("retained comparison requires HAC and SUP manifests in that order")
	}
	for i, path := range paths {
		read, receipt := loadRetained(t, path)
		wantModel := []string{"dna_r10.4.1_e8.2_400bps_hac@v6.0.0", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"}[i]
		if receipt["model"] != wantModel {
			t.Fatalf("retained model order mismatch: got %v want %s", receipt["model"], wantModel)
		}
		interval := []readtrim.Interval{{Start: 38, End: 1653}, {Start: 31, End: 1650}}[i]
		retained = append(retained, oracle.TrimCase{Read: read, Interval: interval, Leading: 10})
		receipt["interval"] = interval
		pins = append(pins, receipt)
	}
	dir := filepath.Join(attempt, "retained-captures")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := oracle.Compare(executable, dir, retained, nil); err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(pins, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "capture-pins.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	t.Log("retained HAC [38,1653) and SUP [31,1650) exact trim output; no default-pipeline claim")
}
