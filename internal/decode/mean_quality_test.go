//go:build cgo

package decode

import (
	"math"
	"strings"
	"testing"
)

func TestMeanQScore(t *testing.T) {
	for _, tc := range []struct {
		q    string
		want float32
	}{{"", 0}, {"!", 1}, {"I", 40}, {"S", 50}, {"~", 50}, {"\x7f", 50}} {
		got, err := MeanQScore(tc.q)
		if err != nil || math.Float32bits(got) != math.Float32bits(tc.want) {
			t.Fatalf("%q got %v %v want %v", tc.q, got, err, tc.want)
		}
	}
	for _, q := range []string{" ", "\x80", strings.Repeat("I", 60) + "\n"} {
		if _, err := MeanQScore(q); err == nil {
			t.Fatalf("invalid quality %q accepted", q)
		}
	}
}
