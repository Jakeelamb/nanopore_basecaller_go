package model

import (
	"encoding/json"
	"testing"
)

func TestSUPManifest(t *testing.T) {
	var receipts []WeightReceipt
	if err := json.Unmarshal(supWeightManifest, &receipts); err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 139 {
		t.Fatalf("weight count %d", len(receipts))
	}
	seen := map[string]bool{}
	elements := int64(0)
	for _, r := range receipts {
		if seen[r.Name] || len(r.SHA256) != 64 {
			t.Fatal(r)
		}
		seen[r.Name] = true
		n := int64(1)
		for _, d := range r.Shape {
			if d < 1 {
				t.Fatal(r)
			}
			n *= d
		}
		elements += n
	}
	if elements != 78718144 {
		t.Fatalf("elements %d", elements)
	}
	for i := 0; i < 18; i++ {
		if receipts[10+7*i+6].Shape[0] != 512 {
			t.Fatal("wrong transformer weight order")
		}
	}
	if receipts[136].Name != "upsample.linear.weight.tensor" || receipts[138].Name != "crf.linear.weight.tensor" {
		t.Fatal("wrong final weight order")
	}
}

func TestAttentionSplitBoundaries(t *testing.T) {
	for _, time := range []int64{1, 7, 64, 128, 260, 1024, 4096} {
		splits := attentionSplits(time, 127, 128)
		if len(splits) < 1 || len(splits) > 12 {
			t.Fatal(time, splits)
		}
		end := int64(0)
		for _, s := range splits {
			if s.qb != end || s.qe <= s.qb || s.qe > time || s.kvb != max(int64(0), s.qb-128) || s.kve != min(time, s.qe+127) {
				t.Fatal(time, s)
			}
			mask := splitMask(s, 127, 128)
			for q := s.qb; q < s.qe; q++ {
				for k := s.kvb; k < s.kve; k++ {
					want := byte(0)
					if k-q >= -127 && k-q <= 128 {
						want = 1
					}
					if mask[(q-s.qb)*(s.kve-s.kvb)+k-s.kvb] != want {
						t.Fatal(time, s, q, k)
					}
				}
			}
			end = s.qe
		}
		if end != time {
			t.Fatal(time, end)
		}
	}
	s := attentionSplits(260, 127, 128)[0]
	if s.qe != 24 || s.kve != 151 {
		t.Fatal(s)
	}
	// The full mask permits key 151 for query 23. Split 0 stops before key 151.
	if 151-23 > 128 || s.kve != 151 {
		t.Fatal("fixture does not expose excluded split-edge key")
	}
}
