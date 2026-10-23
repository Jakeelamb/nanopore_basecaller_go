package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedManifest(t *testing.T) {
	var rs []WeightReceipt
	if err := json.Unmarshal(hacWeightManifest, &rs); err != nil {
		t.Fatal(err)
	}
	if len(rs) != 50 {
		t.Fatalf("weight count %d", len(rs))
	}
	seen := map[string]bool{}
	total := int64(0)
	for _, r := range rs {
		if seen[r.Name] || len(r.SHA256) != 64 {
			t.Fatalf("bad receipt %v", r)
		}
		seen[r.Name] = true
		n := int64(1)
		for _, d := range r.Shape {
			if d < 1 {
				t.Fatal(r)
			}
			n *= d
		}
		total += n
	}
	if total != 9808240 {
		t.Fatalf("elements %d", total)
	}
	if rs[0].Name != "0.conv.weight.tensor" || rs[48].Name != "11.linear.weight.tensor" || rs[49].Name != "12.linear.weight.tensor" {
		t.Fatal("wrong source weight order")
	}
}

func TestUnknownConfig(t *testing.T) {
	for _, b := range [][]byte{nil, []byte("[model]\n"), []byte(HACConfigSHA256)} {
		if _, err := Identify(b); err == nil {
			t.Fatal("unknown config accepted")
		}
	}
	if _, err := Open(nil, "missing"); err == nil {
		t.Fatal("nil engine accepted")
	}
}

func TestConfigPins(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE to check exact stock config bytes")
	}
	models := filepath.Join(filepath.Dir(filepath.Dir(root)), "models")
	for _, name := range []string{"dna_r10.4.1_e8.2_400bps_hac@v6.0.0", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"} {
		data, err := os.ReadFile(filepath.Join(models, name, "config.toml"))
		if err != nil {
			t.Fatal(err)
		}
		spec, err := Identify(data)
		if err != nil {
			t.Fatal(err)
		}
		if spec.Name != name || spec.Stride != 6 || spec.MeanQStart != 60 {
			t.Fatal(spec)
		}
		if _, err = Identify(append(data, '\n')); err == nil {
			t.Fatal("mutated config accepted")
		}
	}
}
