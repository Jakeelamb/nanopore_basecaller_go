package model

import (
	"errors"
	"testing"
)

func TestWeightManifestIndependentAdmittedModels(t *testing.T) {
	for _, item := range []struct {
		architecture Architecture
		count        int
	}{{HAC, 50}, {SUP, 139}} {
		first, err := WeightManifest(item.architecture)
		if err != nil || len(first) != item.count {
			t.Fatalf("%s manifest: count=%d error=%v", item.architecture, len(first), err)
		}
		name, sha, dimension := first[0].Name, first[0].SHA256, first[0].Shape[0]
		first[0].Name = "foreign.tensor"
		first[0].SHA256 = "changed"
		first[0].Shape[0]++
		second, err := WeightManifest(item.architecture)
		if err != nil || second[0].Name != name || second[0].SHA256 != sha || second[0].Shape[0] != dimension {
			t.Fatal("caller mutated authoritative weight metadata", err)
		}
	}
	if _, err := WeightManifest(Architecture("unqualified")); !errors.Is(err, ErrUnsupported) {
		t.Fatal(err)
	}
}
