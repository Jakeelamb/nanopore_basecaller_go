package bamverify

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestTagsOracle compiles the pinned Dorado tag emitters against htslib and
// compares every case. It is a native tag-emission differential only: no GPU,
// model, caller or biological claim is made.
func TestTagsOracle(t *testing.T) {
	src := filepath.Join("..", "..", "external", "dorado-source")
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("pinned Dorado source is required at %s: %v", src, err)
	}
	cxx := "g++"
	if _, err := lookPath(cxx); err != nil {
		t.Fatalf("%s is required for the tags oracle: %v", cxx, err)
	}
	dir := filepath.Join(t.TempDir(), "tags-oracle")
	receipt, err := VerifyTagsOracle(src, dir, cxx)
	if err != nil {
		for _, c := range receipt.Cases {
			t.Logf("%s ported=%v passed=%v native=%v go=%v detail=%s",
				c.Name, c.Ported, c.Passed, c.NativeTags, c.GoTags, c.Detail)
		}
		t.Fatal(err)
	}
	if len(receipt.Cases) != len(TagsOracleCases()) {
		t.Fatalf("cases %d", len(receipt.Cases))
	}
	var ported, gaps, rejected int
	for _, c := range receipt.Cases {
		switch {
		case c.Ported:
			ported++
			if !c.ByteIdentical {
				t.Fatalf("%s is ported but not byte-identical: %s", c.Name, c.Detail)
			}
		case len(c.Gap) > 0:
			gaps++
		default:
			rejected++
		}
	}
	if ported != 12 || gaps != 0 || rejected != 1 {
		t.Fatalf("case mix ported=%d gaps=%d rejected=%d", ported, gaps, rejected)
	}
	// Every conditional path the writer claims must be covered by a ported case.
	for _, want := range []string{"duplex-read", "barcode", "rna-polytail", "rna-polytail-window"} {
		found := false
		for _, c := range receipt.Cases {
			if c.Name == want && c.Ported && c.ByteIdentical {
				found = true
			}
		}
		if !found {
			t.Fatalf("%s is not a ported byte-identical case", want)
		}
	}
	if receipt.SplicedSHA == "" || receipt.BinarySHA == "" || receipt.FixtureSHA == "" {
		t.Fatal("receipt is missing a pin")
	}
}

func lookPath(name string) (string, error) {
	return exec.LookPath(name)
}
