package recordcompare

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const first = "r1\t0\tchr1\t1\t60\t2M\t*\t0\t0\tAC\t!I\tNM:i:0\tRG:Z:g"
const second = "r1\t2048\tchr2\t9\t20\t2M\t*\t0\t0\tTG\tII\tSA:Z:chr1,1,+,2M,60,0;"

func parseTest(t *testing.T, s string) *Records {
	t.Helper()
	r, e := Parse(strings.NewReader(s))
	if e != nil {
		t.Fatal(e)
	}
	return r
}
func TestOrderingAndHeaders(t *testing.T) {
	a := parseTest(t, "@HD\tVN:1.6\n"+first+"\n"+second+"\n")
	b := parseTest(t, "@HD\tVN:1.5\n"+second+"\n"+strings.Replace(first, "NM:i:0\tRG:Z:g", "RG:Z:g\tNM:i:0", 1)+"\n")
	r := Compare(a, b)
	if !r.CanonicalEqual || r.ByteIdentical || r.Left.CanonicalSHA256 != r.Right.CanonicalSHA256 || r.Left.HeaderSHA256 == r.Right.HeaderSHA256 {
		t.Fatalf("unexpected report %+v", r)
	}
	if r.Left.Records != 2 || r.Left.Headers != 1 || !r.HeadersExcluded {
		t.Fatal("wrong counts")
	}
}
func TestChanges(t *testing.T) {
	for _, change := range []struct{ old, new string }{{"AC", "AG"}, {"!I", "II"}, {"NM:i:0", "NM:i:1"}, {"\t60\t", "\t59\t"}, {"\t0\tchr1", "\t16\tchr1"}} {
		a := parseTest(t, first+"\n")
		b := parseTest(t, strings.Replace(first, change.old, change.new, 1)+"\n")
		r := Compare(a, b)
		if r.CanonicalEqual || r.DifferentRecords != 2 || r.LeftOnly != 1 || r.RightOnly != 1 {
			t.Fatalf("change %+v escaped %+v", change, r)
		}
	}
}
func TestMultiplicity(t *testing.T) {
	a := parseTest(t, first+"\n"+first+"\n"+second+"\n")
	b := parseTest(t, first+"\n"+second+"\n")
	r := Compare(a, b)
	if r.CanonicalEqual || r.DifferentRecords != 1 || r.LeftOnly != 1 || r.RightOnly != 0 || len(r.Examples) != 1 || r.Examples[0].LeftCount != 2 {
		t.Fatalf("unexpected %+v", r)
	}
}
func TestMalformed(t *testing.T) {
	fields := strings.Split(first, "\t")
	invalid := []string{"", "r\t0", first + "\t", first + "\tNM:i:2", first + "\tXX:q:1", first + "\tXX:i:no", first + "\tXX:B:C,256", first + "\tXX:H:abc", first + "\tX:i:2"}
	for _, tc := range []struct {
		field int
		value string
	}{{0, "r name"}, {1, "-1"}, {1, "65536"}, {3, "-1"}, {3, "2147483648"}, {4, "256"}, {4, "x"}, {5, "M2"}, {5, ""}, {7, "x"}, {8, "2147483648"}, {9, "A?"}, {10, "I"}} {
		xs := append([]string{}, fields...)
		xs[tc.field] = tc.value
		invalid = append(invalid, strings.Join(xs, "\t"))
	}
	for _, s := range invalid {
		if _, e := Canonical(s); e == nil {
			t.Fatalf("accepted malformed %q", s)
		}
	}
	if _, e := Parse(strings.NewReader(first + "\n@HD\tVN:1.6\n")); e == nil {
		t.Fatal("accepted header after record")
	}
}
func TestTagTypes(t *testing.T) {
	for _, tag := range []string{"XA:A:!", "XB:i:-2147483648", "XC:i:4294967295", "XD:f:1.25", "XE:Z:", "XF:H:00aF", "XG:B:c,-128,127", "XH:B:C,0,255", "XI:B:s,-32768,32767", "XJ:B:S,0,65535", "XK:B:i,-2147483648,2147483647", "XL:B:I,0,4294967295", "XM:B:f,1.2,-2"} {
		if _, e := Canonical(first + "\t" + tag); e != nil {
			t.Fatal(e)
		}
	}
}
func TestPathHashes(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.sam"), filepath.Join(dir, "b.sam")
	if e := os.WriteFile(a, []byte(first+"\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(b, []byte(first+"\n"), 0600); e != nil {
		t.Fatal(e)
	}
	r, e := ComparePaths(a, b)
	if e != nil || !r.ByteIdentical || !r.CanonicalEqual {
		t.Fatalf("%+v %v", r, e)
	}
	if e := os.WriteFile(b, []byte("@CO\tdifferent header\n"+first+"\n"), 0600); e != nil {
		t.Fatal(e)
	}
	r, e = ComparePaths(a, b)
	if e != nil || r.ByteIdentical || !r.CanonicalEqual {
		t.Fatalf("%+v %v", r, e)
	}
}
func TestBoundedExamples(t *testing.T) {
	var input strings.Builder
	for i := 0; i < MaxExamples+3; i++ {
		input.WriteString(strings.Replace(first, "r1", strings.Repeat("a", 200)+string(rune('a'+i)), 1) + "\tXX:Z:" + strings.Repeat("x", 1000) + "\n")
	}
	r := Compare(parseTest(t, input.String()), parseTest(t, ""))
	if len(r.Examples) != MaxExamples || r.DifferentRecords != MaxExamples+3 {
		t.Fatalf("unexpected %+v", r)
	}
	for _, e := range r.Examples {
		if !e.Truncated || len(e.Prefix) != MaxExampleBytes {
			t.Fatal("unbounded example")
		}
	}
}
func TestBAMDecode(t *testing.T) {
	if _, e := exec.LookPath("samtools"); e != nil {
		t.Skip("samtools unavailable")
	}
	dir := t.TempDir()
	sam, bam := filepath.Join(dir, "input.sam"), filepath.Join(dir, "stdout")
	if e := os.WriteFile(sam, []byte("@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:100\n"+first+"\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if out, e := exec.Command("samtools", "view", "-b", "-o", bam, sam).CombinedOutput(); e != nil {
		t.Fatalf("%v %s", e, out)
	}
	r, e := ComparePaths(sam, bam)
	if e != nil || !r.CanonicalEqual || r.ByteIdentical {
		t.Fatalf("%+v %v", r, e)
	}
}
