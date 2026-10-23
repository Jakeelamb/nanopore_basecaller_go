//go:build cgo

package caller

import (
	"bufio"
	"bytes"
	"context"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestCapturedNativeEndpoint(t *testing.T) {
	manifest, stock := os.Getenv("CALLER_CAPTURE_MANIFEST"), os.Getenv("CALLER_STOCK_FASTQ")
	if manifest == "" || stock == "" {
		t.Skip("set CALLER_CAPTURE_MANIFEST and CALLER_STOCK_FASTQ to replay the retained genuine CPU endpoint; this test never runs inference")
	}
	got, err := Replay(context.Background(), manifest)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(stock)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(lines) != 4 || !strings.HasPrefix(lines[0], "@") || !strings.HasPrefix(lines[2], "+") {
		t.Fatal("stock control is not exactly one FASTQ record")
	}
	fields := strings.Fields(lines[0][1:])
	if len(fields) == 0 || got.ID != fields[0] || got.Sequence != lines[1] || got.Qstring != lines[3] {
		t.Fatal("captured Go endpoint differs from retained stock UUID, sequence or quality")
	}
	raw, err := os.ReadFile(stock)
	if err != nil {
		t.Fatal(err)
	}
	var formatted bytes.Buffer
	if err := got.WriteFASTQ(&formatted); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(formatted.Bytes(), raw) {
		t.Fatal("replayed FASTQ differs from retained stock raw bytes")
	}
	second, err := Replay(context.Background(), manifest)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(second, got) {
		t.Fatal("replay changed sequence, quality or moves")
	}
	t.Logf("exact stock raw FASTQ bytes; read=%s bases=%d; Go moves replay exact; native moves unavailable", got.ID, len(got.Sequence))
}
