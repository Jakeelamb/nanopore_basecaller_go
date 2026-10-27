package decodergpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestHistoricalProducerRootBoundedAuthentication(t *testing.T) {
	dir := t.TempDir()
	module := filepath.Join(dir, "selected", "go.mod")
	actual := filepath.Join(dir, "producer", "go.mod")
	body, err := json.Marshal(map[string]string{"GOMOD": module})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		body     []byte
		size     int64
		wrongPin bool
		wantErr  string
	}{
		{name: "bounded", body: body},
		{name: "at-limit", body: body, size: maxPinnedMetadataBytes},
		{name: "oversized", body: body, size: maxPinnedMetadataBytes + 1, wantErr: "exceeds bounded input"},
		{name: "wrong-pin", body: body, wrongPin: true, wantErr: "pinned path/bytes changed"},
		{name: "malformed", body: []byte(`{"GOMOD":`), wantErr: "unexpected end of JSON input"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".json")
			file, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { file.Close() })
			hash := sha256.New()
			writer := io.MultiWriter(file, hash)
			if _, err := writer.Write(tc.body); err != nil {
				t.Fatal(err)
			}
			padding := bytes.Repeat([]byte{' '}, 32<<10)
			for remaining := tc.size - int64(len(tc.body)); remaining > 0; {
				n := min(remaining, int64(len(padding)))
				if _, err := writer.Write(padding[:n]); err != nil {
					t.Fatal(err)
				}
				remaining -= n
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			pin := hex.EncodeToString(hash.Sum(nil))
			if tc.wrongPin {
				pin = strings.Repeat("0", 64)
			}
			g := GoClosure{
				Commands:  []runner.CommandReceipt{{}, {Stdout: path, StdoutSHA: pin}, {}},
				Selection: map[string]string{module: actual},
				Pins:      map[string]string{actual: strings.Repeat("1", 64)},
			}
			root, err := historicalProducerRoot(g)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) || root != "" {
					t.Fatalf("expected %q and empty root, got root=%q err=%v", tc.wantErr, root, err)
				}
			} else if err != nil || root != filepath.Dir(actual) {
				t.Fatalf("expected selected producer root %q, got root=%q err=%v", filepath.Dir(actual), root, err)
			}
		})
	}
}
