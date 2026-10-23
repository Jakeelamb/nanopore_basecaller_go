package stitch

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	stitchoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/stitch"
)

func hexField(data []byte) string {
	if len(data) == 0 {
		return "-"
	}
	return hex.EncodeToString(data)
}

func TestPinnedNativeStitch(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE to pinned Dorado checkout for untouched C++ algorithm differential; ReadCommon is shimmed")
	}
	output := os.Getenv("STITCH_ARTIFACT_DIR")
	if output == "" {
		output = t.TempDir()
	}
	executable, err := stitchoracle.Build(root, output)
	if err != nil {
		t.Fatal(err)
	}
	cases := append(upstreamCases(), randomizedCases(t)...)
	var input bytes.Buffer
	for _, entry := range cases {
		fmt.Fprintf(&input, "%d %d %t %d\n", entry.config.RawSamples, entry.config.Stride, entry.config.Variable, len(entry.chunks))
		for _, c := range entry.chunks {
			fmt.Fprintf(&input, "%d %d %d %s %s %s\n", c.Offset, c.RawSize, c.Granularity,
				hexField([]byte(c.Sequence)), hexField([]byte(c.Qstring)), hexField(c.Moves))
		}
	}
	// C++ formatted bool extraction expects 0/1, so canonicalize those tokens.
	inputBytes := bytes.ReplaceAll(bytes.ReplaceAll(input.Bytes(), []byte(" true "), []byte(" 1 ")), []byte(" false "), []byte(" 0 "))
	command := exec.Command(executable)
	command.Stdin = bytes.NewReader(inputBytes)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	nativeOutput, err := command.Output()
	if err != nil {
		t.Fatalf("native algorithm failed: %v\n%s", err, stderr.Bytes())
	}
	scanner := bufio.NewScanner(bytes.NewReader(nativeOutput))
	scanner.Buffer(make([]byte, 4096), 16<<20)
	for i, entry := range cases {
		if !scanner.Scan() {
			t.Fatalf("missing native case %d: %v", i, scanner.Err())
		}
		fields := strings.Fields(scanner.Text())
		if len(fields) != 3 {
			t.Fatalf("case %d: invalid native output %q", i, scanner.Text())
		}
		values := make([][]byte, 3)
		for j, field := range fields {
			if field == "-" {
				values[j] = []byte{}
				continue
			}
			values[j], err = hex.DecodeString(field)
			if err != nil {
				t.Fatal(err)
			}
		}
		want := Result{string(values[0]), string(values[1]), values[2]}
		got, err := Stitch(entry.config, entry.chunks)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if got.Sequence != want.Sequence || got.Qstring != want.Qstring || !bytes.Equal(got.Moves, want.Moves) {
			t.Fatalf("case %d native mismatch: got %#v want %#v", i, got, want)
		}
	}
	if scanner.Scan() || scanner.Err() != nil {
		t.Fatalf("extra native output or scan error: %v", scanner.Err())
	}
	if os.Getenv("STITCH_ARTIFACT_DIR") != "" {
		for name, data := range map[string][]byte{"inputs.txt": inputBytes, "native-output.txt": nativeOutput, "native-stderr.txt": stderr.Bytes()} {
			if err := os.WriteFile(filepath.Join(output, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("%d cases matched sequence, quality and moves against untouched stitch.cpp; ReadCommon container shim only; oracle at %s", len(cases), output)
}

func TestNativeSourceMutationRejected(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE for pinned-source rejection control")
	}
	copyRoot := t.TempDir()
	for name := range stitchoracle.SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(copyRoot, name)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := stitchoracle.VerifySources(copyRoot); err != nil {
		t.Fatal(err)
	}
	name := filepath.Join(copyRoot, "dorado/read_pipeline/base/stitch.cpp")
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 1
	if err := os.WriteFile(name, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := stitchoracle.Build(copyRoot, t.TempDir()); err == nil {
		t.Fatal("changed source reached native build")
	}
}
