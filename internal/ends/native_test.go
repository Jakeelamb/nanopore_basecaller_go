package ends

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func nativeLines(t *testing.T, input string) [][]byte {
	t.Helper()
	exe := os.Getenv("ENDS_NATIVE_ORACLE")
	if exe == "" {
		t.Skip("set ENDS_NATIVE_ORACLE to the source-pinned oracle from verification/ends")
	}
	cmd := exec.Command(exe)
	cmd.Stdin = strings.NewReader(input)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	started := time.Now().UTC()
	out, err := cmd.Output()
	finished := time.Now().UTC()
	if root := os.Getenv("ENDS_NATIVE_EVIDENCE"); root != "" {
		dir := filepath.Join(root, t.Name())
		if mkdirErr := os.Mkdir(dir, 0700); mkdirErr != nil {
			t.Fatal(mkdirErr)
		}
		for name, data := range map[string][]byte{"input": []byte(input), "stdout": out, "stderr": stderr.Bytes()} {
			if writeErr := os.WriteFile(filepath.Join(dir, name), data, 0600); writeErr != nil {
				t.Fatal(writeErr)
			}
		}
		exit := -1
		if cmd.ProcessState != nil {
			exit = cmd.ProcessState.ExitCode()
		}
		data, marshalErr := json.MarshalIndent(map[string]any{"args": cmd.Args, "started": started, "finished": finished, "exit_code": exit, "input_sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(input))), "stdout_sha256": fmt.Sprintf("%x", sha256.Sum256(out)), "stderr_sha256": fmt.Sprintf("%x", sha256.Sum256(stderr.Bytes()))}, "", "  ")
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		if writeErr := os.WriteFile(filepath.Join(dir, "receipt.json"), append(data, '\n'), 0600); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	if err != nil {
		t.Fatalf("native command: %v %s", err, stderr.Bytes())
	}
	return bytes.Split(bytes.TrimSpace(out), []byte{'\n'})
}

func TestNativeBestResult(t *testing.T) {
	var cases [][]End
	cases = append(cases, nil, []End{{.5, "first", 2, 10}, {.5, "second", 1, 9}}, []End{{.5, "first", 2, 10}, {.5, "longer", 0, 10}})
	for _, score := range []float32{.4, .5, .6, .7, .8, .9, 1} {
		for _, delta := range []float32{-.100001, -.1, -.099999, 0, .099999, .1, .100001} {
			for _, span := range []int{2, 8, 12} {
				cases = append(cases, []End{{score, "first", 2, 10}, {score + delta, "second", 1, 1 + span}, {score + .099999, "third", 0, 15}})
			}
		}
	}
	var in strings.Builder
	for _, ends := range cases {
		fmt.Fprintf(&in, "BEST %d", len(ends))
		for _, end := range ends {
			fmt.Fprintf(&in, " %d %d %d %s", math.Float32bits(end.Score), end.Start, end.End, end.Name)
		}
		in.WriteByte('\n')
	}
	lines := nativeLines(t, in.String())
	if len(lines) != len(cases) {
		t.Fatal("native result count mismatch")
	}
	for i, ends := range cases {
		var want End
		if err := json.Unmarshal(lines[i], &want); err != nil {
			t.Fatal(err)
		}
		if got := bestResult(ends); got != want {
			t.Fatalf("case %d %+v got %+v want %+v", i, ends, got, want)
		}
	}
	t.Logf("%d native score and span candidate controls passed", len(cases))
}

func TestNativeTrimInterval(t *testing.T) {
	var cases []struct {
		pair   Pair
		length int
	}
	for _, score := range []float32{-1, 0, math.Float32frombits(math.Float32bits(.8) - 1), .8, math.Float32frombits(math.Float32bits(.8) + 1), 1} {
		for _, length := range []int{0, 1, 10, 20} {
			for _, position := range [][4]int{{-1, -1, -1, -1}, {0, 0, 0, 0}, {0, 4, 5, 9}, {0, 8, 5, 9}, {0, 5, 6, 9}} {
				cases = append(cases, struct {
					pair   Pair
					length int
				}{Pair{End{score, "front", position[0], position[1]}, End{score, "rear", position[2], position[3]}}, length})
			}
		}
	}
	var in strings.Builder
	for _, c := range cases {
		fmt.Fprintf(&in, "TRIM %d", c.length)
		for _, end := range []End{c.pair.Front, c.pair.Rear} {
			fmt.Fprintf(&in, " %d %d %d %s", math.Float32bits(end.Score), end.Start, end.End, end.Name)
		}
		in.WriteByte('\n')
	}
	lines := nativeLines(t, in.String())
	if len(lines) != len(cases) {
		t.Fatal("native trim count mismatch")
	}
	for i, c := range cases {
		var want Detection
		if err := json.Unmarshal(lines[i], &want); err != nil {
			t.Fatal(err)
		}
		pair, interval := trimInterval(c.pair, c.length)
		if pair != want.Accepted || interval != want.Retained {
			t.Fatalf("case %d got %+v %+v want %+v", i, pair, interval, want)
		}
	}
	t.Logf("%d native threshold and interval controls passed", len(cases))
}
