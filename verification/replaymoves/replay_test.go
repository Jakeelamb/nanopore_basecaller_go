package replaymoves

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	decodeoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/decode"
	stitchoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/stitch"
)

func TestPayloadRejections(t *testing.T) {
	dir := t.TempDir()
	data := []byte("retained bits")
	if e := os.WriteFile(filepath.Join(dir, "bits"), data, 0600); e != nil {
		t.Fatal(e)
	}
	valid := Payload{"bits", digest(data), int64(len(data))}
	if _, e := readPayload(dir, valid); e != nil {
		t.Fatal(e)
	}
	for name, p := range map[string]Payload{"escape": {"../bits", valid.SHA256, valid.Bytes}, "short": {"bits", valid.SHA256, valid.Bytes - 1}, "hash": {"bits", strings.Repeat("0", 64), valid.Bytes}} {
		t.Run(name, func(t *testing.T) {
			if _, e := readPayload(dir, p); e == nil {
				t.Fatal("invalid payload accepted")
			}
		})
	}
	if e := os.Symlink(filepath.Join(dir, "bits"), filepath.Join(dir, "link")); e != nil {
		t.Fatal(e)
	}
	if _, e := readPayload(dir, Payload{"link", valid.SHA256, valid.Bytes}); e == nil {
		t.Fatal("symlink accepted")
	}
	if _, e := floats(dir, valid, 3); e == nil {
		t.Fatal("wrong float shape accepted")
	}
}
func TestReplayRequiresFreshDirectory(t *testing.T) {
	if _, e := Replay(context.Background(), Config{Source: "unused", Output: t.TempDir(), Captures: []string{"unused"}}); e == nil {
		t.Fatal("existing output accepted")
	}
}
func TestNativeDecodeProtocolRejectsTruncation(t *testing.T) {
	if _, e := readDecode(bufioReader("41 21 2 1")); e == nil {
		t.Fatal("truncated output accepted")
	}
}
func bufioReader(s string) *bufio.Reader { return bufio.NewReader(strings.NewReader(s)) }
func copyCapture(t *testing.T, path string) string {
	t.Helper()
	dir := t.TempDir()
	entries, e := os.ReadDir(filepath.Dir(path))
	if e != nil {
		t.Fatal(e)
	}
	for _, entry := range entries {
		if !entry.Type().IsRegular() {
			t.Fatal("capture entry is not a regular file")
		}
		src, e := os.Open(filepath.Join(filepath.Dir(path), entry.Name()))
		if e != nil {
			t.Fatal(e)
		}
		dst, e := os.Create(filepath.Join(dir, entry.Name()))
		if e != nil {
			src.Close()
			t.Fatal(e)
		}
		_, e = io.Copy(dst, src)
		src.Close()
		closeErr := dst.Close()
		if e != nil {
			t.Fatal(e)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	return filepath.Join(dir, filepath.Base(path))
}
func TestRetainedCaptureRejections(t *testing.T) {
	path := os.Getenv("REPLAY_MOVES_CAPTURE")
	if path == "" {
		t.Skip("set REPLAY_MOVES_CAPTURE for copied genuine capture rejection controls")
	}
	for _, kind := range []string{"changed-score", "short-score", "self-consistent-short-score", "changed-normalized", "path-escape", "changed-move", "changed-stride"} {
		t.Run(kind, func(t *testing.T) {
			copyPath := copyCapture(t, path)
			b, e := os.ReadFile(copyPath)
			if e != nil {
				t.Fatal(e)
			}
			var m manifest
			if e = json.Unmarshal(b, &m); e != nil {
				t.Fatal(e)
			}
			switch kind {
			case "changed-score", "short-score", "self-consistent-short-score":
				p := &m.Chunks[0].Scores
				file := filepath.Join(filepath.Dir(copyPath), p.File)
				data, e := os.ReadFile(file)
				if e != nil {
					t.Fatal(e)
				}
				if kind == "changed-score" {
					data[0] ^= 1
				} else {
					data = data[:len(data)-4]
				}
				if e = os.WriteFile(file, data, 0600); e != nil {
					t.Fatal(e)
				}
				if kind == "self-consistent-short-score" {
					p.Bytes = int64(len(data))
					p.SHA256 = digest(data)
				}
			case "changed-normalized":
				b, e := readPayload(filepath.Dir(copyPath), m.Input)
				if e != nil {
					t.Fatal(e)
				}
				var in struct{ Normalized Payload }
				if e = json.Unmarshal(b, &in); e != nil {
					t.Fatal(e)
				}
				file := filepath.Join(filepath.Dir(copyPath), in.Normalized.File)
				data, e := os.ReadFile(file)
				if e != nil {
					t.Fatal(e)
				}
				data[0] ^= 1
				if e = os.WriteFile(file, data, 0600); e != nil {
					t.Fatal(e)
				}
			case "path-escape":
				m.Chunks[0].Scores.File = "../scores"
			case "changed-move":
				m.Chunks[0].Decoded.Moves[0] ^= 1
			case "changed-stride":
				b, e := readPayload(filepath.Dir(copyPath), m.Run)
				if e != nil {
					t.Fatal(e)
				}
				var run map[string]any
				if e = json.Unmarshal(b, &run); e != nil {
					t.Fatal(e)
				}
				run["Chunks"].(map[string]any)["Stride"] = float64(3)
				changed, e := json.Marshal(run)
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(filepath.Join(filepath.Dir(copyPath), m.Run.File), changed, 0600); e != nil {
					t.Fatal(e)
				}
				m.Run.Bytes = int64(len(changed))
				m.Run.SHA256 = digest(changed)
			}
			if e = saveJSON(copyPath, m); e != nil {
				t.Fatal(e)
			}
			if _, e = load(context.Background(), copyPath); e == nil {
				t.Fatal("altered capture accepted")
			}
			t.Logf("%s rejected: %v", kind, e)
		})
	}
}
func TestNativeSourceMutationRejected(t *testing.T) {
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		t.Skip("set DORADO_SOURCE for original-source mutation controls")
	}
	if e := sourcePins(source); e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{decodeoracle.SourcePath, "dorado/read_pipeline/base/stitch.cpp"} {
		t.Run(filepath.Base(name), func(t *testing.T) {
			dir := t.TempDir()
			names := map[string]bool{decodeoracle.SourcePath: true}
			for name := range stitchoracle.SourcePins {
				names[name] = true
			}
			for name := range names {
				b, e := os.ReadFile(filepath.Join(source, name))
				if e != nil {
					t.Fatal(e)
				}
				p := filepath.Join(dir, name)
				if e = os.MkdirAll(filepath.Dir(p), 0700); e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(p, b, 0600); e != nil {
					t.Fatal(e)
				}
			}
			p := filepath.Join(dir, name)
			b, e := os.ReadFile(p)
			if e != nil {
				t.Fatal(e)
			}
			b[len(b)/2] ^= 1
			if e = os.WriteFile(p, b, 0600); e != nil {
				t.Fatal(e)
			}
			if e = sourcePins(dir); e == nil {
				t.Fatal("changed source accepted")
			}
			if name == decodeoracle.SourcePath {
				_, e = decodeoracle.Build(dir, t.TempDir())
			} else {
				_, e = stitchoracle.Build(dir, t.TempDir())
			}
			if e == nil {
				t.Fatal("changed source reached native compiler")
			}
			t.Log(e)
		})
	}
}
func TestRetainedNativeMoves(t *testing.T) {
	paths := os.Getenv("REPLAY_MOVES_CAPTURES")
	if paths == "" {
		t.Skip("set REPLAY_MOVES_CAPTURES to path-list of genuine captures for native replay")
	}
	output := os.Getenv("REPLAY_MOVES_OUTPUT")
	if output == "" {
		output = filepath.Join(t.TempDir(), "attempt")
	}
	r, e := Replay(context.Background(), Config{Source: os.Getenv("DORADO_SOURCE"), Output: output, Captures: filepath.SplitList(paths)})
	if e != nil {
		t.Fatal(e)
	}
	if len(r.Captures) != len(filepath.SplitList(paths)) {
		t.Fatal("missing capture")
	}
	for _, c := range r.Captures {
		t.Logf("%s chunks=%d bases=%d move_bytes=%d moves_sha256=%s", c.Path, len(c.Chunks), c.Stitched.Bases, c.Stitched.MoveBytes, c.Stitched.MovesSHA256)
	}
	input := filepath.Join(output, "truncated-native-input")
	if e = os.Mkdir(input, 0700); e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(input, "input.txt"), []byte("2048 256 32 1120403456 1073741824 3197737370 1065520988\n0\n"), 0600); e != nil {
		t.Fatal(e)
	}
	execution, e := run(context.Background(), filepath.Join(output, "decode-build/decode-oracle"), filepath.Join(input, "input.txt"), input)
	if e == nil || execution.ExitCode != 2 {
		t.Fatalf("short native protocol was not rejected: %+v %v", execution, e)
	}
	stderr, e := os.ReadFile(filepath.Join(input, "native-stderr.txt"))
	if e != nil {
		t.Fatal(e)
	}
	if !bytes.Contains(stderr, []byte("truncated float bits")) {
		t.Fatalf("wrong short protocol rejection: %s", stderr)
	}
	t.Logf("fresh replay and short native-input control at %s", output)
}
