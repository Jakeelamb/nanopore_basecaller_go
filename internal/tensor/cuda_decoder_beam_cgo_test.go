//go:build cgo && linux

package tensor

import (
	"crypto/sha256"
	"encoding/binary"
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

func DecoderBeamFakeEngineForTest(t *testing.T) *Engine { return fakeStorageEngine(t) }

func runDecoderBeamFakeCC(t *testing.T, cmd *exec.Cmd) ([]byte, error) {
	t.Helper()
	root := os.Getenv("D03_FAKE_CC_RECEIPTS")
	if root == "" {
		return cmd.CombinedOutput()
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(root, "fake-cc-")
	if err != nil {
		t.Fatal(err)
	}
	pins := map[string]string{}
	for _, arg := range cmd.Args {
		if strings.HasSuffix(arg, ".c") {
			data, err := os.ReadFile(arg)
			if err != nil {
				t.Fatal(err)
			}
			pins[arg] = fmt.Sprintf("%x", sha256.Sum256(data))
			if err := os.WriteFile(filepath.Join(dir, "input.c"), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	start := time.Now().UTC()
	output, failure := cmd.CombinedOutput()
	exit := 0
	if failure != nil {
		exit = -1
		if e, ok := failure.(*exec.ExitError); ok {
			exit = e.ExitCode()
		}
	}
	receipt := map[string]any{"argv": cmd.Args, "test": t.Name(), "start_utc": start, "end_utc": time.Now().UTC(), "exit": exit, "source_pins": pins, "combined_sha256": fmt.Sprintf("%x", sha256.Sum256(output))}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "receipt.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "combined"), output, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("host fake compiler receipt %s", dir)
	return output, failure
}

func TestDecoderBeamGroupedNumericMarshallingThroughFakeLoader(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	program, err := d.DecoderBeamProgram(s)
	if err != nil || string(program) != "beam" {
		t.Fatalf("retained fake program=%q err=%v", program, err)
	}
	rows := func(dtype DType, width int64, values []uint32) *Storage {
		data := make([]byte, 2*int(width)*dtype.Size())
		for i, v := range values {
			if dtype == DTypeF16 {
				binary.LittleEndian.PutUint16(data[2*i:], uint16(v))
			} else {
				binary.LittleEndian.PutUint32(data[4*i:], v)
			}
		}
		h, err := d.NewStorage(s, dtype, []int64{2, width}, []KoiTag{TagN, TagC}, data)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := h.Close(); err != nil {
				t.Error(err)
			}
		})
		return h
	}
	floatBits := func(v ...float32) []uint32 {
		b := make([]uint32, len(v))
		for i, x := range v {
			b[i] = math.Float32bits(x)
		}
		return b
	}
	a := rows(DTypeF32, 2, floatBits(1, 2, 3, 4))
	out := rows(DTypeF32, 1, nil)
	for _, v := range []struct {
		code float32
		call func(*Stream, *Storage, *Storage) error
	}{
		{3, d.DecoderBeamLogAddPairs}, {4, d.DecoderBeamSubPairs}, {5, d.DecoderBeamMidpoints}, {6, d.DecoderBeamMaxPairs},
	} {
		if err := v.call(s, a, out); err != nil {
			t.Fatal(err)
		}
		data, err := out.Download(s)
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range []float32{3 + v.code, 7 + v.code} {
			if binary.LittleEndian.Uint32(data[4*i:]) != math.Float32bits(want) {
				t.Fatalf("op%g pair row%d differs", v.code, i)
			}
		}
	}
	previous := rows(DTypeF32, 1, floatBits(11, 12))
	transitions := rows(DTypeF16, 4, []uint32{0x3400, 1, 2, 3, 0x3800, 4, 5, 6})
	guides := rows(DTypeF32, 5, floatBits(21, 22, 23, 24, 25, 31, 32, 33, 34, 35))
	expanded := rows(DTypeF32, 6, nil)
	if err := d.DecoderBeamExpand5(s, previous, transitions, guides, expanded, 1.25, -.75); err != nil {
		t.Fatal(err)
	}
	data, err := expanded.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{2, 11, 0x3400, 21, 1.25, -.75, 2, 12, 0x3800, 31, 1.25, -.75} {
		if binary.LittleEndian.Uint32(data[4*i:]) != math.Float32bits(want) {
			t.Fatalf("expand field%d differs", i)
		}
	}
	if err := d.DecoderBeamLogCuts(s, previous, out); err != nil {
		t.Fatal(err)
	}
	data, err = out.Download(s)
	if err != nil || binary.LittleEndian.Uint32(data) != math.Float32bits(12) {
		t.Fatal("log operation dispatch differs", err)
	}
	if err := d.DecoderBeamSubPairs(s, a, a); err == nil {
		t.Fatal("writable alias admitted")
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.DecoderBeamSubPairs(s, a, out); err == nil {
		t.Fatal("closed output admitted")
	}
}

func TestDecoderBeamLoaderRejectsEitherMissingGroupedSymbolWithoutNativeBackend(t *testing.T) {
	for _, missing := range []string{"nt_cuda_decoder_beam_program", "nt_cuda_decoder_beam_numbers"} {
		t.Run(missing, func(t *testing.T) {
			dir := t.TempDir()
			source := filepath.Join(dir, "missing-beam.c")
			lib := filepath.Join(dir, "missing-beam.so")
			body := fakeStorageTypes + fakeStorageBackend + fakeSUPBackend
			body = strings.Replace(body, "int "+missing+"(", "int omitted_"+missing+"(", 1)
			for _, name := range fakeStorageStubs {
				body += "void " + name + "(void){}\n"
			}
			if err := os.WriteFile(source, []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			if data, err := runDecoderBeamFakeCC(t, exec.Command("cc", "-shared", "-fPIC", source, "-o", lib)); err != nil {
				t.Fatalf("fake shim compile %s %v", data, err)
			}
			if _, err := Open(lib); err == nil || !strings.Contains(err.Error(), "missing ABI symbol "+missing) {
				t.Fatalf("missing symbol accepted or wrong rejection %v", err)
			}
		})
	}
}
