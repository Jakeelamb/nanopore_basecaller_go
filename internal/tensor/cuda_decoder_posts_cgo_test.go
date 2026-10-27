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

func DecoderPostFakeEngineForTest(t *testing.T) *Engine { return fakeStorageEngine(t) }

func runDecoderPostFakeCC(t *testing.T, cmd *exec.Cmd) ([]byte, error) {
	t.Helper()
	root := os.Getenv("D04_FAKE_CC_RECEIPTS")
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

func TestDecoderPostClosedNumericMarshallingThroughFakeLoader(t *testing.T) {
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
		if e := s.Close(); e != nil {
			t.Error(e)
		}
		if e := d.Close(); e != nil {
			t.Error(e)
		}
	})
	program, err := d.DecoderPosteriorProgram(s)
	if err != nil || string(program) != "fake" {
		t.Fatal("program projection", string(program), err)
	}
	rows := func(dtype DType, n, width int64, words []uint32) *Storage {
		data := make([]byte, int(n*width)*dtype.Size())
		for i, x := range words {
			if dtype == DTypeF16 {
				binary.LittleEndian.PutUint16(data[2*i:], uint16(x))
			} else {
				binary.LittleEndian.PutUint32(data[4*i:], x)
			}
		}
		h, e := d.NewStorage(s, dtype, []int64{n, width}, []KoiTag{TagN, TagC}, data)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() {
			if e := h.Close(); e != nil {
				t.Error(e)
			}
		})
		return h
	}
	f := math.Float32bits
	check := func(h *Storage, want []uint32) {
		data, e := h.Download(s)
		if e != nil {
			t.Fatal(e)
		}
		for i, x := range want {
			if binary.LittleEndian.Uint32(data[4*i:]) != x {
				t.Fatalf("numeric field%d got%x want%x", i, binary.LittleEndian.Uint32(data[4*i:]), x)
			}
		}
	}
	score := rows(DTypeF16, 64, 1024, []uint32{0x3400})
	prev := rows(DTypeF32, 64, 256, []uint32{f(1.5)})
	alpha := rows(DTypeF32, 64, 256, nil)
	if e := d.DecoderForwardFrame(s, score, prev, alpha, 1.25, -.75); e != nil {
		t.Fatal(e)
	}
	check(alpha, []uint32{f(1.25), f(-.75), f(0x3400), f(1.5)})
	beta := rows(DTypeF32, 64, 256, nil)
	post := rows(DTypeF16, 64, 256, nil)
	if e := d.DecoderPosteriorRow(s, prev, beta, post); e != nil {
		t.Fatal(e)
	}
	data, downloadErr := post.Download(s)
	if downloadErr != nil || binary.LittleEndian.Uint16(data) != 0x2e00 {
		t.Fatal("posterior operands swapped", downloadErr)
	}
	selected := rows(DTypeF16, 2, 9, []uint32{0x3400, 0x3000, 0, 0, 0, 0, 0, 0, 0x3800, 0x3800, 0, 0x3400, 0, 0, 0, 0, 0, 0x3400})
	counts := rows(DTypeI32, 2, 1, []uint32{2, 3})
	out := rows(DTypeF32, 2, 1, nil)
	if e := d.DecoderQualityProbabilities(s, selected, counts, out); e != nil {
		t.Fatal(e)
	}
	check(out, []uint32{f(.1953125), f(.1875)})
	probs := rows(DTypeF32, 2, 3, []uint32{f(.25), f(.5), f(.75), f(.5), f(.25), f(1)})
	groupCounts := rows(DTypeI32, 2, 1, []uint32{3, 3})
	chars, er := d.NewStorage(s, DTypeU8, []int64{2, 1}, []KoiTag{TagN, TagC}, []byte{0, 0})
	if er != nil {
		t.Fatal(er)
	}
	t.Cleanup(func() {
		if e := chars.Close(); e != nil {
			t.Error(e)
		}
	})
	if e := d.DecoderQualityGroups(s, probs, groupCounts, out, chars, 1.25, -.5); e != nil {
		t.Fatal(e)
	}
	check(out, []uint32{f(2.75), f(3.5)})
	raw, er := chars.Download(s)
	if er != nil || len(raw) != 2 || raw[0] != 37 || raw[1] != 37 {
		t.Fatal("group auxiliary output", raw, er)
	}
	summaryCounts := rows(DTypeI32, 2, 2, []uint32{6, 3, 6, 0})
	summary := rows(DTypeF32, 2, 3, nil)
	if e := d.DecoderQualitySummaries(s, probs, summaryCounts, summary, chars); e != nil {
		t.Fatal(e)
	}
	check(summary, []uint32{f(1.75), f(18), f(-.5), f(2.5), f(6), 0x7f800000})
	raw, er = chars.Download(s)
	if er != nil || raw[0] != 1 || raw[1] != 1 {
		t.Fatal("summary write markers", raw, er)
	}
	alias, er := probs.View(0, []int64{2, 1}, []int64{1, 1}, []KoiTag{TagN, TagC})
	if er != nil {
		t.Fatal(er)
	}
	defer alias.Close()
	if e := d.DecoderQualityGroups(s, probs, groupCounts, alias, chars, 1, 0); e == nil {
		t.Fatal("writable alias admitted")
	}
	if e := out.Close(); e != nil {
		t.Fatal(e)
	}
	if e := d.DecoderQualityProbabilities(s, selected, counts, out); e == nil {
		t.Fatal("closed output admitted")
	}
}

func TestDecoderPostMandatorySymbolRejectedByFakeLoader(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "missing-post.c")
	library := filepath.Join(dir, "missing-post.so")
	body := fakeStorageTypes + strings.Replace(fakeStorageBackend, "int nt_cuda_decoder_post_numbers(", "int missing_nt_cuda_decoder_post_numbers(", 1) + fakeSUPBackend
	for _, name := range fakeStorageStubs {
		body += "void " + name + "(void){}\n"
	}
	if e := os.WriteFile(source, []byte(body), 0600); e != nil {
		t.Fatal(e)
	}
	if output, e := runDecoderPostFakeCC(t, exec.Command("cc", "-shared", "-fPIC", source, "-o", library)); e != nil {
		t.Fatal(string(output), e)
	}
	if _, e := Open(library); e == nil || !strings.Contains(e.Error(), "missing ABI symbol nt_cuda_decoder_post_numbers") {
		t.Fatal("mandatory closed-op symbol accepted", e)
	}
}
