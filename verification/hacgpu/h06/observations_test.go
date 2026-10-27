package h06

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHalfObservationSpecCoversEveryCacheTimestepAndLayer(t *testing.T) {
	spec := ObservationSpec()
	if len(spec) != 936 {
		t.Fatalf("event count %d", len(spec))
	}
	seen := map[string]bool{}
	counts := map[string]int{}
	for _, v := range spec {
		key := fmt.Sprintf("%s/%d/%d/%s", v.Stage, v.Layer, v.Tick, v.Role)
		if seen[key] {
			t.Fatalf("duplicate event %+v", v)
		}
		seen[key] = true
		counts[v.Stage]++
		if v.Stage == "step" {
			wantInput, wantHidden, wantOutput := 2+v.Tick, v.Tick, 1+v.Tick
			if v.Layer%2 == 0 {
				wantInput, wantHidden, wantOutput = 32-v.Tick, 34-v.Tick, 33-v.Tick
			}
			if v.Input != wantInput || v.Hidden != wantHidden || v.Output != wantOutput {
				t.Fatalf("wrong source direction %+v", v)
			}
		}
	}
	for stage, n := range map[string]int{"cache": 28, "step": 896, "before": 2, "layer": 7, "final": 3} {
		if counts[stage] != n {
			t.Fatalf("%s count %d", stage, counts[stage])
		}
	}
	if spec[len(spec)-3].Bytes != ArenaBytes || spec[len(spec)-2].Bytes != InputBytes {
		t.Fatal("final owner extents differ")
	}
}

func TestHalfObservationManifestAcceptsCompleteStreamAndRejectsPayloadChange(t *testing.T) {
	dir := t.TempDir()
	var manifest strings.Builder
	manifest.WriteString("nanopore-hac-half-tensors/v1\n")
	files := map[int]string{}
	for _, v := range ObservationSpec() {
		source, ok := files[v.Bytes]
		if !ok {
			source = filepath.Join(dir, fmt.Sprintf("zero-%d", v.Bytes))
			if err := os.WriteFile(source, make([]byte, v.Bytes), 0600); err != nil {
				t.Fatal(err)
			}
			files[v.Bytes] = source
		}
		name := fmt.Sprintf("%s-l%d-t%d-%s.bin", v.Stage, v.Layer, v.Tick, v.Role)
		if err := os.Link(source, filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
		pin, err := artifactPin(source)
		if err != nil {
			t.Fatal(err)
		}
		dims := make([]string, len(v.Shape))
		for i, d := range v.Shape {
			dims[i] = fmt.Sprint(d)
		}
		fmt.Fprintf(&manifest, "%s\t%d\t%d\t%d\t%d\t%d\t%s\tHalf\t%s\t%d\t%s\t%s\n", v.Stage, v.Layer, v.Tick, v.Input, v.Hidden, v.Output, v.Role, strings.Join(dims, ","), v.Bytes, pin.SHA256, name)
	}
	if err := os.WriteFile(filepath.Join(dir, "tensors.tsv"), []byte(manifest.String()), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := ReadObservations(dir)
	if err != nil || len(got) != 936 {
		t.Fatalf("complete stream: count=%d err=%v", len(got), err)
	}
	f, err := os.OpenFile(files[262144], os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.WriteAt([]byte{1}, 0)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("mutate fixture: %v %v", err, closeErr)
	}
	if _, err = ReadObservations(dir); err == nil {
		t.Fatal("changed payload accepted")
	}
}

func TestHalfObservationManifestRejectsBeforePayloadTraversal(t *testing.T) {
	for name, body := range map[string]string{
		"schema":    "other\n",
		"truncated": "nanopore-hac-half-tensors/v1\n",
		"role":      "nanopore-hac-half-tensors/v1\ncache\t0\t-1\t-1\t-1\t-1\twrong\tHalf\t1024,128\t262144\tbad\t../../outside\n",
		"schedule":  "nanopore-hac-half-tensors/v1\ncache\t1\t-1\t-1\t-1\t-1\tdown-ih\tHalf\t1024,128\t262144\tbad\t../../outside\n",
		"geometry":  "nanopore-hac-half-tensors/v1\ncache\t0\t-1\t-1\t-1\t-1\tdown-ih\tHalf\t128,1024\t262144\tbad\t../../outside\n",
		"path":      "nanopore-hac-half-tensors/v1\ncache\t0\t-1\t-1\t-1\t-1\tdown-ih\tHalf\t1024,128\t262144\tbad\t../../outside\n",
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "tensors.tsv"), []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadObservations(dir); err == nil {
				t.Fatal("malformed observation accepted")
			}
		})
	}
}
