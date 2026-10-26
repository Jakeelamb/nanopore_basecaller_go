package h06

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

type Observation struct {
	Stage, Role                        string
	Layer, Tick, Input, Hidden, Output int
	Shape                              []int64
	Bytes                              int
	File                               nativeproof.Pin
}

// ObservationSpec fixes the source observer's entire ordered event stream.
// The first gates snapshot is taken after the up GEMM, before the cell update.
func ObservationSpec() []Observation {
	var out []Observation
	add := func(stage, role string, layer, tick, ti, th, to int, shape ...int64) {
		n := 2
		for _, d := range shape {
			n *= int(d)
		}
		out = append(out, Observation{Stage: stage, Role: role, Layer: layer, Tick: tick, Input: ti, Hidden: th, Output: to, Shape: shape, Bytes: n})
	}
	for layer := 0; layer < 7; layer++ {
		for _, role := range []string{"down-ih", "down-hh"} {
			add("cache", role, layer, -1, -1, -1, -1, 1024, 128)
		}
		add("cache", "up", layer, -1, -1, -1, -1, 256, 4096)
		add("cache", "bias", layer, -1, -1, -1, -1, 4096)
		for tick := 0; tick < Frames; tick++ {
			ti, th, to := 2+tick, tick, 1+tick
			if layer%2 == 0 {
				ti, th, to = Frames-tick, Frames-tick+2, Frames-tick+1
			}
			if layer == 0 && tick == 0 {
				add("before", "gates", layer, tick, ti, th, to, 64, 4096)
				add("before", "cell", layer, tick, ti, th, to, 64, 1024)
			}
			add("step", "down", layer, tick, ti, th, to, 64, 256)
			add("step", "gates", layer, tick, ti, th, to, 64, 4096)
			for _, role := range []string{"cell", "output"} {
				add("step", role, layer, tick, ti, th, to, 64, 1024)
			}
		}
		add("layer", "inout", layer, -1, -1, -1, -1, 35, 64, 1024)
	}
	add("final", "arena", -1, -1, -1, -1, -1, ArenaBytes/2)
	add("final", "inout", -1, -1, -1, -1, -1, 35, 64, 1024)
	add("final", "hidden-ntc", -1, -1, -1, -1, -1, 64, 32, 1024)
	return out
}

// ReadObservations authenticates payloads and exact event metadata. Callers must
// separately authenticate producer execution and this manifest's own digest.
func ReadObservations(dir string) ([]Observation, error) {
	f, err := os.Open(filepath.Join(dir, "tensors.tsv"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	if !s.Scan() || s.Text() != "nanopore-hac-half-tensors/v1" {
		return nil, fmt.Errorf("half observation schema differs")
	}
	out := ObservationSpec()
	for i := range out {
		if !s.Scan() {
			return nil, fmt.Errorf("half observation %d missing", i)
		}
		v := strings.Split(s.Text(), "\t")
		if len(v) != 12 || v[0] != out[i].Stage || v[6] != out[i].Role || v[7] != "Half" {
			return nil, fmt.Errorf("half observation %d role differs", i)
		}
		for j, want := range []int{out[i].Layer, out[i].Tick, out[i].Input, out[i].Hidden, out[i].Output} {
			got, e := strconv.Atoi(v[j+1])
			if e != nil || got != want {
				return nil, fmt.Errorf("half observation %d schedule differs", i)
			}
		}
		var shape []int64
		for _, d := range strings.Split(v[8], ",") {
			n, e := strconv.ParseInt(d, 10, 64)
			if e != nil {
				return nil, e
			}
			shape = append(shape, n)
		}
		n, e := strconv.Atoi(v[9])
		if e != nil || n != out[i].Bytes || !slices.Equal(shape, out[i].Shape) {
			return nil, fmt.Errorf("half observation %d geometry differs", i)
		}
		name := fmt.Sprintf("%s-l%d-t%d-%s.bin", out[i].Stage, out[i].Layer, out[i].Tick, out[i].Role)
		if v[11] != name {
			return nil, fmt.Errorf("half observation %d payload path differs", i)
		}
		entry, e := os.Lstat(filepath.Join(dir, name))
		if e != nil {
			return nil, e
		}
		if !entry.Mode().IsRegular() || entry.Size() != int64(n) {
			return nil, fmt.Errorf("half observation %d requires regular exact-size payload", i)
		}
		pin, e := artifactPin(filepath.Join(dir, name))
		if e != nil {
			return nil, e
		}
		info, e := os.Stat(pin.Path)
		if e != nil {
			return nil, e
		}
		if pin.SHA256 != v[10] || info.Size() != int64(n) {
			return nil, fmt.Errorf("half observation %d payload differs", i)
		}
		out[i].File = pin
	}
	if s.Scan() {
		return nil, fmt.Errorf("unexpected half observation tail")
	}
	if err = s.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
