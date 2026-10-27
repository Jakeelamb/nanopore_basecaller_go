package h06

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

type Comparison struct {
	Stage, Role        string
	Layer, Tick, Bytes int
	Native, Go         nativeproof.Pin
	Equal              bool
}

func compareObservation(out string, want Observation, body []byte) (c Comparison, err error) {
	c = Comparison{Stage: want.Stage, Role: want.Role, Layer: want.Layer, Tick: want.Tick, Bytes: len(body), Native: want.File}
	if len(body) != want.Bytes {
		return c, fmt.Errorf("half comparison extent differs: %s/%s", want.Stage, want.Role)
	}
	expected, err := os.ReadFile(want.File.Path)
	if err != nil {
		return c, err
	}
	if digest(expected) != want.File.SHA256 || len(expected) != want.Bytes {
		return c, fmt.Errorf("half native payload changed: %s", want.File.Path)
	}
	name := fmt.Sprintf("%s-l%d-t%d-%s.bin", want.Stage, want.Layer, want.Tick, want.Role)
	if err = os.WriteFile(filepath.Join(out, name), body, 0600); err != nil {
		return c, err
	}
	c.Go, err = artifactPin(filepath.Join(out, name))
	if err != nil {
		return c, err
	}
	c.Equal = bytes.Equal(body, expected)
	if !c.Equal {
		return c, fmt.Errorf("half numerical mismatch: %s layer=%d tick=%d role=%s", want.Stage, want.Layer, want.Tick, want.Role)
	}
	return c, nil
}

// Replay compares one production Go recurrence call. Admission, authenticated
// producer execution, device identity and runtime custody belong to its caller.
// Only the input is seeded; no future native intermediate enters the Go arena.
func Replay(ctx context.Context, d *tensor.Device, s *tensor.Stream, input []byte, caches []model.FLSTMHalfCache, observed []Observation, out string) (comparisons []Comparison, err error) {
	if len(input) != InputBytes || len(caches) != 7 || len(observed) != len(ObservationSpec()) {
		return nil, fmt.Errorf("fixed half replay inputs required")
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return nil, err
	}
	lookup := map[string]Observation{}
	key := func(stage string, layer, tick int, role string) string {
		return fmt.Sprintf("%s/%d/%d/%s", stage, layer, tick, role)
	}
	for _, v := range observed {
		lookup[key(v.Stage, v.Layer, v.Tick, v.Role)] = v
	}
	compare := func(stage string, layer, tick int, role string, body []byte) error {
		want, ok := lookup[key(stage, layer, tick, role)]
		if !ok {
			return fmt.Errorf("missing half native observation")
		}
		c, e := compareObservation(out, want, body)
		comparisons = append(comparisons, c)
		return e
	}
	for i, c := range caches {
		snapshot := c.Snapshot()
		for _, v := range []struct {
			role   string
			values []uint16
		}{{"down-ih", snapshot.DownIH}, {"down-hh", snapshot.DownHH}, {"up", snapshot.Up}, {"bias", snapshot.Bias}} {
			if err = compare("cache", i, -1, v.role, model.HalfBytes(v.values)); err != nil {
				return comparisons, err
			}
		}
	}
	stack, err := model.NewHACHalfStack(d, s, caches)
	if stack != nil {
		defer func() { err = errors.Join(err, stack.Close()) }()
	}
	if err != nil {
		return comparisons, err
	}
	seed := make([]byte, ArenaBytes)
	copy(seed, input)
	arena, err := d.NewStorage(s, tensor.DTypeF16, []int64{ArenaBytes / 2}, []tensor.KoiTag{tensor.TagC}, seed)
	if err != nil {
		return comparisons, err
	}
	defer func() { err = errors.Join(err, arena.Close()) }()
	inout, err := arena.View(0, []int64{35, 64, 1024}, []int64{65536, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	if err != nil {
		return comparisons, err
	}
	defer func() { err = errors.Join(err, inout.Close()) }()
	temp, err := arena.View(InputBytes/2, []int64{TempBytes / 2}, []int64{1}, []tensor.KoiTag{tensor.TagC})
	if err != nil {
		return comparisons, err
	}
	defer func() { err = errors.Join(err, temp.Close()) }()
	plan, err := model.PlanHACHalfStack(Batch, Frames)
	if err != nil {
		return comparisons, err
	}
	step := 0
	result, err := stack.RunObserved(ctx, s, inout, temp, plan, func(v model.HACHalfObservation) error {
		layer, tick := step/Frames, step%Frames
		want := lookup[key("step", layer, tick, "output")]
		if v.Layer != layer || v.InputRow != int64(want.Input) || v.HiddenRow != int64(want.Hidden) || v.OutputRow != int64(want.Output) {
			return fmt.Errorf("Go half observer schedule differs")
		}
		for _, item := range []struct {
			role    string
			storage *tensor.Storage
		}{{"down", v.Down}, {"gates", v.Gates}, {"cell", v.Cell}, {"output", v.Output}} {
			body, e := item.storage.Download(s)
			if e != nil {
				return e
			}
			if e = compare("step", layer, tick, item.role, body); e != nil {
				return e
			}
		}
		if tick == Frames-1 {
			body, e := inout.Download(s)
			if e != nil {
				return e
			}
			if e = compare("layer", layer, -1, "inout", body); e != nil {
				return e
			}
		}
		step++
		return nil
	})
	if result != nil {
		defer func() { err = errors.Join(err, result.Close()) }()
	}
	if err != nil {
		return comparisons, err
	}
	if step != 7*Frames {
		return comparisons, fmt.Errorf("Go half replay step count differs")
	}
	for _, item := range []struct {
		role    string
		storage *tensor.Storage
	}{{"arena", arena}, {"inout", inout}} {
		body, e := item.storage.Download(s)
		if e != nil {
			return comparisons, e
		}
		if err = compare("final", -1, -1, item.role, body); err != nil {
			return comparisons, err
		}
	}
	raw, err := inout.Download(s)
	if err != nil {
		return comparisons, err
	}
	ntc := make([]byte, 64*32*1024*2)
	for n := 0; n < Batch; n++ {
		for t := 0; t < Frames; t++ {
			src := ((t+2)*Batch + n) * Channels * 2
			dst := (n*Frames + t) * Channels * 2
			copy(ntc[dst:dst+Channels*2], raw[src:src+Channels*2])
		}
	}
	if err = compare("final", -1, -1, "hidden-ntc", ntc); err != nil {
		return comparisons, err
	}
	if len(comparisons) != 934 {
		return comparisons, fmt.Errorf("Go half replay comparison count differs: %d", len(comparisons))
	}
	return comparisons, nil
}
