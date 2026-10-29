// Package graph compares the production Go SUP graph with the retained fixed
// source observations under an independently selected audited runtime.
package graph

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

type WitnessAnchors struct {
	Capture, Prepared, SourceBuild, SourceRuntime supgpu.Pin
}
type Comparison struct {
	Event, Role, GoSHA256, NativeSHA256 string
	Bytes                               int
	Equal                               bool
	ContractSHA256, BackingSHA256       string
}
type Receipt struct {
	Status         string
	Events         int
	Comparisons    []Comparison
	ScoreSHA256    string
	EventContracts []string
}
type Witness struct {
	capture  supgpu.Capture
	prepared supgpu.Prepared
	root     string
}

func digest(data []byte) string           { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }
func eventContract(e supgpu.Event) string { b, _ := json.Marshal(e); return digest(b) }
func readPin(root string, p supgpu.Pin) ([]byte, error) {
	path := p.Path
	if !filepath.IsAbs(path) {
		if filepath.Clean(path) != path || path == ".." || len(path) > 3 && path[:3] == "../" {
			return nil, fmt.Errorf("payload escapes witness")
		}
		path = filepath.Join(root, path)
	}
	return readSized(path, p, 64<<20)
}

// ReadWitness authenticates retained data only. It invokes no compiler,
// library loader, device query, archive reader, or model graph.
func ReadWitness(o WitnessAnchors) (Witness, error) {
	for _, p := range []supgpu.Pin{o.Capture, o.Prepared, o.SourceBuild, o.SourceRuntime} {
		if !filepath.IsAbs(p.Path) {
			return Witness{}, fmt.Errorf("absolute external witness anchors required")
		}
		if _, err := readPin("", p); err != nil {
			return Witness{}, err
		}
	}
	// Reject unknown/duplicate/null original metadata before inherited readers.
	var prepared supgpu.Prepared
	if err := readJSON(unsized(o.Prepared), &prepared); err != nil {
		return Witness{}, err
	}
	var capture supgpu.Capture
	if err := readJSON(unsized(o.Capture), &capture); err != nil {
		return Witness{}, err
	}
	var build supgpu.Build
	if err := readJSON(unsized(o.SourceBuild), &build); err != nil {
		return Witness{}, err
	}
	var runtime supgpu.RuntimePlan
	if err := readJSON(unsized(o.SourceRuntime), &runtime); err != nil {
		return Witness{}, err
	}
	// Authenticate every prospective prepared file before ReadPrepared can read
	// its fixture/signal/config/weights. These are data, never runtime authority.
	inputs := append([]supgpu.Pin{prepared.Fixture, prepared.Signal, prepared.Input, prepared.Config, prepared.Profile}, prepared.Weights...)
	inputs = append(inputs, prepared.Sources...)
	for _, p := range inputs {
		actual, err := sizedPin(unsized(p))
		if err != nil || actual != p {
			return Witness{}, fmt.Errorf("prepared input differs: %s: %v", p.Path, err)
		}
	}
	if len(capture.Commands) != 2 {
		return Witness{}, fmt.Errorf("source command pair absent")
	}
	p, err := supgpu.ReadPrepared(o.Prepared.Path, o.Prepared.SHA256)
	if err != nil {
		return Witness{}, err
	}
	if p.Geometry.Batch != 32 || p.Geometry.Chunk != 768 || p.Geometry.EncoderT != 64 || !p.UseF8 || !p.UseI8 || p.RemoveBits != 4 {
		return Witness{}, fmt.Errorf("witness is not fixed SUP F8/I8 default geometry")
	}
	root := filepath.Dir(o.Capture.Path)
	if filepath.Base(o.Capture.Path) != "capture.json" {
		return Witness{}, fmt.Errorf("capture anchor must identify capture.json")
	}
	c, err := supgpu.Import(root, o.Prepared.Path, o.Prepared.SHA256, o.SourceBuild.Path, o.SourceBuild.SHA256, o.SourceRuntime.Path, o.SourceRuntime.SHA256)
	if err != nil {
		return Witness{}, err
	}
	if _, err = readPin("", o.Capture); err != nil {
		return Witness{}, err
	}
	return Witness{c, p, root}, nil
}
func dtypeName(t tensor.DType) string {
	switch t {
	case tensor.DTypeF16:
		return "Half"
	case tensor.DTypeF32:
		return "Float"
	case tensor.DTypeI8:
		return "Char"
	case tensor.DTypeI32:
		return "Int"
	case tensor.DTypeE4M3:
		return "Float8_e4m3fn"
	}
	return "None"
}
func intTags(tags []tensor.KoiTag) []int {
	out := make([]int, len(tags))
	for i, v := range tags {
		out[i] = int(v)
	}
	return out
}

// CompareObservation accepts only equal full logical and backing payloads.
// No native intermediate is returned to the production graph.
func CompareObservation(root string, want supgpu.Event, got model.SUPGPUObservation) ([]Comparison, error) {
	origin := model.SUPGPUUploaded
	if strings.HasSuffix(want.Name, ".cache.before_remove_bits") {
		origin = model.SUPGPUPreparedCache
	}
	if got.Origin != origin {
		return nil, fmt.Errorf("observation provenance differs at %s", want.Name)
	}
	if got.Name != want.Name || got.Layer != want.Layer || len(got.Tensors) != len(want.Tensors) || len(got.Descriptors) != len(want.Descriptors) {
		return nil, fmt.Errorf("observation order or membership differs at %s", want.Name)
	}
	out := make([]Comparison, 0, len(want.Tensors))
	for i, n := range want.Tensors {
		g := got.Tensors[i]
		if g.Name != n.Name || dtypeName(g.DType) != n.DType || !slices.Equal(g.Shape, n.Shape) || !slices.Equal(g.Strides, n.Strides) || !slices.Equal(intTags(g.Tags), n.Tags) || int(g.ScaleTag) != n.ScaleTag || n.OffsetBytes != 0 {
			return out, fmt.Errorf("tensor descriptor differs at %s/%s", want.Name, n.Name)
		}
		data, err := readPin(root, n.Data)
		if err != nil {
			return out, err
		}
		backing, err := readPin(root, n.Backing)
		if err != nil {
			return out, err
		}
		sha := digest(g.Data)
		equal := slices.Equal(g.Data, data) && slices.Equal(g.Data, backing)
		out = append(out, Comparison{Event: want.Name, Role: n.Name, GoSHA256: sha, NativeSHA256: n.Data.SHA256, Bytes: len(g.Data), Equal: equal, ContractSHA256: eventContract(want), BackingSHA256: n.Backing.SHA256})
		if !equal {
			return out, fmt.Errorf("tensor bytes differ at %s/%s", want.Name, n.Name)
		}
	}
	for i, n := range want.Descriptors {
		g := got.Descriptors[i]
		if g.Name != n.Name || dtypeName(g.DType) != n.DType || !slices.Equal(g.Shape, n.Shape) || !slices.Equal(g.Strides, n.Strides) || !slices.Equal(intTags(g.Tags), n.Tags) || int(g.ScaleTag) != n.ScaleTag || dtypeName(g.ScaleDType) != n.ScaleDType {
			return out, fmt.Errorf("Koi descriptor differs at %s/%s", want.Name, n.Name)
		}
	}
	return out, nil
}

// compareForward performs one production forward using only prepared signal.
func compareForward(ctx context.Context, w Witness, m *model.SUPGPU, s *tensor.Stream, input *tensor.Storage) (r Receipt, err error) {
	r.Status = "failed"
	original, e := input.Download(s)
	if e != nil {
		return r, e
	}
	if digest(original) != w.prepared.Input.SHA256 || int64(len(original)) != w.prepared.Input.Bytes {
		return r, fmt.Errorf("graph input differs from independently prepared native signal")
	}
	observer := func(g model.SUPGPUObservation) error {
		if r.Events >= len(w.capture.Events) {
			return fmt.Errorf("extra production observation")
		}
		comparisons, e := CompareObservation(w.root, w.capture.Events[r.Events], g)
		r.Comparisons = append(r.Comparisons, comparisons...)
		if e != nil {
			return e
		}
		r.Events++
		r.EventContracts = append(r.EventContracts, eventContract(w.capture.Events[r.Events-1]))
		return nil
	}
	plan, err := model.PlanSUPGPU(model.FixedSUPGPUProfile())
	if err != nil {
		return r, err
	}
	result, err := m.ForwardObserved(ctx, s, input, plan, observer)
	if err != nil {
		return r, err
	}
	defer func() {
		err = errors.Join(err, result.Close())
		if err != nil {
			r.Status = "failed"
		}
	}()
	if r.Events != 191 || len(r.Comparisons) != 733 {
		return r, fmt.Errorf("incomplete fixed source observation coverage")
	}
	scores, err := result.Scores.Download(s)
	if err != nil {
		return r, err
	}
	r.ScoreSHA256 = digest(scores)
	if r.ScoreSHA256 != w.capture.Reference.SHA256 {
		return r, fmt.Errorf("original source final scores differ")
	}
	r.Status = "numerical_equal"
	return r, nil
}
