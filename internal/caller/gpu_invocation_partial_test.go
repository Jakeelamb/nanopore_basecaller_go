package caller

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func partialInvocationFixture(t *testing.T, architecture model.Architecture, size, real int) (*gpuInvocation, scheduler.Batch, gpudecode.QualityBatch) {
	t.Helper()
	spec, err := model.Pinned(architecture)
	if err != nil {
		t.Fatal(err)
	}
	n, lane := 64, gpubatch.HAC
	if architecture == model.SUP {
		n, lane = 32, gpubatch.SUP
	}
	key := gpuProfileKey{string(architecture), size, n, false}
	r := &gpuRun{spec: spec, cfg: GPUConfig{BatchSize: n}, recipes: []gpuRecipe{{chunkSize: uint64(size)}}, pending: make(map[string]gpuChunkRef), bundle: gpuBundle{record: gpuBundleRecord{profiles: map[gpuProfileKey]bool{key: true}, partialProfiles: map[gpuProfileKey]bool{key: true}}}}
	b := scheduler.Batch{Lane: lane, Real: real, PaddedSlots: n, PaddedSamples: n * size}
	quality := gpudecode.QualityBatch{Chunks: make([]gpudecode.QualityChunk, n)}
	for j := 0; j < n; j++ {
		id := fmt.Sprintf("chunk-%d", j)
		offset := uint64(j%3 + 1)
		p := &gpuParent{read: pod5.Read{ID: pod5.ID{byte(j + 1)}}, normalized: signal.Normalized{Samples: make([]uint16, offset+3)}, chunks: []gpuSourceChunk{{}, {offset: offset, rawLength: 3, granularity: uint64(size)}}}
		copy(p.normalized.Samples[offset:], []uint16{0x3c00 + uint16(j), 0x3800, 0x3400})
		r.pending[id] = gpuChunkRef{parent: p, index: 1}
		if j < real {
			b.Placements = append(b.Placements, gpubatch.Placement{Chunk: gpubatch.Chunk{ID: id, Length: 3}, Slot: j, Offset: j * size, Length: size})
		}
		moves := make([]byte, size/int(spec.Stride))
		moves[0], moves[2] = 1, 1
		quality.Chunks[j] = gpudecode.QualityChunk{Moves: moves, Sequence: []byte{'A', byte('A' + j%4), 0xff}, Q: []byte{'I', byte('!' + j), 0xff}}
	}
	return &gpuInvocation{run: r}, b, quality
}

func TestGPUFixedPartialRealMembership(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		for _, size := range []int{12288, 6144} {
			t.Run(fmt.Sprintf("%s/chunk%d", architecture, size), func(t *testing.T) {
				_, full, _ := partialInvocationFixture(t, architecture, size, 1)
				for real := 1; real <= full.PaddedSlots; real++ {
					i, b, quality := partialInvocationFixture(t, architecture, size, real)
					for j := real; j < len(quality.Chunks); j++ {
						quality.Chunks[j] = gpudecode.QualityChunk{Moves: []byte{2}, Sequence: []byte("ABSENT"), Q: []byte("!!!!!!")}
					}
					work, err := i.decodedWork(b, quality)
					if err != nil || len(work) != real {
						t.Fatalf("Real%d membership=%d: %v", real, len(work), err)
					}
					for j, w := range work {
						ref := i.run.pending[b.Placements[j].Chunk.ID]
						c := ref.parent.chunks[ref.index]
						if w.ReadID != ref.parent.read.ID.String() || w.ChunkIndex != 1 || w.Offset != c.offset || w.RawLength != 3 || w.Granularity != uint64(size) || w.Sequence != string([]byte{'A', byte('A' + j%4)}) || w.Qstring != string([]byte{'I', byte('!' + j)}) || !reflect.DeepEqual(w.Moves, quality.Chunks[j].Moves) {
							t.Fatalf("Real%d slot%d changed identity, coordinates or decoded content: %+v", real, j, w)
						}
					}
				}
			})
		}
	}
}

func TestGPUFixedPartialRejectsMalformedMembership(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		for _, size := range []int{12288, 6144} {
			for _, name := range []string{"zero", "overflow", "quality-short", "quality-long", "quality-real", "duplicate", "unknown", "slot", "offset", "length", "raw-length", "profile", "index", "bounds", "missing-placement", "physical-slots", "physical-samples", "variable", "lane", "active-moves", "active-extent", "active-storage"} {
				t.Run(fmt.Sprintf("%s/%d/%s", architecture, size, name), func(t *testing.T) {
					i, b, quality := partialInvocationFixture(t, architecture, size, 2)
					ref := i.run.pending[b.Placements[0].Chunk.ID]
					switch name {
					case "zero":
						b.Real, b.Placements = 0, nil
					case "overflow":
						b.Real = b.PaddedSlots + 1
					case "quality-short":
						quality.Chunks = quality.Chunks[:len(quality.Chunks)-1]
					case "quality-long":
						quality.Chunks = append(quality.Chunks, gpudecode.QualityChunk{})
					case "quality-real":
						quality.Chunks = quality.Chunks[:b.Real]
					case "duplicate":
						b.Placements[1].Chunk.ID = b.Placements[0].Chunk.ID
					case "unknown":
						b.Placements[0].Chunk.ID = "absent"
					case "slot":
						b.Placements[0].Slot = 1
					case "offset":
						b.Placements[0].Offset = 1
					case "length":
						b.Placements[0].Length--
					case "raw-length":
						b.Placements[0].Chunk.Length++
					case "profile":
						ref.parent.profile = 1
					case "index":
						ref.index = -1
						i.run.pending[b.Placements[0].Chunk.ID] = ref
					case "bounds":
						ref.parent.chunks[1].offset = uint64(len(ref.parent.normalized.Samples))
					case "missing-placement":
						b.Placements = b.Placements[:1]
					case "physical-slots":
						b.PaddedSlots--
					case "physical-samples":
						b.PaddedSamples--
					case "variable":
						b.Variable = true
					case "lane":
						b.Lane = 0
					case "active-moves":
						quality.Chunks[0].Moves[0] = 2
					case "active-extent":
						quality.Chunks[0].Moves = quality.Chunks[0].Moves[:1]
					case "active-storage":
						quality.Chunks[0].Q = nil
					}
					if work, err := i.decodedWork(b, quality); err == nil || work != nil {
						t.Fatalf("malformed batch produced work: %v", err)
					}
					if name != "quality-short" && name != "quality-long" && name != "quality-real" && name != "active-moves" && name != "active-extent" && name != "active-storage" {
						if _, err := i.prepareInput(b); err == nil {
							t.Fatal("malformed input accepted")
						}
					}
				})
			}
		}
	}
}

// This test body also compiles against c373 without any new private seams or
// approval fields. A nil device provides a host-only storage-boundary sentinel.
func TestGPUFixedDuplicateCallStopsBeforeStorage(t *testing.T) {
	spec, err := model.Pinned(model.HAC)
	if err != nil {
		t.Fatal(err)
	}
	p := &gpuParent{normalized: signal.Normalized{Samples: []uint16{0x3c00, 0x3800, 0x3400}}, chunks: []gpuSourceChunk{{rawLength: 3, granularity: 12288}}}
	r := &gpuRun{spec: spec, cfg: GPUConfig{BatchSize: 64}, recipes: []gpuRecipe{{chunkSize: 12288}}, pending: map[string]gpuChunkRef{"same": {parent: p}}}
	b := scheduler.Batch{Lane: gpubatch.HAC, Real: 64, PaddedSlots: 64, PaddedSamples: 64 * 12288}
	for slot := 0; slot < 64; slot++ {
		b.Placements = append(b.Placements, gpubatch.Placement{Chunk: gpubatch.Chunk{ID: "same", Length: 3}, Slot: slot, Offset: slot * 12288, Length: 12288})
	}
	err = (&gpuInvocation{run: r}).Call(context.Background(), b)
	if errors.Is(err, tensor.ErrClosed) {
		t.Fatal("duplicate real identity reached storage boundary")
	}
	if err == nil || err.Error() != "GPU input placement changed profile, identity or fixed slot" {
		t.Fatalf("duplicate membership was not rejected at input boundary: %v", err)
	}
}

func TestGPUFixedPartialFreshPhysicalInput(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		for _, size := range []int{12288, 6144} {
			i, full, _ := partialInvocationFixture(t, architecture, size, 1)
			var previous []byte
			for _, real := range []int{full.PaddedSlots, 1, full.PaddedSlots - 1, full.PaddedSlots} {
				_, b, _ := partialInvocationFixture(t, architecture, size, real)
				data, err := i.prepareInput(b)
				if err != nil || len(data) != full.PaddedSamples*2 {
					t.Fatalf("physical input extent: %v", err)
				}
				for sample := 0; sample < b.PaddedSamples; sample++ {
					slot, within := sample/size, sample%size
					want := uint16(0)
					if slot < real {
						want = []uint16{0x3c00 + uint16(slot), 0x3800, 0x3400}[within%3]
					}
					if got := binary.LittleEndian.Uint16(data[sample*2:]); got != want {
						t.Fatalf("%s chunk%d Real%d sample%d got%x want%x", architecture, size, real, sample, got, want)
					}
				}
				if previous != nil {
					previous[0] = 0xff
					if data[0] == 0xff {
						t.Fatal("input aliases preceding call")
					}
				}
				previous = data
			}
		}
	}
}

func TestGPUFixedPartialIndependentApproval(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		for _, size := range []int{12288, 6144} {
			i, b, _ := partialInvocationFixture(t, architecture, size, 1)
			key := gpuProfileKey{string(architecture), size, b.PaddedSlots, false}
			for _, partial := range []map[gpuProfileKey]bool{nil, {key: false}, {{string(architecture), 192, b.PaddedSlots, false}: false}} {
				i.run.bundle.record.partialProfiles = partial
				if err := i.Call(context.Background(), b); !errors.Is(err, ErrGPUUnqualified) {
					t.Fatalf("ordinary full approval admitted partial before native work: %v", err)
				}
				_, full, _ := partialInvocationFixture(t, architecture, size, b.PaddedSlots)
				if _, err := i.prepareInput(full); err != nil {
					t.Fatalf("full batch newly held: %v", err)
				}
			}
			i.run.bundle.record.partialProfiles = map[gpuProfileKey]bool{key: true}
			if _, err := i.prepareInput(b); err != nil {
				t.Fatal(err)
			}
			for _, bad := range []gpuProfileKey{{string(architecture), size, b.PaddedSlots, true}, {string(architecture), 192, b.PaddedSlots, false}, {string(architecture), size, 1, false}, {"unknown", size, b.PaddedSlots, false}} {
				record := gpuBundleRecord{profiles: map[gpuProfileKey]bool{bad: true}, partialProfiles: map[gpuProfileKey]bool{bad: true}}
				if !errors.Is(record.validatePartialProfiles(), ErrGPUUnqualified) {
					t.Fatal("malformed partial approval accepted", bad)
				}
			}
			for _, fullApproval := range []map[gpuProfileKey]bool{nil, {key: false}} {
				i.run.bundle.record.profiles = fullApproval
				if err := i.Call(context.Background(), b); !errors.Is(err, ErrGPUUnqualified) {
					t.Fatalf("partial approval without full approval reached native work: %v", err)
				}
				if _, err := newGPUInvocation(i.run, 0); !errors.Is(err, ErrGPUUnqualified) {
					t.Fatalf("malformed record reached invocation initialization: %v", err)
				}
			}
		}
	}
	if len(approvedGPUBundles) != 0 {
		t.Fatal("host tests cannot approve production bundles")
	}
}
