package graph

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type childWire struct {
	Schema, Status, Error, Started, Finished string
	Request                                  nativeproof.Pin
	Consumer                                 selectedBuildWire
	Backend                                  closedbuild.ArtifactOptions
	CPU13                                    nativeproof.Pin
	Sources                                  WitnessAnchors
	Executable                               nativeproof.Pin
	Invocation                               []string
	Inputs                                   []supgpu.Pin
	Numerical                                Receipt
	ForwardCalls                             int
	Cleanup                                  bool
}

// RunGuardedChild opens the private phase pipes before any request/backend/model
// read. It never acquires the external launcher's stock lock.
func RunGuardedChild(ctx context.Context, pin nativeproof.Pin) (result Result, err error) {
	if ctx == nil {
		return result, fmt.Errorf("nil context")
	}
	if err = ctx.Err(); err != nil {
		return
	}
	phases, err := supgpu.OpenPhaseClient()
	if err != nil {
		return result, err
	}
	// Until the attempt exists, all failures still close the inherited client.
	ownedPhase := true
	defer func() {
		if ownedPhase {
			err = errors.Join(err, phases.Close())
		}
	}()
	root, err := currentRoot()
	if err != nil {
		return
	}
	q, err := readRequest(pin, root)
	if err != nil {
		return
	}
	if err = actualSUP(q, pin, true); err != nil {
		return
	}
	if err = runner.CheckRoot(); err != nil {
		return
	}
	o := q.wire
	if err = requireFresh(o.Out); err != nil {
		return
	}
	if err = os.Mkdir(o.Out, 0700); err != nil {
		return
	}
	r := childWire{Schema: "nanopore-sup-tiny-numerical/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Request: pin, Consumer: o.Consumer, Backend: o.Backend, CPU13: o.CPU13, Sources: o.Sources, Executable: q.binary.Output(), Invocation: slices.Clone(os.Args), Inputs: slices.Clone(q.inputs), Numerical: Receipt{Status: "failed", Comparisons: []Comparison{}, EventContracts: []string{}}}
	var engine *tensor.Engine
	var device *tensor.Device
	var stream *tensor.Stream
	var m *model.SUPGPU
	var input *tensor.Storage
	defer func() {
		var cleanup error
		if input != nil {
			cleanup = errors.Join(cleanup, input.Close())
		}
		if m != nil {
			cleanup = errors.Join(cleanup, m.Close())
		}
		if stream != nil {
			cleanup = errors.Join(cleanup, stream.Close())
		}
		if device != nil {
			cleanup = errors.Join(cleanup, device.Close())
		}
		if engine != nil {
			cleanup = errors.Join(cleanup, engine.Close())
		}
		cleanup = errors.Join(cleanup, phases.Close())
		ownedPhase = false
		r.Cleanup = cleanup == nil
		err = errors.Join(err, cleanup)
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
			r.Status = "failed"
		} else {
			r.Status = "tiny_numerical_equal"
		}
		path := filepath.Join(o.Out, "consumer.json")
		err = errors.Join(err, writeJSON(path, r))
		if err == nil {
			var p nativeproof.Pin
			p, err = pinFile(path)
			if err == nil {
				result = resultFor(p, r.Numerical)
			}
		}
	}()
	// Source capture's clean command overrides contain none of the four optional
	// numerical selectors. Derive presence/value from its authenticated command,
	// while leaving selected LD_AUDIT/LD_LIBRARY_PATH untouched.
	for _, key := range []string{"DORADO_MATMUL_FP16_MODE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		value, present := sourceEnvironment(q.witness, key)
		if present {
			err = os.Setenv(key, value)
		} else {
			err = os.Unsetenv(key)
		}
		if err != nil {
			return
		}
	}
	if err = phases.Checkpoint("pre-model"); err != nil {
		return
	}
	// Re-authenticate mutable model/input bytes at the deserialization boundary.
	for _, p := range q.inputs {
		v, e := sizedPin(unsized(p))
		if e != nil || v != p {
			return result, fmt.Errorf("pre-model live pin changed: %s: %v", p.Path, e)
		}
	}
	engine, err = tensor.Open(o.Backend.Output.Path)
	if err != nil {
		return
	}
	weights, err := model.LoadSUPGPUWeights(ctx, engine, o.Model)
	if err != nil {
		return
	}
	receipts := weights.Receipts()
	if len(receipts) != 139 {
		return result, fmt.Errorf("loaded SUP weight membership differs")
	}
	for i, p := range q.witness.prepared.Weights {
		if receipts[i].Name != filepath.Base(p.Path) || receipts[i].SHA256 != p.SHA256 {
			return result, fmt.Errorf("loaded SUP weight differs: %s", p.Path)
		}
	}
	device, err = engine.NewCUDADevice(o.Ordinal)
	if err != nil {
		return
	}
	stream, err = device.NewStream()
	if err != nil {
		return
	}
	// Keep a nonnil partially constructed model so the cleanup above retries its
	// retained owners after a constructor cleanup error.
	m, err = model.NewSUPGPU(ctx, device, stream, weights)
	if err != nil {
		return
	}
	signal, err := readPin("", q.witness.prepared.Input)
	if err != nil {
		return
	}
	input, err = device.NewStorage(stream, tensor.DTypeF16, []int64{32, 1, 768}, []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}, signal)
	if err != nil {
		return
	}
	if err = phases.Checkpoint("pre-forward"); err != nil {
		return
	}
	r.ForwardCalls++
	r.Numerical, err = compareForward(ctx, q.witness, m, stream, input)
	if err != nil {
		return
	}
	if err = validateComparisons(q.witness, r.Numerical); err != nil {
		return
	}
	err = phases.Checkpoint("post-forward")
	return
}
func sourceEnvironment(w Witness, key string) (string, bool) {
	for _, entry := range w.capture.Commands[0].Environment {
		k, v, ok := strings.Cut(entry, "=")
		if ok && k == key {
			return v, true
		}
	}
	return "", false
}
