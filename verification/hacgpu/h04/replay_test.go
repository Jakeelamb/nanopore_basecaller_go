package h04

import (
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func writeSyntheticCaches(t *testing.T, c, k int64, layers int) (string, hacgpu.FLSTMCaches, Receipt) {
	t.Helper()
	dir := t.TempDir()
	capture := hacgpu.FLSTMCaches{Schema: hacgpu.FLSTMCacheSchema, Status: "captured", Case: "synthetic-host-test",
		SourceCommit: hacgpu.SourceCommit, Batch: 64, Chunk: 192, Features: 1, LSTMLayers: layers,
		LSTMSize: int(c), LSTMInnerDim: int(k), C: c, K: k, FirstReverse: true,
		InputSHA256: hashBytes(nil), ModelPackage: "synthetic", Quantize: hacgpu.FLSTMQuantizeContract{
			Levels: 256, Dim: 1, QuantMax: 127, QuantMin: -127, WeightDType: "Char", ScaleDType: "Float", InputDType: "Half", Round: "half_to_even"}}
	old := Receipt{C: c, K: k, LSTMLayers: layers, Batch: 64, Chunk: 192, FirstReverse: true, Case: capture.Case,
		Features: 1, ModelPackage: "synthetic", InputSHA: capture.InputSHA256, CaptureInput: capture.InputSHA256,
		Quantize: capture.Quantize, Capture: dir}
	for i := 0; i < layers; i++ {
		layer := hacgpu.FLSTMCacheLayer{Layer: i, Reverse: i%2 == 0, ScaleMin: 1, ScaleMax: 1}
		prior := LayerResult{Index: i, Reverse: layer.Reverse, Equal: true,
			Domain: QuantDomain{ScaleMin: 1, ScaleMax: 1, CaptureScaleMin: 1, CaptureScaleMax: 1, Matches: true}}
		for j, role := range hacgpu.FLSTMCacheRoleNames() {
			tensor := hacgpu.FLSTMCacheTensor{Role: role, DType: "Char", Shape: []int64{k, c}, Contiguous: true}
			elem := int64(1)
			switch {
			case strings.HasPrefix(role, "up-"):
				tensor.Shape = []int64{4 * c, 2 * k}
			}
			if strings.HasSuffix(role, "scale") {
				tensor.DType, elem = "Half", 2
				if strings.Contains(role, "quant-") {
					tensor.DType, elem = "Float", 4
				}
				tensor.Shape = tensor.Shape[:1]
			}
			if strings.Contains(role, "bias") {
				tensor.DType, elem, tensor.Shape = "Half", 2, []int64{4 * c}
			}
			count := elem
			for _, d := range tensor.Shape {
				count *= d
			}
			body := make([]byte, count)
			tensor.File = filepath.Base(dir) + "-" + strings.Repeat("x", i+1) + "-" + role + ".bin"
			body[0] = byte(j + 1)
			tensor.SHA256 = hashBytes(body)
			if err := os.WriteFile(filepath.Join(dir, tensor.File), body, 0644); err != nil {
				t.Fatal(err)
			}
			layer.Tensors = append(layer.Tensors, tensor)
			prior.Tensors = append(prior.Tensors, TensorCompare{Role: role, DType: tensor.DType, Shape: append([]int64(nil), tensor.Shape...),
				Bytes: count, GoSHA256: tensor.SHA256, NativeSHA: tensor.SHA256, Equal: true})
		}
		capture.Layers = append(capture.Layers, layer)
		old.Layers = append(old.Layers, prior)
	}
	writeCacheJSON(t, dir, capture)
	return dir, capture, old
}

func writeCacheJSON(t *testing.T, dir string, c hacgpu.FLSTMCaches) {
	t.Helper()
	body, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "caches.json"), body, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestRetainedCacheImportRejectsChangedMetadata(t *testing.T) {
	dir, valid, _ := writeSyntheticCaches(t, 16, 32, 2)
	if _, err := hacgpu.ImportFLSTMCaches(dir); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		edit func(*hacgpu.FLSTMCaches)
	}{
		{"source", func(c *hacgpu.FLSTMCaches) { c.SourceCommit = "changed" }},
		{"quantizer", func(c *hacgpu.FLSTMCaches) { c.Quantize.Dim = 0 }},
		{"direction", func(c *hacgpu.FLSTMCaches) { c.Layers[1].Reverse = true }},
		{"shape", func(c *hacgpu.FLSTMCaches) { c.Layers[0].Tensors[0].Shape[0]++ }},
		{"dtype", func(c *hacgpu.FLSTMCaches) { c.Layers[0].Tensors[0].DType = "Half" }},
		{"filename", func(c *hacgpu.FLSTMCaches) { c.Layers[0].Tensors[0].File = "../escape" }},
		{"scale-domain", func(c *hacgpu.FLSTMCaches) { c.Layers[0].ScaleMin = -1 }},
		{"overflow", func(c *hacgpu.FLSTMCaches) { c.C, c.LSTMSize = 1<<62, 1<<62 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, _ := json.Marshal(valid)
			var c hacgpu.FLSTMCaches
			if err := json.Unmarshal(raw, &c); err != nil {
				t.Fatal(err)
			}
			tc.edit(&c)
			writeCacheJSON(t, dir, c)
			if _, err := hacgpu.ImportFLSTMCaches(dir); err == nil {
				t.Fatal("changed metadata accepted")
			}
		})
	}
}

func TestRetainedCacheRejectsPayloadChangedAfterImport(t *testing.T) {
	dir, _, _ := writeSyntheticCaches(t, 16, 32, 1)
	c, err := hacgpu.ImportFLSTMCaches(dir)
	if err != nil {
		t.Fatal(err)
	}
	tensor := c.Layers[0].Tensors[0]
	path := filepath.Join(dir, tensor.File)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body[0] ^= 1
	if err = os.WriteFile(path, body, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = c.Layers[0].Bytes(dir, tensor.Role); err == nil {
		t.Fatal("changed payload accepted after import")
	}
	if _, err = hacgpu.ImportFLSTMCaches(dir); err == nil {
		t.Fatal("changed payload accepted during import")
	}
}

func TestRetainedPairsRejectChangedHistoricalWitness(t *testing.T) {
	_, c, old := writeSyntheticCaches(t, 1024, 128, 7)
	pairs, err := retainedPairs(old, c)
	if err != nil || len(pairs) != 98 {
		t.Fatalf("pairs=%d error=%v", len(pairs), err)
	}
	for _, field := range []string{"go-digest", "native-digest", "equality", "shape", "layer"} {
		t.Run(field, func(t *testing.T) {
			raw, _ := json.Marshal(old)
			var changed Receipt
			if err := json.Unmarshal(raw, &changed); err != nil {
				t.Fatal(err)
			}
			switch field {
			case "go-digest":
				changed.Layers[0].Tensors[0].GoSHA256 = hashBytes(nil)
			case "native-digest":
				changed.Layers[0].Tensors[0].NativeSHA = hashBytes(nil)
			case "equality":
				changed.Layers[0].Tensors[0].Equal = false
			case "shape":
				changed.Layers[0].Tensors[0].Shape[0]++
			case "layer":
				changed.Layers[0].Index = 1
			}
			if _, err := retainedPairs(changed, c); err == nil {
				t.Fatal("changed historical witness accepted")
			}
		})
	}
}

func TestReplayPinsRejectChangedIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "identity")
	body := []byte("pinned identity")
	if err := os.WriteFile(path, body, 0644); err != nil {
		t.Fatal(err)
	}
	pin := hashBytes(body)
	if err := replayPin(path, pin); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("changed identity"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := replayPin(path, pin); err == nil {
		t.Fatal("changed identity accepted")
	}
}

func TestVerifyConsumerRejectsUnadmittedLoad(t *testing.T) {
	_, err := VerifyConsumer(ConsumerOptions{Backend: "must-not-load"})
	if err == nil || !strings.Contains(err.Error(), "coordinator admission") {
		t.Fatalf("unadmitted loader returned %v", err)
	}
}

func TestVerifyConsumerHeldLockRejectsBeforeNativeLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host.lock")
	if err := os.WriteFile(path, []byte("preserved"), 0600); err != nil {
		t.Fatal(err)
	}
	held, err := runner.Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	r, err := verifyConsumer(ConsumerOptions{Admitted: true, Backend: "must-not-load"}, path)
	if !errors.Is(err, runner.ErrAdmissionBlocked) || r.Attempt != "" {
		t.Fatalf("held lock did not reject before capture and native loading, attempt=%q error=%v", r.Attempt, err)
	}
}

func TestObservedGoPayloadRejectsChangedBytes(t *testing.T) {
	dir := t.TempDir()
	body := []byte("observed paired bytes")
	pair := TensorCompare{Bytes: int64(len(body)), GoSHA256: hashBytes(body), NativeSHA: hashBytes(body), Equal: true,
		GoPayload: filepath.Join(dir, "go.bin"), NativePayload: filepath.Join(dir, "native.bin")}
	for _, path := range []string{pair.GoPayload, pair.NativePayload} {
		if err := os.WriteFile(path, body, 0644); err != nil {
			t.Fatal(err)
		}
	}
	if err := verifyObservedPair(pair); err != nil {
		t.Fatal(err)
	}
	body[0] ^= 1
	if err := os.WriteFile(pair.GoPayload, body, 0644); err != nil {
		t.Fatal(err)
	}
	if err := verifyObservedPair(pair); err == nil {
		t.Fatal("changed observed Go payload accepted")
	}
}

func TestConsumerSourcePinsExcludeSiblingAndIncludePacker(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{"internal/signal", "internal/tensor", "internal/model", "verification/tensor/runner",
		"verification/hacgpu", "verification/hacgpu/h02", "verification/hacgpu/h04", "verification/hacgpu/h04/cmd/replay", "verification/hacgpu/h05"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0755); err != nil {
			t.Fatal(err)
		}
	}
	for _, file := range []string{"go.mod", "internal/model/hac_weights.json", "internal/model/sup_weights.json",
		"internal/model/hac_gpu_flstm_weights.go", "verification/hacgpu/h04/replay_test.go", "internal/tensor/abi.h"} {
		if err := os.WriteFile(filepath.Join(root, file), []byte("pinned"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	before, err := replaySourcePins(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "verification/hacgpu/h05/replay.go"), []byte("unrelated"), 0644); err != nil {
		t.Fatal(err)
	}
	after, err := replaySourcePins(root)
	if err != nil || !maps.Equal(before, after) {
		t.Fatalf("unrelated sibling changed consumer identity: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal/model/hac_gpu_flstm_weights.go"), []byte("changed packer"), 0644); err != nil {
		t.Fatal(err)
	}
	after, err = replaySourcePins(root)
	if err != nil || maps.Equal(before, after) {
		t.Fatalf("packer change did not change consumer identity: %v", err)
	}
}
