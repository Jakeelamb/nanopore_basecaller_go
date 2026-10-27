package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func observationCache() SUPEncoderGPUCache {
	c := SUPEncoderGPUCache{before: map[string]supGPUBuffer{}, after: map[string]supGPUBuffer{}}
	for _, name := range []string{"qkv_half", "qkv_int8", "qkv_scale", "proj_weight", "proj_bias", "norm1_weight", "norm2_weight", "fc1_half", "fc1_fp8", "fc2_fp8"} {
		c.before[name] = supGPUBuffer{dtype: tensor.DTypeF16, shape: []int64{1}, data: []byte{1, 0x34}}
		c.after[name] = supGPUBuffer{dtype: tensor.DTypeF16, shape: []int64{1}, data: []byte{0, 0x34}}
	}
	return c
}

func TestSUPGPUCacheObservationReadsUploadedOwners(t *testing.T) {
	c := observationCache()
	before, err := supCacheObservations(c, false, func(string) (SUPGPUBufferSnapshot, error) {
		t.Fatal("host-before observation fetched uploaded after cache")
		return SUPGPUBufferSnapshot{}, nil
	})
	if err != nil || len(before) != 10 || before[0].Data[0] != 1 {
		t.Fatalf("host snapshot lost: %v %v", before, err)
	}
	var reads []string
	after, err := supCacheObservations(c, true, func(name string) (SUPGPUBufferSnapshot, error) {
		reads = append(reads, name)
		v := c.after[name].snapshot(name)
		v.Data[0] = 2 // Finite changed device byte must reach the observer.
		return v, nil
	})
	if err != nil || len(after) != 10 || len(reads) != 10 {
		t.Fatalf("uploaded snapshot incomplete: %v %v", reads, err)
	}
	for i, v := range after {
		if v.Name != reads[i] || v.Data[0] != 2 || c.after[v.Name].data[0] != 0 {
			t.Fatal("observer substituted prepared bytes or mutated cache")
		}
	}
	if reflect.DeepEqual(after[0].Data, c.Snapshot(true)[0].Data) {
		t.Fatal("uploaded mutation hidden by prepared cache")
	}
}

func TestSUPGPUCacheObservationRejectsForeignMetadataAndStops(t *testing.T) {
	c := observationCache()
	for _, mutate := range []func(*SUPGPUBufferSnapshot){
		func(v *SUPGPUBufferSnapshot) { v.Name = "foreign" },
		func(v *SUPGPUBufferSnapshot) { v.DType = tensor.DTypeI8 },
		func(v *SUPGPUBufferSnapshot) { v.Shape = []int64{2} },
		func(v *SUPGPUBufferSnapshot) { v.Strides = []int64{2} },
		func(v *SUPGPUBufferSnapshot) { v.Tags = []tensor.KoiTag{tensor.TagC} },
		func(v *SUPGPUBufferSnapshot) { v.Data = nil },
	} {
		calls := 0
		if _, err := supCacheObservations(c, true, func(name string) (SUPGPUBufferSnapshot, error) {
			calls++
			v := c.after[name].snapshot(name)
			mutate(&v)
			return v, nil
		}); err == nil || calls != 1 {
			t.Fatalf("bad cache metadata admitted/continued: calls=%d err=%v", calls, err)
		}
	}
	stop := errors.New("cache download failed")
	calls := 0
	_, err := supCacheObservations(c, true, func(name string) (SUPGPUBufferSnapshot, error) { calls++; return SUPGPUBufferSnapshot{}, stop })
	if !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("read error hidden/continued calls=%d err=%v", calls, err)
	}
	if _, err = supCacheObservations(c, true, nil); err == nil {
		t.Fatal("missing device reader accepted")
	}
}
