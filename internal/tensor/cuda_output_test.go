//go:build cgo && linux

package tensor

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

func outputFixture(t *testing.T) (*Device, *Stream, FactorisedLinearOperands, FactorisedLinearGeometry) {
	t.Helper()
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	o := FactorisedLinearOperands{}
	slots := []**Storage{&o.Input, &o.DownWeight, &o.DownScale, &o.Down, &o.UpWeight, &o.Output}
	shapes := [][]int64{{32, 64, 1024}, {128, 1024}, {128}, {64 * 32 * 128}, {1024, 128}, {64, 32, 1024}}
	types := []DType{DTypeI8, DTypeI8, DTypeF16, DTypeF16, DTypeF16, DTypeF16}
	tags := [][]KoiTag{{TagT, TagN, TagC}, {'K', TagC}, {'K'}, {TagC}, {TagC, 'K'}, {TagN, TagT, TagC}}
	for i, p := range slots {
		count := 1
		for _, x := range shapes[i] {
			count *= int(x)
		}
		*p, err = d.NewStorage(s, types[i], shapes[i], tags[i], make([]byte, count*types[i].Size()))
		if err != nil {
			t.Fatal(err)
		}
	}
	return d, s, o, FactorisedLinearGeometry{64, 32}
}
func TestFactorisedLinearMarshalling(t *testing.T) {
	d, s, o, g := outputFixture(t)
	if err := d.ValidateFactorisedLinear(s, o, g); err != nil {
		t.Fatal(err)
	}
	if err := d.FactorisedLinear(s, o, g); err != nil {
		t.Fatal(err)
	}
	b, err := o.Output.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	for i, h := range o.storages() {
		if b[i] != byte(h.state.id) {
			t.Fatalf("operand %d not forwarded", i)
		}
	}
	if binary.LittleEndian.Uint64(b[8:]) != 64 || binary.LittleEndian.Uint64(b[16:]) != 32 || binary.LittleEndian.Uint32(b[24:]) != 1 || binary.LittleEndian.Uint32(b[28:]) != math.Float32bits(5) {
		t.Fatal("geometry, parity or scale not forwarded")
	}
	wide := FactorisedLinearGeometry{N: 1<<32 + 64, T: 32}
	if err := d.FactorisedLinearUnchecked(s, o, wide, 1, 5); err != nil {
		t.Fatal(err)
	}
	b, err = o.Output.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint64(b[8:]) != uint64(wide.N) {
		t.Fatal("unchecked ABI truncated the signed 64-bit geometry before native validation")
	}
}
func TestFactorisedLinearGeometryAdmission(t *testing.T) {
	for _, g := range []FactorisedLinearGeometry{{64, 32}, {128, 64}} {
		if err := g.validate(); err != nil {
			t.Fatalf("source tiled geometry rejected: %+v %v", g, err)
		}
	}
	for _, g := range []FactorisedLinearGeometry{{1, 32}, {64, 48}, {0, 32}, {64, 0}, {1<<32 + 64, 32}, {64, 1<<32 + 32}, {math.MaxInt64, math.MaxInt64}} {
		if err := g.validate(); err == nil {
			t.Fatalf("invalid geometry accepted: %+v", g)
		}
	}
}
func TestFactorisedLinearRejectsPhysicalOperands(t *testing.T) {
	d, s, o, g := outputFixture(t)
	for i := 0; i < 6; i++ {
		bad := o
		slots := []**Storage{&bad.Input, &bad.DownWeight, &bad.DownScale, &bad.Down, &bad.UpWeight, &bad.Output}
		*slots[i] = nil
		if err := d.FactorisedLinear(s, bad, g); !errors.Is(err, ErrClosed) {
			t.Fatalf("nil %d: %v", i, err)
		}
	}
	bad := o
	bad.Down = o.DownScale
	if err := d.FactorisedLinear(s, bad, g); err == nil {
		t.Fatal("short down accepted")
	}
	bad = o
	bad.Input = o.Output
	if err := d.FactorisedLinear(s, bad, g); err == nil {
		t.Fatal("half input accepted")
	}
	wrongTags, err := o.Input.View(0, []int64{32, 64, 1024}, []int64{64 * 1024, 1024, 1}, []KoiTag{TagN, TagT, TagC})
	if err != nil {
		t.Fatal(err)
	}
	defer wrongTags.Close()
	bad = o
	bad.Input = wrongTags
	if err := d.FactorisedLinear(s, bad, g); err == nil {
		t.Fatal("wrong input tags accepted")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.FactorisedLinear(s, o, g); !errors.Is(err, ErrClosed) {
		t.Fatal("closed stream accepted")
	}
}
func TestFactorisedLinearMixedByteOverlapAndAdjacency(t *testing.T) {
	d, s, o, g := outputFixture(t)
	const bytes = 524288 + 4194304
	root, err := d.NewStorage(s, DTypeU8, []int64{bytes}, []KoiTag{TagC}, make([]byte, bytes))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	down, err := root.Reinterpret(DTypeF16, 0, []int64{262144}, []int64{1}, []KoiTag{TagC})
	if err != nil {
		t.Fatal(err)
	}
	defer down.Close()
	out, err := root.Reinterpret(DTypeF16, 262144, []int64{64, 32, 1024}, []int64{32768, 1024, 1}, []KoiTag{TagN, TagT, TagC})
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	o.Down = down
	o.Output = out
	if err := d.ValidateFactorisedLinear(s, o, g); err != nil {
		t.Fatalf("exact adjacency rejected: %v", err)
	}
	input, err := root.Reinterpret(DTypeI8, 262144, []int64{32, 64, 1024}, []int64{65536, 1024, 1}, []KoiTag{TagT, TagN, TagC})
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	o.Input = input
	if err := d.ValidateFactorisedLinear(s, o, g); err == nil {
		t.Fatal("mixed I8/F16 byte overlap accepted")
	}
}
