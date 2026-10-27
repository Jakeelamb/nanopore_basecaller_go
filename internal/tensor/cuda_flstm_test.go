//go:build cgo && linux

package tensor

import (
	"errors"
	"math"
	"testing"
)

func flstmFixture(t *testing.T) (*Device, *Stream, FactorisedLSTMOperands, FactorisedLSTMGeometry) {
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
	o := FactorisedLSTMOperands{}
	slots := []**Storage{&o.InOut, &o.Workspace, &o.Down, &o.DownScale, &o.DownIH, &o.DownHH, &o.DownScaleIH, &o.DownScaleHH, &o.Up, &o.UpScale, &o.UpBias}
	shapes := [][]int64{{37, 64, 1024}, {128}, {1081344}, {8448}, {128, 1024}, {128, 1024}, {128}, {128}, {4096, 256}, {4096}, {4096}}
	types := []DType{DTypeI8, DTypeU8, DTypeU8, DTypeU8, DTypeI8, DTypeI8, DTypeF16, DTypeF16, DTypeI8, DTypeF16, DTypeF16}
	for i, p := range slots {
		n := 1
		for _, x := range shapes[i] {
			n *= int(x)
		}
		*p = mustSmallConvStorage(t, d, s, types[i], shapes[i], n)
	}
	return d, s, o, FactorisedLSTMGeometry{64, 32, -1, 1}
}
func TestFactorisedLSTMMarshalling(t *testing.T) {
	d, s, o, g := flstmFixture(t)
	if err := d.ValidateFactorisedLSTM(s, o, g); err != nil {
		t.Fatal(err)
	}
	if err := d.FactorisedLSTM(s, o, g); err != nil {
		t.Fatal(err)
	}
	b, err := o.InOut.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	for i, h := range o.storages() {
		if b[i] != byte(h.state.id) {
			t.Fatalf("operand %d forwarded as %d want %d", i, b[i], h.state.id)
		}
	}
	for i, v := range []byte{64, 32, 255, 1} {
		if b[11+i] != v {
			t.Fatalf("geometry %d got %d want %d", i, b[11+i], v)
		}
	}
}
func TestFactorisedLSTMRejectsOperands(t *testing.T) {
	d, s, o, g := flstmFixture(t)
	for i := 0; i < 11; i++ {
		bad := o
		slots := []**Storage{&bad.InOut, &bad.Workspace, &bad.Down, &bad.DownScale, &bad.DownIH, &bad.DownHH, &bad.DownScaleIH, &bad.DownScaleHH, &bad.Up, &bad.UpScale, &bad.UpBias}
		*slots[i] = nil
		if err := d.FactorisedLSTM(s, bad, g); !errors.Is(err, ErrClosed) {
			t.Fatalf("nil operand %d: %v", i, err)
		}
	}
	bad := o
	bad.DownHH = o.DownScaleHH
	if err := d.FactorisedLSTM(s, bad, g); err == nil {
		t.Fatal("wrong dtype accepted")
	}
	bad = o
	bad.DownHH = o.Up
	if err := d.FactorisedLSTM(s, bad, g); err == nil {
		t.Fatal("wrong shape accepted")
	}
	bad = o
	bad.Down = o.DownScale
	if err := d.FactorisedLSTM(s, bad, g); err == nil {
		t.Fatal("wrong scratch accepted")
	}
	foreign, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	if err = foreign.Close(); err != nil {
		t.Fatal(err)
	}
	if err = d.FactorisedLSTM(foreign, o, g); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed stream accepted: %v", err)
	}
}
func TestFactorisedLSTMRejectsGeometry(t *testing.T) {
	for _, g := range []FactorisedLSTMGeometry{{0, 32, -1, 1}, {63, 32, -1, 1}, {64, 0, -1, 1}, {64, 32, 0, 1}, {64, 32, -1, 2}, {64, 32, -1, 0}, {64, 32, 1, 1}, {2147483584, math.MaxInt32, -1, 1}} {
		if err := g.validate(); err == nil {
			t.Fatalf("accepted %+v", g)
		}
	}
}
func TestFactorisedLSTMMixedDtypeOverlap(t *testing.T) {
	d, s, o, g := flstmFixture(t)
	// A U8 workspace over F16 cache bytes overlaps despite different element units.
	v, err := o.UpScale.Reinterpret(DTypeU8, 0, []int64{128}, []int64{1}, []KoiTag{TagC})
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	o.Workspace = v
	if err := d.ValidateFactorisedLSTM(s, o, g); err == nil {
		t.Fatal("mixed dtype cache/workspace overlap accepted")
	}
}
