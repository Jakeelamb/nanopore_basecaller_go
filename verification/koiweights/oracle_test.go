package koiweights

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestDataBoundary(t *testing.T) {
	p := filepath.Join(t.TempDir(), "half.bin")
	b := make([]byte, 4)
	binary.LittleEndian.PutUint16(b, 0x7bff)
	binary.LittleEndian.PutUint16(b[2:], 0x8000)
	if err := os.WriteFile(p, b, 0644); err != nil {
		t.Fatal(err)
	}
	d, err := data(p, "binary16", []int{2}, []int{1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = validateData(d, "binary16", []int{2}, []int{1}, nil); err != nil {
		t.Fatal(err)
	}
	half, err := ReadHalf(d)
	if err != nil || half[0] != 0x7bff || half[1] != 0x8000 {
		t.Fatal(half, err)
	}
	for _, name := range []string{"dtype", "endian", "rank", "shape", "stride", "bytes", "hash"} {
		t.Run(name, func(t *testing.T) {
			m := d
			switch name {
			case "dtype":
				m.DType = "uint16"
			case "endian":
				m.Endian = "big"
			case "rank":
				m.Shape = []int{1, 2}
			case "shape":
				m.Shape = []int{3}
			case "stride":
				m.Strides = []int{2}
			case "bytes":
				m.Bytes = 3
			case "hash":
				m.SHA256 = WeightHash
			}
			if err := validateData(m, "binary16", []int{2}, []int{1}, nil); err == nil {
				t.Fatal("malformed data accepted")
			}
		})
	}
	b[0] ^= 1
	if err = os.WriteFile(p, b, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = ReadHalf(d); err == nil {
		t.Fatal("changed payload accepted")
	}
}

func TestOutputRejectsOtherModuleBeforeWrite(t *testing.T) {
	root := t.TempDir()
	out := filepath.Join(root, "artifacts/koiweights/no-write")
	if err := admitOutput(root, out); err == nil {
		t.Fatal("foreign module accepted")
	}
	if _, err := os.Stat(filepath.Join(root, "artifacts")); !os.IsNotExist(err) {
		t.Fatal("foreign module created output")
	}
}

func TestNonblockingLock(t *testing.T) {
	p := filepath.Join(t.TempDir(), "existing.lock")
	if err := os.WriteFile(p, nil, 0644); err != nil {
		t.Fatal(err)
	}
	f, err := admit(p)
	if err != nil {
		t.Fatal(err)
	}
	defer release(f)
	if _, err = admit(p); err != ErrAdmissionBlocked {
		t.Fatalf("held lock: %v", err)
	}
	missing := filepath.Join(t.TempDir(), "absent.lock")
	if _, err = admit(missing); !os.IsNotExist(err) {
		t.Fatalf("missing lock: %v", err)
	}
	if _, err = os.Stat(missing); !os.IsNotExist(err) {
		t.Fatal("admission created a lock")
	}
}
