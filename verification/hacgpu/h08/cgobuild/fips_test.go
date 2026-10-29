package cgobuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

const smallFIPSDigest = "03835ab5650386d13df2b795bd30314c43c10d0c63c759cee66ebac2c71deb26"

func smallFIPSELF(t *testing.T) ([]byte, []byte) {
	t.Helper()
	b := make([]byte, 0x600)
	copy(b, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	p16 := func(o int, v uint16) { binary.LittleEndian.PutUint16(b[o:], v) }
	p32 := func(o int, v uint32) { binary.LittleEndian.PutUint32(b[o:], v) }
	p64 := func(o int, v uint64) { binary.LittleEndian.PutUint64(b[o:], v) }
	p16(16, 2)
	p16(18, 62)
	p32(20, 1)
	p64(24, 0x400300)
	p64(32, 64)
	p64(40, 0x400)
	p16(52, 64)
	p16(54, 56)
	p16(56, 2)
	p16(58, 64)
	p16(60, 3)
	p16(62, 1)
	p32(64, 1)
	p32(68, 6)
	p64(80, 0x400000)
	p64(88, 0x400000)
	p64(96, uint64(len(b)))
	p64(104, uint64(len(b)))
	p64(112, 0x1000)
	copy(b[0x380:], "\x00.shstrtab\x00.go.fipsinfo\x00")
	p32(0x440, 1)
	p32(0x444, 3)
	p64(0x458, 0x380)
	p64(0x460, 24)
	p64(0x470, 1)
	p32(0x480, 11)
	p32(0x484, 1)
	p64(0x488, 3)
	p64(0x490, 0x400200)
	p64(0x498, 0x200)
	p64(0x4a0, 120)
	p64(0x4b0, 32)
	copy(b[0x200:], "\xff Go fipsinfo \xff\x00")
	p64(0x230, 0x400200)
	for i, r := range []struct {
		off  int
		data []byte
	}{{0x300, []byte("abc")}, {0x310, []byte("12345")}, {0x320, []byte{0, 255}}, {0x330, []byte("Z")}} {
		copy(b[r.off:], r.data)
		p64(0x238+16*i, 0x400000+uint64(r.off))
		p64(0x240+16*i, 0x400000+uint64(r.off+len(r.data)))
	}
	final := bytes.Clone(b)
	digest, err := hex.DecodeString(smallFIPSDigest)
	if err != nil {
		t.Fatal(err)
	}
	copy(final[0x210:], digest)
	return b, final
}

func TestGo1273FIPSFinalization(t *testing.T) {
	pre, final := smallFIPSELF(t)
	savedPre, savedFinal := bytes.Clone(pre), bytes.Clone(final)
	got, err := verifyGo1273FIPSByteTransition(pre, final)
	if err != nil {
		t.Fatal(err)
	}
	if got.checksumOffset != 0x210 || hex.EncodeToString(got.checksum[:]) != smallFIPSDigest || got.ranges != [4]fipsRange{{0x300, 0x303}, {0x310, 0x315}, {0x320, 0x322}, {0x330, 0x331}} {
		t.Fatalf("unexpected derivation: %+v", got)
	}
	if !bytes.Equal(pre, savedPre) || !bytes.Equal(final, savedFinal) {
		t.Fatal("verifier changed caller bytes")
	}
	for _, delta := range []uint64{0x100000, 0x700000000000} {
		a, b := smallFIPSELF(t)
		for _, o := range []int{24, 80, 88, 0x490, 0x230, 0x238, 0x240, 0x248, 0x250, 0x258, 0x260, 0x268, 0x270} {
			binary.LittleEndian.PutUint64(a[o:], binary.LittleEndian.Uint64(a[o:])+delta)
			binary.LittleEndian.PutUint64(b[o:], binary.LittleEndian.Uint64(b[o:])+delta)
		}
		if _, e := verifyGo1273FIPSByteTransition(a, b); e != nil {
			t.Fatalf("relocated mapping: %v", e)
		}
	}
	a, b := smallFIPSELF(t)
	for _, x := range [][]byte{a, b} {
		copy(x[0x280:0x2f8], x[0x200:0x278])
		clear(x[0x200:0x278])
		binary.LittleEndian.PutUint64(x[0x490:], 0x400280)
		binary.LittleEndian.PutUint64(x[0x498:], 0x280)
		binary.LittleEndian.PutUint64(x[0x2b0:], 0x400280)
	}
	moved, e := verifyGo1273FIPSByteTransition(a, b)
	if e != nil || moved.checksumOffset != 0x290 {
		t.Fatalf("moved section: %+v %v", moved, e)
	}
}

func TestGo1273FIPSRejections(t *testing.T) {
	type control struct {
		name, reason string
		mutate       func(*[]byte, *[]byte)
	}
	both64 := func(off int, value uint64) func(*[]byte, *[]byte) {
		return func(a, b *[]byte) {
			for _, x := range [][]byte{*a, *b} {
				binary.LittleEndian.PutUint64(x[off:], value)
			}
		}
	}
	both16 := func(off int, value uint16) func(*[]byte, *[]byte) {
		return func(a, b *[]byte) {
			for _, x := range [][]byte{*a, *b} {
				binary.LittleEndian.PutUint16(x[off:], value)
			}
		}
	}
	bothByte := func(off int, value byte) func(*[]byte, *[]byte) {
		return func(a, b *[]byte) { (*a)[off] = value; (*b)[off] = value }
	}
	controls := []control{
		{"changed_code", "other final bytes changed", func(a, b *[]byte) { (*b)[0x300] ^= 1 }},
		{"changed_unhashed_byte", "other final bytes changed", func(a, b *[]byte) { (*b)[0x350] ^= 1 }},
		{"changed_preimage_code", "final checksum mismatch", func(a, b *[]byte) { (*a)[0x300] ^= 1 }},
		{"wrong_checksum", "final checksum mismatch", func(a, b *[]byte) { (*b)[0x210] ^= 1 }},
		{"zero_checksum", "final checksum mismatch", func(a, b *[]byte) { clear((*b)[0x210:0x230]) }},
		{"nonzero_preimage", "nonzero preimage checksum", func(a, b *[]byte) { (*a)[0x210] = 1 }},
		{"final_length", "final file length", func(a, b *[]byte) { *b = (*b)[:len(*b)-1] }},
		{"short_header", "ELF file bounds", func(a, b *[]byte) { *a = (*a)[:63]; *b = (*b)[:63] }},
		{"class", "unsupported ELF profile", bothByte(4, 1)},
		{"endianness", "unsupported ELF profile", bothByte(5, 2)},
		{"pie", "unsupported ELF profile", both16(16, 3)},
		{"machine", "unsupported ELF profile", both16(18, 183)},
		{"header_version", "unsupported ELF profile", bothByte(20, 2)},
		{"header_size", "unsupported ELF profile", both16(52, 63)},
		{"section_offset", ".go.fipsinfo address/offset mapping", both64(0x498, 0x220)},
		{"section_size", ".go.fipsinfo layout", both64(0x4a0, 119)},
		{"section_alignment", ".go.fipsinfo layout", both64(0x4b0, 16)},
		{"magic", ".go.fipsinfo magic", bothByte(0x200, 0)},
		{"self_pointer", ".go.fipsinfo self pointer", both64(0x230, 0x400201)},
		{"missing_section", "missing .go.fipsinfo", bothByte(0x480, 1)},
		{"duplicate_section", "duplicate .go.fipsinfo", func(a, b *[]byte) {
			for _, x := range [][]byte{*a, *b} {
				copy(x[0x4c0:0x500], x[0x480:0x4c0])
				binary.LittleEndian.PutUint16(x[60:], 4)
			}
		}},
		{"section_table_truncated", "section table bounds/layout", both64(40, 0x580)},
		{"section_table_overflow", "section table bounds/layout", both64(40, ^uint64(0)-63)},
		{"section_table_stride", "section table bounds/layout", both16(58, 63)},
		{"section_extended_count", "section table bounds/layout", both16(60, 0)},
		{"program_table_truncated", "program table bounds/layout", both64(32, 0x5e0)},
		{"program_table_overflow", "program table bounds/layout", both64(32, ^uint64(0)-55)},
		{"program_table_stride", "program table bounds/layout", both16(54, 55)},
		{"program_extended_count", "program table bounds/layout", both16(56, 0xffff)},
		{"section_payload_overrun", "section bounds/layout", both64(0x4a0, ^uint64(0))},
		{"section_names_overrun", "section name table bounds/layout", both64(0x460, 0x1000)},
		{"section_name_offset", "section name bounds", bothByte(0x480, 255)},
		{"segment_file_overrun", "segment bounds/layout", both64(96, 0x601)},
		{"segment_offset_overflow", "segment bounds/layout", both64(72, ^uint64(0))},
		{"segment_virtual_overflow", "segment bounds/layout", both64(104, ^uint64(0))},
		{"segment_filesz_memsz", "load segment layout", both64(104, 0x500)},
		{"unmapped_range", "FIPS range 0: range mapping count 0", both64(0x240, 0x400601)},
		{"memory_only_range", "FIPS range 0: range mapping count 0", func(a, b *[]byte) { both64(104, 0x800)(a, b); both64(0x240, 0x400601)(a, b) }},
		{"reversed_range", "FIPS range 0: reversed range", both64(0x240, 0x4002ff)},
		{"range_file_overrun", "FIPS range 0: range mapping count 0", both64(0x240, ^uint64(0))},
		{"ambiguous_range", "FIPS range 0: range mapping count 2", func(a, b *[]byte) {
			for _, x := range [][]byte{*a, *b} {
				p := 120
				binary.LittleEndian.PutUint32(x[p:], 1)
				binary.LittleEndian.PutUint64(x[p+8:], 0x300)
				binary.LittleEndian.PutUint64(x[p+16:], 0x400300)
				binary.LittleEndian.PutUint64(x[p+32:], 3)
				binary.LittleEndian.PutUint64(x[p+40:], 3)
			}
		}},
		{"info_overlap", "FIPS range overlaps info", func(a, b *[]byte) { both64(0x238, 0x400200)(a, b); both64(0x240, 0x400201)(a, b) }},
		{"little_endian_framing", "final checksum mismatch", func(a, b *[]byte) {
			v, _ := hex.DecodeString("82d2178558d9199eeae4bd09ee89e5d0aa2a9d1d8836cfa932340894bfded2b7")
			copy((*b)[0x210:], v)
		}},
		{"lengths_before_data_framing", "final checksum mismatch", func(a, b *[]byte) {
			v, _ := hex.DecodeString("782066083d23aaaa43a8f23d7d6e351530db91894e8f4280cf42fd794fe90fbc")
			copy((*b)[0x210:], v)
		}},
	}
	for _, c := range controls {
		t.Run(c.name, func(t *testing.T) {
			pre, final := smallFIPSELF(t)
			c.mutate(&pre, &final)
			savedPre, savedFinal := bytes.Clone(pre), bytes.Clone(final)
			got, err := verifyGo1273FIPSByteTransition(pre, final)
			if err == nil || !strings.Contains(err.Error(), c.reason) {
				t.Fatalf("expected %q, got %v", c.reason, err)
			}
			if got != (fipsByteTransition{}) {
				t.Fatal("rejection returned a partial proof")
			}
			if !bytes.Equal(pre, savedPre) || !bytes.Equal(final, savedFinal) {
				t.Fatal("rejection mutated caller bytes")
			}
		})
	}
}

func TestGo1273FIPSGenuineRetainedPair(t *testing.T) {
	keys := []string{"H08_FIPS_PREIMAGE", "H08_FIPS_FINAL", "H08_FIPS_PREIMAGE_SHA256", "H08_FIPS_FINAL_SHA256"}
	values := make([]string, len(keys))
	count := 0
	for i, k := range keys {
		values[i] = os.Getenv(k)
		if values[i] != "" {
			count++
		}
	}
	if count == 0 {
		t.Skip("retained pair requires explicit controller paths and hashes")
	}
	if count != len(keys) {
		t.Fatal("all four retained-pair environment values are required")
	}
	var data [2][]byte
	for i := range data {
		expected, err := hex.DecodeString(values[i+2])
		if err != nil || len(expected) != 32 {
			t.Fatalf("invalid selected SHA256 for %s", keys[i])
		}
		data[i], err = os.ReadFile(values[i])
		if err != nil {
			t.Fatal(err)
		}
		got := sha256.Sum256(data[i])
		if !bytes.Equal(got[:], expected) {
			t.Fatalf("selected digest differs for %s", keys[i])
		}
	}
	got, err := verifyGo1273FIPSByteTransition(data[0], data[1])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("derived checksum offset=%d checksum=%x ranges=%+v", got.checksumOffset, got.checksum, got.ranges)
}

func TestGo1273FIPSTruncatedPrefixes(t *testing.T) {
	pre, final := smallFIPSELF(t)
	for n := 0; n < len(pre); n++ {
		if _, err := verifyGo1273FIPSByteTransition(pre[:n], final[:n]); err == nil {
			t.Fatalf("accepted truncated prefix of %d bytes", n)
		}
	}
}
