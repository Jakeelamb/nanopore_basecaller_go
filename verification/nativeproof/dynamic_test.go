package nativeproof

import (
	"debug/elf"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestInspectELFBindsRuntimeDynamicStringTable(t *testing.T) {
	dir := t.TempDir()
	for _, mutation := range []string{"dual-table", "forged-section-address", "ambiguous-load-mapping", "duplicate-load-mapping", "missing-strtab", "missing-strsz", "duplicate-strtab", "conflicting-strtab", "duplicate-strsz", "conflicting-strsz", "short-strsz", "zero-strsz", "oversize-strsz", "unmapped-strtab", "memory-only-strtab", "address-overflow", "load-address-overflow", "load-offset-overflow", "dynamic-offset-overflow", "string-offset-at-size", "missing-null", "nonzero-null", "semantic-after-null"} {
		p := writeELF(t, filepath.Join(dir, mutation), fixture{needed: []string{"libdeclared.so"}, extraDynamic: 3, interpreter: "/test/loader"})
		body, err := os.ReadFile(p.Path)
		if err != nil {
			t.Fatal(err)
		}
		sh := int(binary.LittleEndian.Uint64(body[40:]))
		strtab := sh + 2*64
		dynsec := sh + 3*64
		dyn := int(binary.LittleEndian.Uint64(body[dynsec+24:]))
		dynsize := int(binary.LittleEndian.Uint64(body[dynsec+32:]))
		strAt := int(binary.LittleEndian.Uint64(body[strtab+24:]))
		strSize := int(binary.LittleEndian.Uint64(body[strtab+32:]))
		entry := func(tag elf.DynTag) int {
			for i := dyn; i < dyn+dynsize; i += 16 {
				if binary.LittleEndian.Uint64(body[i:]) == uint64(tag) {
					return i
				}
			}
			t.Fatalf("missing fixture tag %v", tag)
			return 0
		}
		stab, sz, null := entry(elf.DT_STRTAB), entry(elf.DT_STRSZ), entry(elf.DT_NULL)
		set := func(at int, val uint64) { binary.LittleEndian.PutUint64(body[at:], val) }
		appendTable := func(change bool) int {
			at := len(body)
			table := append([]byte(nil), body[strAt:strAt+strSize]...)
			if change {
				copy(table[1:], "libhiddenxx.so")
			}
			body = append(body, table...)
			return at
		}
		switch mutation {
		case "dual-table", "forged-section-address":
			at := appendTable(true)
			set(stab+8, 0x400000+uint64(at))
			set(64+32, uint64(len(body)))
			set(64+40, uint64(len(body)))
			if mutation == "forged-section-address" {
				set(strtab+16, 0x400000+uint64(at))
			}
		case "ambiguous-load-mapping", "duplicate-load-mapping":
			at := strAt
			if mutation == "ambiguous-load-mapping" {
				at = appendTable(true)
			}
			binary.LittleEndian.PutUint32(body[176:], uint32(elf.PT_LOAD))
			set(176+8, uint64(at))
			set(176+16, 0x400000+uint64(strAt))
			set(176+32, uint64(strSize))
			set(176+40, uint64(strSize))
		case "missing-strtab":
			set(stab, uint64(elf.DT_DEBUG))
		case "missing-strsz":
			set(sz, uint64(elf.DT_DEBUG))
		case "duplicate-strtab", "conflicting-strtab":
			at := entry(elf.DT_DEBUG)
			set(at, uint64(elf.DT_STRTAB))
			val := binary.LittleEndian.Uint64(body[stab+8:])
			if mutation == "conflicting-strtab" {
				val++
			}
			set(at+8, val)
		case "duplicate-strsz", "conflicting-strsz":
			at := entry(elf.DT_DEBUG)
			set(at, uint64(elf.DT_STRSZ))
			val := uint64(strSize)
			if mutation == "conflicting-strsz" {
				val++
			}
			set(at+8, val)
		case "short-strsz":
			set(sz+8, uint64(strSize-1))
		case "zero-strsz":
			set(sz+8, 0)
		case "oversize-strsz":
			set(sz+8, ^uint64(0))
		case "unmapped-strtab":
			set(stab+8, 0x400000+uint64(len(body))+4096)
		case "memory-only-strtab":
			at := appendTable(false)
			set(stab+8, 0x400000+uint64(at))
			set(strtab+16, 0x400000+uint64(at))
			set(strtab+24, uint64(at))
			set(64+40, uint64(len(body)))
		case "address-overflow":
			set(stab+8, ^uint64(0)-3)
		case "load-address-overflow":
			set(64+16, ^uint64(0)-3)
		case "load-offset-overflow":
			set(64+8, ^uint64(0)-3)
		case "dynamic-offset-overflow":
			set(120+8, ^uint64(0)-3)
			set(dynsec+24, ^uint64(0)-3)
		case "string-offset-at-size":
			set(dyn+8, uint64(strSize))
		case "missing-null":
			set(null, uint64(elf.DT_DEBUG))
		case "nonzero-null":
			set(null+8, 1)
		case "semantic-after-null":
			at := entry(elf.DT_DEBUG)
			set(at, uint64(elf.DT_NULL))
			set(at+16, uint64(elf.DT_NEEDED))
			set(at+24, 1)
		}
		if err = os.WriteFile(p.Path, body, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = InspectELF(p.Path); err == nil {
			t.Fatalf("runtime string table/termination disagreement accepted: %s", mutation)
		}
		t.Logf("rejected %s: %v", mutation, err)
	}
}

func TestInspectELFAcceptsBoundedNullPadding(t *testing.T) {
	p := writeELF(t, filepath.Join(t.TempDir(), "padded"), fixture{needed: []string{"libvalid.so"}, extraDynamic: 3})
	body, err := os.ReadFile(p.Path)
	if err != nil {
		t.Fatal(err)
	}
	sh := int(binary.LittleEndian.Uint64(body[40:]))
	dyn := int(binary.LittleEndian.Uint64(body[sh+3*64+24:]))
	size := int(binary.LittleEndian.Uint64(body[sh+3*64+32:]))
	for at := dyn; at < dyn+size; at += 16 {
		if binary.LittleEndian.Uint64(body[at:]) == uint64(elf.DT_DEBUG) {
			binary.LittleEndian.PutUint64(body[at:], uint64(elf.DT_NULL))
		}
	}
	if err = os.WriteFile(p.Path, body, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := InspectELF(p.Path)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Needed) != 1 || m.Needed[0] != "libvalid.so" {
		t.Fatalf("runtime strings differ: %+v", m)
	}
}
