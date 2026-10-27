package nativeproof

import (
	"debug/elf"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectELFRejectsPageCongruentPartialAlias(t *testing.T) {
	p := writeELF(t, filepath.Join(t.TempDir(), "page-alias"), fixture{needed: []string{"libdeclared.so"}, interpreter: "/test/loader"})
	old, err := os.ReadFile(p.Path)
	if err != nil {
		t.Fatal(err)
	}
	oldSections := int(binary.LittleEndian.Uint64(old[40:]))
	body := make([]byte, 0x9000)
	copy(body, old)
	copy(body[0x7000:], old[oldSections:oldSections+4*64])
	set := func(at int, v uint64) { binary.LittleEndian.PutUint64(body[at:], v) }
	set(40, 0x7000)
	strtab, dynsec := 0x7000+2*64, 0x7000+3*64
	set(strtab+16, 0x401000)
	set(strtab+24, 0x1000)
	set(strtab+32, 0x2000)
	set(dynsec+16, 0x406000)
	set(dynsec+24, 0x6000)
	set(dynsec+32, 64)
	for i, entry := range []struct {
		tag   elf.DynTag
		value uint64
	}{{elf.DT_NEEDED, 0x1001}, {elf.DT_STRTAB, 0x401000}, {elf.DT_STRSZ, 0x2000}, {elf.DT_NULL, 0}} {
		set(0x6000+i*16, uint64(entry.tag))
		set(0x6000+i*16+8, entry.value)
	}
	copy(body[0x2001:], "libdeclared.so\x00")
	copy(body[0x8001:], "libhiddenxx.so\x00")
	set(64+32, 0x9000)
	set(64+40, 0x9000)
	set(64+48, 0x1000)
	set(120+8, 0x6000)
	set(120+16, 0x406000)
	set(120+32, 64)
	set(120+40, 64)
	if err = os.WriteFile(p.Path, body, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := InspectELF(p.Path)
	if err != nil || len(m.Needed) != 1 || m.Needed[0] != "libdeclared.so" {
		t.Fatalf("valid primary page mapping failed: %+v %v", m, err)
	}
	// A later, page-congruent load remaps only the string table's second page.
	binary.LittleEndian.PutUint32(body[176:], uint32(elf.PT_LOAD))
	binary.LittleEndian.PutUint32(body[180:], uint32(elf.PF_R))
	set(176+8, 0x8000)
	set(176+16, 0x402000)
	set(176+32, 0x1000)
	set(176+40, 0x1000)
	set(176+48, 0x1000)
	if err = os.WriteFile(p.Path, body, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = InspectELF(p.Path); err == nil {
		t.Fatal("strict partial PT_LOAD alias accepted the original dependency while the later page declares different runtime strings")
	}
	t.Logf("rejected page-congruent partial string-table alias: %v", err)
}

func TestLoadedFileOffsetExhaustiveMemoryIntervals(t *testing.T) {
	const base = uint64(0x400000)
	cases, rejected, accepted := 0, 0, 0
	for start := uint64(4); start <= 8; start++ {
		for size := uint64(1); size <= 6; size++ {
			for otherStart := uint64(0); otherStart <= 16; otherStart++ {
				for memory := uint64(0); memory <= 8; memory++ {
					for _, file := range []uint64{0, memory / 2, memory} {
						e := &elf.File{Progs: []*elf.Prog{
							{ProgHeader: elf.ProgHeader{Type: elf.PT_LOAD, Vaddr: base, Filesz: 64, Memsz: 64}},
							{ProgHeader: elf.ProgHeader{Type: elf.PT_LOAD, Vaddr: base + otherStart, Off: 128, Filesz: file, Memsz: memory}},
						}}
						// Independent byte-membership oracle covers every bounded
						// prefix, suffix, interior, full, BSS, empty and adjacent form.
						overlap := false
						for b := start; b < start+size; b++ {
							if b >= otherStart && b < otherStart+memory {
								overlap = true
							}
						}
						offset, err := loadedFileOffset(e, 256, base+start, size)
						cases++
						if overlap {
							if err == nil {
								t.Fatalf("competing memory accepted: query[%d,%d) other[%d,%d) file=%d", start, start+size, otherStart, otherStart+memory, file)
							}
							rejected++
						} else {
							if err != nil || offset != start {
								t.Fatalf("disjoint/empty/adjacent memory rejected: query[%d,%d) other[%d,%d) file=%d offset=%d err=%v", start, start+size, otherStart, otherStart+memory, file, offset, err)
							}
							accepted++
						}
					}
				}
			}
		}
	}
	if cases != 13770 || accepted == 0 || rejected == 0 {
		t.Fatalf("incomplete interval coverage: %d %d %d", cases, accepted, rejected)
	}
	t.Logf("interval oracle: %d cases, %d competing-memory rejections, %d disjoint/empty/adjacent positives", cases, rejected, accepted)
}

func TestInspectELFChecksPartialMemoryForBothDynamicRanges(t *testing.T) {
	for _, query := range []string{"dynamic", "strtab"} {
		for _, shape := range []string{"prefix", "suffix", "interior", "single-byte", "BSS-prefix", "BSS-suffix", "BSS-interior", "BSS-single-byte", "adjacent-before", "adjacent-after"} {
			p := writeELF(t, filepath.Join(t.TempDir(), "partial"), fixture{needed: []string{"libdeclared.so"}, interpreter: "/test/loader"})
			body, err := os.ReadFile(p.Path)
			if err != nil {
				t.Fatal(err)
			}
			sh := int(binary.LittleEndian.Uint64(body[40:]))
			sec := sh + 3*64
			if query == "strtab" {
				sec = sh + 2*64
			}
			address, size := binary.LittleEndian.Uint64(body[sec+16:]), binary.LittleEndian.Uint64(body[sec+32:])
			start, memory := address, uint64(0)
			switch strings.TrimPrefix(shape, "BSS-") {
			case "prefix":
				start, memory = address-1, 2
			case "suffix":
				start, memory = address+size-1, 2
			case "interior":
				start, memory = address+1, size-2
			case "single-byte":
				start, memory = address+size/2, 1
			case "adjacent-before":
				start, memory = address-1, 1
			case "adjacent-after":
				start, memory = address+size, 1
			}
			file := memory
			if strings.HasPrefix(shape, "BSS-") {
				file = 0
			}
			offset := len(body)
			body = append(body, make([]byte, memory)...)
			binary.LittleEndian.PutUint32(body[176:], uint32(elf.PT_LOAD))
			binary.LittleEndian.PutUint32(body[180:], uint32(elf.PF_R))
			for at, value := range map[int]uint64{176 + 8: uint64(offset), 176 + 16: start, 176 + 32: file, 176 + 40: memory, 176 + 48: 1} {
				binary.LittleEndian.PutUint64(body[at:], value)
			}
			if err = os.WriteFile(p.Path, body, 0600); err != nil {
				t.Fatal(err)
			}
			_, err = InspectELF(p.Path)
			if strings.HasPrefix(shape, "adjacent-") {
				if err != nil {
					t.Fatalf("valid %s/%s adjacency rejected: %v", query, shape, err)
				}
				t.Logf("accepted %s/%s nonoverlap", query, shape)
			} else {
				if err == nil {
					t.Fatalf("%s/%s partial loaded-memory overlap accepted", query, shape)
				}
				t.Logf("rejected %s/%s: %v", query, shape, err)
			}
		}
	}
}
