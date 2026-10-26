package nativeproof

import (
	"bytes"
	"debug/elf"
	"fmt"
	"io"
)

// loadedFileOffset requires exactly one intersecting loaded memory extent,
// which must back the complete queried range with file bytes.
// Every extent uses subtraction bounds before addition or integer conversion.
func loadedFileOffset(e *elf.File, fileSize, address, size uint64) (uint64, error) {
	if size == 0 || size > ^uint64(0)-address {
		return 0, fmt.Errorf("invalid or overflowing loaded address extent")
	}
	var offset uint64
	count := 0
	end := address + size
	for _, p := range e.Progs {
		if p.Type != elf.PT_LOAD {
			continue
		}
		if p.Filesz > p.Memsz || p.Off > fileSize || p.Filesz > fileSize-p.Off || p.Memsz > ^uint64(0)-p.Vaddr {
			return 0, fmt.Errorf("invalid PT_LOAD file or memory extent")
		}
		if p.Memsz == 0 || p.Vaddr >= end || address >= p.Vaddr+p.Memsz {
			continue
		}
		count++
		if address < p.Vaddr {
			return 0, fmt.Errorf("loaded range intersects a strict partial PT_LOAD memory extent")
		}
		delta := address - p.Vaddr
		if delta > p.Filesz || size > p.Filesz-delta {
			return 0, fmt.Errorf("loaded range intersects PT_LOAD memory without complete file backing")
		}
		offset = p.Off + delta
	}
	if count != 1 {
		return 0, fmt.Errorf("loaded range has %d intersecting PT_LOAD memory extents", count)
	}
	return offset, nil
}

func readFileRange(r io.ReaderAt, fileSize, offset, size uint64) ([]byte, error) {
	if offset > fileSize || size > fileSize-offset {
		return nil, fmt.Errorf("ELF metadata range escapes file")
	}
	body, err := io.ReadAll(io.NewSectionReader(r, int64(offset), int64(size)))
	if err == nil && uint64(len(body)) != size {
		err = io.ErrUnexpectedEOF
	}
	return body, err
}

// The supported section description must match the actual runtime program and
// DT_STRTAB/DT_STRSZ ranges. Strings are decoded from that checked file range,
// not through debug/elf.DynString's independently linked section interpretation.
func runtimeDynamicStrings(e *elf.File, r io.ReaderAt, fileSize uint64) (map[elf.DynTag][]string, error) {
	out := map[elf.DynTag][]string{}
	var program *elf.Prog
	for _, p := range e.Progs {
		if p.Type == elf.PT_DYNAMIC {
			if program != nil {
				return nil, fmt.Errorf("repeated PT_DYNAMIC")
			}
			program = p
		}
	}
	var section *elf.Section
	for _, s := range e.Sections {
		if s.Type == elf.SHT_DYNAMIC {
			if section != nil || s.Flags&elf.SHF_COMPRESSED != 0 {
				return nil, fmt.Errorf("repeated or compressed SHT_DYNAMIC")
			}
			section = s
		}
	}
	if program == nil && section == nil {
		return out, nil
	}
	if program == nil || section == nil || program.Off != section.Offset || program.Filesz != section.Size || program.Vaddr != section.Addr || program.Memsz < program.Filesz {
		return nil, fmt.Errorf("missing or conflicting PT_DYNAMIC/SHT_DYNAMIC binding")
	}
	entrySize := uint64(8)
	if e.Class == elf.ELFCLASS64 {
		entrySize = 16
	}
	if section.Entsize != entrySize || section.Size == 0 || section.Size%entrySize != 0 {
		return nil, fmt.Errorf("invalid dynamic entry size or extent")
	}
	offset, err := loadedFileOffset(e, fileSize, program.Vaddr, program.Filesz)
	if err != nil || offset != program.Off {
		return nil, fmt.Errorf("dynamic program is not its declared loaded file range: %v", err)
	}
	data, err := readFileRange(r, fileSize, offset, program.Filesz)
	if err != nil {
		return nil, err
	}
	values := map[elf.DynTag][]uint64{}
	terminated := false
	for at := uint64(0); at < uint64(len(data)); at += entrySize {
		var tag elf.DynTag
		var value uint64
		if entrySize == 16 {
			tag, value = elf.DynTag(e.ByteOrder.Uint64(data[at:])), e.ByteOrder.Uint64(data[at+8:])
		} else {
			tag, value = elf.DynTag(e.ByteOrder.Uint32(data[at:])), uint64(e.ByteOrder.Uint32(data[at+4:]))
		}
		if tag == elf.DT_NULL {
			if value != 0 {
				return nil, fmt.Errorf("nonzero DT_NULL value")
			}
			terminated = true
			continue
		}
		if terminated {
			return nil, fmt.Errorf("semantic dynamic entry after DT_NULL")
		}
		switch tag {
		case elf.DT_FILTER, elf.DT_AUXILIARY, elf.DT_AUDIT, elf.DT_DEPAUDIT:
			return nil, fmt.Errorf("unsupported dynamic loading tag %s", tag)
		case elf.DT_STRTAB, elf.DT_STRSZ, elf.DT_NEEDED, elf.DT_SONAME, elf.DT_RPATH, elf.DT_RUNPATH:
			values[tag] = append(values[tag], value)
		}
	}
	if !terminated {
		return nil, fmt.Errorf("dynamic stream has no DT_NULL")
	}
	if len(values[elf.DT_STRTAB]) != 1 || len(values[elf.DT_STRSZ]) != 1 {
		return nil, fmt.Errorf("exactly one DT_STRTAB and DT_STRSZ required")
	}
	if uint64(section.Link) >= uint64(len(e.Sections)) {
		return nil, fmt.Errorf("invalid dynamic string-table section link")
	}
	stringsSection := e.Sections[section.Link]
	address, size := values[elf.DT_STRTAB][0], values[elf.DT_STRSZ][0]
	stringOffset, err := loadedFileOffset(e, fileSize, address, size)
	if err != nil {
		return nil, err
	}
	if stringsSection.Type != elf.SHT_STRTAB || stringsSection.Flags&elf.SHF_COMPRESSED != 0 || stringsSection.Addr != address || stringsSection.Offset != stringOffset || stringsSection.Size != size {
		return nil, fmt.Errorf("section and runtime dynamic string tables disagree")
	}
	table, err := readFileRange(r, fileSize, stringOffset, size)
	if err != nil {
		return nil, err
	}
	if table[0] != 0 || table[len(table)-1] != 0 {
		return nil, fmt.Errorf("dynamic string table lacks boundary terminators")
	}
	for _, tag := range []elf.DynTag{elf.DT_NEEDED, elf.DT_SONAME, elf.DT_RPATH, elf.DT_RUNPATH} {
		if tag != elf.DT_NEEDED && len(values[tag]) > 1 {
			return nil, fmt.Errorf("repeated dynamic string tag %s", tag)
		}
		for _, at := range values[tag] {
			if at >= uint64(len(table)) {
				return nil, fmt.Errorf("dynamic string offset outside DT_STRSZ")
			}
			end := bytes.IndexByte(table[at:], 0)
			if end < 0 {
				return nil, fmt.Errorf("unterminated dynamic string")
			}
			out[tag] = append(out[tag], string(table[at:at+uint64(end)]))
		}
	}
	return out, nil
}
