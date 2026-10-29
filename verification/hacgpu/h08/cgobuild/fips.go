package cgobuild

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

type fipsRange struct{ start, end uint64 }
type fipsByteTransition struct {
	checksumOffset uint64
	checksum       [32]byte
	ranges         [4]fipsRange
}

// Algorithm references (new references, not old-build inputs):
// cmd/link/internal/ld/fips140.go sha256 f214648c1a2ec04ea5483812ceec13cec96aa1a1acd9f95d2882b6373ef432c5
// cmd/internal/obj/fips140.go sha256 25fba80147d77aa812720b5c4ec71304b1547491ff59f0221c12a3c504236137
func verifyGo1273FIPSByteTransition(preimage, final []byte) (fipsByteTransition, error) {
	reject := func(reason string) (fipsByteTransition, error) {
		return fipsByteTransition{}, fmt.Errorf("Go1.27.3 FIPS: %s", reason)
	}
	if len(preimage) < 64 || uint64(len(preimage)) > 1<<30 {
		return reject("ELF file bounds")
	}
	if len(final) != len(preimage) {
		return reject("final file length")
	}
	le := binary.LittleEndian
	u16 := func(off uint64) uint16 { return le.Uint16(preimage[off : off+2]) }
	u32 := func(off uint64) uint32 { return le.Uint32(preimage[off : off+4]) }
	u64 := func(off uint64) uint64 { return le.Uint64(preimage[off : off+8]) }
	fits := func(off, size uint64) bool { return off <= uint64(len(preimage)) && size <= uint64(len(preimage))-off }
	adds := func(a, b uint64) bool { return b <= ^uint64(0)-a }
	aligned := func(a uint64) bool { return a == 0 || a&(a-1) == 0 }
	if !bytes.Equal(preimage[:16], []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}) || u16(16) != 2 || u16(18) != 62 || u32(20) != 1 || u32(48) != 0 || u16(52) != 64 {
		return reject("unsupported ELF profile")
	}
	ph, sh := u64(32), u64(40)
	pn, sn, stringsIndex := uint64(u16(56)), uint64(u16(60)), uint64(u16(62))
	if u16(54) != 56 || pn == 0 || pn == 0xffff || ph < 64 || !fits(ph, pn*56) {
		return reject("program table bounds/layout")
	}
	if u16(58) != 64 || sn < 2 || stringsIndex == 0 || stringsIndex >= sn || sh < 64 || !fits(sh, sn*64) {
		return reject("section table bounds/layout")
	}
	overlaps := func(a, b, c, d uint64) bool { return a < d && c < b }
	if overlaps(ph, ph+pn*56, sh, sh+sn*64) {
		return reject("overlapping ELF tables")
	}
	if !bytes.Equal(preimage[sh:sh+64], make([]byte, 64)) {
		return reject("section zero layout")
	}
	type segment struct{ off, addr, size uint64 }
	var loads []segment
	for i := uint64(0); i < pn; i++ {
		p := ph + i*56
		typ, off, addr, filesz, memsz, align := u32(p), u64(p+8), u64(p+16), u64(p+32), u64(p+40), u64(p+48)
		if !fits(off, filesz) || !adds(addr, memsz) || !adds(addr, filesz) || !aligned(align) {
			return reject("segment bounds/layout")
		}
		if typ == 1 {
			if filesz > memsz || (align > 1 && off%align != addr%align) {
				return reject("load segment layout")
			}
			loads = append(loads, segment{off, addr, filesz})
		}
	}
	nameSection := sh + stringsIndex*64
	nameOff, nameSize := u64(nameSection+24), u64(nameSection+32)
	if u32(nameSection+4) != 3 || nameSize == 0 || !fits(nameOff, nameSize) {
		return reject("section name table bounds/layout")
	}
	names := preimage[nameOff : nameOff+nameSize]
	if names[0] != 0 || names[len(names)-1] != 0 {
		return reject("section name table layout")
	}
	var infoOff, infoAddr uint64
	found := false
	for i := uint64(1); i < sn; i++ {
		s := sh + i*64
		name, typ, flags, addr, off, size, link, align, entsize := uint64(u32(s)), u32(s+4), u64(s+8), u64(s+16), u64(s+24), u64(s+32), uint64(u32(s+40)), u64(s+48), u64(s+56)
		if !adds(addr, size) || !aligned(align) || link >= sn || (entsize != 0 && size%entsize != 0) || off > uint64(len(preimage)) || (typ != 8 && !fits(off, size)) {
			return reject("section bounds/layout")
		}
		if name >= nameSize {
			return reject("section name bounds")
		}
		end := bytes.IndexByte(names[name:], 0)
		if end < 0 {
			return reject("unterminated section name")
		}
		if string(names[name:name+uint64(end)]) != ".go.fipsinfo" {
			continue
		}
		if found {
			return reject("duplicate .go.fipsinfo")
		}
		found = true
		if typ != 1 || flags&2 == 0 || flags & ^uint64(3) != 0 || size != 120 || link != 0 || u32(s+44) != 0 || entsize != 0 || align != 32 || off%32 != 0 || addr%32 != 0 {
			return reject(".go.fipsinfo layout")
		}
		infoOff, infoAddr = off, addr
	}
	if !found {
		return reject("missing .go.fipsinfo")
	}
	uniqueLoadRange := func(start, end uint64) (fipsRange, error) {
		if start > end {
			return fipsRange{}, fmt.Errorf("reversed range")
		}
		count := 0
		var r fipsRange
		for _, p := range loads {
			if p.addr <= start && end <= p.addr+p.size {
				count++
				r = fipsRange{p.off + (start - p.addr), p.off + (end - p.addr)}
			}
		}
		if count != 1 {
			return fipsRange{}, fmt.Errorf("range mapping count %d", count)
		}
		if !fits(r.start, r.end-r.start) {
			return fipsRange{}, fmt.Errorf("range file overrun")
		}
		return r, nil
	}
	infoMapping, err := uniqueLoadRange(infoAddr, infoAddr+120)
	if err != nil || infoMapping.start != infoOff || infoMapping.end != infoOff+120 {
		return reject(".go.fipsinfo address/offset mapping")
	}
	if !bytes.Equal(preimage[infoOff:infoOff+16], []byte("\xff Go fipsinfo \xff\x00")) {
		return reject(".go.fipsinfo magic")
	}
	if u64(infoOff+48) != infoAddr {
		return reject(".go.fipsinfo self pointer")
	}
	result := fipsByteTransition{checksumOffset: infoOff + 16}
	if !bytes.Equal(preimage[result.checksumOffset:result.checksumOffset+32], make([]byte, 32)) {
		return reject("nonzero preimage checksum")
	}
	h := hmac.New(sha256.New, make([]byte, 32))
	h.Write([]byte("go fips object v1\n"))
	for i := range result.ranges {
		start, end := u64(infoOff+56+uint64(i)*16), u64(infoOff+64+uint64(i)*16)
		r, e := uniqueLoadRange(start, end)
		if e != nil {
			return reject(fmt.Sprintf("FIPS range %d: %v", i, e))
		}
		if overlaps(r.start, r.end, infoOff, infoOff+120) {
			return reject("FIPS range overlaps info")
		}
		result.ranges[i] = r
		var size [8]byte
		binary.BigEndian.PutUint64(size[:], r.end-r.start)
		h.Write(size[:])
		h.Write(preimage[r.start:r.end])
	}
	copy(result.checksum[:], h.Sum(nil))
	at := result.checksumOffset
	if !bytes.Equal(final[at:at+32], result.checksum[:]) {
		return reject("final checksum mismatch")
	}
	if !bytes.Equal(preimage[:at], final[:at]) || !bytes.Equal(preimage[at+32:], final[at+32:]) {
		return reject("other final bytes changed")
	}
	return result, nil
}
