package koiquant

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

type Fixture struct {
	Name, Domain string
	Input        koiweights.Data
}

var fixtureNames = []string{
	"genuine", "axes", "round-clip", "scale-neighbors", "finite-extremes",
	"all-half-patterns", "signed-zero", "tiny-range", "infinities", "nans",
}

func fixtureDomain(name string) string {
	if name == "genuine" {
		return "genuine-unclassified"
	}
	switch name {
	case "axes", "round-clip", "scale-neighbors", "finite-extremes":
		return "finite-observation"
	default:
		return "excluded-domain-observation"
	}
}

// fixtureHalf creates raw bit patterns, without implementing quantization math.
func fixtureHalf(name string) ([]uint16, error) {
	out := make([]uint16, Elements)
	for r := 0; r < Elements/512; r++ {
		for c := 0; c < 512; c++ {
			var bits uint16
			switch name {
			case "axes":
				bits = uint16(0x3000 + (r*17+c)%0x2400)
				if (r+c)%2 != 0 {
					bits |= 0x8000
				}
			case "round-clip":
				// Max 128 makes the expected scale exactly one, pending native observation.
				bits = []uint16{0x5800, 0xd800, 0x57f0, 0xd7f0, 0x3800, 0xb800, 0x3e00, 0xbe00, 0x57ef, 0x57f1, 0x0001, 0x8001}[c%12]
			case "scale-neighbors":
				bits = []uint16{0x3bff, 0x3c00, 0x3c01, 0x3fff, 0x4000, 0x4001}[r%6]
				if c%2 != 0 {
					bits |= 0x8000
				}
			case "finite-extremes":
				bits = []uint16{0x7bff, 0xfbff, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0, 0x8000}[c%10]
			case "all-half-patterns":
				bits = uint16(r*512 + c)
			case "signed-zero":
				bits = uint16(c%2) << 15
			case "tiny-range":
				bits = []uint16{1, 0x03ff, 0x0400, 0x0800}[r%4]
				if c%2 != 0 {
					bits |= 0x8000
				}
			case "infinities":
				bits = []uint16{0x7c00, 0xfc00, 0x3c00, 0xbc00}[c%4]
			case "nans":
				bits = []uint16{0x7c01, 0x7e00, 0xfc01, 0xfe00, 0x3c00}[c%5]
			default:
				return nil, fmt.Errorf("unknown fixture %q", name)
			}
			out[r*512+c] = bits
		}
	}
	return out, nil
}

func halfBytes(values []uint16) []byte {
	b := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(b[2*i:], value)
	}
	return b
}

func writeHalf(path string, values []uint16) (koiweights.Data, error) {
	b := halfBytes(values)
	if err := os.WriteFile(path, b, 0644); err != nil {
		return koiweights.Data{}, err
	}
	return koiweights.Data{Path: path, DType: "binary16", Endian: "little", Shape: []int{3, 8, 64, 512}, Strides: []int{262144, 32768, 512, 1}, Bytes: int64(len(b)), SHA256: digest(b)}, nil
}

func writeFixtures(out string, genuine []uint16) ([]Fixture, error) {
	fixtures := make([]Fixture, 0, len(fixtureNames))
	for _, name := range fixtureNames {
		values := genuine
		if name != "genuine" {
			var err error
			values, err = fixtureHalf(name)
			if err != nil {
				return nil, err
			}
		}
		d, err := writeHalf(filepath.Join(out, "inputs", name+"-u16le.bin"), values)
		if err != nil {
			return nil, err
		}
		fixtures = append(fixtures, Fixture{Name: name, Domain: fixtureDomain(name), Input: d})
	}
	return fixtures, nil
}
