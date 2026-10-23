package cudafixture

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const Schema = "nanopore-cuda-half-case/v1"

type Allocation struct {
	File     string `json:"file"`
	Elements uint64 `json:"storage_elements"`
	SHA      string `json:"sha256"`
}
type View struct {
	Allocation string  `json:"allocation"`
	Shape      []int64 `json:"shape"`
	Strides    []int64 `json:"strides"`
	Offset     uint64  `json:"offset_elements"`
}
type Case struct {
	Schema      string                `json:"schema"`
	Name        string                `json:"name"`
	Mode        string                `json:"mode"`
	DType       string                `json:"dtype"`
	Endian      string                `json:"endian"`
	Allocations map[string]Allocation `json:"allocations"`
	Views       map[string]View       `json:"views"`
}

func digest(raw []byte) string { s := sha256.Sum256(raw); return hex.EncodeToString(s[:]) }
func safeFile(name string) bool {
	return name != "" && filepath.Base(name) == name && name != "." && name != ".."
}
func (c Case) Validate() error {
	if !safeFile(c.Name) {
		return errors.New("invalid fixture name")
	}
	if c.Schema != Schema || c.DType != "f16" || c.Endian != "little" {
		return errors.New("wrong fixture schema, dtype or endian")
	}
	if c.Mode != "copy" && c.Mode != "gemm" {
		return errors.New("unsupported fixture endpoint")
	}
	if len(c.Allocations) == 0 || len(c.Allocations) > 3 || len(c.Views) != 3 {
		return errors.New("wrong allocation or view count")
	}
	for _, a := range c.Allocations {
		if !safeFile(a.File) || a.Elements == 0 || a.Elements > uint64(math.MaxInt/2) || len(a.SHA) != 64 {
			return errors.New("invalid allocation")
		}
		if _, err := hex.DecodeString(a.SHA); err != nil {
			return err
		}
	}
	for _, role := range []string{"a", "b", "c"} {
		v, ok := c.Views[role]
		if !ok {
			return errors.New("missing operand view")
		}
		a, ok := c.Allocations[v.Allocation]
		if !ok {
			return errors.New("unknown allocation")
		}
		if len(v.Shape) == 0 || len(v.Shape) > 8 || len(v.Shape) != len(v.Strides) {
			return errors.New("bad fixture rank")
		}
		end := v.Offset
		count := uint64(1)
		for i, n := range v.Shape {
			if n <= 0 || v.Strides[i] < 0 {
				return errors.New("bad fixture dimension or stride")
			}
			if count > uint64(math.MaxInt/2)/uint64(n) {
				return errors.New("fixture count overflow")
			}
			count *= uint64(n)
			if uint64(n-1) > 0 && uint64(v.Strides[i]) > (math.MaxUint64-end)/uint64(n-1) {
				return errors.New("fixture span overflow")
			}
			end += uint64(n-1) * uint64(v.Strides[i])
		}
		if end >= a.Elements {
			return errors.New("fixture view outside allocation")
		}
		if c.Mode == "gemm" && (len(v.Shape) != 2 || v.Strides[1] != 1 || v.Strides[0] < v.Shape[1] || v.Shape[0] > math.MaxInt32 || v.Shape[1] > math.MaxInt32 || v.Strides[0] > math.MaxInt32) {
			return errors.New("unsupported GEMM fixture layout")
		}
	}
	if c.Mode == "gemm" {
		a, b, out := c.Views["a"], c.Views["b"], c.Views["c"]
		if a.Shape[1] != b.Shape[0] || a.Shape[0] != out.Shape[0] || b.Shape[1] != out.Shape[1] {
			return errors.New("fixture GEMM dimensions differ")
		}
	}
	return nil
}
func Load(path string) (Case, map[string][]uint16, error) {
	var c Case
	raw, err := os.ReadFile(path)
	if err != nil {
		return c, nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&c); err != nil {
		return c, nil, err
	}
	var extra any
	if err = dec.Decode(&extra); err != io.EOF {
		return c, nil, errors.New("trailing fixture data")
	}
	if err = c.Validate(); err != nil {
		return c, nil, err
	}
	data := map[string][]uint16{}
	for name, a := range c.Allocations {
		p := filepath.Join(filepath.Dir(path), a.File)
		info, err := os.Lstat(p)
		if err != nil {
			return c, nil, err
		}
		if !info.Mode().IsRegular() || info.Size() < 0 || uint64(info.Size()) != a.Elements*2 {
			return c, nil, errors.New("raw fixture byte count or file kind mismatch")
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return c, nil, err
		}
		if digest(b) != a.SHA {
			return c, nil, errors.New("raw fixture SHA256 mismatch")
		}
		bits := make([]uint16, a.Elements)
		for i := range bits {
			bits[i] = binary.LittleEndian.Uint16(b[i*2:])
		}
		data[name] = bits
	}
	return c, data, nil
}
func Write(dir string, c Case, data map[string][]uint16) error {
	c.Schema = Schema
	c.DType = "f16"
	c.Endian = "little"
	c.Allocations = map[string]Allocation{}
	for name, bits := range data {
		if !safeFile(name) {
			return errors.New("bad allocation name")
		}
		raw := make([]byte, len(bits)*2)
		for i, b := range bits {
			binary.LittleEndian.PutUint16(raw[i*2:], b)
		}
		nameFile := name + ".u16le"
		c.Allocations[name] = Allocation{nameFile, uint64(len(bits)), digest(raw)}
	}
	if err := c.Validate(); err != nil {
		return err
	}
	if err := os.Mkdir(dir, 0755); err != nil {
		return err
	}
	for name, bits := range data {
		raw := make([]byte, len(bits)*2)
		for i, b := range bits {
			binary.LittleEndian.PutUint16(raw[i*2:], b)
		}
		if err := os.WriteFile(filepath.Join(dir, c.Allocations[name].File), raw, 0644); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "case.json"), append(raw, '\n'), 0644)
}
func Generate(dir string) error {
	if err := os.Mkdir(dir, 0755); err != nil {
		return err
	}
	seed := uint64(0x48414c46)
	next := func() uint16 {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		exponents := []uint16{9, 11, 13, 15, 17}
		return uint16(seed>>32)&0x8000 | exponents[(seed>>16)%5]<<10 | uint16(seed)&0x3ff
	}
	makeCase := func(name string, m, n, k, as, bs, cs int, ao, bo, co int, shared bool) error {
		a := make([]uint16, ao+(m-1)*as+k+7)
		b := make([]uint16, bo+(k-1)*bs+n+7)
		out := make([]uint16, co+(m-1)*cs+n+7)
		for i := range a {
			a[i] = next()
		}
		for i := range b {
			b[i] = next()
		}
		for i := range out {
			out[i] = 0x3555
		}
		if name == "contiguous" {
			copy(a[ao:], []uint16{0, 0x8000, 0x3c00, 0xbc00})
			copy(b[bo:], []uint16{0, 0x8000, 0x3c00, 0xbc00})
		}
		data := map[string][]uint16{"a": a, "b": b, "c": out}
		ca := "c"
		if shared {
			co = len(a) + 5
			both := append(append(a, make([]uint16, 5)...), out...)
			data = map[string][]uint16{"a": both, "b": b}
			ca = "a"
		}
		view := func(all string, r, c, stride, off int) View {
			return View{all, []int64{int64(r), int64(c)}, []int64{int64(stride), 1}, uint64(off)}
		}
		c := Case{Name: name, Mode: "gemm", Views: map[string]View{"a": view("a", m, k, as, ao), "b": view("b", k, n, bs, bo), "c": view(ca, m, n, cs, co)}}
		return Write(filepath.Join(dir, name), c, data)
	}
	all := make([]uint16, 65536)
	for i := range all {
		all[i] = uint16(i)
	}
	v := View{"all", []int64{65536}, []int64{1}, 0}
	if err := Write(filepath.Join(dir, "all-half-bits"), Case{Name: "all-half-bits", Mode: "copy", Views: map[string]View{"a": v, "b": v, "c": v}}, map[string][]uint16{"all": all}); err != nil {
		return err
	}
	for _, spec := range []struct {
		name                            string
		m, n, k, as, bs, cs, ao, bo, co int
		shared                          bool
	}{
		{"contiguous", 16, 16, 32, 32, 16, 16, 0, 0, 0, false}, {"rectangular", 3, 7, 5, 5, 7, 7, 0, 0, 0, false}, {"padded", 64, 8, 16, 20, 12, 16, 3, 5, 7, false}, {"disjoint-shared", 16, 16, 32, 32, 16, 16, 3, 5, 0, true},
	} {
		if err := makeCase(spec.name, spec.m, spec.n, spec.k, spec.as, spec.bs, spec.cs, spec.ao, spec.bo, spec.co, spec.shared); err != nil {
			return err
		}
	}
	for _, k := range []int{32, 64, 128, 256} {
		for i := 0; i < 8; i++ {
			name := fmt.Sprintf("precision-k%d-%02d", k, i)
			if err := makeCase(name, 16, 16, k, k, 16, 16, 0, 0, 0, false); err != nil {
				return err
			}
		}
	}
	return nil
}
