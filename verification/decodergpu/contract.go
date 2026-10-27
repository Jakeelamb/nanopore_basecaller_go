// Package decodergpu prepares a verification-only fixed HAC decoder oracle.
// It does not expose a production decoder or execute a native tool.
package decodergpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
)

const (
	N            = 64
	T            = 32
	C            = 1024
	States       = C / 4
	BeamWidth    = 32
	ScoresBytes  = N * T * C * 2
	GuidesBytes  = N * (T + 1) * C
	HistoryBytes = N * (T + 1) * 4 * BeamWidth
	AuxBytes     = GuidesBytes + HistoryBytes
)

type Descriptor [4]int32
type Scalars struct {
	ScoreClamp, QScale, QShift, BeamCut, Blank uint32
	MovePad                                    int32
}
type Span struct{ Offset, Bytes int64 }
type Buffer struct {
	Name, DType, Initialization string
	Bytes                       int64
}
type Stage struct {
	Name, Function string
	Compare        map[string][]Span
	Exclusions     map[string]string
}
type Contract struct {
	Schema         string
	Shape, Strides [3]int64
	Scalars        Scalars
	Descriptors    [N]Descriptor
	Buffers        []Buffer
	Stages         []Stage
}

func FixedContract() Contract {
	c := Contract{Schema: "nanopore-fixed-hac-decoder-contract/v1", Shape: [3]int64{N, T, C}, Strides: [3]int64{T * C, C, 1}, Scalars: Scalars{ScoreClamp: 0, QScale: math.Float32bits(1.02), QShift: math.Float32bits(-0.3), BeamCut: math.Float32bits(100), Blank: math.Float32bits(2)}}
	for i := range c.Descriptors {
		c.Descriptors[i] = Descriptor{int32(i * T), T, int32(i * T), 0}
	}
	c.Buffers = []Buffer{{"chunks", "i32", "fixed descriptor bytes", N * 16}, {"results", "raw-u32", "uninitialized", N * 32}, {"scores", "f16", "immutable original copied to destructive working allocation", ScoresBytes}, {"aux", "u8", "uninitialized", AuxBytes}, {"path", "i32", "zero", N * (T + 1) * 4}, {"moves", "u8", "zero", N * T}, {"sequence", "u8", "zero", N * T}, {"qstring", "u8", "zero", N * T}}
	for i, name := range []string{"back-guide", "beam-search", "posterior", "decode"} {
		functions := []string{"host_back_guide_step", "host_beam_search_step", "host_compute_posts_step", "host_run_decode"}
		stage := Stage{Name: name, Function: functions[i], Compare: map[string][]Span{"chunks": {{0, N * 16}}, "scores": {{0, ScoresBytes}}, "aux": {{0, GuidesBytes}}, "path": {{0, N * (T + 1) * 4}}, "moves": {{0, N * T}}, "sequence": {{0, N * T}}, "qstring": {{0, N * T}}}, Exclusions: map[string]string{"aux-history": "history slot population is not established; retain raw only"}}
		if i == 0 {
			stage.Exclusions["results"] = "no result stores in back-guide"
		} else {
			for chunk := 0; chunk < N; chunk++ {
				stage.Compare["results"] = append(stage.Compare["results"], Span{int64(chunk * 32), 8}, Span{int64(chunk*32 + 12), 4})
			}
			stage.Exclusions["results"] = "compare only beam-written words0,1,3; later provisional fields raw only"
		}
		c.Stages = append(c.Stages, stage)
	}
	return c
}
func (c Contract) Validate() error {
	if !reflect.DeepEqual(c, FixedContract()) {
		return fmt.Errorf("decoder preparation admits only the pinned finite fixed HAC contract")
	}
	return nil
}
func (c Contract) Parameters() []byte {
	b := make([]byte, 52)
	copy(b, "D01FIX01")
	values := []uint32{1, N, T, C, BeamWidth, c.Scalars.ScoreClamp, c.Scalars.QScale, c.Scalars.QShift, c.Scalars.BeamCut, c.Scalars.Blank, uint32(c.Scalars.MovePad)}
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[8+i*4:], v)
	}
	return b
}
func (c Contract) DescriptorBytes() []byte {
	b := make([]byte, N*16)
	for i, d := range c.Descriptors {
		for j, x := range d {
			binary.LittleEndian.PutUint32(b[i*16+j*4:], uint32(x))
		}
	}
	return b
}
func ValidateScores(payload []byte, want string) error {
	if len(payload) != ScoresBytes {
		return fmt.Errorf("scores need exactly %d bytes", ScoresBytes)
	}
	for i := 0; i < len(payload); i += 2 {
		if binary.LittleEndian.Uint16(payload[i:])&0x7c00 == 0x7c00 {
			return fmt.Errorf("nonfinite half score at element%d", i/2)
		}
	}
	if digest(payload) != want {
		return fmt.Errorf("scores digest differs from immutable H01 input")
	}
	return nil
}
func ValidateWire(parameters, descriptors []byte) error {
	c := FixedContract()
	if !reflect.DeepEqual(parameters, c.Parameters()) || !reflect.DeepEqual(descriptors, c.DescriptorBytes()) {
		return fmt.Errorf("fixed scalar bits/descriptors changed before native execution")
	}
	return nil
}
