package intervaltrimoracle

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
)

type TrimCase struct {
	Read     readtrim.MuxRead
	Interval readtrim.Interval
	Leading  uint64
}
type FilterCase struct {
	Sequence, Qstring string
	Start             int
	Threshold         float32
}

func put(b *bytes.Buffer, values ...any) {
	for _, v := range values {
		if err := binary.Write(b, binary.LittleEndian, v); err != nil {
			panic(err)
		}
	}
}

func Compare(executable, dir string, trims []TrimCase, filters []FilterCase) error {
	var input, expected bytes.Buffer
	for i, c := range trims {
		r := c.Read
		put(&input, uint8(0), uint32(len(r.Sequence)))
		input.WriteString(r.Sequence)
		input.WriteString(r.Qstring)
		put(&input, uint32(c.Interval.Start), uint32(c.Interval.End), c.Leading, r.NumMinknowEvents, r.PreTrimSequenceLength, r.IsMuxChange, r.Stride, uint32(len(r.Moves)))
		input.Write(r.Moves)
		put(&input, uint32(len(r.Signal)), r.Signal)
		got, err := readtrim.ApplyInterval(r, c.Interval, c.Leading)
		if err != nil {
			return fmt.Errorf("trim case %d: %w", i, err)
		}
		r = got.Read
		put(&expected, uint32(len(r.Sequence)))
		expected.WriteString(r.Sequence)
		expected.WriteString(r.Qstring)
		put(&expected, uint32(len(r.Moves)))
		expected.Write(r.Moves)
		put(&expected, uint32(len(r.Signal)), r.Signal, got.LeadingTrimSamples, got.NumSamples, r.NumMinknowEvents, r.PreTrimSequenceLength, r.IsMuxChange, r.Stride)
	}
	for i, c := range filters {
		put(&input, uint8(1), uint32(len(c.Sequence)))
		input.WriteString(c.Sequence)
		input.WriteString(c.Qstring)
		put(&input, uint64(c.Start), math.Float32bits(c.Threshold))
		got, err := readtrim.Filter(c.Sequence, c.Qstring, c.Start, c.Threshold)
		if err != nil {
			return fmt.Errorf("filter case %d: %w", i, err)
		}
		put(&expected, got.Keep, math.Float32bits(got.MeanQ))
	}
	for name, data := range map[string][]byte{"inputs.bin": input.Bytes(), "go-output.bin": expected.Bytes()} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			return err
		}
	}
	if err := commandReceipt(dir, "native", exec.Command(executable), input.Bytes()); err != nil {
		return err
	}
	actual, err := os.ReadFile(filepath.Join(dir, "native.stdout"))
	if err != nil {
		return err
	}
	if !bytes.Equal(actual, expected.Bytes()) {
		return fmt.Errorf("native interval/filter mismatch got %s want %s", hash(actual), hash(expected.Bytes()))
	}
	return writeJSON(filepath.Join(dir, "comparison.json"), map[string]any{"trim_cases": len(trims), "filter_cases": len(filters), "exact_bytes": true, "input_sha256": hash(input.Bytes()), "output_sha256": hash(actual)})
}
