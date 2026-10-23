package koiquant

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

type StageDifference struct {
	Fixture, Stage, DType string
	Elements, Different   int
	FirstIndex            int
	FirstCPU, FirstCUDA   uint32
}

type Comparison struct {
	Schema, Started, Finished string
	CPU, CUDA                 Pin
	ConsumerSources           map[string]string
	Stages                    []StageDifference
}

func dataDifference(fixture, stage string, a, b koiweights.Data) (d StageDifference, err error) {
	d = StageDifference{Fixture: fixture, Stage: stage, DType: a.DType, FirstIndex: -1}
	if a.DType != b.DType || a.Bytes != b.Bytes || !reflect.DeepEqual(a.Shape, b.Shape) || !reflect.DeepEqual(a.Strides, b.Strides) {
		return d, fmt.Errorf("CPU/CUDA stage storage differs")
	}
	width, err := width(a.DType)
	if err != nil {
		return d, err
	}
	left, err := os.ReadFile(a.Path)
	if err != nil {
		return d, err
	}
	right, err := os.ReadFile(b.Path)
	if err != nil {
		return d, err
	}
	if len(left) != int(a.Bytes) || len(right) != int(b.Bytes) {
		return d, fmt.Errorf("invalid comparison payload length")
	}
	d.Elements = len(left) / int(width)
	bits := func(data []byte) uint32 {
		switch width {
		case 1:
			return uint32(data[0])
		case 2:
			return uint32(binary.LittleEndian.Uint16(data))
		default:
			return binary.LittleEndian.Uint32(data)
		}
	}
	for i := 0; i < d.Elements; i++ {
		av, bv := bits(left[i*int(width):]), bits(right[i*int(width):])
		if av != bv {
			d.Different++
			if d.FirstIndex < 0 {
				d.FirstIndex, d.FirstCPU, d.FirstCUDA = i, av, bv
			}
		}
	}
	return d, nil
}

func checkImported(path string) (m Imported, raw RawCapture, err error) {
	if err = readClosed(path, &m); err != nil {
		return m, raw, err
	}
	if m.Schema != ImportSchema || m.Status != "complete_observations_validated" || len(m.Observations) != len(fixtureNames) || len(m.ConsumerSources) != len(sourceFiles) {
		return m, raw, fmt.Errorf("incomplete imported observation")
	}
	if err = validateClocks(m.Started, m.Finished); err != nil {
		return m, raw, err
	}
	if err = validatePin(m.Raw); err != nil {
		return m, raw, err
	}
	if err = readClosed(m.Raw.Path, &raw); err != nil {
		return m, raw, err
	}
	if err = validateRaw(m.Raw.Path, raw); err != nil {
		return m, raw, err
	}
	if err = validatePin(m.Ranges.Table); err != nil {
		return m, raw, err
	}
	if m.Ranges.UniqueRanges != 31743 || m.Ranges.Rows != 32256 || m.Ranges.PaddingRows != 513 {
		return m, raw, fmt.Errorf("incomplete finite range summary")
	}
	for i, o := range m.Observations {
		f := raw.Prepared.Fixtures[i]
		actual, e := importStages(filepath.Dir(m.Raw.Path), f.Name, raw.Device, raw.Ordinal, f.Input, false)
		if e != nil {
			return m, raw, e
		}
		actual.Status = "passed"
		if !reflect.DeepEqual(actual, o) {
			return m, raw, fmt.Errorf("import differs from retained raw observation %s", f.Name)
		}
	}
	return m, raw, nil
}

// Compare preserves raw bit differences for every stage and dtype, including nonfinite values.
func Compare(root, cpuPath, cudaPath, out string) (m Comparison, err error) {
	cpuPath, err = filepath.Abs(cpuPath)
	if err != nil {
		return m, err
	}
	cudaPath, err = filepath.Abs(cudaPath)
	if err != nil {
		return m, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return m, err
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return m, err
	}
	if err = admitOutput(root, out); err != nil {
		return m, err
	}
	cpu, cr, err := checkImported(cpuPath)
	if err != nil {
		return m, err
	}
	cuda, gr, err := checkImported(cudaPath)
	if err != nil {
		return m, err
	}
	if cr.Device != "cpu" || gr.Device != "cuda" || cr.Build.SHA256 != gr.Build.SHA256 || !reflect.DeepEqual(cr.Prepared.Generated, gr.Prepared.Generated) || !reflect.DeepEqual(cr.ProducerSources, gr.ProducerSources) {
		return m, fmt.Errorf("CPU/CUDA producer mismatch")
	}
	m = Comparison{Schema: "nanopore-sup-qkv-i8/compare-v1", Started: clock(), Stages: []StageDifference{}}
	m.CPU, err = filePin(cpuPath)
	if err != nil {
		return m, err
	}
	m.CUDA, err = filePin(cudaPath)
	if err != nil {
		return m, err
	}
	m.ConsumerSources, err = currentSources(root)
	if err != nil {
		return m, err
	}
	for i, a := range cpu.Observations {
		b := cuda.Observations[i]
		if a.Name != b.Name {
			return m, fmt.Errorf("fixture comparison mismatch")
		}
		for _, spec := range StageSpecs() {
			d, e := dataDifference(a.Name, spec.Name, a.Stages[spec.Name], b.Stages[spec.Name])
			if e != nil {
				return m, e
			}
			m.Stages = append(m.Stages, d)
		}
	}
	m.Finished = clock()
	if err = os.Mkdir(out, 0755); err != nil {
		return m, err
	}
	err = writeJSON(filepath.Join(out, "manifest.json"), m)
	return m, err
}
