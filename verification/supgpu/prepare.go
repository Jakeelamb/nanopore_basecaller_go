package supgpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/gpuprofile"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

const fixtureSHA = "1bb772a5b6e68ecaf2fe6911809ae3cbb49e6341b2dd479285ff0cbacf1621ec"
const developmentSignalSHA = "931078657d3ef96ef9faeacd3fb0dad3197abeff4bf577a9cf8710b2b99bce4b"
const developmentRead = "bb76c31a-36f0-48a0-abd2-107a10e1ade0"

var sourcePins = []struct{ path, sha string }{
	{"dorado/nn/TxModules.cpp", "9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701"},
	{"dorado/nn/ConvStack.cpp", "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2"},
	{"dorado/basecall/model/TxModel.cpp", "fb9184a3755819242c820c24cb9c8976022f88e22908fa2097f2f0a4047a20ed"},
	{"dorado/basecall/crf_utils.cpp", "0758b6f4d57693566ab15cb0332255ff75ea01ac4f529c04329cb1aac13caf0f"},
	{"dorado/torch_utils/tensor_utils.cpp", "7e535e83d13225e1622cda6d16f842641ea466633009ad03e695d7af7362b018"},
	{"dorado/config/include/config/BasecallModelConfig.h", "55f5abcc032582d23370ff54c6b7645d1914dc9c458e0f57d56b0d15d23be5d8"},
}

type PrepareOptions struct{ Dorado, Model, Fixture, Profile, Target, Out string }

// Prepare fixes the first caller-normalized fixed case from the actual config.
// The retained profile selects FP8. It does not execute a capability query.
func Prepare(o PrepareOptions) (Prepared, error) {
	p := Prepared{Schema: PreparedSchema, SourceCommit: SourceCommit, Case: "fixed-development-n32-minimum", Cohort: "HG002-development", UseI8: true, UseF8: true, RemoveBits: 4, Trim: 10, Weights: []Pin{}, Sources: []Pin{}}
	var err error
	for _, expected := range sourcePins {
		v, e := pin(filepath.Join(o.Dorado, expected.path))
		if e != nil {
			return p, e
		}
		if v.SHA256 != expected.sha {
			return p, fmt.Errorf("source changed: %s", expected.path)
		}
		p.Sources = append(p.Sources, v)
	}
	targetRaw, err := os.ReadFile(o.Target)
	if err != nil {
		return p, err
	}
	target, err := gpuprofile.DecodeTarget(bytes.NewReader(targetRaw))
	if err != nil {
		return p, err
	}
	tp, err := pin(o.Target)
	if err != nil {
		return p, err
	}
	p.Sources = append(p.Sources, tp)
	p.Profile, err = pin(o.Profile)
	if err != nil {
		return p, err
	}
	if p.Profile.SHA256 != ProfileSHA {
		return p, fmt.Errorf("unrecognized retained P01 observation")
	}
	var profile struct {
		Status      string                 `json:"status"`
		Observation gpuprofile.Observation `json:"observation"`
	}
	raw, err := os.ReadFile(o.Profile)
	if err != nil {
		return p, err
	}
	if err = json.Unmarshal(raw, &profile); err != nil {
		return p, err
	}
	if profile.Status != "predicate_observation_passed" {
		return p, fmt.Errorf("P01 observation did not pass")
	}
	if _, err = gpuprofile.Bind(target, profile.Observation); err != nil {
		return p, err
	}
	if profile.Observation.E4M3Status == nil || *profile.Observation.E4M3Status != 0 {
		return p, fmt.Errorf("retained target did not select FP8")
	}
	p.DeviceUUID = target.DeviceUUID
	p.Config, err = pin(filepath.Join(o.Model, "config.toml"))
	if err != nil {
		return p, err
	}
	if p.Config.SHA256 != gpuprofile.SUPConfig || filepath.Base(filepath.Clean(o.Model)) != ModelName {
		return p, fmt.Errorf("unsupported SUP package/config")
	}
	config, err := os.ReadFile(p.Config.Path)
	if err != nil {
		return p, err
	}
	stride, mean, stdev, err := configGeometry(string(config))
	if err != nil {
		return p, err
	}
	p.Geometry = Geometry{Batch: 32, Chunk: stride * 64, EncoderT: 64, ConvStride: stride, CallerChunkGranularity: stride * 64, Overlap: 600}
	p.MeanBits, p.StdevBits = math.Float32bits(float32(mean)), math.Float32bits(float32(stdev))
	for _, name := range WeightNames() {
		v, e := pin(filepath.Join(o.Model, name))
		if e != nil {
			return p, e
		}
		p.Weights = append(p.Weights, v)
	}
	p.Fixture, err = pin(o.Fixture)
	if err != nil {
		return p, err
	}
	if p.Fixture.SHA256 != fixtureSHA {
		return p, fmt.Errorf("unrecognized retained POD5 fixture manifest")
	}
	fixtureRaw, err := os.ReadFile(p.Fixture.Path)
	if err != nil {
		return p, err
	}
	var fixture struct {
		Fixtures []struct {
			Name       string `json:"name"`
			SourcePath string `json:"source_path"`
			Reads      []struct {
				ID         string            `json:"id"`
				SignalFile string            `json:"signal_file"`
				SignalSHA  string            `json:"signal_sha256"`
				FloatBits  map[string]uint32 `json:"float_bits"`
			} `json:"reads"`
		} `json:"fixtures"`
	}
	if err = json.Unmarshal(fixtureRaw, &fixture); err != nil {
		return p, err
	}
	found := false
	for _, f := range fixture.Fixtures {
		for _, r := range f.Reads {
			if r.ID != developmentRead {
				continue
			}
			if found || f.Name != "hg002_smoke" || !strings.Contains(f.SourcePath, "/HG002/PAW70337/") || r.SignalSHA != developmentSignalSHA {
				return p, fmt.Errorf("fixture identity mismatch")
			}
			found = true
			p.ReadID = r.ID
			p.Signal, err = pin(filepath.Join(filepath.Dir(p.Fixture.Path), r.SignalFile))
			if err != nil {
				return p, err
			}
			if p.Signal.SHA256 != r.SignalSHA {
				return p, fmt.Errorf("retained signal changed")
			}
			p.CalibrationScaleBits = r.FloatBits["CalibrationScale"]
			p.CalibrationOffsetBits = r.FloatBits["CalibrationOffset"]
		}
	}
	if !found {
		return p, fmt.Errorf("genuine development fixture not found")
	}
	input, chunks, err := preparedInput(p)
	if err != nil {
		return p, err
	}
	p.Chunks = chunks
	if err = os.MkdirAll(filepath.Dir(o.Out), 0755); err != nil {
		return p, err
	}
	if err = os.Mkdir(o.Out, 0755); err != nil {
		return p, err
	}
	inputPath := filepath.Join(o.Out, "input.f16le")
	if err = os.WriteFile(inputPath, input, 0644); err != nil {
		return p, err
	}
	p.Input, err = pin(inputPath)
	if err != nil {
		return p, err
	}
	if err = writeJSON(filepath.Join(o.Out, "prepared.json"), p); err != nil {
		return p, err
	}
	return p, nil
}

func WeightNames() []string {
	names := []string{}
	for i := 0; i < 5; i++ {
		for _, suffix := range []string{"weight", "bias"} {
			names = append(names, fmt.Sprintf("conv.%d.conv.%s.tensor", i, suffix))
		}
	}
	for i := 0; i < 18; i++ {
		for _, name := range []string{"self_attn.Wqkv.weight", "self_attn.out_proj.weight", "self_attn.out_proj.bias", "ff.fc1.weight", "ff.fc2.weight", "norm1.weight", "norm2.weight"} {
			names = append(names, fmt.Sprintf("transformer_encoder.%d.%s.tensor", i, name))
		}
	}
	return append(names, "upsample.linear.weight.tensor", "upsample.linear.bias.tensor", "crf.linear.weight.tensor")
}

func configGeometry(raw string) (stride int, mean, stdev float64, err error) {
	stride = 1
	convs := 0
	section := ""
	standardise := false
	hasMean, hasStdev := false, false
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			section = line
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if section == "[[model.encoder.conv.sublayers]]" && key == "stride" {
			v, e := strconv.Atoi(value)
			if e != nil || v <= 0 || v > 16 {
				return 0, 0, 0, fmt.Errorf("bad conv stride")
			}
			stride *= v
			convs++
		}
		if section == "[standardisation]" {
			switch key {
			case "standardise":
				standardise = value == "1"
			case "mean":
				mean, err = strconv.ParseFloat(value, 64)
				hasMean = true
			case "stdev":
				stdev, err = strconv.ParseFloat(value, 64)
				hasStdev = true
			}
			if err != nil {
				return 0, 0, 0, err
			}
		}
	}
	if convs != 5 || stride != 12 || !standardise || !hasMean || !hasStdev || stdev <= 0 {
		return 0, 0, 0, fmt.Errorf("unexpected SUP convolution/scaling config")
	}
	return stride, mean, stdev, nil
}

func preparedInput(p Prepared) ([]byte, []Chunk, error) {
	raw, err := os.ReadFile(p.Signal.Path)
	if err != nil {
		return nil, nil, err
	}
	samples, err := hacgpu.DecodeInt16LE(raw)
	if err != nil {
		return nil, nil, err
	}
	n := hacgpu.Normalisation{Mean: float64(math.Float32frombits(p.MeanBits)), Stdev: float64(math.Float32frombits(p.StdevBits)), Scaling: float64(math.Float32frombits(p.CalibrationScaleBits)), Offset: float64(math.Float32frombits(p.CalibrationOffsetBits)), TrimStart: p.Trim}
	bits, err := n.Normalise(samples)
	if err != nil {
		return nil, nil, err
	}
	starts, err := chunk.Fixed(uint64(len(bits)), chunk.Config{ChunkSize: uint64(p.Geometry.Chunk), Stride: uint64(p.Geometry.ConvStride), Overlap: uint64(p.Geometry.Overlap), Granularity: uint64(p.Geometry.CallerChunkGranularity)})
	if err != nil {
		return nil, nil, err
	}
	if len(starts) < p.Geometry.Batch {
		return nil, nil, fmt.Errorf("genuine signal lacks 32 source-planned chunks")
	}
	input := make([]byte, p.Geometry.Batch*p.Geometry.Chunk*2)
	chunks := []Chunk{}
	for i, start := range starts[:p.Geometry.Batch] {
		end := start + uint64(p.Geometry.Chunk)
		if end > uint64(len(bits)) {
			return nil, nil, fmt.Errorf("selected chunk requires padding")
		}
		part := input[i*p.Geometry.Chunk*2 : (i+1)*p.Geometry.Chunk*2]
		for j, v := range bits[start:end] {
			binary.LittleEndian.PutUint16(part[2*j:], v)
		}
		sum := sha256.Sum256(part)
		chunks = append(chunks, Chunk{start, end, hex.EncodeToString(sum[:])})
	}
	return input, chunks, nil
}

// ReadPrepared authenticates all declared inputs and re-derives the input bytes.
func ReadPrepared(path, sha string) (Prepared, error) {
	var p Prepared
	if err := readPinnedJSON(path, sha, &p); err != nil {
		return p, err
	}
	if p.Schema != PreparedSchema || p.SourceCommit != SourceCommit || p.Case != "fixed-development-n32-minimum" || p.Cohort != "HG002-development" || p.ReadID != developmentRead || p.DeviceUUID != gpuprofile.DeviceUUID || !p.UseI8 || !p.UseF8 || p.RemoveBits != 4 || p.Trim != 10 {
		return p, fmt.Errorf("unsupported prepared identity/dispatch")
	}
	if p.Geometry != (Geometry{32, 768, 64, 12, 768, 600}) || p.Config.SHA256 != gpuprofile.SUPConfig || p.Profile.SHA256 != ProfileSHA || p.Fixture.SHA256 != fixtureSHA || p.Signal.SHA256 != developmentSignalSHA {
		return p, fmt.Errorf("prepared geometry or input identity changed")
	}
	all := append([]Pin{p.Fixture, p.Signal, p.Input, p.Config, p.Profile}, p.Weights...)
	all = append(all, p.Sources...)
	seen := map[string]bool{}
	for _, v := range all {
		if seen[v.Path] {
			return p, fmt.Errorf("duplicate pin %s", v.Path)
		}
		seen[v.Path] = true
		if err := verifyPin("", v); err != nil {
			return p, err
		}
	}
	if len(p.Weights) != 139 || len(p.Sources) < len(sourcePins) {
		return p, fmt.Errorf("missing weight/source pins")
	}
	for i, name := range WeightNames() {
		if filepath.Base(p.Weights[i].Path) != name {
			return p, fmt.Errorf("weight binding %d changed", i)
		}
	}
	for i, s := range sourcePins {
		if !strings.HasSuffix(p.Sources[i].Path, "/"+s.path) || p.Sources[i].SHA256 != s.sha {
			return p, fmt.Errorf("source binding %d changed", i)
		}
	}
	config, err := os.ReadFile(p.Config.Path)
	if err != nil {
		return p, err
	}
	_, mean, stdev, err := configGeometry(string(config))
	if err != nil {
		return p, err
	}
	if p.MeanBits != math.Float32bits(float32(mean)) || p.StdevBits != math.Float32bits(float32(stdev)) || p.CalibrationScaleBits != 1044352829 || p.CalibrationOffsetBits != 3280404480 {
		return p, fmt.Errorf("scaling metadata changed")
	}
	got, err := os.ReadFile(p.Input.Path)
	if err != nil {
		return p, err
	}
	want, chunks, err := preparedInput(p)
	if err != nil {
		return p, err
	}
	if !bytes.Equal(got, want) {
		return p, fmt.Errorf("normalized input does not derive from retained signal")
	}
	if len(chunks) != len(p.Chunks) {
		return p, fmt.Errorf("chunk membership changed")
	}
	for i, c := range chunks {
		if c != p.Chunks[i] {
			return p, fmt.Errorf("chunk %d changed", i)
		}
	}
	return p, nil
}
