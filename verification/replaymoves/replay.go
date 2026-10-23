// Package replaymoves verifies native moves from retained CRF tensors without inference.
package replaymoves

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
	decodeoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/decode"
	stitchoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/stitch"
)

type Config struct {
	Source, Output string
	Captures       []string
}
type Payload struct {
	File, SHA256 string
	Bytes        int64
}
type savedChunk struct {
	Blocks, States                   int
	Decoded                          stitch.Chunk
	Scores, Forward, Backward, Posts Payload
}
type manifest struct {
	Schema     int
	Run, Input Payload
	Chunks     []savedChunk
	Output     Payload
}
type capture struct {
	Path, SHA256 string
	Manifest     manifest
	Options      decode.Options
	Config       stitch.Config
	Expected     stitch.Result
}
type Execution struct {
	Args                                    []string
	Started, Finished                       time.Time
	ExitCode                                int
	Error                                   string
	InputSHA256, OutputSHA256, StderrSHA256 string
}
type Endpoint struct {
	SequenceSHA256, QualitySHA256, MovesSHA256 string
	Bases, MoveBytes                           int
}
type ChunkResult struct {
	Index, Blocks, States        int
	Endpoint                     Endpoint
	StatesSHA256, QualBitsSHA256 string
	FinalScoreBits               uint32
}
type CaptureResult struct {
	Path, ManifestSHA256 string
	Payloads             map[string]Payload
	Chunks               []ChunkResult
	Stitched             Endpoint
}
type Receipt struct {
	Started, Finished    time.Time
	Passed               bool
	Error, Qualification string
	Pins                 map[string]string
	Builds               map[string]Execution
	Executions           map[string]Execution
	Captures             []CaptureResult
}

func digest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func hashFile(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	_, e = io.Copy(h, f)
	return fmt.Sprintf("%x", h.Sum(nil)), e
}
func saveJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
func readPayload(dir string, p Payload) ([]byte, error) {
	if p.File == "" || filepath.Base(p.File) != p.File || p.Bytes < 0 || p.Bytes > 1<<30 || len(p.SHA256) != 64 {
		return nil, errors.New("invalid payload reference")
	}
	info, e := os.Lstat(filepath.Join(dir, p.File))
	if e != nil {
		return nil, e
	}
	if !info.Mode().IsRegular() || info.Size() != p.Bytes {
		return nil, errors.New("payload size or file type differs")
	}
	b, e := os.ReadFile(filepath.Join(dir, p.File))
	if e != nil {
		return nil, e
	}
	if int64(len(b)) != p.Bytes || digest(b) != p.SHA256 {
		return nil, errors.New("payload SHA256 differs")
	}
	return b, nil
}
func load(ctx context.Context, path string) (capture, error) {
	// The maintained caller replay validates all schema fields, input bits, model pins,
	// shapes and Go endpoints. It does not load weights or invoke inference.
	out, e := caller.Replay(ctx, path)
	if e != nil {
		return capture{}, e
	}
	b, e := os.ReadFile(path)
	if e != nil {
		return capture{}, e
	}
	c := capture{Path: path, SHA256: digest(b), Expected: stitch.Result{Sequence: out.Sequence, Qstring: out.Qstring, Moves: out.Moves}}
	if e = json.Unmarshal(b, &c.Manifest); e != nil {
		return c, e
	}
	dir := filepath.Dir(path)
	b, e = readPayload(dir, c.Manifest.Run)
	if e != nil {
		return c, e
	}
	var run struct {
		Decode decode.Options
		Chunks struct{ Stride uint64 }
	}
	if e = json.Unmarshal(b, &run); e != nil {
		return c, e
	}
	c.Options = run.Decode
	b, e = readPayload(dir, c.Manifest.Input)
	if e != nil {
		return c, e
	}
	var input struct{ NormalizedSamples uint64 }
	if e = json.Unmarshal(b, &input); e != nil {
		return c, e
	}
	c.Config = stitch.Config{RawSamples: input.NormalizedSamples, Stride: run.Chunks.Stride}
	return c, nil
}
func floats(dir string, p Payload, n int) ([]float32, error) {
	if n < 0 || n > 1<<27 || p.Bytes != int64(n)*4 {
		return nil, errors.New("float payload shape differs")
	}
	b, e := readPayload(dir, p)
	if e != nil {
		return nil, e
	}
	a := make([]float32, n)
	for i := range a {
		a[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return a, nil
}
func writeDecode(w io.Writer, in decode.Input, o decode.Options) error {
	if _, e := fmt.Fprintf(w, "%d %d %d %d %d %d %d\n", in.Blocks, in.States, o.BeamWidth, math.Float32bits(o.BeamCut), math.Float32bits(o.BlankScore), math.Float32bits(o.QShift), math.Float32bits(o.QScale)); e != nil {
		return e
	}
	var scratch [16]byte
	for _, a := range [][]float32{in.Scores, in.BackGuides, in.Posts} {
		for _, v := range a {
			b := strconv.AppendUint(scratch[:0], uint64(math.Float32bits(v)), 10)
			b = append(b, ' ')
			if _, e := w.Write(b); e != nil {
				return e
			}
		}
		if _, e := io.WriteString(w, "\n"); e != nil {
			return e
		}
	}
	return nil
}
func readDecode(r *bufio.Reader) (decode.Result, error) {
	var seq, q string
	var n int
	if _, e := fmt.Fscan(r, &seq, &q, &n); e != nil {
		return decode.Result{}, e
	}
	if n < 1 || n > 100000 {
		return decode.Result{}, errors.New("invalid native blocks")
	}
	s, e := hex.DecodeString(seq)
	if e != nil {
		return decode.Result{}, e
	}
	quality, e := hex.DecodeString(q)
	if e != nil {
		return decode.Result{}, e
	}
	out := decode.Result{Sequence: string(s), Qstring: string(quality), Moves: make([]uint8, n), States: make([]int32, n), QualData: make([]float32, 4*n)}
	for i := range out.Moves {
		if _, e = fmt.Fscan(r, &out.Moves[i]); e != nil {
			return out, e
		}
	}
	for i := range out.States {
		if _, e = fmt.Fscan(r, &out.States[i]); e != nil {
			return out, e
		}
	}
	var bits uint32
	if _, e = fmt.Fscan(r, &bits); e != nil {
		return out, e
	}
	out.FinalScore = math.Float32frombits(bits)
	for i := range out.QualData {
		if _, e = fmt.Fscan(r, &bits); e != nil {
			return out, e
		}
		out.QualData[i] = math.Float32frombits(bits)
	}
	return out, nil
}
func eof(r io.Reader) error {
	var extra string
	_, e := fmt.Fscan(r, &extra)
	if e != io.EOF {
		return fmt.Errorf("unexpected native output or parse error: %q %v", extra, e)
	}
	return nil
}
func endpoint(r stitch.Result) Endpoint {
	return Endpoint{digest([]byte(r.Sequence)), digest([]byte(r.Qstring)), digest(r.Moves), len(r.Sequence), len(r.Moves)}
}
func discrete(a decode.Result, b stitch.Chunk) bool {
	return a.Sequence == b.Sequence && a.Qstring == b.Qstring && bytes.Equal(a.Moves, b.Moves)
}
func exactDecode(a, b decode.Result) bool {
	if a.Sequence != b.Sequence || a.Qstring != b.Qstring || !bytes.Equal(a.Moves, b.Moves) || !reflect.DeepEqual(a.States, b.States) || math.Float32bits(a.FinalScore) != math.Float32bits(b.FinalScore) || len(a.QualData) != len(b.QualData) {
		return false
	}
	for i := range a.QualData {
		if math.Float32bits(a.QualData[i]) != math.Float32bits(b.QualData[i]) {
			return false
		}
	}
	return true
}
func run(ctx context.Context, exe, input, dir string) (Execution, error) {
	args := []string{exe}
	in, e := os.Open(input)
	if e != nil {
		return Execution{}, e
	}
	defer in.Close()
	out, e := os.Create(filepath.Join(dir, "native-output.txt"))
	if e != nil {
		return Execution{}, e
	}
	defer out.Close()
	stderr, e := os.Create(filepath.Join(dir, "native-stderr.txt"))
	if e != nil {
		return Execution{}, e
	}
	defer stderr.Close()
	bounded, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(bounded, exe)
	cmd.Stdin = in
	cmd.Stdout = out
	cmd.Stderr = stderr
	r := Execution{Args: args, Started: time.Now().UTC(), ExitCode: -1}
	e = cmd.Run()
	r.Finished = time.Now().UTC()
	if cmd.ProcessState != nil {
		r.ExitCode = cmd.ProcessState.ExitCode()
	}
	if e != nil {
		r.Error = e.Error()
	}
	var pinErr error
	r.InputSHA256, pinErr = hashFile(input)
	e = errors.Join(e, pinErr)
	r.OutputSHA256, pinErr = hashFile(out.Name())
	e = errors.Join(e, pinErr)
	r.StderrSHA256, pinErr = hashFile(stderr.Name())
	e = errors.Join(e, pinErr)
	return r, errors.Join(e, saveJSON(filepath.Join(dir, "run-receipt.json"), r))
}
func sourcePins(source string) error {
	if e := decodeoracle.VerifySources(source); e != nil {
		return e
	}
	return stitchoracle.VerifySources(source)
}
func identity(root string, pins map[string]string) error {
	pins["go_runtime"] = runtime.Version()
	pins["go_platform"] = runtime.GOOS + "/" + runtime.GOARCH
	for _, dir := range []string{"internal", "verification"} {
		if e := filepath.WalkDir(filepath.Join(root, dir), func(path string, d os.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.IsDir() {
				return nil
			}
			ext := filepath.Ext(path)
			if ext != ".go" && ext != ".cpp" && ext != ".c" && ext != ".h" && ext != ".json" {
				return nil
			}
			h, e := hashFile(path)
			if e != nil {
				return e
			}
			rel, e := filepath.Rel(root, path)
			if e != nil {
				return e
			}
			pins[rel] = h
			return nil
		}); e != nil {
			return e
		}
	}
	for _, name := range []string{"go.mod", "go.sum"} {
		path := filepath.Join(root, name)
		if _, e := os.Stat(path); os.IsNotExist(e) {
			continue
		}
		h, e := hashFile(path)
		if e != nil {
			return e
		}
		pins[name] = h
	}
	return nil
}
func executablePins(exe string, pins map[string]string) error {
	h, e := hashFile(exe)
	if e != nil {
		return e
	}
	pins[exe] = h
	b, e := exec.Command("ldd", exe).CombinedOutput()
	if e != nil {
		return fmt.Errorf("ldd: %w: %s", e, b)
	}
	pins[exe+" ldd"] = string(b)
	for _, field := range strings.Fields(string(b)) {
		if strings.HasPrefix(field, "/") {
			h, e = hashFile(field)
			if e != nil {
				return e
			}
			pins[field] = h
		}
	}
	return nil
}

// Replay always creates a new output directory and rebuilds both original-source
// oracles. No success cache or inference producer exists in this package.
func Replay(ctx context.Context, cfg Config) (receipt Receipt, err error) {
	receipt = Receipt{Started: time.Now().UTC(), Qualification: "retained tensors; original-source float CPU beam search and untouched stitch.cpp with ReadCommon shim; no model rerun, stock CUDA12/GPU or biological qualification", Pins: map[string]string{}, Builds: map[string]Execution{}, Executions: map[string]Execution{}}
	if ctx == nil || cfg.Source == "" || cfg.Output == "" || len(cfg.Captures) == 0 {
		return receipt, errors.New("context, source, new output directory and captures required")
	}
	if err = os.Mkdir(cfg.Output, 0700); err != nil {
		return receipt, err
	}
	defer func() {
		receipt.Finished = time.Now().UTC()
		receipt.Passed = err == nil
		if err != nil {
			receipt.Error = err.Error()
		}
		err = errors.Join(err, saveJSON(filepath.Join(cfg.Output, "receipt.json"), receipt))
	}()
	if err = sourcePins(cfg.Source); err != nil {
		return receipt, err
	}
	for name, want := range stitchoracle.SourcePins {
		receipt.Pins["dorado/"+name] = want
	}
	receipt.Pins["dorado/"+decodeoracle.SourcePath] = decodeoracle.SourceSHA256
	receipt.Pins["dorado_commit"] = decodeoracle.SourceCommit
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return receipt, errors.New("cannot locate verifier source")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "../.."))
	if err = identity(root, receipt.Pins); err != nil {
		return receipt, err
	}
	compiler, e := exec.LookPath("c++")
	if e != nil {
		return receipt, e
	}
	if err = executablePins(compiler, receipt.Pins); err != nil {
		return receipt, err
	}
	version, e := exec.Command("c++", "--version").CombinedOutput()
	if e != nil {
		return receipt, e
	}
	receipt.Pins["compiler_version"] = string(version)
	goCompiler, e := exec.LookPath("go")
	if e != nil {
		return receipt, e
	}
	if receipt.Pins["go_compiler_sha256"], e = hashFile(goCompiler); e != nil {
		return receipt, e
	}
	goEnv, e := exec.Command("go", "env", "-json", "CGO_ENABLED", "CC", "CXX", "CGO_CFLAGS", "CGO_CPPFLAGS", "CGO_CXXFLAGS", "CGO_LDFLAGS", "GOFLAGS", "GOEXPERIMENT").CombinedOutput()
	if e != nil {
		return receipt, e
	}
	receipt.Pins["go_build_environment"] = string(goEnv)
	self, e := os.Executable()
	if e != nil {
		return receipt, e
	}
	if err = executablePins(self, receipt.Pins); err != nil {
		return receipt, err
	}
	executables := map[string]string{}
	for _, build := range []struct {
		name  string
		build func(string, string) (string, error)
	}{{"decode", decodeoracle.Build}, {"stitch", stitchoracle.Build}} {
		r := Execution{Args: []string{build.name + "oracle.Build", cfg.Source, filepath.Join(cfg.Output, build.name+"-build")}, Started: time.Now().UTC(), ExitCode: -1}
		exe, e := build.build(cfg.Source, filepath.Join(cfg.Output, build.name+"-build"))
		r.Finished = time.Now().UTC()
		if e != nil {
			r.Error = e.Error()
		} else {
			r.ExitCode = 0
		}
		receipt.Builds[build.name] = r
		if err = saveJSON(filepath.Join(cfg.Output, build.name+"-build-result.json"), r); err != nil {
			return receipt, err
		}
		if e != nil {
			return receipt, e
		}
		executables[build.name] = exe
		if err = executablePins(exe, receipt.Pins); err != nil {
			return receipt, err
		}
	}
	for i, path := range cfg.Captures {
		var result CaptureResult
		result, err = replayCapture(ctx, path, filepath.Join(cfg.Output, fmt.Sprintf("capture-%02d", i)), executables, receipt.Executions)
		receipt.Captures = append(receipt.Captures, result)
		if err != nil {
			return receipt, err
		}
	}
	if err = sourcePins(cfg.Source); err != nil {
		return receipt, err
	}
	after := map[string]string{}
	if err = identity(root, after); err != nil {
		return receipt, err
	}
	for k, v := range after {
		if receipt.Pins[k] != v {
			return receipt, fmt.Errorf("verifier source changed during replay: %s", k)
		}
	}
	return receipt, nil
}
func replayCapture(ctx context.Context, path, dir string, exes map[string]string, executions map[string]Execution) (result CaptureResult, err error) {
	result.Path = path
	c, e := load(ctx, path)
	if e != nil {
		return result, e
	}
	result.ManifestSHA256 = c.SHA256
	result.Payloads = map[string]Payload{}
	for _, p := range []Payload{c.Manifest.Run, c.Manifest.Input, c.Manifest.Output} {
		result.Payloads[p.File] = p
	}
	for _, p := range []Payload{c.Manifest.Run, c.Manifest.Input} {
		b, e := readPayload(filepath.Dir(path), p)
		if e != nil {
			return result, e
		}
		var nested struct{ ModelConfig, Raw, Normalized Payload }
		if e = json.Unmarshal(b, &nested); e != nil {
			return result, e
		}
		for _, item := range []Payload{nested.ModelConfig, nested.Raw, nested.Normalized} {
			if item.File != "" {
				result.Payloads[item.File] = item
			}
		}
	}
	if e = os.Mkdir(dir, 0700); e != nil {
		return result, e
	}
	nativeChunks := make([]stitch.Chunk, len(c.Manifest.Chunks))
	for i, s := range c.Manifest.Chunks {
		for _, p := range []Payload{s.Scores, s.Forward, s.Backward, s.Posts} {
			result.Payloads[p.File] = p
		}
		in := decode.Input{Blocks: s.Blocks, States: s.States}
		in.Scores, e = floats(filepath.Dir(path), s.Scores, s.Blocks*4*s.States)
		if e != nil {
			return result, e
		}
		in.BackGuides, e = floats(filepath.Dir(path), s.Backward, (s.Blocks+1)*s.States)
		if e != nil {
			return result, e
		}
		in.Posts, e = floats(filepath.Dir(path), s.Posts, (s.Blocks+1)*s.States)
		if e != nil {
			return result, e
		}
		chunkDir := filepath.Join(dir, fmt.Sprintf("chunk-%02d", i))
		if e = os.Mkdir(chunkDir, 0700); e != nil {
			return result, e
		}
		input := filepath.Join(chunkDir, "input.txt")
		f, e := os.Create(input)
		if e != nil {
			return result, e
		}
		w := bufio.NewWriterSize(f, 1<<20)
		e = writeDecode(w, in, c.Options)
		e = errors.Join(e, w.Flush(), f.Close())
		if e != nil {
			return result, e
		}
		goResult, e := decode.Decode(in, c.Options)
		if e != nil {
			return result, e
		}
		if e = saveJSON(filepath.Join(chunkDir, "go-output.json"), goResult); e != nil {
			return result, e
		}
		execution, e := run(ctx, exes["decode"], input, chunkDir)
		executions[chunkDir] = execution
		if e != nil {
			return result, e
		}
		b, e := os.ReadFile(filepath.Join(chunkDir, "native-output.txt"))
		if e != nil {
			return result, e
		}
		reader := bufio.NewReader(bytes.NewReader(b))
		native, e := readDecode(reader)
		if e != nil {
			return result, e
		}
		if e = eof(reader); e != nil {
			return result, e
		}
		if !discrete(native, s.Decoded) || !exactDecode(native, goResult) {
			return result, fmt.Errorf("chunk %d native/Go/captured endpoint or native/Go diagnostic bits differ", i)
		}
		nativeChunks[i] = s.Decoded
		nativeChunks[i].Sequence = native.Sequence
		nativeChunks[i].Qstring = native.Qstring
		nativeChunks[i].Moves = native.Moves
		stateBits := make([]byte, 4*len(native.States))
		qualBits := make([]byte, 4*len(native.QualData))
		for j, v := range native.States {
			binary.LittleEndian.PutUint32(stateBits[j*4:], uint32(v))
		}
		for j, v := range native.QualData {
			binary.LittleEndian.PutUint32(qualBits[j*4:], math.Float32bits(v))
		}
		result.Chunks = append(result.Chunks, ChunkResult{i, s.Blocks, s.States, endpoint(stitch.Result{Sequence: native.Sequence, Qstring: native.Qstring, Moves: native.Moves}), digest(stateBits), digest(qualBits), math.Float32bits(native.FinalScore)})
	}
	stitchDir := filepath.Join(dir, "stitch")
	if e = os.Mkdir(stitchDir, 0700); e != nil {
		return result, e
	}
	var input bytes.Buffer
	fmt.Fprintf(&input, "%d %d 0 %d\n", c.Config.RawSamples, c.Config.Stride, len(nativeChunks))
	for _, s := range nativeChunks {
		fmt.Fprintf(&input, "%d %d %d %x %x %x\n", s.Offset, s.RawSize, s.Granularity, s.Sequence, s.Qstring, s.Moves)
	}
	inputPath := filepath.Join(stitchDir, "input.txt")
	if e = os.WriteFile(inputPath, input.Bytes(), 0600); e != nil {
		return result, e
	}
	execution, e := run(ctx, exes["stitch"], inputPath, stitchDir)
	executions[stitchDir] = execution
	if e != nil {
		return result, e
	}
	b, e := os.ReadFile(filepath.Join(stitchDir, "native-output.txt"))
	if e != nil {
		return result, e
	}
	fields := strings.Fields(string(b))
	if len(fields) != 3 {
		return result, errors.New("invalid stitch output")
	}
	values := make([][]byte, 3)
	for i, s := range fields {
		values[i], e = hex.DecodeString(s)
		if e != nil {
			return result, e
		}
	}
	native := stitch.Result{Sequence: string(values[0]), Qstring: string(values[1]), Moves: values[2]}
	goResult, e := stitch.Stitch(c.Config, nativeChunks)
	if e != nil {
		return result, e
	}
	if e = saveJSON(filepath.Join(stitchDir, "go-output.json"), goResult); e != nil {
		return result, e
	}
	if !reflect.DeepEqual(native, goResult) || !reflect.DeepEqual(native, c.Expected) {
		return result, errors.New("native/Go/captured stitched sequence, quality or moves differ")
	}
	result.Stitched = endpoint(native)
	// Revalidate raw payload existence and all manifest pins before qualifying.
	again, e := load(ctx, path)
	if e != nil {
		return result, e
	}
	if again.SHA256 != c.SHA256 {
		return result, errors.New("capture changed during replay")
	}
	return result, nil
}
