// Package splitteroracle compiles unchanged Dorado decision functions with ATen CPU Half.
package splitteroracle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
const EdlibCommit = "42aa8fa7051fb2aa1c4f144c5eda22a0e753f026"

var SourcePins = map[string]string{
	"dorado/splitter/include/splitter/DuplexReadSplitter.h":                       "c70d8cb1bbc1e80f1bb55fe5fc1fcb4a75436c964f3bac9db98a7582d8e350b3",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/SimplexRead.h": "a6e225079228a55ff0077477bf7d7d0b2c74e9b8d2441c659dee522da745b663",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h":  "a8ff4bced80c61f07070d6f5ed17ad3c2367c4f7d2a4564f54b5a7e47aec01cb",
	"dorado/splitter/DuplexReadSplitter.cpp":                                      "13e8a88e4aecda1dcb44eb58e2d8aa30dec76b7202e10f680dd1f6419710200e",
	"dorado/splitter/splitter_utils.cpp":                                          "33a055cf5c8ef3db49500c992e9c832682e4c175a231ae2b854d5705de8a9a48",
	"dorado/splitter/splitter_utils.h":                                            "53dc01aefc071c6e7270ba8dcbec0643636213f8771197a96529dc4678e2b79b",
	"dorado/splitter/include/splitter/ReadSplitter.h":                             "348579bf0eb2aeec8867079311730a38d91c5975ea5ad051a723dce5c0fa4641",
	"dorado/utils/sequence_utils.cpp":                                             "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2",
}
var EdlibPins = map[string]string{
	"edlib/src/edlib.cpp":   "a9c2b4a44820453d4991b129b373d04d82a59fc0aaabfad3d8da838c380ab0bd",
	"edlib/include/edlib.h": "62d0529a87aa6e701c767fb9ce3786c759068f454c129641b2be38da6237ccb5",
}

type Config struct{ Source, Edlib, Torch string }

func hash(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }
func saveJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}
func verify(root string, pins map[string]string) error {
	for path, want := range pins {
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			return err
		}
		if got := hash(data); got != want {
			return fmt.Errorf("source hash mismatch for %s: %s != %s", path, got, want)
		}
	}
	return nil
}
func VerifySources(c Config) error {
	if err := verify(c.Source, SourcePins); err != nil {
		return err
	}
	return verify(c.Edlib, EdlibPins)
}

func sourceBody(data []byte, signature string) ([]byte, error) {
	start := bytes.Index(data, []byte(signature))
	if start < 0 {
		return nil, fmt.Errorf("missing source signature %s", signature)
	}
	open := bytes.IndexByte(data[start:], '{')
	if open < 0 {
		return nil, fmt.Errorf("missing source body %s", signature)
	}
	open += start
	depth := 0
	for i := open; i < len(data); i++ {
		if data[i] == '{' {
			depth++
		}
		if data[i] == '}' {
			depth--
			if depth == 0 {
				end := i + 1
				if end < len(data) && data[end] == ';' {
					end++
				}
				return bytes.Clone(data[start:end]), nil
			}
		}
	}
	return nil, fmt.Errorf("unclosed source body %s", signature)
}

type limitedBuffer struct {
	data  bytes.Buffer
	limit int
}

func (w *limitedBuffer) Write(p []byte) (int, error) {
	if len(p) > w.limit-w.Len() {
		return 0, fmt.Errorf("native output exceeds byte limit")
	}
	return w.data.Write(p)
}

func (w *limitedBuffer) Bytes() []byte { return w.data.Bytes() }
func (w *limitedBuffer) Len() int      { return w.data.Len() }

// Command preserves raw process output and clocks even when execution fails.
func Command(dir, name string, cmd *exec.Cmd, input []byte) ([]byte, error) {
	cmd.Stdin = bytes.NewReader(input)
	stdout, stderr := &limitedBuffer{limit: 512 << 20}, &limitedBuffer{limit: 4 << 20}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	started := time.Now().UTC()
	runErr := cmd.Run()
	finished := time.Now().UTC()
	code := -1
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	for path, content := range map[string][]byte{name + ".stdout": stdout.Bytes(), name + ".stderr": stderr.Bytes()} {
		if err := os.WriteFile(filepath.Join(dir, path), content, 0600); err != nil {
			return nil, err
		}
	}
	receipt := map[string]any{"command": cmd.Args, "started": started, "finished": finished, "exit_code": code, "stdin_sha256": hash(input), "stdout_sha256": hash(stdout.Bytes()), "stderr_sha256": hash(stderr.Bytes())}
	if runErr != nil {
		receipt["error"] = runErr.Error()
	}
	if err := saveJSON(filepath.Join(dir, name+".receipt.json"), receipt); err != nil {
		return nil, err
	}
	if runErr != nil {
		return stdout.Bytes(), fmt.Errorf("%s: %w: %s", name, runErr, stderr.Bytes())
	}
	return stdout.Bytes(), nil
}

// Build requires a fresh directory. It copies pinned full source and extracts
// complete bodies unchanged. Read/class/logger declarations are test-only ABI shims.
func Build(c Config, output string) (string, error) {
	if err := os.Mkdir(output, 0700); err != nil {
		return "", err
	}
	if err := VerifySources(c); err != nil {
		_ = saveJSON(filepath.Join(output, "rejection.json"), map[string]any{"utc": time.Now().UTC(), "error": err.Error()})
		return "", err
	}
	for _, group := range []struct {
		root, prefix string
		pins         map[string]string
	}{{c.Source, "source", SourcePins}, {c.Edlib, "edlib", EdlibPins}} {
		for path := range group.pins {
			data, err := os.ReadFile(filepath.Join(group.root, path))
			if err != nil {
				return "", err
			}
			dst := filepath.Join(output, group.prefix, path)
			if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
				return "", err
			}
			if err := os.WriteFile(dst, data, 0600); err != nil {
				return "", err
			}
		}
	}
	pins := map[string]string{}
	for _, s := range []struct{ path, signature, name string }{
		{"dorado/splitter/splitter_utils.h", "inline constexpr int16_t fast_half_comparable(", "fast_half_comparable"},
		{"dorado/splitter/splitter_utils.h", "template <class FilterF>\nauto filter_ranges(", "filter_ranges"},
		{"dorado/splitter/splitter_utils.cpp", "template <typename T>\nSampleRanges<T> detect_pore_signal(", "detect_pore_signal"},
		{"dorado/utils/sequence_utils.cpp", "std::vector<uint64_t> move_cum_sums(", "move_cum_sums"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "float qscore_mean(", "qscore_mean"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "std::optional<PosRange> find_best_adapter_match(", "find_best_adapter_match"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "struct DuplexReadSplitter::ExtRead {", "ExtRead"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "DuplexReadSplitter::ExtRead DuplexReadSplitter::create_ext_read(", "create_ext_read"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "PosRanges DuplexReadSplitter::possible_pore_regions(", "possible_pore_regions"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "bool DuplexReadSplitter::check_nearby_adapter(", "check_nearby_adapter"},
	} {
		data, err := os.ReadFile(filepath.Join(c.Source, s.path))
		if err != nil {
			return "", err
		}
		slice, err := sourceBody(data, s.signature)
		if err != nil {
			return "", err
		}
		name := s.name + ".slice.cpp"
		if err := os.WriteFile(filepath.Join(output, name), slice, 0600); err != nil {
			return "", err
		}
		pins[name] = hash(slice)
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate oracle.cpp")
	}
	driver, err := os.ReadFile(filepath.Join(filepath.Dir(here), "oracle.cpp"))
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "oracle.cpp"), driver, 0600); err != nil {
		return "", err
	}
	pins["oracle.cpp"] = hash(driver)
	compiler, err := exec.LookPath("c++")
	if err != nil {
		return "", err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return "", err
	}
	compilerBytes, err := os.ReadFile(compiler)
	if err != nil {
		return "", err
	}
	version, err := Command(output, "compiler-version", exec.Command(compiler, "--version"), nil)
	if err != nil {
		return "", err
	}
	executable := filepath.Join(output, "splitter-oracle")
	lib := filepath.Join(c.Torch, "lib")
	args := []string{"-std=c++20", "-O2", "-Wall", "-Wextra", "-ffp-contract=off", "-MMD", "-MF", filepath.Join(output, "headers.d"), "-I" + filepath.Join(c.Torch, "include"), "-I" + filepath.Join(c.Torch, "include/torch/csrc/api/include"), "-I" + filepath.Join(output, "edlib/edlib/include"), "-I" + output, filepath.Join(output, "oracle.cpp"), filepath.Join(output, "edlib/edlib/src/edlib.cpp"), "-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10", "-o", executable}
	buildManifest := map[string]any{"source_commit": SourceCommit, "source_pins": SourcePins, "edlib_commit": EdlibCommit, "edlib_pins": EdlibPins, "slices_and_driver": pins, "compiler": compiler, "compiler_sha256": hash(compilerBytes), "compiler_version": string(version), "command": append([]string{compiler}, args...), "qualification": "unchanged source bodies; original settings; real CPU ATen Half; source-built edlib; all-candidate probe changes only top_candidates to INT32_MAX; no children, model calls or GPU"}
	if err := saveJSON(filepath.Join(output, "manifest.json"), buildManifest); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	if _, err := Command(output, "build", exec.CommandContext(ctx, compiler, args...), nil); err != nil {
		return "", err
	}
	exe, err := os.ReadFile(executable)
	if err != nil {
		return "", err
	}
	buildManifest["executable_sha256"] = hash(exe)
	dependencies, err := Command(output, "ldd", exec.Command("ldd", executable), nil)
	if err != nil {
		return "", err
	}
	runtimePins := map[string]string{}
	for _, field := range strings.Fields(string(dependencies)) {
		if strings.HasPrefix(field, "/") {
			data, err := os.ReadFile(field)
			if err != nil {
				return "", err
			}
			runtimePins[field] = hash(data)
		}
	}
	buildManifest["runtime_pins"] = runtimePins
	// A dependency scan records all translation-unit headers, including ATen.
	for i, source := range []string{filepath.Join(output, "oracle.cpp"), filepath.Join(output, "edlib/edlib/src/edlib.cpp")} {
		scan := []string{"-std=c++20", "-MM", "-I" + filepath.Join(c.Torch, "include"), "-I" + filepath.Join(c.Torch, "include/torch/csrc/api/include"), "-I" + filepath.Join(output, "edlib/edlib/include"), "-I" + output, source}
		data, err := Command(output, fmt.Sprintf("headers-%d", i), exec.Command(compiler, scan...), nil)
		if err != nil {
			return "", err
		}
		joined := strings.ReplaceAll(string(data), "\\\n", " ")
		colon := strings.IndexByte(joined, ':')
		if colon < 0 {
			return "", fmt.Errorf("invalid header dependency output")
		}
		for _, path := range strings.Fields(joined[colon+1:]) {
			b, err := os.ReadFile(path)
			if err != nil {
				return "", err
			}
			pins[path] = hash(b)
		}
	}
	if err := VerifySources(c); err != nil {
		return "", err
	}
	if err := saveJSON(filepath.Join(output, "manifest.json"), buildManifest); err != nil {
		return "", err
	}
	return executable, nil
}

const maxElements = 1 << 26

func EncodeInput(reads []splitter.Input) ([]byte, error) {
	if len(reads) > 100000 {
		return nil, fmt.Errorf("read count exceeds oracle limit")
	}
	var data bytes.Buffer
	put := func(v any) { _ = binary.Write(&data, binary.LittleEndian, v) }
	put(uint32(len(reads)))
	for _, r := range reads {
		if len(r.Signal) > maxElements || len(r.Moves) > maxElements || len(r.Sequence) > maxElements {
			return nil, fmt.Errorf("input count exceeds oracle limit")
		}
		if _, err := splitter.FindSimplexPA(r); err != nil {
			return nil, err
		}
		put(uint32(len(r.Sequence)))
		data.WriteString(r.Sequence)
		data.WriteString(r.Qstring)
		put(r.Stride)
		put(uint32(len(r.Moves)))
		data.Write(r.Moves)
		put(uint32(len(r.Signal)))
		put(r.Signal)
		if data.Len() > 512<<20 {
			return nil, fmt.Errorf("input exceeds oracle byte limit")
		}
	}
	return data.Bytes(), nil
}

func encodeExpected(reads []splitter.Input) ([]byte, []splitter.Decision, error) {
	var data bytes.Buffer
	put := func(v any) { _ = binary.Write(&data, binary.LittleEndian, v) }
	ranges := func(rs []splitter.Interval) {
		put(uint32(len(rs)))
		for _, r := range rs {
			put(r.Start)
			put(r.End)
		}
	}
	put(uint32(len(reads)))
	decisions := make([]splitter.Decision, 0, len(reads))
	for _, in := range reads {
		out, err := splitter.FindSimplexPA(in)
		if err != nil {
			return nil, nil, err
		}
		decisions = append(decisions, out)
		put(uint32(len(out.Clusters)))
		for _, c := range out.Clusters {
			put(c.Start)
			put(c.End)
			put(c.Argmax)
			put(c.Peak)
		}
		put(uint32(len(out.MoveSums)))
		put(out.MoveSums)
		ranges(out.RawCandidates)
		put(uint32(len(out.RawCandidates)))
		for _, r := range out.RawCandidates {
			found := false
			for _, p := range out.PoreCandidates {
				if p.Rejection == splitter.Eligible && p.Range == r {
					put(math.Float32bits(p.MeanQ))
					found = true
					break
				}
			}
			if !found {
				return nil, nil, fmt.Errorf("missing local quality trace")
			}
		}
		ranges(out.Candidates)
		put(uint32(len(out.AdapterChecks)))
		for _, c := range out.AdapterChecks {
			put(c.Pore.Start)
			put(c.Pore.End)
			put(c.Search.Start)
			put(c.Search.End)
			if c.Found {
				put(uint8(1))
			} else {
				put(uint8(0))
			}
			put(c.Match.Start)
			put(c.Match.End)
		}
		ranges(out.Spacers)
	}
	return data.Bytes(), decisions, nil
}

func Compare(executable, dir string, reads []splitter.Input) error {
	if err := os.Mkdir(dir, 0700); err != nil {
		return err
	}
	input, err := EncodeInput(reads)
	if err != nil {
		return err
	}
	expected, decisions, err := encodeExpected(reads)
	if err != nil {
		return err
	}
	for name, data := range map[string][]byte{"inputs.bin": input, "go-output.bin": expected} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			return err
		}
	}
	if err := saveJSON(filepath.Join(dir, "go-decisions.json"), decisions); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	actual, runErr := Command(dir, "native", exec.CommandContext(ctx, executable), input)
	comparison := map[string]any{"cases": len(reads), "input_sha256": hash(input), "go_output_sha256": hash(expected), "native_output_sha256": hash(actual), "exact": runErr == nil && bytes.Equal(actual, expected), "compared": "clusters, inclusive move sums, all eligible intervals, local mean float32 bits, top-ten intervals, adapter search/match intervals, selected spacers"}
	if err := saveJSON(filepath.Join(dir, "comparison.json"), comparison); err != nil {
		return err
	}
	if runErr != nil {
		return runErr
	}
	if !bytes.Equal(actual, expected) {
		i := 0
		for i < len(actual) && i < len(expected) && actual[i] == expected[i] {
			i++
		}
		return fmt.Errorf("native output differs at byte %d, native bytes %d, Go bytes %d", i, len(actual), len(expected))
	}
	return nil
}

// CheckHalfTable compares the unchanged native integer ordering for every bit pattern.
func CheckHalfTable(executable, dir string) error {
	if err := os.Mkdir(dir, 0700); err != nil {
		return err
	}
	actual, err := Command(dir, "half-table", exec.Command(executable, "--half-table"), nil)
	if err != nil {
		return err
	}
	if len(actual) != 2+65536*2 {
		return fmt.Errorf("invalid half table output size")
	}
	if binary.LittleEndian.Uint16(actual) != 0x419a {
		return fmt.Errorf("native PA threshold half differs")
	}
	for bits := 0; bits < 65536; bits++ {
		em := int16(bits & 0x7fff)
		if bits&0x8000 != 0 {
			em = -em
		}
		if int16(binary.LittleEndian.Uint16(actual[2+bits*2:])) != em {
			return fmt.Errorf("half ordering differs at %04x", bits)
		}
	}
	return saveJSON(filepath.Join(dir, "comparison.json"), map[string]any{"patterns": 65536, "threshold_bits": "419a", "exact": true, "output_sha256": hash(actual), "nonfinite": "ordering helper only; production inputs reject nonfinite samples"})
}

var _ io.Writer = (*limitedBuffer)(nil)
