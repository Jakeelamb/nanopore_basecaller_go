// Package koiweights captures and replays the fixed CPU SUP QKV half-weight oracle.
package koiweights

import (
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"time"
)

const SourceHash = "9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701"
const SourceRevision = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
const WeightHash = "b2260c11cadaf34d4bc14f0a9875829a51ecad7f4f6268604b2d3ae8d848f47d"
const ArchiveHash = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"
const HostLock = "/home/jake/Projects/nanopore_basecaller/results/stock/host.lock"
const FixtureEnv = "NANOPORE_KOI_QKV_MANIFEST"
const Elements = 3 * 8 * 64 * 512

//go:embed native/oracle.cpp
var adapter string

var producerFiles = []string{"verification/koiweights/oracle.go", "verification/koiweights/oracle_test.go", "verification/koiweights/native/oracle.cpp", "verification/koiweights/cmd/main.go"}
var consumerFiles = []string{"internal/model/koi_qkv.go", "internal/model/koi_qkv_test.go", "internal/model/koi_qkv_native_test.go"}
var ranges = [][2]int{{99, 106}, {445, 452}, {462, 464}}
var tiledShape = []int{3, 8, 4, 64, 16, 8}
var tiledStrides = []int{262144, 32768, 8192, 128, 8, 1}
var tiledTags = []string{"KOI_DIM_MAT_Q_K_V", "H", "D", "C", "d", "c"}

var ErrAdmissionBlocked = errors.New("admission_blocked")

type Block struct {
	Start, End   int
	SHA256, Text string
}
type Data struct {
	Path, DType, Endian string
	Shape, Strides      []int
	Tags                []string
	Bytes               int64
	SHA256              string
}
type Fixture struct {
	Name                               string
	Input, Interleaved, Tiled, Rounded Data
}
type Command struct {
	Phase                                      string
	Args                                       []string
	Started, Finished                          string
	Exit                                       int
	Stdout, Stderr, StdoutSHA256, StderrSHA256 string
}
type Manifest struct {
	ConsumerTools                                                                                            map[string]string
	Schema                                                                                                   int
	Kind, Status, Error, Qualification                                                                       string
	Started, Finished                                                                                        string
	SourceRevision, SourcePath, SourceSHA256, TensorManifest, TensorManifestSHA256, WeightPath, WeightSHA256 string
	TorchRoot, ArchivePath, ArchiveSHA256                                                                    string
	DevOptions                                                                                               map[string]int
	Blocks                                                                                                   []Block
	ProducerSources, ConsumerSources, ProducerInputs, Generated, Runtime                                     map[string]string
	AllHalf, AllRounded, GenuineFloat                                                                        Data
	Fixtures                                                                                                 []Fixture
	Commands                                                                                                 []Command
	Capture, CaptureSHA256                                                                                   string
	Controls                                                                                                 map[string]string
}
type tensorReceipt struct {
	Torch      string            `json:"torch_root"`
	Archive    string            `json:"archive"`
	ArchiveSHA string            `json:"archive_sha256"`
	Status     string            `json:"status"`
	Revision   string            `json:"dorado_commit"`
	Members    map[string]string `json:"extracted_archive_member_sha256"`
	Inputs     map[string]string `json:"input_sha256"`
	Runtime    map[string]string `json:"runtime_sha256"`
}
type Options struct{ Root, Source, TensorManifest, Weight, Out, Replay string }

func Hash(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func checkHash(path, want string) error {
	if len(want) != 64 {
		return fmt.Errorf("invalid hash pin %s", path)
	}
	got, e := Hash(path)
	if e != nil {
		return e
	}
	if got != want {
		return fmt.Errorf("hash pin mismatch %s: got %s want %s", path, got, want)
	}
	return nil
}
func pin(dst map[string]string, path string) error {
	h, e := Hash(path)
	if e == nil {
		dst[path] = h
	}
	return e
}
func pins(root string, files []string) (map[string]string, error) {
	out := map[string]string{}
	for _, f := range files {
		p := filepath.Join(root, f)
		if e := pin(out, p); e != nil {
			return nil, e
		}
	}
	return out, nil
}
func checkPins(pins map[string]string) error {
	keys := make([]string, 0, len(pins))
	for p := range pins {
		keys = append(keys, p)
	}
	sort.Strings(keys)
	for _, p := range keys {
		if e := checkHash(p, pins[p]); e != nil {
			return e
		}
	}
	return nil
}
func clock() string { return time.Now().UTC().Format(time.RFC3339Nano) }
func absolute(path string) (string, error) {
	if path == "" {
		return "", errors.New("empty path")
	}
	return filepath.Abs(path)
}

func admitOutput(root, out string) error {
	b, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil || !strings.HasPrefix(string(b), "module github.com/Jakeelamb/nanopore_basecaller_go\n") {
		return errors.New("run from the Go port module root; output was not created")
	}
	parent := filepath.Join(root, "artifacts", "koiweights")
	if filepath.Dir(out) != parent {
		return fmt.Errorf("output must be a fresh direct child of %s", parent)
	}
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return err
	}
	realArtifacts, err := filepath.EvalSymlinks(filepath.Join(root, "artifacts"))
	if err != nil {
		return err
	}
	if realArtifacts != filepath.Join(realRoot, "artifacts") {
		return errors.New("artifacts symlink escapes the Go module")
	}
	if err = os.MkdirAll(parent, 0755); err != nil {
		return err
	}
	realParent, err := filepath.EvalSymlinks(parent)
	if err != nil {
		return err
	}
	if realParent != filepath.Join(realRoot, "artifacts", "koiweights") {
		return errors.New("output parent symlink escapes the Go artifacts directory")
	}
	return os.Mkdir(out, 0755)
}
func saveManifest(path string, m *Manifest) error {
	b, e := json.MarshalIndent(m, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func readManifest(path string) (Manifest, error) {
	var m Manifest
	b, e := os.ReadFile(path)
	if e == nil {
		e = json.Unmarshal(b, &m)
	}
	return m, e
}
func cleanEnv() []string {
	var out []string
	for _, s := range os.Environ() {
		key := strings.SplitN(s, "=", 2)[0]
		if key != "LD_PRELOAD" && key != "LD_LIBRARY_PATH" && key != "LD_AUDIT" && key != "LIBRARY_PATH" && key != "CPATH" && key != "CPLUS_INCLUDE_PATH" && key != "C_INCLUDE_PATH" && key != "OMP_NUM_THREADS" && key != "MKL_NUM_THREADS" && key != FixtureEnv {
			out = append(out, s)
		}
	}
	return append(out, "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1")
}
func run(m *Manifest, root, out, label string, env []string, args ...string) error {
	c := Command{Phase: label, Args: args, Started: clock(), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
	stdout, e := os.OpenFile(c.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if e != nil {
		return e
	}
	defer stdout.Close()
	stderr, e := os.OpenFile(c.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if e != nil {
		return e
	}
	defer stderr.Close()
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = root
	cmd.Env = append(cleanEnv(), env...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	e = cmd.Run()
	c.Finished = clock()
	if cmd.ProcessState != nil {
		c.Exit = cmd.ProcessState.ExitCode()
	}
	c.StdoutSHA256, _ = Hash(c.Stdout)
	c.StderrSHA256, _ = Hash(c.Stderr)
	m.Commands = append(m.Commands, c)
	if e != nil {
		return fmt.Errorf("%s failed: %w; see %s", label, e, c.Stderr)
	}
	return nil
}

func Extract(path string) ([]Block, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	if digest(b) != SourceHash {
		return nil, fmt.Errorf("source pin mismatch %s", path)
	}
	lines := strings.Split(string(b), "\n")
	var out []Block
	for _, r := range ranges {
		txt := strings.Join(lines[r[0]-1:r[1]], "\n") + "\n"
		out = append(out, Block{r[0], r[1], digest([]byte(txt)), txt})
	}
	return out, nil
}
func generated(blocks []Block) string {
	body := adapter
	for i, name := range []string{"ROUNDING_BLOCK", "INTERLEAVE_BLOCK", "TILE_BLOCK"} {
		body = strings.Replace(body, "// "+name, blocks[i].Text, 1)
	}
	return body
}
func admit(path string) (*os.File, error) {
	f, e := os.OpenFile(path, os.O_RDONLY, 0)
	if e != nil {
		return nil, e
	}
	if e = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); e != nil {
		f.Close()
		if errors.Is(e, syscall.EWOULDBLOCK) || errors.Is(e, syscall.EAGAIN) {
			return nil, ErrAdmissionBlocked
		}
		return nil, e
	}
	return f, nil
}
func release(f *os.File) { _ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN); _ = f.Close() }

func distribution(path string) (tensorReceipt, error) {
	var t tensorReceipt
	b, e := os.ReadFile(path)
	if e == nil {
		e = json.Unmarshal(b, &t)
	}
	if e != nil {
		return t, e
	}
	if t.Status != "built" || t.ArchiveSHA != ArchiveHash || t.Revision != SourceRevision || len(t.Members) == 0 || len(t.Runtime) == 0 {
		return t, errors.New("unqualified CPU13 distribution receipt")
	}
	return t, nil
}
func distributionPins(t tensorReceipt) (map[string]string, error) {
	p := map[string]string{t.Archive: ArchiveHash}
	for member, h := range t.Members {
		if !strings.HasPrefix(member, "libtorch/") {
			return nil, fmt.Errorf("invalid archive member %s", member)
		}
		p[filepath.Join(filepath.Dir(t.Torch), filepath.FromSlash(member))] = h
	}
	for path, h := range t.Runtime {
		p[path] = h
	}
	compiler, e := exec.LookPath("g++")
	if e != nil {
		return nil, e
	}
	compiler, e = filepath.EvalSymlinks(compiler)
	if e != nil {
		return nil, e
	}
	h := t.Inputs[compiler]
	if h == "" {
		return nil, errors.New("compiler missing qualified input pin")
	}
	p[compiler] = h
	return p, checkPins(p)
}
func writeHalf(path string, n int, fn func(int) uint16) error {
	b := make([]byte, n*2)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(b[i*2:], fn(i))
	}
	return os.WriteFile(path, b, 0644)
}
func data(path, dtype string, shape, strides []int, tags []string) (Data, error) {
	h, e := Hash(path)
	if e != nil {
		return Data{}, e
	}
	s, e := os.Stat(path)
	if e != nil {
		return Data{}, e
	}
	return Data{path, dtype, "little", shape, strides, tags, s.Size(), h}, nil
}
func fixture(out, name string) (Fixture, error) {
	f := Fixture{Name: name}
	var e error
	f.Input, e = data(filepath.Join(out, "inputs", name+"-u16le.bin"), "binary16", []int{1536, 512}, []int{512, 1}, nil)
	if e != nil {
		return f, e
	}
	f.Interleaved, e = data(filepath.Join(out, "expected", name+"-interleaved-u16le.bin"), "binary16", []int{3, 8, 64, 512}, []int{262144, 32768, 512, 1}, nil)
	if e != nil {
		return f, e
	}
	f.Tiled, e = data(filepath.Join(out, "expected", name+"-tiled-u16le.bin"), "binary16", tiledShape, tiledStrides, tiledTags)
	if e != nil {
		return f, e
	}
	f.Rounded, e = data(filepath.Join(out, "expected", name+"-rounded-u16le.bin"), "binary16", tiledShape, tiledStrides, tiledTags)
	return f, e
}
func depPins(path string, t tensorReceipt, dst map[string]string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	s := strings.ReplaceAll(string(b), "\\\n", "")
	colon := strings.Index(s, ":")
	if colon < 0 {
		return errors.New("invalid compiler depfile")
	}
	for _, p := range strings.Fields(s[colon+1:]) {
		if strings.HasPrefix(p, t.Torch+string(filepath.Separator)) {
			member := "libtorch/" + filepath.ToSlash(strings.TrimPrefix(p, t.Torch+string(filepath.Separator)))
			want := t.Members[member]
			if want == "" {
				return fmt.Errorf("unqualified compile input %s", p)
			}
			if e = checkHash(p, want); e != nil {
				return e
			}
		}
		if e = pin(dst, p); e != nil {
			return e
		}
	}
	return nil
}
func runtimePins(path string, t tensorReceipt) (map[string]string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	out := map[string]string{}
	for _, line := range strings.Split(string(b), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		p := fields[5]
		if !strings.HasPrefix(p, "/") || !strings.Contains(filepath.Base(p), ".so") {
			continue
		}
		lower := strings.ToLower(filepath.Base(p))
		if strings.Contains(lower, "cuda") || strings.Contains(lower, "cudnn") || strings.Contains(lower, "koi") || strings.Contains(lower, "nvrtc") {
			return nil, fmt.Errorf("forbidden GPU runtime %s", p)
		}
		if want, ok := t.Runtime[p]; ok {
			if e = checkHash(p, want); e != nil {
				return nil, e
			}
		} else if strings.HasPrefix(p, t.Torch+"/") {
			return nil, fmt.Errorf("unqualified Torch runtime %s", p)
		}
		if e = pin(out, p); e != nil {
			return nil, e
		}
	}
	for _, name := range []string{"libtorch_cpu.so", "libc10.so"} {
		p := filepath.Join(t.Torch, "lib", name)
		if out[p] != t.Runtime[p] || out[p] == "" {
			return nil, fmt.Errorf("missing actual qualified runtime %s", p)
		}
	}
	return out, nil
}

func Capture(o Options) (ret error) {
	var e error
	for _, p := range []*string{&o.Root, &o.Source, &o.TensorManifest, &o.Weight, &o.Out} {
		*p, e = absolute(*p)
		if e != nil {
			return e
		}
	}
	if e = admitOutput(o.Root, o.Out); e != nil {
		return e
	}
	m := Manifest{Schema: 1, Kind: "capture", Status: "failed", Started: clock(), SourceRevision: SourceRevision, SourcePath: filepath.Join(o.Source, "dorado/nn/TxModules.cpp"), SourceSHA256: SourceHash, TensorManifest: o.TensorManifest, WeightPath: o.Weight, WeightSHA256: WeightHash, DevOptions: map[string]int{"C": 512, "H": 8, "D": 64, "remove_bits": 4, "remove_bits_qkv": 4}, Qualification: "CPU extracted source-block half layout and integer-rounding parity only; no GPU, quantization, graph, inference, biological or performance acceptance", ProducerInputs: map[string]string{}, Generated: map[string]string{}}
	defer func() {
		m.Finished = clock()
		if ret != nil {
			m.Error = ret.Error()
			if errors.Is(ret, ErrAdmissionBlocked) {
				m.Status = "admission_blocked"
			}
		}
		if se := saveManifest(filepath.Join(o.Out, "manifest.json"), &m); ret == nil {
			ret = se
		}
	}()
	m.Blocks, e = Extract(m.SourcePath)
	if e != nil {
		return e
	}
	if e = checkHash(o.Weight, WeightHash); e != nil {
		return e
	}
	t, e := distribution(o.TensorManifest)
	if e != nil {
		return e
	}
	m.TensorManifestSHA256, e = Hash(o.TensorManifest)
	if e != nil {
		return e
	}
	m.ProducerSources, e = pins(o.Root, producerFiles)
	if e != nil {
		return e
	}
	m.ConsumerSources, e = pins(o.Root, consumerFiles)
	if e != nil {
		return e
	}
	lock, e := admit(HostLock)
	if e != nil {
		return e
	}
	defer func() {
		if lock != nil {
			release(lock)
		}
	}()
	if e = run(&m, o.Root, o.Out, "source-revision", nil, "git", "-C", o.Source, "rev-parse", "HEAD"); e != nil {
		return e
	}
	revision, e := os.ReadFile(filepath.Join(o.Out, "source-revision.stdout"))
	if e != nil {
		return e
	}
	if strings.TrimSpace(string(revision)) != SourceRevision {
		return errors.New("Dorado source revision mismatch")
	}
	m.ProducerInputs, e = distributionPins(t)
	if e != nil {
		return e
	}
	m.TorchRoot = t.Torch
	m.ArchivePath = t.Archive
	m.ArchiveSHA256 = ArchiveHash
	compiler, _ := exec.LookPath("g++")
	compiler, _ = filepath.EvalSymlinks(compiler)
	for _, dir := range []string{"inputs", "expected"} {
		if e = os.Mkdir(filepath.Join(o.Out, dir), 0755); e != nil {
			return e
		}
	}
	if e = writeHalf(filepath.Join(o.Out, "inputs/all-half-u16le.bin"), 65536, func(i int) uint16 { return uint16(i) }); e != nil {
		return e
	}
	if e = writeHalf(filepath.Join(o.Out, "inputs/qkv-half-patterns-u16le.bin"), Elements, func(i int) uint16 { return uint16(i) }); e != nil {
		return e
	}
	if e = writeHalf(filepath.Join(o.Out, "inputs/qkv-axes-u16le.bin"), Elements, func(i int) uint16 {
		q, h, d, c := i/(8*64*512), i/(64*512)%8, i/512%64, i%512
		return uint16(q*21101 + h*2053 + d*521 + (c/8)*17 + c%8)
	}); e != nil {
		return e
	}
	cpp := filepath.Join(o.Out, "oracle.cpp")
	if e = os.WriteFile(cpp, []byte(generated(m.Blocks)), 0644); e != nil {
		return e
	}
	if e = pin(m.Generated, cpp); e != nil {
		return e
	}
	if e = run(&m, o.Root, o.Out, "compiler", nil, compiler, "--version"); e != nil {
		return e
	}
	exe, dep := filepath.Join(o.Out, "oracle"), filepath.Join(o.Out, "oracle.d")
	lib := filepath.Join(t.Torch, "lib")
	args := []string{compiler, "-std=c++17", "-O1", "-g0", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-I" + filepath.Join(t.Torch, "include"), "-I" + filepath.Join(t.Torch, "include/torch/csrc/api/include"), "-MD", "-MF", dep, cpp, "-o", exe, "-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10"}
	if e = run(&m, o.Root, o.Out, "compile", nil, args...); e != nil {
		return e
	}
	if e = depPins(dep, t, m.ProducerInputs); e != nil {
		return e
	}
	for _, p := range []string{dep, exe} {
		if e = pin(m.Generated, p); e != nil {
			return e
		}
	}
	if e = run(&m, o.Root, o.Out, "linkage", nil, "ldd", exe); e != nil {
		return e
	}
	if e = run(&m, o.Root, o.Out, "producer", nil, exe, o.Out, o.Weight); e != nil {
		return e
	}
	if e = checkHash(o.Weight, WeightHash); e != nil {
		return e
	}
	m.Runtime, e = runtimePins(filepath.Join(o.Out, "loaded-maps.txt"), t)
	if e != nil {
		return e
	}
	if e = pin(m.Generated, filepath.Join(o.Out, "loaded-maps.txt")); e != nil {
		return e
	}
	for _, name := range []string{"qkv-axes", "qkv-half-patterns", "genuine"} {
		f, er := fixture(o.Out, name)
		if er != nil {
			return er
		}
		m.Fixtures = append(m.Fixtures, f)
	}
	m.AllHalf, e = data(filepath.Join(o.Out, "inputs/all-half-u16le.bin"), "binary16", []int{65536}, []int{1}, nil)
	if e != nil {
		return e
	}
	m.AllRounded, e = data(filepath.Join(o.Out, "expected/all-half-rounded-u16le.bin"), "binary16", []int{65536}, []int{1}, nil)
	if e != nil {
		return e
	}
	m.GenuineFloat, e = data(filepath.Join(o.Out, "inputs/genuine-f32le.bin"), "float32", []int{1536, 512}, []int{512, 1}, nil)
	if e != nil {
		return e
	}
	if e = checkPins(m.ProducerSources); e != nil {
		return e
	}
	if e = checkPins(m.ProducerInputs); e != nil {
		return e
	}
	m.Status = "captured"
	if e = saveManifest(filepath.Join(o.Out, "manifest.json"), &m); e != nil {
		return e
	}
	// Release admission after the last native work; replay and Go checks do not use Torch.
	release(lock)
	lock = nil
	return captureConsumer(o, &m)
}

func validateData(d Data, dtype string, shape, strides []int, tags []string) error {
	if d.DType != dtype || d.Endian != "little" || !reflect.DeepEqual(d.Shape, shape) || !reflect.DeepEqual(d.Strides, strides) || !reflect.DeepEqual(d.Tags, tags) {
		return fmt.Errorf("invalid dtype/endian/shape/strides/tags for %s", d.Path)
	}
	bytes := int64(2)
	if dtype == "float32" {
		bytes = 4
	}
	for _, n := range shape {
		bytes *= int64(n)
	}
	if d.Bytes != bytes {
		return fmt.Errorf("invalid declared byte count %s", d.Path)
	}
	info, e := os.Stat(d.Path)
	if e != nil {
		return e
	}
	if info.Size() != bytes {
		return fmt.Errorf("invalid actual byte count %s", d.Path)
	}
	return checkHash(d.Path, d.SHA256)
}
func validatePayloads(m Manifest) error {
	if e := validateData(m.AllHalf, "binary16", []int{65536}, []int{1}, nil); e != nil {
		return e
	}
	if e := validateData(m.AllRounded, "binary16", []int{65536}, []int{1}, nil); e != nil {
		return e
	}
	if e := validateData(m.GenuineFloat, "float32", []int{1536, 512}, []int{512, 1}, nil); e != nil {
		return e
	}
	if len(m.Fixtures) != 3 {
		return errors.New("require exactly three full QKV fixtures")
	}
	for i, name := range []string{"qkv-axes", "qkv-half-patterns", "genuine"} {
		f := m.Fixtures[i]
		if f.Name != name {
			return errors.New("fixture identity mismatch")
		}
		for _, v := range []struct {
			d              Data
			shape, strides []int
			tags           []string
		}{{f.Input, []int{1536, 512}, []int{512, 1}, nil}, {f.Interleaved, []int{3, 8, 64, 512}, []int{262144, 32768, 512, 1}, nil}, {f.Tiled, tiledShape, tiledStrides, tiledTags}, {f.Rounded, tiledShape, tiledStrides, tiledTags}} {
			if e := validateData(v.d, "binary16", v.shape, v.strides, v.tags); e != nil {
				return e
			}
		}
	}
	return nil
}

// ValidateCapture verifies current producer bytes and every saved producer input.
// Consumer code is pinned by each fresh comparison receipt, independently of capture.
func ValidateCapture(path, root string) (Manifest, error) {
	m, e := readManifest(path)
	if e != nil {
		return m, e
	}
	if m.Schema != 1 || m.Kind != "capture" || (m.Status != "captured" && m.Status != "verified" && m.Status != "consumer_failed") || m.SourceRevision != SourceRevision || m.SourceSHA256 != SourceHash || m.WeightSHA256 != WeightHash || m.ArchiveSHA256 != ArchiveHash {
		return m, errors.New("invalid capture identity or status")
	}
	if !reflect.DeepEqual(m.DevOptions, map[string]int{"C": 512, "H": 8, "D": 64, "remove_bits": 4, "remove_bits_qkv": 4}) {
		return m, errors.New("invalid fixed dev options")
	}
	blocks, e := Extract(m.SourcePath)
	if e != nil {
		return m, e
	}
	if !reflect.DeepEqual(blocks, m.Blocks) {
		return m, errors.New("extracted source block mismatch")
	}
	if e = checkHash(m.WeightPath, WeightHash); e != nil {
		return m, e
	}
	if e = checkHash(m.TensorManifest, m.TensorManifestSHA256); e != nil {
		return m, e
	}
	t, e := distribution(m.TensorManifest)
	if e != nil {
		return m, e
	}
	if m.TorchRoot != t.Torch || m.ArchivePath != t.Archive {
		return m, errors.New("distribution path mismatch")
	}
	current, e := pins(root, producerFiles)
	if e != nil {
		return m, e
	}
	if !reflect.DeepEqual(current, m.ProducerSources) {
		return m, errors.New("current producer source pins differ from capture")
	}
	// All qualified members must appear, even if a malicious manifest drops a pin.
	for member, h := range t.Members {
		p := filepath.Join(filepath.Dir(t.Torch), filepath.FromSlash(member))
		if m.ProducerInputs[p] != h {
			return m, fmt.Errorf("missing qualified input pin %s", p)
		}
	}
	if m.ProducerInputs[t.Archive] != ArchiveHash {
		return m, errors.New("missing distribution archive pin")
	}
	compiler, e := exec.LookPath("g++")
	if e != nil {
		return m, e
	}
	compiler, e = filepath.EvalSymlinks(compiler)
	if e != nil {
		return m, e
	}
	if m.ProducerInputs[compiler] != t.Inputs[compiler] || m.ProducerInputs[compiler] == "" {
		return m, errors.New("compiler pin mismatch")
	}
	for p, h := range t.Runtime {
		if m.ProducerInputs[p] != h {
			return m, fmt.Errorf("missing qualified runtime pin %s", p)
		}
	}
	for _, p := range []map[string]string{m.ProducerInputs, m.Generated, m.Runtime} {
		if len(p) == 0 {
			return m, errors.New("empty producer pin map")
		}
		if e = checkPins(p); e != nil {
			return m, e
		}
	}
	dir := filepath.Dir(path)
	for _, name := range []string{"oracle.cpp", "oracle", "oracle.d", "loaded-maps.txt"} {
		if m.Generated[filepath.Join(dir, name)] == "" {
			return m, fmt.Errorf("missing generated pin %s", name)
		}
	}
	deps := map[string]string{}
	if e = depPins(filepath.Join(dir, "oracle.d"), t, deps); e != nil {
		return m, e
	}
	for p, h := range deps {
		if m.ProducerInputs[p] != h {
			return m, fmt.Errorf("dependency pin mismatch %s", p)
		}
	}
	observed, e := runtimePins(filepath.Join(dir, "loaded-maps.txt"), t)
	if e != nil {
		return m, e
	}
	if !reflect.DeepEqual(observed, m.Runtime) {
		return m, errors.New("loaded runtime map mismatch")
	}
	cpp := filepath.Join(dir, "oracle.cpp")
	if m.Generated[cpp] != digest([]byte(generated(blocks))) {
		return m, errors.New("generated source pin mismatch")
	}
	if e = checkHash(cpp, digest([]byte(generated(blocks)))); e != nil {
		return m, e
	}
	if len(m.Commands) < 4 {
		return m, errors.New("missing native command receipts")
	}
	producer := false
	for _, c := range m.Commands {
		if c.Exit != 0 && c.Phase != "consumer" {
			return m, errors.New("failed command in capture")
		}
		if e = checkHash(c.Stdout, c.StdoutSHA256); e != nil {
			return m, e
		}
		if e = checkHash(c.Stderr, c.StderrSHA256); e != nil {
			return m, e
		}
		if len(c.Args) > 0 && c.Args[0] == filepath.Join(dir, "oracle") {
			producer = true
		}
	}
	if !producer {
		return m, errors.New("missing completed native producer")
	}
	if e = validatePayloads(m); e != nil {
		return m, e
	}
	return m, nil
}

func ReadHalf(d Data) ([]uint16, error) {
	if d.DType != "binary16" || d.Endian != "little" {
		return nil, errors.New("half reader requires little-endian binary16")
	}
	if e := checkHash(d.Path, d.SHA256); e != nil {
		return nil, e
	}
	b, e := os.ReadFile(d.Path)
	if e != nil {
		return nil, e
	}
	if len(b)%2 != 0 || int64(len(b)) != d.Bytes {
		return nil, errors.New("half reader byte count")
	}
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[i*2:])
	}
	return out, nil
}

func consumer(root, out, path string, m *Manifest) error {
	var e error
	m.ConsumerSources, e = pins(root, consumerFiles)
	if e != nil {
		return e
	}
	goexe, e := exec.LookPath("go")
	if e != nil {
		return e
	}
	goexe, e = filepath.EvalSymlinks(goexe)
	if e != nil {
		return e
	}
	m.ConsumerTools = map[string]string{}
	if e = pin(m.ConsumerTools, goexe); e != nil {
		return e
	}
	if e = run(m, root, out, "consumer", []string{FixtureEnv + "=" + path}, goexe, "test", "-json", "-count=1", "./internal/model", "-run", "^TestSUPQKVF16"); e != nil {
		return e
	}
	log, e := os.ReadFile(filepath.Join(out, "consumer.stdout"))
	if e != nil {
		return e
	}
	ran, passed := false, false
	for _, line := range strings.Split(string(log), "\n") {
		if line == "" {
			continue
		}
		var event struct{ Action, Test string }
		if e = json.Unmarshal([]byte(line), &event); e != nil {
			return e
		}
		if event.Test == "TestSUPQKVF16Native" {
			switch event.Action {
			case "run":
				ran = true
			case "pass":
				passed = true
			case "skip":
				return errors.New("native consumer skipped")
			}
		}
	}
	if !ran || !passed {
		return errors.New("required native consumer did not pass")
	}
	return checkPins(m.ConsumerSources)
}
func captureConsumer(o Options, m *Manifest) error {
	path := filepath.Join(o.Out, "manifest.json")
	if e := consumer(o.Root, o.Out, path, m); e != nil {
		m.Status = "consumer_failed"
		return e
	}
	m.Controls = map[string]string{}
	if e := controls(o.Root, o.Out, path, m.Controls); e != nil {
		m.Status = "consumer_failed"
		return e
	}
	m.Status = "verified"
	return nil
}

// Replay never builds or runs the native producer. It creates a new consumer receipt.
func Replay(o Options) (ret error) {
	var e error
	for _, p := range []*string{&o.Root, &o.Replay, &o.Out} {
		*p, e = absolute(*p)
		if e != nil {
			return e
		}
	}
	if e = admitOutput(o.Root, o.Out); e != nil {
		return e
	}
	m := Manifest{Schema: 1, Kind: "replay", Status: "failed", Started: clock(), Capture: o.Replay, ProducerInputs: map[string]string{}, Controls: map[string]string{}}
	defer func() {
		m.Finished = clock()
		if ret != nil {
			m.Error = ret.Error()
		}
		if se := saveManifest(filepath.Join(o.Out, "manifest.json"), &m); ret == nil {
			ret = se
		}
	}()
	m.CaptureSHA256, e = Hash(o.Replay)
	if e != nil {
		return e
	}
	if _, e = ValidateCapture(o.Replay, o.Root); e != nil {
		return e
	}
	if e = consumer(o.Root, o.Out, o.Replay, &m); e != nil {
		return e
	}
	if e = controls(o.Root, o.Out, o.Replay, m.Controls); e != nil {
		return e
	}
	if e = checkHash(o.Replay, m.CaptureSHA256); e != nil {
		return e
	}
	m.Status = "verified"
	return nil
}

func controls(root, out, path string, results map[string]string) error {
	base, e := readManifest(path)
	if e != nil {
		return e
	}
	if e = os.Mkdir(filepath.Join(out, "controls"), 0755); e != nil {
		return e
	}
	for _, name := range []string{"dtype", "endian", "shape", "foreign-shape", "byte-count", "output-dtype", "output-shape", "output-stride", "output-tags"} {
		b, _ := json.Marshal(base)
		var m Manifest
		_ = json.Unmarshal(b, &m)
		switch name {
		case "dtype":
			m.Fixtures[0].Input.DType = "uint16"
		case "endian":
			m.Fixtures[0].Input.Endian = "big"
		case "shape":
			m.Fixtures[0].Input.Shape = []int{3, 8, 64, 512}
		case "foreign-shape":
			m.Fixtures[0].Input.Shape = []int{1152, 384}
		case "byte-count":
			m.Fixtures[0].Input.Bytes--
		case "output-dtype":
			m.Fixtures[0].Rounded.DType = "float32"
		case "output-shape":
			m.Fixtures[0].Rounded.Shape = []int{3, 8, 4, 16, 64, 8}
		case "output-stride":
			m.Fixtures[0].Rounded.Strides[0]++
		case "output-tags":
			m.Fixtures[0].Rounded.Tags[0] = "Q"
		}
		if e = validatePayloads(m); e == nil {
			return fmt.Errorf("negative control accepted %s", name)
		}
		results[name] = e.Error()
	}
	for _, name := range []string{"input-byte", "output-byte", "source-byte", "weight-hash", "library-pin", "distribution-pin"} {
		var source, want string
		switch name {
		case "input-byte":
			source, want = base.Fixtures[0].Input.Path, base.Fixtures[0].Input.SHA256
		case "output-byte":
			source, want = base.Fixtures[0].Rounded.Path, base.Fixtures[0].Rounded.SHA256
		case "source-byte":
			source, want = base.SourcePath, SourceHash
		case "weight-hash":
			source, want = base.WeightPath, strings.Repeat("0", 64)
		case "library-pin":
			source, want = filepath.Join(base.TorchRoot, "lib/libc10.so"), strings.Repeat("0", 64)
		case "distribution-pin":
			var t tensorReceipt
			t.Status = "built"
			t.ArchiveSHA = strings.Repeat("0", 64)
			b, _ := json.Marshal(t)
			p := filepath.Join(out, "controls", name+".json")
			if e = os.WriteFile(p, b, 0644); e != nil {
				return e
			}
			_, e = distribution(p)
			if e == nil {
				return errors.New("wrong distribution pin accepted")
			}
			results[name] = e.Error()
			continue
		}
		if strings.HasSuffix(name, "-byte") {
			b, er := os.ReadFile(source)
			if er != nil {
				return er
			}
			b[0] ^= 1
			p := filepath.Join(out, "controls", name+".bin")
			if e = os.WriteFile(p, b, 0644); e != nil {
				return e
			}
			source = p
		}
		if name == "source-byte" {
			_, e = Extract(source)
		} else {
			e = checkHash(source, want)
		}
		if e == nil {
			return fmt.Errorf("negative control accepted %s", name)
		}
		results[name] = e.Error()
	}
	// This isolated lock exercises the same nonblocking admission primitive without stock writes.
	p := filepath.Join(out, "controls", "held.lock")
	f, e := os.OpenFile(p, os.O_CREATE|os.O_EXCL|os.O_RDONLY, 0644)
	if e != nil {
		return e
	}
	f.Close()
	holder, e := admit(p)
	if e != nil {
		return e
	}
	_, e = admit(p)
	release(holder)
	if !errors.Is(e, ErrAdmissionBlocked) {
		return errors.New("held lock was not rejected")
	}
	results["held-lock-primitive"] = e.Error()
	b, e := json.MarshalIndent(results, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(filepath.Join(out, "controls.json"), append(b, '\n'), 0644)
}

// AdmissionControl holds the original lock and proves capture stops before commands.
func AdmissionControl(o Options) error {
	root, e := absolute(o.Root)
	if e != nil {
		return e
	}
	o.Root = root
	lock, e := admit(HostLock)
	if e != nil {
		return e
	}
	defer release(lock)
	e = Capture(o)
	if !errors.Is(e, ErrAdmissionBlocked) {
		return fmt.Errorf("capture did not report held stock lock: %v", e)
	}
	m, e := readManifest(filepath.Join(o.Out, "manifest.json"))
	if e != nil {
		return e
	}
	if m.Status != "admission_blocked" || len(m.Commands) != 0 || len(m.Generated) != 0 {
		return errors.New("held-lock capture ran work")
	}
	return nil
}

// Mutants runs altered Go sources through overlays. Production files stay unchanged.
func Mutants(o Options) (ret error) {
	var e error
	for _, p := range []*string{&o.Root, &o.Out} {
		*p, e = absolute(*p)
		if e != nil {
			return e
		}
	}
	if e = admitOutput(o.Root, o.Out); e != nil {
		return e
	}
	m := Manifest{Schema: 1, Kind: "mutants", Status: "failed", Started: clock(), Controls: map[string]string{}, Generated: map[string]string{}}
	defer func() {
		m.Finished = clock()
		if ret != nil {
			m.Error = ret.Error()
		}
		if se := saveManifest(filepath.Join(o.Out, "manifest.json"), &m); ret == nil {
			ret = se
		}
	}()
	source := filepath.Join(o.Root, "internal/model/koi_qkv.go")
	b, e := os.ReadFile(source)
	if e != nil {
		return e
	}
	m.ConsumerSources, e = pins(o.Root, consumerFiles)
	if e != nil {
		return e
	}
	const rounding = "(out.interleavedHalf[dst+c] + 8) & 0xfff0"
	mutants := []struct{ name, from, to string }{
		{"omitted-qk-interleave", "q < 2", "q < 0"},
		{"v-interleave", "q < 2", "q < 3"},
		{"wrong-tile-order", "*16+d%16)*8", "*16+d/16)*8"},
		{"early-rounding", "copy(out.interleavedHalf[dst:dst+512], rawHalf[src:src+512])", "copy(out.interleavedHalf[dst:dst+512], rawHalf[src:src+512]); for i := dst; i < dst+512; i++ { out.interleavedHalf[i] = (out.interleavedHalf[i]+8)&0xfff0 }"},
		{"saturation", rounding, "(out.interleavedHalf[dst+c] + 8) & 0xfff0; if out.interleavedHalf[dst+c]&0x7fff >= 0x7bf8 && out.interleavedHalf[dst+c]&0x7fff < 0x7c00 { out.tiledRoundedHalf[tiled] = (out.interleavedHalf[dst+c]&0x8000)|0x7bf0 }"},
		{"round-to-even", rounding, "uint16((uint32(out.interleavedHalf[dst+c])+7+((uint32(out.interleavedHalf[dst+c])>>4)&1))&0xfff0)"},
	}
	goexe, e := exec.LookPath("go")
	if e != nil {
		return e
	}
	for _, mutant := range mutants {
		if strings.Count(string(b), mutant.from) != 1 {
			return fmt.Errorf("mutant %s target no longer unique", mutant.name)
		}
		altered := strings.Replace(string(b), mutant.from, mutant.to, 1)
		file := filepath.Join(o.Out, mutant.name+".go.txt")
		if e = os.WriteFile(file, []byte(altered), 0644); e != nil {
			return e
		}
		overlay, _ := json.Marshal(struct{ Replace map[string]string }{map[string]string{source: file}})
		config := filepath.Join(o.Out, mutant.name+".json")
		if e = os.WriteFile(config, overlay, 0644); e != nil {
			return e
		}
		for _, p := range []string{file, config} {
			if e = pin(m.Generated, p); e != nil {
				return e
			}
		}
		er := run(&m, o.Root, o.Out, mutant.name, nil, goexe, "test", "-json", "-count=1", "-overlay", config, "./internal/model", "-run", "^TestSUPQKVF16")
		if er == nil {
			return fmt.Errorf("mutant %s survived", mutant.name)
		}
		log, e := os.ReadFile(filepath.Join(o.Out, mutant.name+".stdout"))
		if e != nil {
			return e
		}
		failedTest := false
		for _, line := range strings.Split(string(log), "\n") {
			var event struct{ Action, Test string }
			if json.Unmarshal([]byte(line), &event) == nil && event.Action == "fail" && strings.HasPrefix(event.Test, "TestSUPQKVF16") {
				failedTest = true
			}
		}
		if !failedTest {
			return fmt.Errorf("mutant %s failed without a behavioral test failure", mutant.name)
		}
		m.Controls[mutant.name] = "rejected by behavioral Go test"
	}
	if e = checkPins(m.ConsumerSources); e != nil {
		return e
	}
	m.Status = "verified"
	return nil
}
