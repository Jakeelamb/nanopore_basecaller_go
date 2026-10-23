// Package gpuprofile freezes dispatch inputs before weight caches exist.
package gpuprofile

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
const HAC = "dna_r10.4.1_e8.2_400bps_hac@v6.0.0"
const SUP = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
const HACConfig = "e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f"
const SUPConfig = "ad82d7c369deddc0d1e69367302b8e002c306a9770ae4304a3cbb70715fcce49"
const Source13 = "source-cuda13"
const DeviceUUID = "807d7ae76a99c79512c223fc801d38e7"

type Environment struct {
	Present bool   `json:"present"`
	Value   string `json:"value"`
}

type DevOptions struct {
	KoiUseI8   bool `json:"koi_use_i8"`
	KoiUseF8   bool `json:"koi_use_f8"`
	UseUSwiGLU bool `json:"use_uswiglu"`
	RemoveBits int  `json:"remove_bits"`
}

func DefaultDevOptions() DevOptions { return DevOptions{true, true, false, 4} }

type ModelPin struct {
	Name         string `json:"name"`
	ConfigSHA256 string `json:"config_sha256"`
	WeightCount  int    `json:"weight_count"`
}

type Target struct {
	Schema             string      `json:"schema"`
	SourceCommit       string      `json:"source_commit"`
	RuntimeFamily      string      `json:"runtime_family"`
	DeviceUUID         string      `json:"device_uuid"`
	CUDAVisibleDevices Environment `json:"cuda_visible_devices"`
	Models             [2]ModelPin `json:"models"`
	DevOptions         DevOptions  `json:"dev_options"`
	BatchPolicy        string      `json:"batch_policy"`
	Kit                string      `json:"kit"`
}

func DecodeTarget(r io.Reader) (Target, error) {
	var t Target
	raw, err := io.ReadAll(r)
	if err != nil {
		return t, err
	}
	if err = decodeClosed(raw, &t); err != nil {
		return t, err
	}
	return t, t.Validate()
}

func (t Target) Validate() error {
	if t.Schema != "nanopore-gpu-target/v1" || t.SourceCommit != SourceCommit {
		return fmt.Errorf("unsupported schema or source revision")
	}
	if t.RuntimeFamily != Source13 {
		return fmt.Errorf("runtime family requires separate qualification")
	}
	if t.DeviceUUID != DeviceUUID || t.CUDAVisibleDevices != (Environment{}) {
		return fmt.Errorf("device or CUDA_VISIBLE_DEVICES differs from target")
	}
	if t.Models != ([2]ModelPin{{HAC, HACConfig, 50}, {SUP, SUPConfig, 139}}) {
		return fmt.Errorf("unsupported model/config/weight-count pins")
	}
	if t.DevOptions != DefaultDevOptions() {
		return fmt.Errorf("default precision options must remain enabled and unchanged")
	}
	if t.BatchPolicy != "explicit-positive-multiple" || t.Kit != "SQK-LSK114" {
		return fmt.Errorf("unsupported batch policy or kit")
	}
	return nil
}

// Nil query results mean unobserved, not capability false.
type Observation struct {
	RuntimeFamily      string      `json:"runtime_family"`
	DeviceUUID         string      `json:"device_uuid"`
	CUDAVisibleDevices Environment `json:"cuda_visible_devices"`
	ComputeMajor       int         `json:"compute_major"`
	ComputeMinor       int         `json:"compute_minor"`
	SMCount            int         `json:"sm_count"`
	F16Status          *int        `json:"f16_status"`
	E4M3Status         *int        `json:"e4m3_status"`
}

// BackendIdentity has only value fields and exposes a copy of its observations.
type BackendIdentity struct {
	target                       Target
	major, minor, sms, f16, e4m3 int
}

func Bind(t Target, o Observation) (BackendIdentity, error) {
	if err := t.Validate(); err != nil {
		return BackendIdentity{}, err
	}
	if o.RuntimeFamily != t.RuntimeFamily || o.DeviceUUID != t.DeviceUUID || o.CUDAVisibleDevices != t.CUDAVisibleDevices {
		return BackendIdentity{}, fmt.Errorf("runtime identity differs from frozen target")
	}
	if o.F16Status == nil || o.E4M3Status == nil || o.ComputeMajor <= 0 || o.ComputeMinor < 0 || o.SMCount <= 0 {
		return BackendIdentity{}, fmt.Errorf("capability observation incomplete")
	}
	if *o.F16Status < 0 || *o.F16Status > 3 || *o.E4M3Status < 0 || *o.E4M3Status > 3 {
		return BackendIdentity{}, fmt.Errorf("invalid Koi query status")
	}
	// Only NOT_SUPPORTED is an observed false capability. Other errors are failures.
	if *o.F16Status > 1 || *o.E4M3Status > 1 {
		return BackendIdentity{}, fmt.Errorf("Koi capability query failed")
	}
	return BackendIdentity{t, o.ComputeMajor, o.ComputeMinor, o.SMCount, *o.F16Status, *o.E4M3Status}, nil
}

func (b BackendIdentity) Target() Target { return b.target }

type Shape struct {
	Kind                                               string
	C, K, Heads, WindowUpper, WindowLower, FeedForward int
}

func IsSUP(s Shape) bool {
	return s.Kind == "tx" && s.C == 512 && s.Heads == 8 && s.WindowUpper == 127 && s.WindowLower == 128 && s.FeedForward == 2048
}
func CanRunFLSTM(major, sms int) bool { return major >= 8 && sms >= 9 }
func VariableChunksSupported(s Shape, major, sms int, cudaBuild, orin bool, deviceCount int) bool {
	if !cudaBuild || orin || deviceCount == 0 {
		return false
	}
	switch s.Kind {
	case "lstm":
		return major >= 8 && s.C > 128 && s.C <= 1024 && s.C%128 == 0
	case "flstm":
		return CanRunFLSTM(major, sms) && s.C == 1024 && s.K == 128
	case "tx":
		return major >= 8 && IsSUP(s)
	}
	return false
}

type Request struct {
	Model        string     `json:"model"`
	ConfigSHA256 string     `json:"config_sha256"`
	Batch        int        `json:"batch"`
	FixedChunks  bool       `json:"fixed_chunks"`
	DisableTrim  bool       `json:"disable_trim"`
	DisableSplit bool       `json:"disable_split"`
	Output       string     `json:"output"`
	Kit          string     `json:"kit"`
	DevOptions   DevOptions `json:"dev_options"`
}

type DispatchPlan struct {
	backend                      BackendIdentity
	request                      Request
	hacFused, supTiled, fp8, vcs bool
}

type Selection struct {
	HACFused       bool `json:"hac_fused"`
	SUPTiled       bool `json:"sup_tiled"`
	FP8            bool `json:"fp8"`
	Int8           bool `json:"int8"`
	VariableChunks bool `json:"variable_chunks"`
}

func Prepare(b BackendIdentity, r Request) (DispatchPlan, error) {
	if err := CheckRequest(b.target, r); err != nil {
		return DispatchPlan{}, err
	}
	var s Shape
	switch r.Model {
	case HAC:
		s = Shape{Kind: "flstm", C: 1024, K: 128}
	case SUP:
		s = Shape{Kind: "tx", C: 512, Heads: 8, WindowUpper: 127, WindowLower: 128, FeedForward: 2048}
	}
	p := DispatchPlan{backend: b, request: r}
	p.hacFused = s.Kind == "flstm" && CanRunFLSTM(b.major, b.sms)
	p.supTiled = IsSUP(s) && b.f16 == 0
	if IsSUP(s) && !p.supTiled {
		return DispatchPlan{}, fmt.Errorf("generic SUP requires additional cards")
	}
	p.fp8 = p.supTiled && b.e4m3 == 0
	p.vcs = !r.FixedChunks
	if p.vcs && !VariableChunksSupported(s, b.major, b.sms, true, false, 1) {
		return DispatchPlan{}, fmt.Errorf("default VCS unsupported; do not silently disable")
	}
	return p, nil
}

func (p DispatchPlan) Backend() BackendIdentity { return p.backend }
func (p DispatchPlan) Request() Request         { return p.request }
func (p DispatchPlan) Selection() Selection {
	return Selection{p.hacFused, p.supTiled, p.fp8, p.supTiled, p.vcs}
}

// CheckRequest validates CLI inputs without claiming a runtime observation.
func CheckRequest(t Target, r Request) error {
	if err := t.Validate(); err != nil {
		return err
	}
	if r.DevOptions != t.DevOptions || r.Kit != t.Kit {
		return fmt.Errorf("unsupported precision options or kit")
	}
	if r.Output != "fastq" && r.Output != "bam" {
		return fmt.Errorf("only FASTQ and unaligned BAM supported")
	}
	granularity := 64
	switch r.Model {
	case HAC:
		if r.ConfigSHA256 != HACConfig {
			return fmt.Errorf("HAC config hash differs")
		}
	case SUP:
		if r.ConfigSHA256 != SUPConfig {
			return fmt.Errorf("SUP config hash differs")
		}
		granularity = 32
	default:
		return fmt.Errorf("unsupported model")
	}
	if r.Batch <= 0 || r.Batch%granularity != 0 {
		return fmt.Errorf("batch must be a positive multiple of %d", granularity)
	}
	return nil
}

const BasicSHA = "15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae"
const BuildSHA = "f41faaed7ba863f3d16c5a5f08cb373bf588794ec73fe25d1620359863b9d0dd"
const BackendSHA = "d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7"
const KoiHeaderSHA = "8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88"
const KoiLibrarySHA = "17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f"

type ProbeOptions struct {
	Target                               Target
	Basic, Build, KoiRoot, CUDARoot, Out string
	ModelRoot                            string
	Admitted                             bool
	Invocation                           []string
}

type QualifiedBuild struct {
	ABI            int               `json:"abi_version"`
	Status         string            `json:"status"`
	Family         string            `json:"cuda_family"`
	Output         string            `json:"output"`
	OutputSHA      string            `json:"output_sha256"`
	Inputs         map[string]string `json:"input_sha256"`
	Runtime        map[string]string `json:"runtime_sha256"`
	ToolkitVersion int               `json:"cuda_toolkit_header_version"`
}

type NativePredicates struct {
	HACFused   bool `json:"hac_fused"`
	SUPTiled   bool `json:"sup_tiled"`
	FP8        bool `json:"fp8"`
	Int8       bool `json:"int8"`
	HACVCS     bool `json:"hac_vcs"`
	SUPVCS     bool `json:"sup_vcs"`
	DefaultVCS bool `json:"default_vcs"`
	RemoveBits int  `json:"remove_bits"`
}

type RawProbe struct {
	DeviceUUID  string           `json:"device_uuid"`
	Ordinal     int              `json:"device_ordinal"`
	DeviceCount int              `json:"visible_device_count"`
	Major       int              `json:"compute_major"`
	Minor       int              `json:"compute_minor"`
	SMs         int              `json:"sm_count"`
	Driver      int              `json:"driver_version"`
	Runtime     int              `json:"runtime_version"`
	CVDPresent  bool             `json:"cuda_visible_devices_present"`
	F16         int              `json:"f16_status"`
	E4M3        int              `json:"e4m3_status"`
	Predicates  NativePredicates `json:"predicates"`
}

type ProbeReceipt struct {
	Compile     ProbeCompile            `json:"compile"`
	Schema      string                  `json:"schema"`
	Status      string                  `json:"status"`
	Error       string                  `json:"error"`
	Started     string                  `json:"started_utc"`
	Finished    string                  `json:"finished_utc"`
	Invocation  []string                `json:"invocation"`
	Target      Target                  `json:"target"`
	Environment map[string]Environment  `json:"environment"`
	Inputs      map[string]string       `json:"input_sha256"`
	Runtime     map[string]string       `json:"runtime_sha256"`
	Commands    []runner.CommandReceipt `json:"commands"`
	Raw         *RawProbe               `json:"raw"`
	Observation *Observation            `json:"observation"`
	Matrix      []Request               `json:"request_matrix"`
}

const ProfileSchema = "nanopore-gpu-profile-capture/v2"

var consumerSources = []string{
	"go.mod", "provenance/gpu-target.json", "docs/gpu-target.md",
	"verification/gpuprofile/profile.go", "verification/gpuprofile/profile_test.go",
	"verification/gpuprofile/cmd/main.go", "verification/gpuprofile/native/probe.cpp",
	"verification/tensor/runner/command.go", "verification/tensor/runner/admission.go",
}

type ProbeCompile struct {
	Compiler  string `json:"compiler"`
	KoiRoot   string `json:"koi_root"`
	CUDARoot  string `json:"cuda_root"`
	Directory string `json:"directory"`
}

func (c ProbeCompile) binary() string     { return filepath.Join(c.Directory, "gpuprofile-probe") }
func (c ProbeCompile) depfile() string    { return filepath.Join(c.Directory, "probe.d") }
func (c ProbeCompile) header() string     { return filepath.Join(c.Directory, "predicates.h") }
func (c ProbeCompile) koiHeader() string  { return filepath.Join(c.KoiRoot, "include/koi.h") }
func (c ProbeCompile) koiLibrary() string { return filepath.Join(c.KoiRoot, "lib/libkoi.a") }

func (c ProbeCompile) argv() []string {
	return []string{c.Compiler, "-std=c++20", "-O1", "-pthread", "-DDORADO_CUDA_BUILD=1", "-DDORADO_ORIN=0", "-I" + c.Directory, "-I" + filepath.Join(c.KoiRoot, "include"), "-I" + filepath.Join(c.CUDARoot, "include"), "-MD", "-MF", c.depfile(), "verification/gpuprofile/native/probe.cpp", c.koiLibrary(), "-L" + filepath.Join(c.CUDARoot, "lib64"), "-Wl,-rpath," + filepath.Join(c.CUDARoot, "lib64"), "-lcudart", "-lcublas", "-lcublasLt", "-ldl", "-o", c.binary()}
}

func readDependencies(path, binary string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	target, text, ok := strings.Cut(strings.ReplaceAll(string(raw), "\\\n", ""), ":")
	if !ok || strings.TrimSpace(target) != binary || strings.ContainsAny(text, "\\:$#") {
		return nil, fmt.Errorf("invalid probe dependency target or syntax")
	}
	paths := strings.Fields(text)
	if len(paths) == 0 {
		return nil, fmt.Errorf("empty probe dependency closure")
	}
	for _, p := range paths {
		if filepath.Clean(p) != p {
			return nil, fmt.Errorf("unclean probe dependency path")
		}
	}
	return paths, nil
}

// validateCompileClosure is shared by capture and import, before qualification in import.
func validateCompileClosure(m ProbeReceipt) error {
	c := m.Compile
	for _, p := range []string{c.Compiler, c.KoiRoot, c.CUDARoot, c.Directory} {
		if !filepath.IsAbs(p) || filepath.Clean(p) != p || strings.ContainsAny(p, " \t\r\n\\:$#") {
			return fmt.Errorf("invalid typed compile path")
		}
	}
	if len(m.Commands) != 3 || !reflect.DeepEqual(m.Commands[0].Args, []string{c.Compiler, "--version"}) || !reflect.DeepEqual(m.Commands[1].Args, c.argv()) || !reflect.DeepEqual(m.Commands[2].Args, []string{c.binary(), filepath.Join(c.Directory, "probe.maps")}) {
		return fmt.Errorf("compile argv does not bind pinned probe")
	}
	mandatory := append([]string{}, consumerSources...)
	mandatory = append(mandatory, c.Compiler, c.binary(), c.header(), c.depfile(), c.koiHeader(), c.koiLibrary(), filepath.Join(c.CUDARoot, "include/cuda.h"), filepath.Join(c.Directory, "probe.maps"))
	for _, p := range mandatory {
		h := m.Inputs[p]
		if len(h) != 64 {
			return fmt.Errorf("missing mandatory consumer/probe pin %s", p)
		}
		if _, err := hex.DecodeString(h); err != nil {
			return fmt.Errorf("invalid mandatory input pin %s", p)
		}
	}
	if m.Inputs[c.koiHeader()] != KoiHeaderSHA || m.Inputs[c.koiLibrary()] != KoiLibrarySHA {
		return fmt.Errorf("typed Koi inputs differ from qualified pins")
	}
	if err := checkPin(c.depfile(), m.Inputs[c.depfile()]); err != nil {
		return err
	}
	paths, err := readDependencies(c.depfile(), c.binary())
	if err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, p := range paths {
		seen[p] = true
		if len(m.Inputs[p]) != 64 {
			return fmt.Errorf("missing compiler dependency pin %s", p)
		}
		if _, err := hex.DecodeString(m.Inputs[p]); err != nil {
			return fmt.Errorf("invalid dependency pin %s", p)
		}
	}
	for _, p := range []string{"verification/gpuprofile/native/probe.cpp", c.header(), c.koiHeader(), filepath.Join(c.CUDARoot, "include/cuda_runtime_api.h")} {
		if !seen[p] {
			return fmt.Errorf("depfile omits required probe dependency %s", p)
		}
	}
	return nil
}

func checkPin(path, want string) error {
	if len(want) != 64 {
		return fmt.Errorf("missing SHA256 pin for %s", path)
	}
	if _, err := hex.DecodeString(want); err != nil {
		return err
	}
	got, err := runner.SHA(path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("pin mismatch: %s", path)
	}
	return nil
}

func qualify(basicPath, buildPath string) (QualifiedBuild, error) {
	var b QualifiedBuild
	if err := checkPin(basicPath, BasicSHA); err != nil {
		return b, err
	}
	if err := checkPin(buildPath, BuildSHA); err != nil {
		return b, err
	}
	raw, err := os.ReadFile(buildPath)
	if err != nil {
		return b, err
	}
	if err = json.Unmarshal(raw, &b); err != nil {
		return b, err
	}
	var basic struct {
		ABI     int               `json:"abi_version"`
		Status  string            `json:"status"`
		Inputs  map[string]string `json:"input_sha256"`
		Runtime map[string]string `json:"runtime_sha256"`
	}
	raw, err = os.ReadFile(basicPath)
	if err != nil {
		return b, err
	}
	if err = json.Unmarshal(raw, &basic); err != nil {
		return b, err
	}
	if b.ABI != 4 || b.Status != "built" || b.Family != "13.0" || b.OutputSHA != BackendSHA || basic.ABI != 4 || basic.Status != "source_differential_passed" || basic.Inputs[b.Output] != BackendSHA {
		return b, fmt.Errorf("unqualified ABI4 runtime")
	}
	matched := false
	for path, hash := range basic.Inputs {
		if filepath.Base(path) == "manifest.json" && hash == BuildSHA {
			matched = true
		}
	}
	if !matched || len(b.Runtime) == 0 {
		return b, fmt.Errorf("qualification chain incomplete")
	}
	if err = checkPin(b.Output, BackendSHA); err != nil {
		return b, err
	}
	for path, hash := range b.Runtime {
		if err = checkPin(path, hash); err != nil {
			return b, err
		}
	}
	for path, hash := range basic.Runtime {
		if old := b.Runtime[path]; old != "" && old != hash {
			return b, fmt.Errorf("conflicting qualified runtime")
		}
		if err = checkPin(path, hash); err != nil {
			return b, err
		}
		b.Runtime[path] = hash
	}
	return b, nil
}

var predicatePins = map[string]string{
	"dorado/nn/KoiUtils.cpp":            "9f8ae183210f552692cad2b940cb7c659914f854ed9da50602b5cd989f22802f",
	"dorado/nn/TxModules.cpp":           "9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701",
	"dorado/api/runner_creation.cpp":    "a8868f710a2be3ff6bff2629e19a99df37398c755479dc88c8cd26fc90b00010",
	"dorado/cli/cli_lib/basecaller.cpp": "33ef3b531810a538f9131239c3b2a0114a1a212956ce658b6b39ef6afb391764",
}

func between(s, start, end string) (string, error) {
	i := strings.Index(s, start)
	if i < 0 {
		return "", fmt.Errorf("missing source marker %s", start)
	}
	s = s[i+len(start):]
	j := strings.Index(s, end)
	if j < 0 {
		return "", fmt.Errorf("missing source terminator %s", end)
	}
	return s[:j], nil
}

// PredicateHeader embeds pinned source expressions; adapters supply only their context.
func PredicateHeader(source string) ([]byte, error) {
	files := map[string]string{}
	for rel, sha := range predicatePins {
		path := filepath.Join(source, rel)
		if err := checkPin(path, sha); err != nil {
			return nil, err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		files[rel] = string(b)
	}
	koi := files["dorado/nn/KoiUtils.cpp"]
	internal, err := between(koi, "namespace {", "}  // namespace")
	if err != nil {
		return nil, err
	}
	tx := files["dorado/nn/TxModules.cpp"]
	_, tx, ok := strings.Cut(tx, "TxEncoderStackImpl::TxEncoderStackImpl")
	if !ok {
		return nil, fmt.Errorf("missing encoder constructor")
	}
	sup, err := between(tx, "const bool is_sup_model =", ";")
	if err != nil {
		return nil, err
	}
	tiled, err := between(tx, "use_koi_tiled =", ";")
	if err != nil {
		return nil, err
	}
	i8, err := between(tx, "use_i8 =", ";")
	if err != nil {
		return nil, err
	}
	f8, err := between(files["dorado/nn/TxModules.cpp"], "bool use_f8 =", ";")
	if err != nil {
		return nil, err
	}
	rounding, err := between(files["dorado/nn/TxModules.cpp"], "int default_remove =", ";")
	if err != nil {
		return nil, err
	}
	vcs, err := between(files["dorado/api/runner_creation.cpp"], "bool check_variable_chunk_sizes_supported(", "\nstd::pair<")
	if err != nil {
		return nil, err
	}
	cli := files["dorado/cli/cli_lib/basecaller.cpp"]
	defaultIndex := strings.LastIndex(cli, ".variable_chunk_sizes =")
	if defaultIndex < 0 {
		return nil, fmt.Errorf("missing CLI VCS default")
	}
	defaultVCS, err := between(cli[defaultIndex:], ".variable_chunk_sizes =", ",\n")
	if err != nil {
		return nil, err
	}
	txVCS, err := between(koi, "bool koi_can_run_tx_vcs(const int device_id) {", "\n}")
	if err != nil {
		return nil, err
	}
	txVCS = strings.ReplaceAll(txVCS, "at::cuda::getDeviceProperties(device_id)", "&device_prop")
	return []byte("namespace {\n" + internal + "\n}\nnamespace dorado::api {\nbool check_variable_chunk_sizes_supported(" + vcs + "\n}\n" +
		"bool source_tx_vcs([[maybe_unused]] int device_id) { " + txVCS + "\n}\n" +
		"bool source_sup(const Params& params) { return " + sup + "; }\n" +
		"bool source_tiled(const Params& params) { const bool is_sup_model=source_sup(params); const bool is_hyp_model=false; Options options; return " + tiled + "; }\n" +
		"bool source_i8() { return " + i8 + "; }\n" +
		"bool source_f8() { return " + f8 + "; }\n" +
		"int source_remove_bits() { return " + rounding + "; }\n" +
		"bool source_default_vcs() { Parser parser; return " + defaultVCS + "; }\n"), nil
}

func jsonFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}

func clockUTC() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// CaptureProfile is verification-only. Admission and qualification precede the lock.
func CaptureProfile(o ProbeOptions) (m ProbeReceipt, err error) {
	if !o.Admitted {
		return m, fmt.Errorf("runtime admission required; no lock or child execution")
	}
	if err = o.Target.Validate(); err != nil {
		return m, err
	}
	build, err := qualify(o.Basic, o.Build)
	if err != nil {
		return m, err
	}
	if err = runner.CheckRoot(); err != nil {
		return m, err
	}
	env := map[string]Environment{}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "LD_PRELOAD", "LD_LIBRARY_PATH", "DORADO_MATMUL_FP16_MODE"} {
		v, p := os.LookupEnv(key)
		env[key] = Environment{p, v}
	}
	if env["CUDA_VISIBLE_DEVICES"] != o.Target.CUDAVisibleDevices {
		return m, fmt.Errorf("environment differs from target")
	}
	if o.KoiRoot == "" || o.CUDARoot == "" {
		return m, fmt.Errorf("qualified Koi and CUDA roots required")
	}
	for _, p := range []*string{&o.KoiRoot, &o.CUDARoot} {
		*p, err = filepath.Abs(*p)
		if err != nil {
			return m, err
		}
	}
	cudaHeader := filepath.Join(o.CUDARoot, "include/cuda.h")
	cudaBytes, e := os.ReadFile(cudaHeader)
	if e != nil {
		return m, e
	}
	toolkit := 0
	for _, line := range strings.Split(string(cudaBytes), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 3 && fields[0] == "#define" && fields[1] == "CUDA_VERSION" {
			toolkit, e = strconv.Atoi(fields[2])
			if e != nil {
				return m, e
			}
		}
	}
	if toolkit/1000 != 13 || toolkit != build.ToolkitVersion {
		return m, fmt.Errorf("CUDA header differs from accepted toolkit version")
	}
	if o.ModelRoot == "" {
		return m, fmt.Errorf("read-only model root required")
	}
	configs := []string{filepath.Join(o.ModelRoot, HAC, "config.toml"), filepath.Join(o.ModelRoot, SUP, "config.toml")}
	for i, path := range configs {
		if err = checkPin(path, o.Target.Models[i].ConfigSHA256); err != nil {
			return m, err
		}
	}
	header := filepath.Join(o.KoiRoot, "include/koi.h")
	library := filepath.Join(o.KoiRoot, "lib/libkoi.a")
	if err = checkPin(header, KoiHeaderSHA); err != nil {
		return m, err
	}
	if err = checkPin(library, KoiLibrarySHA); err != nil {
		return m, err
	}
	generated, err := PredicateHeader("external/dorado-source")
	if err != nil {
		return m, err
	}
	compiler, err := exec.LookPath("g++")
	if err != nil {
		return m, err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return m, err
	}
	if err = checkPin(compiler, build.Inputs[compiler]); err != nil {
		return m, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return m, err
	}
	defer lock.Close()
	dir, err := runner.NewAttempt(o.Out)
	if err != nil {
		return m, err
	}
	m = ProbeReceipt{Schema: ProfileSchema, Compile: ProbeCompile{Compiler: compiler, KoiRoot: o.KoiRoot, CUDARoot: o.CUDARoot, Directory: dir}, Status: "capture_failed", Started: clockUTC(), Invocation: o.Invocation, Target: o.Target, Environment: env, Inputs: map[string]string{}, Runtime: map[string]string{}, Commands: []runner.CommandReceipt{}, Matrix: []Request{}}
	for key, value := range map[string]string{"LD_LIBRARY_PATH": "", "LD_PRELOAD": "", "DORADO_MATMUL_FP16_MODE": "CUBLAS"} {
		m.Environment[key] = Environment{true, value}
	}
	save := func() error { return jsonFile(filepath.Join(dir, "profile.json"), m) }
	defer func() {
		m.Finished = clockUTC()
		if err != nil {
			m.Error = err.Error()
		}
		e := save()
		if err == nil {
			err = e
		}
	}()
	pin := func(path string) error {
		h, e := runner.SHA(path)
		if e == nil {
			m.Inputs[path] = h
		}
		return e
	}
	for _, path := range append([]string{o.Basic, o.Build, build.Output, header, library}, consumerSources...) {
		if err = pin(path); err != nil {
			return m, err
		}
	}
	if err = pin(cudaHeader); err != nil {
		return m, err
	}
	for _, path := range configs {
		if err = pin(path); err != nil {
			return m, err
		}
	}
	for rel := range predicatePins {
		if err = pin(filepath.Join("external/dorado-source", rel)); err != nil {
			return m, err
		}
	}
	for path, h := range build.Runtime {
		m.Runtime[path] = h
	}
	generatedPath := filepath.Join(dir, "predicates.h")
	if err = os.WriteFile(generatedPath, generated, 0644); err != nil {
		return m, err
	}
	if err = pin(generatedPath); err != nil {
		return m, err
	}
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		r, e := runner.Execute(dir, label, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}, args...)
		m.Commands = append(m.Commands, r)
		// Persist raw command evidence before any interpretation, including failures.
		if se := save(); se != nil {
			return r, se
		}
		if e != nil {
			return r, e
		}
		if e = checkPin(r.Stdout, r.StdoutSHA); e != nil {
			return r, e
		}
		if e = checkPin(r.Stderr, r.StderrSHA); e != nil {
			return r, e
		}
		return r, nil
	}
	if err = pin(compiler); err != nil {
		return m, err
	}
	if _, err = execute("compiler", compiler, "--version"); err != nil {
		return m, err
	}
	binary := m.Compile.binary()
	depfile := m.Compile.depfile()
	if _, err = execute("compile", m.Compile.argv()...); err != nil {
		return m, err
	}
	if err = pin(binary); err != nil {
		return m, err
	}
	dependencies, err := readDependencies(depfile, binary)
	if err != nil {
		return m, err
	}
	for _, path := range append(dependencies, depfile) {
		if err = pin(path); err != nil {
			return m, err
		}
	}
	maps := filepath.Join(dir, "probe.maps")
	command, err := execute("probe", binary, maps)
	if err != nil {
		return m, err
	}
	if err = pin(maps); err != nil {
		return m, err
	}
	if err = validateCompileClosure(m); err != nil {
		return m, err
	}
	if err = validateMaps(maps, build.Runtime, m.Runtime); err != nil {
		return m, err
	}
	// Preserve map pins before attempting to decode capability output.
	if err = save(); err != nil {
		return m, err
	}
	raw, err := os.ReadFile(command.Stdout)
	if err != nil {
		return m, err
	}
	var payload RawProbe
	if err = decodeClosed(raw, &payload); err != nil {
		return m, err
	}
	observation, matrix, err := ValidateProbe(o.Target, payload)
	if err != nil {
		return m, err
	}
	for path, h := range m.Inputs {
		if err = checkPin(path, h); err != nil {
			return m, err
		}
	}
	for path, h := range m.Runtime {
		if err = checkPin(path, h); err != nil {
			return m, err
		}
	}
	m.Raw = &payload
	m.Observation = &observation
	m.Matrix = matrix
	m.Status = "predicate_observation_passed"
	return m, nil
}

func validateMaps(path string, qualified, observed map[string]string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	seenCUDA, seenDriver := false, false
	seen := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		if len(fields) != 6 {
			return fmt.Errorf("unsupported mapped path")
		}
		p := fields[5]
		if strings.HasPrefix(p, "/dev/") {
			continue
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		h, e := runner.SHA(p)
		if e != nil {
			return e
		}
		name := filepath.Base(p)
		gpu := strings.HasPrefix(name, "libcu") || strings.HasPrefix(name, "libnv")
		if strings.HasPrefix(name, "libtorch") || strings.HasPrefix(name, "libc10") || strings.Contains(p, "cuda-12") || strings.Contains(p, "CUDA-12") {
			return fmt.Errorf("foreign runtime mapped: %s", p)
		}
		if gpu {
			matched := qualified[p] == h
			// Runtime manifests can pin a distribution symlink while maps names its target.
			for q, want := range qualified {
				if want == h {
					real, e := filepath.EvalSymlinks(q)
					if e == nil && real == p {
						matched = true
					}
				}
			}
			if !matched {
				return fmt.Errorf("GPU library outside accepted runtime closure: %s", p)
			}
		}
		if strings.HasPrefix(name, "libcudart.so") {
			seenCUDA = true
		}
		if strings.HasPrefix(name, "libcuda.so") {
			seenDriver = true
		}
		observed[p] = h
	}
	// Already qualified entries must not satisfy actual-mapping presence by themselves.
	if !seenCUDA || !seenDriver {
		return fmt.Errorf("missing actual CUDA runtime/driver mappings")
	}
	return nil
}

func ValidateProbe(t Target, p RawProbe) (Observation, []Request, error) {
	var o Observation
	if p.Ordinal != 0 || p.DeviceCount != 1 || p.Runtime/1000 != 13 || p.Driver <= 0 || p.CVDPresent != t.CUDAVisibleDevices.Present {
		return o, nil, fmt.Errorf("probe runtime/device identity mismatch")
	}
	o = Observation{RuntimeFamily: t.RuntimeFamily, DeviceUUID: p.DeviceUUID, CUDAVisibleDevices: t.CUDAVisibleDevices, ComputeMajor: p.Major, ComputeMinor: p.Minor, SMCount: p.SMs, F16Status: &p.F16, E4M3Status: &p.E4M3}
	b, err := Bind(t, o)
	if err != nil {
		return Observation{}, nil, err
	}
	hac := Shape{Kind: "flstm", C: 1024, K: 128}
	sup := Shape{Kind: "tx", C: 512, Heads: 8, WindowUpper: 127, WindowLower: 128, FeedForward: 2048}
	want := NativePredicates{HACFused: CanRunFLSTM(p.Major, p.SMs), SUPTiled: p.F16 == 0, FP8: p.E4M3 == 0, Int8: true, HACVCS: VariableChunksSupported(hac, p.Major, p.SMs, true, false, 1), SUPVCS: VariableChunksSupported(sup, p.Major, p.SMs, true, false, 1), DefaultVCS: true, RemoveBits: 4}
	if p.Predicates != want {
		return Observation{}, nil, fmt.Errorf("native source predicates differ from Go predicates")
	}
	matrix := []Request{}
	for _, model := range []string{HAC, SUP} {
		hash, batch := HACConfig, 64
		if model == SUP {
			hash, batch = SUPConfig, 32
		}
		for _, fixed := range []bool{false, true} {
			for _, trim := range []bool{false, true} {
				for _, split := range []bool{false, true} {
					r := Request{Model: model, ConfigSHA256: hash, Batch: batch, FixedChunks: fixed, DisableTrim: trim, DisableSplit: split, Output: "fastq", Kit: t.Kit, DevOptions: t.DevOptions}
					plan, e := Prepare(b, r)
					if e != nil {
						return Observation{}, nil, e
					}
					expected := Selection{HACFused: model == HAC && want.HACFused, SUPTiled: model == SUP && want.SUPTiled, FP8: model == SUP && want.FP8, Int8: model == SUP && want.Int8, VariableChunks: !fixed}
					if plan.Selection() != expected {
						return Observation{}, nil, fmt.Errorf("dispatch differs for profile matrix")
					}
					matrix = append(matrix, r)
				}
			}
		}
	}
	return o, matrix, nil
}

// VerifyFrozen rejects a changed observation even if it would be valid on its own.
func VerifyFrozen(frozen, candidate ProbeReceipt) error {
	if frozen.Status != "predicate_observation_passed" || frozen.Raw == nil || frozen.Observation == nil {
		return fmt.Errorf("missing frozen observation")
	}
	if candidate.Compile != frozen.Compile || candidate.Status != frozen.Status || candidate.Target != frozen.Target || !reflect.DeepEqual(candidate.Raw, frozen.Raw) || !reflect.DeepEqual(candidate.Observation, frozen.Observation) || !reflect.DeepEqual(candidate.Environment, frozen.Environment) || !reflect.DeepEqual(candidate.Inputs, frozen.Inputs) || !reflect.DeepEqual(candidate.Runtime, frozen.Runtime) || !reflect.DeepEqual(candidate.Matrix, frozen.Matrix) {
		return fmt.Errorf("candidate differs from frozen evidence")
	}
	_, _, err := ValidateProbe(candidate.Target, *candidate.Raw)
	return err
}

func uniqueValue(d *json.Decoder) (any, error) {
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return token, nil
	}
	switch delim {
	case '{':
		out := map[string]any{}
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return nil, e
			}
			key, ok := k.(string)
			if !ok {
				return nil, fmt.Errorf("invalid object key")
			}
			if _, exists := out[key]; exists {
				return nil, fmt.Errorf("duplicate JSON key %s", key)
			}
			v, e := uniqueValue(d)
			if e != nil {
				return nil, e
			}
			out[key] = v
		}
		end, e := d.Token()
		if e != nil || end != json.Delim('}') {
			return nil, fmt.Errorf("invalid JSON object")
		}
		return out, nil
	case '[':
		out := []any{}
		for d.More() {
			v, e := uniqueValue(d)
			if e != nil {
				return nil, e
			}
			out = append(out, v)
		}
		end, e := d.Token()
		if e != nil || end != json.Delim(']') {
			return nil, fmt.Errorf("invalid JSON array")
		}
		return out, nil
	}
	return nil, fmt.Errorf("unexpected JSON delimiter")
}

func requireFields(tree any, t reflect.Type) error {
	if t.Kind() == reflect.Pointer {
		if tree == nil {
			return nil
		}
		return requireFields(tree, t.Elem())
	}
	if tree == nil {
		return fmt.Errorf("null required field of type %s", t)
	}
	switch t.Kind() {
	case reflect.Struct:
		object, ok := tree.(map[string]any)
		if !ok {
			return fmt.Errorf("expected object")
		}
		allowed := map[string]bool{}
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			if name == "" {
				name = field.Name
			}
			allowed[name] = true
			value, exists := object[name]
			if !exists {
				return fmt.Errorf("missing field %s", name)
			}
			if err := requireFields(value, field.Type); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
		for key := range object {
			if !allowed[key] {
				return fmt.Errorf("unknown exact field %s", key)
			}
		}
	case reflect.Slice, reflect.Array:
		array, ok := tree.([]any)
		if !ok {
			return fmt.Errorf("expected array")
		}
		if t.Kind() == reflect.Array && len(array) != t.Len() {
			return fmt.Errorf("wrong fixed array length")
		}
		for _, value := range array {
			if err := requireFields(value, t.Elem()); err != nil {
				return err
			}
		}
	case reflect.Map:
		object, ok := tree.(map[string]any)
		if !ok {
			return fmt.Errorf("expected map")
		}
		for _, value := range object {
			if err := requireFields(value, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeClosed(raw []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	tree, err := uniqueValue(d)
	if err != nil {
		return err
	}
	if _, err = d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	d = json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err = d.Decode(out); err != nil {
		return err
	}
	return requireFields(tree, reflect.TypeOf(out).Elem())
}

// ImportProfile replays retained evidence only. The caller supplies the frozen manifest hash.
func ImportProfile(path, frozenSHA string) (ProbeReceipt, error) {
	var m ProbeReceipt
	if err := checkPin(path, frozenSHA); err != nil {
		return m, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return m, err
	}
	if err = decodeClosed(raw, &m); err != nil {
		return ProbeReceipt{}, err
	}
	if m.Schema != ProfileSchema || m.Status != "predicate_observation_passed" || m.Raw == nil || m.Observation == nil || m.Error != "" {
		return ProbeReceipt{}, fmt.Errorf("capture not passed")
	}
	if _, err = time.Parse(time.RFC3339Nano, m.Started); err != nil {
		return ProbeReceipt{}, err
	}
	finished, err := time.Parse(time.RFC3339Nano, m.Finished)
	if err != nil {
		return ProbeReceipt{}, err
	}
	started, _ := time.Parse(time.RFC3339Nano, m.Started)
	if finished.Before(started) {
		return ProbeReceipt{}, fmt.Errorf("capture clock reversed")
	}
	if len(m.Commands) != 3 || len(m.Inputs) == 0 || len(m.Runtime) == 0 || len(m.Invocation) == 0 {
		return ProbeReceipt{}, fmt.Errorf("incomplete evidence")
	}
	if err = validateCompileClosure(m); err != nil {
		return ProbeReceipt{}, err
	}
	find := func(want string) (string, error) {
		for p, h := range m.Inputs {
			if h == want {
				return p, nil
			}
		}
		return "", fmt.Errorf("missing required input pin %s", want)
	}
	basicPath, err := find(BasicSHA)
	if err != nil {
		return ProbeReceipt{}, err
	}
	buildPath, err := find(BuildSHA)
	if err != nil {
		return ProbeReceipt{}, err
	}
	build, err := qualify(basicPath, buildPath)
	if err != nil {
		return ProbeReceipt{}, err
	}
	for _, h := range []string{BackendSHA, KoiHeaderSHA, KoiLibrarySHA, HACConfig, SUPConfig} {
		if _, err = find(h); err != nil {
			return ProbeReceipt{}, err
		}
	}
	for rel, h := range predicatePins {
		if m.Inputs[filepath.Join("external/dorado-source", rel)] != h {
			return ProbeReceipt{}, fmt.Errorf("missing source predicate pin %s", rel)
		}
	}
	if m.Environment["CUDA_VISIBLE_DEVICES"] != m.Target.CUDAVisibleDevices {
		return ProbeReceipt{}, fmt.Errorf("environment differs")
	}
	if len(m.Environment) != 6 {
		return ProbeReceipt{}, fmt.Errorf("incomplete effective environment")
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "LD_LIBRARY_PATH", "LD_PRELOAD", "DORADO_MATMUL_FP16_MODE"} {
		if _, ok := m.Environment[key]; !ok {
			return ProbeReceipt{}, fmt.Errorf("missing environment key %s", key)
		}
	}
	for key, value := range map[string]string{"LD_LIBRARY_PATH": "", "LD_PRELOAD": "", "DORADO_MATMUL_FP16_MODE": "CUBLAS"} {
		if m.Environment[key] != (Environment{true, value}) {
			return ProbeReceipt{}, fmt.Errorf("effective environment mismatch")
		}
	}
	for p, h := range m.Inputs {
		if err = checkPin(p, h); err != nil {
			return ProbeReceipt{}, err
		}
	}
	for p, h := range m.Runtime {
		if err = checkPin(p, h); err != nil {
			return ProbeReceipt{}, err
		}
	}
	for _, c := range m.Commands {
		if !reflect.DeepEqual(c.Environment, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}) {
			return ProbeReceipt{}, fmt.Errorf("command environment mismatch")
		}
		a, e := time.Parse(time.RFC3339Nano, c.Started)
		if e != nil {
			return ProbeReceipt{}, e
		}
		b, e := time.Parse(time.RFC3339Nano, c.Finished)
		if e != nil {
			return ProbeReceipt{}, e
		}
		if c.Exit != 0 || b.Before(a) || a.Before(started) || b.After(finished) || len(c.Args) == 0 {
			return ProbeReceipt{}, fmt.Errorf("invalid command receipt")
		}
		if err = checkPin(c.Stdout, c.StdoutSHA); err != nil {
			return ProbeReceipt{}, err
		}
		if err = checkPin(c.Stderr, c.StderrSHA); err != nil {
			return ProbeReceipt{}, err
		}
	}
	last := m.Commands[2]
	if len(last.Args) != 2 || filepath.Base(last.Args[0]) != "gpuprofile-probe" || m.Inputs[last.Args[0]] == "" || m.Inputs[last.Args[1]] == "" {
		return ProbeReceipt{}, fmt.Errorf("missing pinned probe binary/maps")
	}
	generated, err := PredicateHeader("external/dorado-source")
	if err != nil {
		return ProbeReceipt{}, err
	}
	headerPath := filepath.Join(filepath.Dir(last.Args[0]), "predicates.h")
	header, err := os.ReadFile(headerPath)
	if err != nil {
		return ProbeReceipt{}, err
	}
	if m.Inputs[headerPath] == "" || !bytes.Equal(header, generated) {
		return ProbeReceipt{}, fmt.Errorf("generated native predicate source differs")
	}
	if build.Inputs[m.Compile.Compiler] != m.Inputs[m.Compile.Compiler] {
		return ProbeReceipt{}, fmt.Errorf("compiler receipt differs from qualification")
	}
	mapped := map[string]string{}
	if err = validateMaps(last.Args[1], build.Runtime, mapped); err != nil {
		return ProbeReceipt{}, err
	}
	for p, h := range mapped {
		if m.Runtime[p] != h {
			return ProbeReceipt{}, fmt.Errorf("missing observed mapping pin %s", p)
		}
	}
	payload, err := os.ReadFile(last.Stdout)
	if err != nil {
		return ProbeReceipt{}, err
	}
	var p RawProbe
	if err = decodeClosed(payload, &p); err != nil {
		return ProbeReceipt{}, err
	}
	if !reflect.DeepEqual(p, *m.Raw) {
		return ProbeReceipt{}, fmt.Errorf("projection differs from raw output")
	}
	o, matrix, err := ValidateProbe(m.Target, p)
	if err != nil {
		return ProbeReceipt{}, err
	}
	if !reflect.DeepEqual(o, *m.Observation) || !reflect.DeepEqual(matrix, m.Matrix) {
		return ProbeReceipt{}, fmt.Errorf("projection differs from replay")
	}
	return m, nil
}
