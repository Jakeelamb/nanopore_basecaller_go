// Package supgpu prepares and imports verification-only SUP GPU observations.
// Import and Prepare never load a native library or query a device.
package supgpu

import "github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"

const (
	SourceCommit   = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
	PreparedSchema = "nanopore-supgpu-prepared/v1"
	CaptureSchema  = "nanopore-supgpu-capture/v2"
	ModelName      = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
	ProfileSHA     = "67582f569c4e9cbcd3f84c2158584a04537302c1215c1adfaf3a375f3c84e829"
)

type Pin struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type Geometry struct {
	Batch                  int `json:"batch"`
	Chunk                  int `json:"chunk"`
	EncoderT               int `json:"encoder_t"`
	ConvStride             int `json:"conv_stride"`
	CallerChunkGranularity int `json:"caller_chunk_granularity"`
	Overlap                int `json:"overlap"`
}

type Chunk struct {
	Start  uint64 `json:"start"`
	End    uint64 `json:"end"`
	SHA256 string `json:"sha256"`
}

// Prepared is the external trust anchor. Its SHA must come from outside the
// capture, and all source, fixture, profile and weight pins are checked again.
type Prepared struct {
	Schema                string   `json:"schema"`
	SourceCommit          string   `json:"source_commit"`
	Case                  string   `json:"case"`
	Geometry              Geometry `json:"geometry"`
	DeviceUUID            string   `json:"device_uuid"`
	UseI8                 bool     `json:"use_i8"`
	UseF8                 bool     `json:"use_f8"`
	RemoveBits            int      `json:"remove_bits"`
	Fixture               Pin      `json:"fixture"`
	Signal                Pin      `json:"signal"`
	ReadID                string   `json:"read_id"`
	Cohort                string   `json:"cohort"`
	CalibrationScaleBits  uint32   `json:"calibration_scale_bits"`
	CalibrationOffsetBits uint32   `json:"calibration_offset_bits"`
	MeanBits              uint32   `json:"mean_bits"`
	StdevBits             uint32   `json:"stdev_bits"`
	Trim                  int      `json:"trim"`
	Input                 Pin      `json:"input"`
	Chunks                []Chunk  `json:"chunks"`
	Config                Pin      `json:"config"`
	Profile               Pin      `json:"profile"`
	Weights               []Pin    `json:"weights"`
	Sources               []Pin    `json:"sources"`
}

// Range is a prospective byte partition of a complete allocation. Excluded
// ranges must name a source rule; an observed mismatch is never a rule.
type Range struct {
	Begin int64  `json:"begin"`
	End   int64  `json:"end"`
	State string `json:"state"`
	Rule  string `json:"rule"`
}

// Tensor retains a contiguous logical copy and the complete allocated storage.
// Strides/offset refer to the allocation, not the contiguous logical payload.
type Tensor struct {
	Name        string  `json:"name"`
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	Strides     []int64 `json:"strides"`
	Tags        []int   `json:"tags"`
	ScaleTag    int     `json:"scale_tag"`
	OffsetBytes int64   `json:"offset_bytes"`
	Data        Pin     `json:"data"`
	Backing     Pin     `json:"backing"`
	Regions     []Range `json:"regions"`
}

type Event struct {
	Name        string       `json:"name"`
	Layer       int          `json:"layer"`
	Tensors     []Tensor     `json:"tensors"`
	Descriptors []Descriptor `json:"descriptors"`
}

type Descriptor struct {
	Name       string  `json:"name"`
	DType      string  `json:"dtype"`
	Shape      []int64 `json:"shape"`
	Strides    []int64 `json:"strides"`
	Tags       []int   `json:"tags"`
	ScaleTag   int     `json:"scale_tag"`
	ScaleDType string  `json:"scale_dtype"`
}

// Build records the entire admitted compilation boundary. Every command keeps
// raw logs and clocks. Depfiles and the link map are inputs to validation.
type Build struct {
	Schema            string                  `json:"schema"`
	Status            string                  `json:"status"`
	PreparedSHA256    string                  `json:"prepared_sha256"`
	Plan              Pin                     `json:"plan"`
	Compiler          Pin                     `json:"compiler"`
	Inputs            []Pin                   `json:"inputs"`
	Depfiles          []Pin                   `json:"depfiles"`
	LinkMaps          []Pin                   `json:"link_maps"`
	Reference         Pin                     `json:"reference"`
	Observer          Pin                     `json:"observer"`
	Audit             Pin                     `json:"audit"`
	Commands          []runner.CommandReceipt `json:"commands"`
	PreflightCommands []runner.CommandReceipt `json:"preflight_commands"`
	PreDepfiles       []Pin                   `json:"precompile_depfiles"`
	Dependencies      [][]string              `json:"dependencies"`
	BeforeInputs      Pin                     `json:"before_compile_inputs"`
	Objects           []Pin                   `json:"objects"`
}

type Boundary struct {
	Name string `json:"name"`
	Data Pin    `json:"data"`
}

type Capture struct {
	Schema         string                  `json:"schema"`
	PreparedSHA256 string                  `json:"prepared_sha256"`
	BuildSHA256    string                  `json:"build_sha256"`
	RuntimePlan    Pin                     `json:"runtime_plan"`
	DeviceUUID     string                  `json:"device_uuid"`
	UseI8          bool                    `json:"use_i8"`
	UseF8          bool                    `json:"use_f8"`
	Reference      Pin                     `json:"reference_scores"`
	Staged         Pin                     `json:"staged_scores"`
	Events         []Event                 `json:"events"`
	Runtime        []Pin                   `json:"runtime"`
	RuntimeMaps    []Pin                   `json:"runtime_maps"`
	Commands       []runner.CommandReceipt `json:"commands"`
	Boundaries     []Boundary              `json:"reference_boundaries"`
	Guards         []Pin                   `json:"guards"`
}
