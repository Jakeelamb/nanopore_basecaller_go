// Package packed prepares source-only packed HAC observation inputs. It grants
// no native build, runtime, GPU, or biological qualification.
package packed

const PreparedSchema = "PackedHACPrepared/v1"
const RequestSchema = "PackedHACRequest/v1"
const GenerationSchema = "PackedHACGeneration/v1"
const MaxArtifactBytes int64 = 64 << 20

type Pin struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// All pins are supplied independently, never learned from a producer response.
// Model/config/weight manifests are opaque custody inputs in this first unit.
type Custody struct {
	Model      Pin `json:"model_manifest"`
	Config     Pin `json:"config"`
	Weights    Pin `json:"weight_manifest"`
	Dataset    Pin `json:"dataset_manifest"`
	CPU21      Pin `json:"cpu21_evidence_manifest"`
	Generation Pin `json:"generation_manifest"`
	// Unchanged helper/config/rounding authorities are custody inputs too.
	Authorities []Pin `json:"source_authorities"`
}

type Geometry struct {
	N             int64 `json:"physical_n"`
	Chunk         int64 `json:"configured_chunk"`
	MaxChunk      int64 `json:"model_max_chunk"`
	InputChannels int64 `json:"input_channels"`
	Channels      int64 `json:"channels"`
	Inner         int64 `json:"inner"`
	Layers        int64 `json:"layers"`
	StrideIn      int64 `json:"stride_in"`
	StrideOut     int64 `json:"stride_out"`
	Granularity   int64 `json:"granularity"`
}

func FirstGeometry() Geometry { return Geometry{64, 12288, 12288, 1, 1024, 128, 7, 6, 6, 6} }

// Pointers distinguish unknown calibration from a recorded zero offset/mean.
type Normalisation struct {
	Algorithm         string   `json:"algorithm"`
	CalibrationScale  *float64 `json:"calibration_scale"`
	CalibrationOffset *float64 `json:"calibration_offset"`
	Mean              *float64 `json:"model_mean"`
	Stdev             *float64 `json:"model_stdev"`
	TrimStart         int64    `json:"trim_start"`
	Provenance        Pin      `json:"provenance"`
}

type Member struct {
	ID                string        `json:"id"`
	ReadID            string        `json:"read_id"`
	Dataset           string        `json:"dataset"`
	Split             string        `json:"split"`
	Source            Pin           `json:"source_fixture"`
	RawParentSamples  int64         `json:"raw_parent_samples"`
	RawOffset         int64         `json:"raw_parent_offset"`
	RawLength         int64         `json:"raw_samples"`
	RoundedLength     int64         `json:"rounded_samples"`
	Normalisation     Normalisation `json:"normalisation"`
	Normalized        Pin           `json:"normalized_f16_le_fixture"`
	NormalizedSamples int64         `json:"normalized_samples"`
	NormalizedOffset  int64         `json:"normalized_offset"`
}

type Recipe struct {
	Workers            int64    `json:"workers"`
	CPUWorkspacePinned bool     `json:"cpu_workspace_pinned_required"`
	UnusedInputFill    uint16   `json:"unused_input_f16_bits"`
	ObservationFills   []uint8  `json:"observation_fills"`
	GuardBytes         int64    `json:"guard_bytes"`
	Assertions         []string `json:"assertions"`
}

type Request struct {
	Schema   string   `json:"schema"`
	Case     string   `json:"case"`
	Split    string   `json:"split"`
	Dataset  string   `json:"dataset"`
	Custody  Custody  `json:"custody"`
	Geometry Geometry `json:"geometry"`
	Members  []Member `json:"members"`
	// ExpectedOrder is independently declared and binds member ordering.
	ExpectedOrder []string `json:"expected_order"`
	Recipe        Recipe   `json:"recipe"`
}

type Segment struct {
	ID             string `json:"id"`
	RawSamples     int64  `json:"raw_samples"`
	RoundedSamples int64  `json:"rounded_samples"`
	InputOffset    int64  `json:"input_offset"`
	ScoreOffset    int64  `json:"score_offset"`
	Scores         int64  `json:"scores"`
}

type UploadSpan struct {
	Role                 string `json:"role"`
	InitializedWords     int64  `json:"initialized_words"`
	CapacityWords        int64  `json:"capacity_words"`
	WorkspaceOffsetWords int64  `json:"workspace_offset_words"`
}

type Plan struct {
	TOut                      int64        `json:"configured_t_out"`
	TLSTM                     int64        `json:"t_lstm"`
	ActiveInput               int64        `json:"active_input_samples"`
	InputCapacity             int64        `json:"input_capacity_samples"`
	ActiveScores              int64        `json:"active_scores"`
	CPUWorkspaceWords         int64        `json:"cpu_workspace_words"`
	InitializedWorkspaceWords int64        `json:"initialized_workspace_words"`
	RecurrentParentBytes      int64        `json:"recurrent_parent_bytes"`
	DownScratchBytes          int64        `json:"down_scratch_bytes"`
	ScaleScratchBytes         int64        `json:"scale_scratch_bytes"`
	WorkspaceScratchBytes     int64        `json:"workspace_scratch_bytes"`
	Segments                  []Segment    `json:"segments"`
	Uploads                   []UploadSpan `json:"uploads_in_source_copy_order"`
}

type Prepared struct {
	Schema     string  `json:"schema"`
	Status     string  `json:"status"`
	RequestPin Pin     `json:"request_pin"`
	Request    Request `json:"request"`
	Plan       Plan    `json:"plan"`
	Input      Pin     `json:"input"`
	Pins       []Pin   `json:"authenticated_inputs"`
	// Nil is an explicit unavailable native closure, never an admission flag.
	NativeClosure *NativeClosure `json:"native_closure"`
	Limits        []string       `json:"limits"`
}

type NativeClosure struct {
	Compiler   Pin `json:"compiler_contract"`
	Link       Pin `json:"link_contract"`
	Runtime    Pin `json:"runtime_contract"`
	Controller Pin `json:"controller_contract"`
	Target     Pin `json:"target_contract"`
	Command    Pin `json:"command_contract"`
	Phase      Pin `json:"phase_contract"`
}
