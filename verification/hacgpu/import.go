package hacgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// dtypeBytes is the element size of every torch dtype name the oracle can emit.
// It is deliberately a closed set: an unknown dtype name is a rejection, not an
// assumption.
var dtypeBytes = map[string]int64{
	"Half": 2, "BFloat16": 2, "Char": 1, "Byte": 1, "Bool": 1,
	"Short": 2, "Int": 4, "Long": 8, "Float": 4, "Double": 8,
}

// StageNames is the ordered stage list for the graph shape the capture declares.
//
// The list is derived from the declared graph, never assumed: one `convN` stage
// per convolution layer, one `flstmN` stage per recurrent layer, then the
// projection(s) and the scores. `CRFModel.cpp:51-70` builds one of two decoders:
// with `can_use_koi && lstm_size == 1024 && decomposition == 128 && tanh_x5` it
// registers a single FactorisedLinearLayer and `linear2` is never constructed;
// otherwise it registers `linear1` and `linear2` separately. HAC v6 pins
// `scale = 5`, so it takes the factorised branch and has no `linear2` stage.
// Import requires the stage list to match exactly, so a missing, extra or
// reordered layer is a hard rejection.
func StageNames(convs, lstmLayers int, linear2 bool) []string {
	names := make([]string, 0, 2+convs+lstmLayers)
	names = append(names, "input")
	for i := 0; i < convs; i++ {
		names = append(names, fmt.Sprintf("conv%d", i))
	}
	for i := 0; i < lstmLayers; i++ {
		names = append(names, fmt.Sprintf("flstm%d", i))
	}
	names = append(names, "linear1")
	if linear2 {
		names = append(names, "linear2")
	}
	return append(names, "scores")
}

// CaptureSchema identifies a native stage capture.
const CaptureSchema = "nanopore-hacgpu-native-stage-capture/v1"

// Stage is one captured graph stage.
type Stage struct {
	Name            string   `json:"name"`
	Kind            string   `json:"kind"`
	Layer           int      `json:"layer"`
	Shape           []int64  `json:"shape"`
	DType           string   `json:"dtype"`
	Layout          string   `json:"layout"`
	OffsetBytes     int64    `json:"offset_bytes"`
	Bytes           int64    `json:"bytes"`
	ReservationByte int64    `json:"reservation_bytes"`
	CurrentBytes    int64    `json:"current_bytes"`
	N               int      `json:"N"`
	T               int      `json:"T"`
	C               int      `json:"C"`
	InputToRevLSTM  bool     `json:"is_input_to_rev_lstm"`
	UnwrittenT      [][2]int `json:"unwritten_t"`
	DataFile        string   `json:"data_file"`
	DataSHA         string   `json:"data_sha256"`
	BackingFile     string   `json:"backing_file"`
	BackingSHA      string   `json:"backing_sha256"`
}

// Capture is the typed view of capture.json.
type Capture struct {
	Schema       string            `json:"schema"`
	Status       string            `json:"status"`
	Case         string            `json:"case"`
	ModelDir     string            `json:"model_dir"`
	ModelPackage string            `json:"model_package"`
	SourceCommit string            `json:"source_commit"`
	Ordinal      int               `json:"ordinal"`
	Batch        int               `json:"batch"`
	Chunk        int               `json:"chunk"`
	Stride       int               `json:"stride"`
	TOut         int               `json:"t_out"`
	Features     int               `json:"features"`
	LSTMLayers   int               `json:"lstm_layers"`
	LSTMSize     int               `json:"lstm_size"`
	LSTMInnerDim int               `json:"lstm_inner_dim"`
	Convs        int               `json:"convs"`
	Scale        float64           `json:"scale"`
	Standardise  bool              `json:"standardise"`
	StdMean      float32           `json:"standardisation_mean"`
	StdStdev     float32           `json:"standardisation_stdev"`
	OutFeatures  int               `json:"out_features"`
	Linear2      bool              `json:"linear2_present"`
	Linear1Kind  string            `json:"linear1_kind"`
	Reservation  int64             `json:"reservation_bytes"`
	BackingElems int64             `json:"backing_elems"`
	BackingDType string            `json:"backing_dtype"`
	BackingAlign int64             `json:"backing_alignment"`
	ScoresShape  []int64           `json:"scores_shape"`
	ScoresDType  string            `json:"scores_dtype"`
	ScoresContig bool              `json:"scores_contiguous"`
	ScoresSHA    string            `json:"scores_sha256"`
	StackSHA     map[string]string `json:"stack_sha256"`
	Stages       []Stage           `json:"stages"`
}

// Import reads and fully verifies a capture directory. Every stage payload and
// every raw parent backing dump is re-hashed; a single corrupt byte, a missing
// stage, an unexpected stage, a bad offset or an inconsistent reservation is a
// hard rejection. This is the only way downstream cards obtain stage bytes.
func Import(dir string) (Capture, error) {
	var c Capture
	raw, err := os.ReadFile(filepath.Join(dir, "capture.json"))
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(raw, &c); err != nil {
		return c, err
	}
	if c.Schema != CaptureSchema {
		return c, fmt.Errorf("capture schema %q is not %q", c.Schema, CaptureSchema)
	}
	if c.Status != "captured" {
		return c, fmt.Errorf("capture status %q is not captured", c.Status)
	}
	if c.SourceCommit != SourceCommit {
		return c, fmt.Errorf("capture source commit %q is not the pinned revision", c.SourceCommit)
	}
	if c.Features <= 0 || c.Chunk <= 0 || c.Batch <= 0 {
		return c, fmt.Errorf("capture geometry is not positive")
	}
	if c.Reservation <= 0 || c.BackingElems <= 0 || c.BackingAlign != 256 {
		return c, fmt.Errorf("capture backing geometry is inconsistent")
	}
	if c.Linear2 && c.Linear1Kind != "plain" {
		return c, fmt.Errorf("capture declares linear2 with linear1 kind %q", c.Linear1Kind)
	}
	if !c.Linear2 && c.Linear1Kind != "factorised" {
		return c, fmt.Errorf("capture has no linear2 but linear1 kind %q", c.Linear1Kind)
	}
	if c.Convs <= 0 || c.LSTMLayers <= 0 {
		return c, fmt.Errorf("capture declares %d convolutions and %d recurrent layers", c.Convs, c.LSTMLayers)
	}
	if c.Batch%64 != 0 {
		return c, fmt.Errorf("capture batch %d is not a multiple of the 64-wide kernel granularity", c.Batch)
	}
	// `host_factorised_linear` tiles the time axis 32 rows at a time. A T_out
	// that is not a multiple of 32 never completes (observed: T_out 48 hangs), so
	// a capture claiming one is not a graph result. Production T_out values are
	// 2048 and 1024, both multiples of 32.
	if c.Stride <= 0 || c.TOut <= 0 || c.TOut%32 != 0 || c.TOut != c.Chunk/c.Stride {
		return c, fmt.Errorf("capture stride %d, T_out %d and chunk %d are not a valid projection geometry",
			c.Stride, c.TOut, c.Chunk)
	}
	want := StageNames(c.Convs, c.LSTMLayers, c.Linear2)
	if len(c.Stages) != len(want) {
		return c, fmt.Errorf("capture has %d stages, want %d", len(c.Stages), len(want))
	}
	seen := map[string]bool{}
	for i, s := range c.Stages {
		if s.Name != want[i] {
			return c, fmt.Errorf("stage %d is %q, want %q", i, s.Name, want[i])
		}
		seen[s.Name] = true
		if err = verifyStage(dir, &c, s); err != nil {
			return c, fmt.Errorf("stage %q: %w", s.Name, err)
		}
	}
	for _, name := range want {
		if !seen[name] {
			return c, fmt.Errorf("capture lacks stage %q", name)
		}
	}
	// The per-layer replay must reproduce the untouched stack outputs byte for
	// byte. The oracle records the stack digests from its untouched run; this
	// check is the reason a per-layer capture can be trusted as the graph.
	for stack, stage := range map[string]string{
		"convs":   fmt.Sprintf("conv%d", c.Convs-1),
		"rnns":    fmt.Sprintf("flstm%d", c.LSTMLayers-1),
		"linear1": "linear1",
	} {
		wantSHA := c.StackSHA[stack]
		if wantSHA == "" {
			return c, fmt.Errorf("capture lacks the untouched %s stack digest", stack)
		}
		got, ok := c.Stage(stage)
		if !ok {
			return c, fmt.Errorf("capture lacks stage %q for the %s cross-check", stage, stack)
		}
		if got.DataSHA != wantSHA {
			return c, fmt.Errorf("per-layer %q digest %s does not reproduce the untouched %s stack %s",
				stage, got.DataSHA, stack, wantSHA)
		}
	}
	scores := c.Stages[len(c.Stages)-1]
	if scores.Name != "scores" || scores.DataSHA != c.ScoresSHA {
		return c, fmt.Errorf("scores stage does not match the recorded scores digest")
	}
	if want := c.StackSHA["scores"]; want == "" || want != scores.DataSHA {
		return c, fmt.Errorf("scores stage does not reproduce the untouched scores stack digest")
	}
	// A batch or T_out the projection kernel cannot tile leaves the output
	// untouched. Reject a constant scores tensor: it is the silent shape of that
	// failure, not a graph result.
	body, err := c.RawPayload(dir, "scores")
	if err != nil {
		return c, err
	}
	distinct := false
	for i := 2; i+1 < len(body); i += 2 {
		if body[i] != body[0] || body[i+1] != body[1] {
			distinct = true
			break
		}
	}
	if !distinct {
		return c, fmt.Errorf("scores stage is a single repeated value, so the graph did not run")
	}
	if !c.Standardise {
		return c, fmt.Errorf("model config does not request standardised scaling")
	}
	if !(c.StdStdev > 0) {
		return c, fmt.Errorf("model config stdev %v is not positive", c.StdStdev)
	}
	if !c.ScoresContig {
		return c, fmt.Errorf("scores are not contiguous")
	}
	if c.ScoresDType != "Half" {
		return c, fmt.Errorf("scores dtype %q is not Half", c.ScoresDType)
	}
	return c, nil
}

func verifyStage(dir string, c *Capture, s Stage) error {
	if s.DataFile == "" || s.BackingFile == "" {
		return fmt.Errorf("stage lacks a payload or backing file")
	}
	if err := checkStageIdentity(s); err != nil {
		return err
	}
	if s.Bytes <= 0 || s.CurrentBytes <= 0 {
		return fmt.Errorf("stage byte counts are not positive")
	}
	if s.ReservationByte != c.Reservation {
		return fmt.Errorf("stage reservation %d differs from capture %d", s.ReservationByte, c.Reservation)
	}
	if s.OffsetBytes < 0 || s.OffsetBytes+s.Bytes > c.BackingElems*2 {
		return fmt.Errorf("stage interval [%d,%d) escapes the backing allocation", s.OffsetBytes, s.OffsetBytes+s.Bytes)
	}
	// The graph legitimately produces more than one dtype: the CUTLASS conv and
	// FLSTM stages are quantized I8, the decoder stages are F16. The dtype is
	// recorded per stage and the byte count must match the recorded shape at that
	// element size, so a mislabelled or mis-sized stage still rejects.
	elemSize, ok := dtypeBytes[s.DType]
	if !ok {
		return fmt.Errorf("stage dtype %q is not a known element type", s.DType)
	}
	if s.N != c.Batch {
		return fmt.Errorf("stage batch %d differs from capture %d", s.N, c.Batch)
	}
	if err := checkUnwritten(s); err != nil {
		return err
	}
	if elems := product(s.Shape); elems*elemSize != s.Bytes {
		return fmt.Errorf("stage shape %v at %d bytes/element does not match %d bytes", s.Shape, elemSize, s.Bytes)
	}
	data, err := os.ReadFile(filepath.Join(dir, s.DataFile))
	if err != nil {
		return err
	}
	if int64(len(data)) != s.Bytes {
		return fmt.Errorf("payload is %d bytes, want %d", len(data), s.Bytes)
	}
	got, err := bytesSHA(data)
	if err != nil {
		return err
	}
	if got != s.DataSHA {
		return fmt.Errorf("payload digest %s does not match the recorded %s", got, s.DataSHA)
	}
	backing, err := os.ReadFile(filepath.Join(dir, s.BackingFile))
	if err != nil {
		return err
	}
	if int64(len(backing)) != c.BackingElems*2 {
		return fmt.Errorf("backing dump is %d bytes, want %d", len(backing), c.BackingElems*2)
	}
	got, err = bytesSHA(backing)
	if err != nil {
		return err
	}
	if got != s.BackingSHA {
		return fmt.Errorf("backing digest %s does not match the recorded %s", got, s.BackingSHA)
	}
	// The stage payload must be the exact slice of the raw parent backing dump.
	if !equal(data, backing[s.OffsetBytes:s.OffsetBytes+s.Bytes]) {
		return fmt.Errorf("payload is not the declared interval of the backing dump")
	}
	return nil
}

// checkUnwritten validates the declared unwritten time rows against the stage's
// layout and shape. They are prospective exclusions: the source never writes
// them, so they are not evidence, but a declaration that escapes the tensor or
// overlaps itself is still a malformed capture.
func checkUnwritten(s Stage) error {
	var extent int64
	switch s.Layout {
	case "NTC":
		if len(s.Shape) != 3 {
			return fmt.Errorf("NTC stage %q has shape %v", s.Name, s.Shape)
		}
		extent = s.Shape[1]
	case "CUTLASS_TNC_I8", "CUTLASS_TNC_F16":
		if len(s.Shape) != 3 {
			return fmt.Errorf("CUTLASS stage %q has shape %v", s.Name, s.Shape)
		}
		extent = s.Shape[0]
	default:
		if len(s.UnwrittenT) > 0 {
			return fmt.Errorf("stage %q declares unwritten rows in layout %q", s.Name, s.Layout)
		}
		return nil
	}
	seen := make([]bool, extent)
	for _, r := range s.UnwrittenT {
		if r[0] < 0 || r[1] < 0 || int64(r[0]+r[1]) > extent {
			return fmt.Errorf("stage %q declares unwritten rows [%d,%d) outside the %d-row T axis",
				s.Name, r[0], r[0]+r[1], extent)
		}
		for t := r[0]; t < r[0]+r[1]; t++ {
			if seen[t] {
				return fmt.Errorf("stage %q declares row %d unwritten twice", s.Name, t)
			}
			seen[t] = true
		}
	}
	return nil
}

// checkStageIdentity requires the recorded kind and layer index to agree with the
// stage name, so a relabelled stage cannot claim to be a different layer.
func checkStageIdentity(s Stage) error {
	switch {
	case s.Name == "input" || s.Name == "scores":
		if s.Kind != s.Name || s.Layer != -1 {
			return fmt.Errorf("stage %q has kind %q layer %d", s.Name, s.Kind, s.Layer)
		}
	case s.Name == "linear1" || s.Name == "linear2":
		if s.Kind != "projection" || s.Layer != -1 {
			return fmt.Errorf("stage %q has kind %q layer %d", s.Name, s.Kind, s.Layer)
		}
	default:
		var prefix string
		switch {
		case strings.HasPrefix(s.Name, "conv"):
			prefix = "conv"
			if s.Kind != "convolution" {
				return fmt.Errorf("stage %q has kind %q", s.Name, s.Kind)
			}
		case strings.HasPrefix(s.Name, "flstm"):
			prefix = "flstm"
			if s.Kind != "recurrent" {
				return fmt.Errorf("stage %q has kind %q", s.Name, s.Kind)
			}
		default:
			return fmt.Errorf("stage %q has no known kind", s.Name)
		}
		if s.Name != fmt.Sprintf("%s%d", prefix, s.Layer) {
			return fmt.Errorf("stage %q has layer index %d", s.Name, s.Layer)
		}
	}
	return nil
}

// Compare proves two admitted captures of the same declared case are
// byte-identical: every stage payload, every raw parent backing dump, the stack
// digests and the scores digest. A capture that is not reproducible cannot be
// differential evidence, so this is the determinism gate.
func Compare(a, b string) (int, error) {
	ca, err := Import(a)
	if err != nil {
		return 0, fmt.Errorf("capture %s: %w", a, err)
	}
	cb, err := Import(b)
	if err != nil {
		return 0, fmt.Errorf("capture %s: %w", b, err)
	}
	if ca.Case != cb.Case || ca.Batch != cb.Batch || ca.Chunk != cb.Chunk ||
		ca.ModelDir != cb.ModelDir || ca.SourceCommit != cb.SourceCommit {
		return 0, fmt.Errorf("captures declare different cases")
	}
	if ca.StageList() != cb.StageList() {
		return 0, fmt.Errorf("captures have different stage lists")
	}
	for i, sa := range ca.Stages {
		sb := cb.Stages[i]
		if sa.DataSHA != sb.DataSHA {
			return 0, fmt.Errorf("stage %q payload differs between captures", sa.Name)
		}
		if sa.BackingSHA != sb.BackingSHA {
			return 0, fmt.Errorf("stage %q backing dump differs between captures", sa.Name)
		}
		if sa.Shape != nil && fmt.Sprint(sa.Shape) != fmt.Sprint(sb.Shape) {
			return 0, fmt.Errorf("stage %q shape differs between captures", sa.Name)
		}
		if sa.OffsetBytes != sb.OffsetBytes || sa.Bytes != sb.Bytes {
			return 0, fmt.Errorf("stage %q interval differs between captures", sa.Name)
		}
	}
	for _, key := range []string{"convs", "rnns", "linear1", "scores"} {
		if ca.StackSHA[key] != cb.StackSHA[key] {
			return 0, fmt.Errorf("untouched %s stack digest differs between captures", key)
		}
	}
	if ca.ScoresSHA != cb.ScoresSHA {
		return 0, fmt.Errorf("scores digest differs between captures")
	}
	return len(ca.Stages), nil
}

// Stage returns a verified stage by name.
func (c Capture) Stage(name string) (Stage, bool) {
	for _, s := range c.Stages {
		if s.Name == name {
			return s, true
		}
	}
	return Stage{}, false
}

// RawPayload reads a verified stage payload exactly as the graph produced it.
func (c Capture) RawPayload(dir, name string) ([]byte, error) {
	s, ok := c.Stage(name)
	if !ok {
		return nil, fmt.Errorf("capture has no stage %q", name)
	}
	return os.ReadFile(filepath.Join(dir, s.DataFile))
}

// Payload reads a verified F16 stage payload as binary16 bits.
func (c Capture) Payload(dir, name string) ([]uint16, error) {
	s, ok := c.Stage(name)
	if !ok {
		return nil, fmt.Errorf("capture has no stage %q", name)
	}
	if s.DType != "Half" {
		return nil, fmt.Errorf("stage %q is %s, not Half", name, s.DType)
	}
	body, err := os.ReadFile(filepath.Join(dir, s.DataFile))
	if err != nil {
		return nil, err
	}
	return DecodeHalfLE(body)
}

// StageList renders the captured stage order for receipts.
func (c Capture) StageList() string {
	names := make([]string, 0, len(c.Stages))
	for _, s := range c.Stages {
		names = append(names, s.Name)
	}
	return strings.Join(names, ",")
}

func product(shape []int64) int64 {
	total := int64(1)
	for _, d := range shape {
		total *= d
	}
	return total
}

func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func bytesSHA(b []byte) (string, error) {
	h := sha256.New()
	if _, err := h.Write(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// SortedKeys is a deterministic key order for receipt maps.
func SortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
