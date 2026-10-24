// Package h03 is the H03 differential consumer: it replays the whole HAC
// convolution stack through the Go caller and compares the final width-19
// stride-6 tanh convolution with the admitted H01 stage capture.
//
// It is verification-only. It is not a producer, it is not part of the caller,
// and it never runs without explicit coordinator admission. The Go caller it
// exercises is internal/model/hac_gpu_lastconv.go; the only native work is the
// named host_convolution / host_linear primitive behind the typed boundary.
package h03

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h02"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// ReceiptSchema identifies one H03 differential attempt.
const ReceiptSchema = "nanopore-hacgpu-lastconv-receipt/v1"

// FillPattern is the byte every element of the declared backing holds before the
// first kernel runs. It is declared here, before execution, so an unwritten
// region can be proven untouched instead of being compared.
const FillPattern = 0x5a

// Options names the admitted capture, the pinned model and the ABI4 backend.
type Options struct {
	Capture  string
	Model    string
	Backend  string
	Out      string
	Ordinal  int
	Admitted bool
	// DevOptions are the source developer options. The zero value is replaced by
	// the pinned defaults so a caller cannot accidentally select a path.
	DevOptions *model.LastConvDevOptions
}

// LayerPlan is the recorded plan of one convolution.
type LayerPlan struct {
	Index              int      `json:"index"`
	CIn                int64    `json:"c_in"`
	COut               int64    `json:"c_out"`
	WinLen             int64    `json:"win_len"`
	Stride             int64    `json:"stride"`
	Padding            int64    `json:"padding"`
	Activation         string   `json:"activation"`
	Layout             string   `json:"layout"`
	OutputTPadding     int64    `json:"output_t_padding"`
	TIn                int64    `json:"t_in"`
	TOut               int64    `json:"t_out"`
	OffsetBytes        int64    `json:"offset_bytes"`
	Bytes              int64    `json:"bytes"`
	AllocationShape    []int64  `json:"allocation_shape"`
	CaptureOffsetBytes int64    `json:"capture_offset_bytes"`
	CaptureShape       []int64  `json:"capture_shape"`
	CaptureLayout      string   `json:"capture_layout"`
	SourceUnwritten    [][2]int `json:"source_unwritten_t"`
	CaptureUnwritten   [][2]int `json:"capture_unwritten_t"`
}

// LastConvPlan is the recorded final-layer view.
type LastConvPlan struct {
	InputToRevLSTM     bool     `json:"is_input_to_rev_lstm"`
	InputOffsetBytes   int64    `json:"input_offset_bytes"`
	InputShape         []int64  `json:"input_shape"`
	OutputOffsetBytes  int64    `json:"output_offset_bytes"`
	OutputShape        []int64  `json:"output_allocation_shape"`
	ViewOffsetElements uint64   `json:"view_offset_elements"`
	ViewOffsetBytes    int64    `json:"view_offset_bytes"`
	ViewShape          []int64  `json:"view_shape"`
	ViewStrides        []int64  `json:"view_strides"`
	SourceUnwritten    [][2]int `json:"source_unwritten_t"`
	CaptureUnwritten   [][2]int `json:"capture_unwritten_t"`
	InNStride          int32    `json:"in_n_stride"`
	OutNStride         int32    `json:"out_n_stride"`
	OutTStride         int32    `json:"out_t_stride"`
	InType             string   `json:"in_type"`
	OutType            string   `json:"out_type"`
	AccumType          string   `json:"accum_type"`
	Activation         string   `json:"activation"`
	WrittenBytes       int64    `json:"written_bytes"`
	Aliases            []Alias  `json:"aliasing"`
}

// Alias records where the last-convolution allocation overlaps an earlier
// allocation. It is interval arithmetic from the plan, not a value comparison of
// a declared-unwritten region.
type Alias struct {
	Layer           int      `json:"earlier_layer"`
	EarlierOffset   int64    `json:"earlier_offset_bytes"`
	EarlierBytes    int64    `json:"earlier_bytes"`
	FirstSharedRow  int64    `json:"first_shared_allocation_row"`
	SharedRows      int64    `json:"shared_rows"`
	OverwrittenRows [][2]int `json:"rows_the_view_overwrites"`
	UntouchedRows   [][2]int `json:"rows_left_holding_the_earlier_output"`
	Preserved       bool     `json:"untouched_rows_still_hold_the_earlier_output"`
	Changed         bool     `json:"overwritten_rows_no_longer_hold_it"`
}

// Packing records the weight-packing differential for one layer.
type Packing struct {
	Index            int     `json:"index"`
	Layout           string  `json:"packed_layout"`
	WeightFileSHA    string  `json:"weight_file_sha256"`
	WeightShape      []int64 `json:"weight_shape"`
	BiasFileSHA      string  `json:"bias_file_sha256"`
	GoWeightSHA      string  `json:"go_packed_weight_sha256"`
	ReferenceSHA     string  `json:"reference_packed_weight_sha256"`
	GoBiasSHA        string  `json:"go_packed_bias_sha256"`
	ReferenceBiasSHA string  `json:"reference_packed_bias_sha256"`
	// Transposed records that the reference oracle emits the H02 [W*C_in, C_out]
	// layout, so H03 proves its [C_out, W*C_in] packer against the transpose of
	// the same native bytes rather than against a new native symbol.
	Transposed bool `json:"reference_compared_transposed"`
	Equal      bool `json:"go_equals_reference"`
}

// Dispatch records which primitive produced the final convolution. A fallback
// success is not fast-path support, so both attempts are named.
type Dispatch struct {
	DevOptions        string `json:"dev_options"`
	FastAttempted     bool   `json:"fast_attempted"`
	FastResult        string `json:"fast_result"`
	FallbackAttempted bool   `json:"fallback_attempted"`
	FallbackResult    string `json:"fallback_result"`
	Path              string `json:"path"`
}

// Unwritten records that the rows outside the last-convolution view are
// accounted for: proven still fill, or declared as an earlier allocation's
// output. They are never value-compared with the capture.
type Unwritten struct {
	Index        int      `json:"index"`
	Rows         [][2]int `json:"declared_unwritten_t"`
	Bytes        int64    `json:"bytes"`
	RowZeroFill  bool     `json:"row_zero_still_fill_pattern"`
	AliasedRows  [][2]int `json:"aliased_rows"`
	AliasedBytes int64    `json:"aliased_bytes"`
	StillFill    bool     `json:"all_other_rows_still_fill_pattern"`
}

// Padding records the two zeroed input bands.
type Padding struct {
	Index   int   `json:"index"`
	Batch   int64 `json:"batch"`
	Rows    int64 `json:"rows_per_band"`
	Bytes   int64 `json:"bytes_zeroed"`
	AllZero bool  `json:"every_band_element_is_zero"`
}

// Lifetime records the view-lifetime and untouched-region proof.
type Lifetime struct {
	BackingStillFill bool       `json:"untouched_backing_still_fill_pattern"`
	FillSpans        [][2]int64 `json:"fill_spans_checked"`
}

// Receipt is the admitted H03 differential record.
type Receipt struct {
	Schema              string                  `json:"schema"`
	Status              string                  `json:"status"`
	Error               string                  `json:"error,omitempty"`
	Started             string                  `json:"started_utc"`
	Finished            string                  `json:"finished_utc"`
	Attempt             string                  `json:"attempt"`
	Case                string                  `json:"case"`
	Capture             string                  `json:"capture_dir"`
	CaptureJSON         string                  `json:"capture_manifest_sha256"`
	CaptureStages       string                  `json:"capture_stage_list"`
	Model               string                  `json:"model_dir"`
	Backend             string                  `json:"backend"`
	BackendSHA          string                  `json:"backend_sha256"`
	BuildManifest       string                  `json:"backend_build_manifest"`
	BuildSHA            string                  `json:"backend_build_manifest_sha256"`
	ABI                 int                     `json:"abi_version"`
	TorchRoot           string                  `json:"torch_root"`
	Batch               int                     `json:"batch"`
	Chunk               int                     `json:"chunk"`
	Features            int64                   `json:"features"`
	Reservation         int64                   `json:"reservation_bytes"`
	FillPattern         uint8                   `json:"fill_pattern"`
	DeclaredPreparation []string                `json:"declared_preparation"`
	DevOptions          string                  `json:"dev_options"`
	Ordinal             int                     `json:"cuda_ordinal"`
	Device              string                  `json:"device_bookend_identity"`
	Environment         map[string]string       `json:"effective_runtime_environment"`
	Plan                []LayerPlan             `json:"plan"`
	LastConv            LastConvPlan            `json:"last_conv"`
	Packing             []Packing               `json:"packing"`
	Dispatch            Dispatch                `json:"dispatch"`
	Chain               []Stage                 `json:"chain"`
	Stages              []Stage                 `json:"stages"`
	InputPadding        Padding                 `json:"input_padding"`
	Unwritten           Unwritten               `json:"unwritten"`
	Lifetime            Lifetime                `json:"lifetime"`
	Inputs              map[string]string       `json:"input_sha256"`
	BackendInputs       int                     `json:"backend_build_inputs_verified"`
	BackendInputsSHA    string                  `json:"backend_build_inputs_sha256"`
	Runtime             map[string]string       `json:"runtime_sha256"`
	Commands            []runner.CommandReceipt `json:"commands"`
}

// Run replays the declared case once and returns the verified receipt.
func Run(o Options) (r Receipt, err error) {
	r = Receipt{
		Schema: ReceiptSchema, Status: "failed",
		Started: time.Now().UTC().Format(time.RFC3339Nano),
		Ordinal: o.Ordinal, FillPattern: FillPattern,
		Inputs: map[string]string{}, Runtime: map[string]string{},
		Environment: map[string]string{},
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		if r.Attempt == "" {
			return
		}
		raw, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(filepath.Join(r.Attempt, "receipt.json"), append(raw, '\n'), 0o644)
		}
		if merr != nil && err == nil {
			err = merr
		}
	}()
	if !o.Admitted {
		return r, fmt.Errorf("H03 replay requires explicit coordinator admission")
	}
	if o.Ordinal < 0 || o.Ordinal > math.MaxInt32 {
		return r, fmt.Errorf("invalid CUDA ordinal %d", o.Ordinal)
	}
	dev := model.DefaultLastConvDevOptions()
	if o.DevOptions != nil {
		dev = *o.DevOptions
	}
	r.DevOptions = dev.String()
	for _, p := range []*string{&o.Capture, &o.Model, &o.Backend, &o.Out} {
		v, aerr := filepath.Abs(*p)
		if aerr != nil {
			return r, aerr
		}
		*p = v
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}

	build, buildPath, err := verifyBackend(o.Backend)
	if err != nil {
		return r, err
	}
	r.Backend, r.BackendSHA = o.Backend, build.OutputSHA
	r.BuildManifest, r.BuildSHA = buildPath, build.ManifestSHA
	r.ABI, r.TorchRoot = build.ABI, build.Torch
	paths := make([]string, 0, len(build.Inputs))
	for path := range build.Inputs {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	var pins strings.Builder
	for _, path := range paths {
		got, herr := fileSHA(path)
		if herr != nil {
			return r, herr
		}
		if got != build.Inputs[path] {
			return r, fmt.Errorf("declared backend build pin changed: %s", path)
		}
		fmt.Fprintf(&pins, "%s %s\n", path, got)
	}
	r.BackendInputs = len(paths)
	r.BackendInputsSHA = hashString(pins.String())

	capture, err := hacgpu.Import(o.Capture)
	if err != nil {
		return r, fmt.Errorf("capture %s: %w", o.Capture, err)
	}
	r.Capture = o.Capture
	r.Case = capture.Case
	r.CaptureStages = capture.StageList()
	r.Batch, r.Chunk, r.Features = capture.Batch, capture.Chunk, int64(capture.Features)
	r.Reservation = capture.Reservation
	r.CaptureJSON, err = fileSHA(filepath.Join(o.Capture, "capture.json"))
	if err != nil {
		return r, err
	}
	r.Inputs[filepath.Join(o.Capture, "capture.json")] = r.CaptureJSON
	for _, s := range capture.Stages {
		for _, name := range []string{s.DataFile, s.BackingFile} {
			if name == "" {
				continue
			}
			path := filepath.Join(o.Capture, name)
			if r.Inputs[path], err = fileSHA(path); err != nil {
				return r, err
			}
		}
	}

	convs, err := h02.ModelConvs(o.Model)
	if err != nil {
		return r, err
	}
	if len(convs) != capture.Convs {
		return r, fmt.Errorf("model declares %d convolutions but the capture has %d", len(convs), capture.Convs)
	}
	params, err := h02.ConvParams(convs)
	if err != nil {
		return r, err
	}
	stack, err := model.PlanHACSmallConvs(int64(capture.Batch), int64(capture.Chunk), int64(capture.Features), capture.Reservation, params)
	if err != nil {
		return r, err
	}
	lastName := fmt.Sprintf("conv%d", len(convs)-1)
	convLast, ok := capture.Stage(lastName)
	if !ok {
		return r, fmt.Errorf("capture has no %s stage", lastName)
	}
	last, err := model.PlanHACLastConv(stack, convLast.InputToRevLSTM)
	if err != nil {
		return r, err
	}
	if r.Plan, err = recordPlan(stack, last, capture); err != nil {
		return r, err
	}
	if r.LastConv, err = recordLastConv(stack, last, convLast); err != nil {
		return r, err
	}
	r.Model = o.Model

	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()

	dir, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r.Attempt = dir
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		rec, rerr := runner.Execute(dir, label, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}, args...)
		r.Commands = append(r.Commands, rec)
		return rec, rerr
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		value, present := os.LookupEnv(key)
		r.Environment[key] = value
		r.Environment[key+".present"] = strconv.FormatBool(present)
	}
	r.Environment["DORADO_MATMUL_FP16_MODE"] = "CUBLAS"
	if _, err = execute("before-gpu-state", "nvidia-smi",
		"--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu",
		"--format=csv,noheader,nounits"); err != nil {
		return r, err
	}

	if err = replay(&r, stack, last, capture, dev); err != nil {
		return r, err
	}

	r.Runtime, err = runtimePins()
	if err != nil {
		return r, err
	}

	state, err := execute("after-gpu-state", "nvidia-smi",
		"--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu",
		"--format=csv,noheader,nounits")
	if err != nil {
		return r, err
	}
	body, err := os.ReadFile(state.Stdout)
	if err != nil {
		return r, err
	}
	r.Device = strings.TrimSpace(string(body))

	r.Status = "lastconv_passed"
	return r, nil
}

// recordPlan turns the Go plan into the receipt's layer table and cross-checks
// every convolution against the capture. A plan whose offset, shape or layout
// differs from the oracle is a rejection, not a warning.
func recordPlan(stack *model.HACSmallConvPlan, last *model.HACLastConvPlan, capture hacgpu.Capture) ([]LayerPlan, error) {
	out := make([]LayerPlan, 0, len(stack.Layers))
	for i, l := range stack.Layers {
		stage, ok := capture.Stage(fmt.Sprintf("conv%d", i))
		if !ok {
			return nil, fmt.Errorf("capture has no conv%d stage", i)
		}
		var shape []int64
		switch l.Params.OutputLayout {
		case model.GPUStorageNTC:
			shape = []int64{l.Geometry.N, l.Geometry.Rows, l.Geometry.C}
		case model.GPUStorageCUTLASSI8, model.GPUStorageCUTLASSF16:
			shape = []int64{l.Geometry.Rows, l.Geometry.N, l.Geometry.C}
		default:
			return nil, fmt.Errorf("conv%d plan layout %s has no declared captured shape", i, l.Params.OutputLayout)
		}
		rec := LayerPlan{
			Index: l.Index, CIn: l.Params.CIn, COut: l.Params.COut, WinLen: l.Params.WinLen,
			Stride: l.Params.Stride, Padding: l.Params.Padding(),
			Activation: l.Params.Activation.String(), Layout: l.Params.OutputLayout.String(),
			OutputTPadding: l.Params.OutputTPadding, TIn: l.TIn, TOut: l.TOut,
			OffsetBytes: l.OffsetBytes, Bytes: l.Geometry.Bytes, AllocationShape: shape,
			CaptureOffsetBytes: stage.OffsetBytes, CaptureShape: stage.Shape, CaptureLayout: stage.Layout,
		}
		if i == len(stack.Layers)-1 {
			rec.SourceUnwritten = last.UnwrittenRows()
			rec.CaptureUnwritten = stage.UnwrittenT
		}
		if rec.OffsetBytes != rec.CaptureOffsetBytes {
			return nil, fmt.Errorf("conv%d plan offset %d differs from the captured %d", i, rec.OffsetBytes, rec.CaptureOffsetBytes)
		}
		if !sameShape(shape, stage.Shape) {
			return nil, fmt.Errorf("conv%d plan allocation %v does not describe the captured %v", i, shape, stage.Shape)
		}
		if rec.Layout != stage.Layout {
			return nil, fmt.Errorf("conv%d plan layout %s differs from the captured %s", i, rec.Layout, stage.Layout)
		}
		if rec.Bytes != stage.Bytes {
			return nil, fmt.Errorf("conv%d plan extent %d differs from the captured %d", i, rec.Bytes, stage.Bytes)
		}
		if i == len(stack.Layers)-1 && !sameUnwrittenRows(rec.SourceUnwritten, stage.UnwrittenT) {
			return nil, fmt.Errorf("conv%d source unwritten rows %v differ from the captured %v", i, rec.SourceUnwritten, stage.UnwrittenT)
		}
		out = append(out, rec)
	}
	return out, nil
}

// recordLastConv records the view arithmetic and every allocation overlap. Both
// are derived from the plan before execution.
func recordLastConv(stack *model.HACSmallConvPlan, last *model.HACLastConvPlan, capture hacgpu.Stage) (LastConvPlan, error) {
	g, err := last.Geometry(model.DefaultLastConvDevOptions())
	if err != nil {
		return LastConvPlan{}, err
	}
	elem := int64(1)
	if last.Output.Dtype == model.GPUDtypeHalf {
		elem = 2
	}
	rowBytes := last.Output.N * last.Output.C * elem
	rec := LastConvPlan{
		InputToRevLSTM:     last.InputToRevLSTM,
		InputOffsetBytes:   last.InputOffsetBytes,
		InputShape:         []int64{last.Input.N, last.Input.Rows, last.Input.C},
		OutputOffsetBytes:  last.OutputOffsetBytes,
		OutputShape:        []int64{last.Output.Rows, last.Output.N, last.Output.C},
		ViewOffsetElements: last.OutputViewOffsetElements(),
		ViewOffsetBytes:    int64(last.OutputViewOffsetElements()) * elem,
		ViewShape:          last.OutputViewShape(),
		ViewStrides:        last.OutputViewStrides(),
		SourceUnwritten:    last.UnwrittenRows(),
		CaptureUnwritten:   capture.UnwrittenT,
		InNStride:          g.InNStride, OutNStride: g.OutNStride, OutTStride: g.OutTStride,
		InType: g.InType.String(), OutType: g.OutType.String(), AccumType: g.AccumType.String(),
		Activation:   g.Activation.String(),
		WrittenBytes: last.Layer.TOut * last.Output.N * last.Output.C * elem,
	}
	if rowBytes <= 0 {
		return rec, fmt.Errorf("last-convolution allocation row is %d bytes", rowBytes)
	}
	if !sameUnwrittenRows(rec.SourceUnwritten, capture.UnwrittenT) {
		return rec, fmt.Errorf("source unwritten rows %v differ from the captured %v", rec.SourceUnwritten, capture.UnwrittenT)
	}
	outStart := last.OutputOffsetBytes
	outEnd := outStart + last.Output.Bytes
	writtenStart := last.OutputViewOffsetElements() / uint64(rowBytes)
	writtenEnd := writtenStart + uint64(last.Layer.TOut)
	for _, earlier := range stack.Layers[:len(stack.Layers)-1] {
		eStart := earlier.OffsetBytes
		eEnd := eStart + earlier.Geometry.Bytes
		if eEnd <= outStart || outEnd <= eStart {
			continue
		}
		delta := eStart - outStart
		if delta%rowBytes != 0 || earlier.Geometry.Bytes%rowBytes != 0 {
			return rec, fmt.Errorf("conv%d allocation [%d,%d) is not row-aligned with the last allocation [%d,%d)",
				earlier.Index, eStart, eEnd, outStart, outEnd)
		}
		first := delta / rowBytes
		rows := earlier.Geometry.Bytes / rowBytes
		rec.Aliases = append(rec.Aliases, Alias{
			Layer: earlier.Index, EarlierOffset: eStart, EarlierBytes: earlier.Geometry.Bytes,
			FirstSharedRow: first, SharedRows: rows,
			OverwrittenRows: intersectRows(first, first+rows, int64(writtenStart), int64(writtenEnd)),
			UntouchedRows:   subtractRows(first, first+rows, int64(writtenStart), int64(writtenEnd)),
		})
	}
	return rec, nil
}

func intersectRows(a0, a1, b0, b1 int64) [][2]int {
	lo, hi := max64(a0, b0), min64(a1, b1)
	if lo >= hi {
		return nil
	}
	return [][2]int{{int(lo), int(hi - lo)}}
}

func subtractRows(a0, a1, b0, b1 int64) [][2]int {
	var out [][2]int
	if a0 < b0 {
		out = append(out, [2]int{int(a0), int(min64(a1, b0) - a0)})
	}
	if a1 > b1 {
		out = append(out, [2]int{int(max64(a0, b1)), int(a1 - max64(a0, b1))})
	}
	return out
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
