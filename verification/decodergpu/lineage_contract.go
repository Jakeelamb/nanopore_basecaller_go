package decodergpu

import (
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
)

const lineageOutputSHA = "d296e6223866ff4b5386ab0b92932bf5330f3977dbb20abb02c07f43596aefdc"
const lineageCaptureSHA = "658c280d1d9316370d2e803978334db6964a76a06a6f7906b2013d164b0261d9"
const lineageEnvironmentError = "lineage actual environment/default cache contract differs"

type LineageLookup string

const (
	LineageNoLookup    LineageLookup = "none"
	LineageFourLookups LineageLookup = "four-original-functions"
)

// Each case has its own process and one original input. Provenance does not
// place the archive, scores, or the decoder executable in the probe link.
type LineageContract struct {
	Schema, Architecture                     string
	Archive, Member, Capture, ExpectedOutput File
	Policy                                   ImagePolicy
	Input                                    DeviceImage
	Lookup                                   LineageLookup
	// The v1 field retains the receipt envelope without its dynamic PID.
	Environment []string
}

func lineageExecEnvironment(envelope []string) ([]string, error) {
	if len(envelope) < 3 || !strings.HasPrefix(envelope[0], "SUPGPU_COMMAND_CWD=") || envelope[1] != "SUPGPU_GATE_FDS=3,4,5,6" {
		return nil, fmt.Errorf("lineage receipt environment annotations differ")
	}
	cwd := strings.TrimPrefix(envelope[0], "SUPGPU_COMMAND_CWD=")
	if !filepath.IsAbs(cwd) || filepath.Clean(cwd) != cwd {
		return nil, fmt.Errorf("lineage receipt working directory differs")
	}
	seen := map[string]bool{}
	for _, entry := range envelope[2:] {
		key, _, ok := strings.Cut(entry, "=")
		if !ok || key == "" || seen[key] || strings.HasPrefix(key, "SUPGPU_") || strings.HasPrefix(key, "CUDA_") {
			return nil, fmt.Errorf("lineage environment changes default CUDA/cache policy")
		}
		seen[key] = true
	}
	return slices.Clone(envelope[2:]), nil
}

func lineagePhases() []string {
	return []string{"pre-model", "pre-load", "post-load", "post-lookup", "post-cleanup"}
}

func (c LineageContract) Validate() error {
	if c.Schema != "nanopore-decoder-load-lineage/v1" || c.Archive.SHA256 != koiSHA || c.Member.SHA256 != "930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f" || c.Capture.SHA256 != lineageCaptureSHA || c.ExpectedOutput.SHA256 != lineageOutputSHA || c.Policy.Member != c.Member || c.Policy.Schema != "nanopore-decoder-device-images/v1" || len(c.Policy.Images) != 2 {
		return fmt.Errorf("lineage original archive/member/capture/output provenance differs")
	}
	if c.Lookup != LineageNoLookup && c.Lookup != LineageFourLookups {
		return fmt.Errorf("lineage lookup mode must be declared before execution")
	}
	if _, err := lineageExecEnvironment(c.Environment); err != nil {
		return err
	}
	want := map[string]string{"sm_120": "292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938", "sm_120a": "8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40"}
	seen := map[string]bool{}
	selected := false
	for _, image := range c.Policy.Images {
		if seen[image.Architecture] || want[image.Architecture] == "" || image.Image.SHA256 != want[image.Architecture] || !reflect.DeepEqual(image.Symbols, map[string]uint32{"back_guide_step": 21, "beam_search_step": 20, "compute_posts_step": 19, "decode_step": 18}) {
			return fmt.Errorf("lineage original image policy differs")
		}
		seen[image.Architecture] = true
		selected = selected || (image.Architecture == c.Architecture && reflect.DeepEqual(image, c.Input))
	}
	if !selected {
		return fmt.Errorf("lineage input is not the declared original architecture image")
	}
	return nil
}

type LineageAPI struct {
	Sequence, Thread, Context, ContextUID, Correlation, Operation uint64
	Module, Function                                              uint64
	Image, Options, OptionValues                                  uint64
	OptionCount                                                   uint32
	Callback, Site                                                uint32
	Result                                                        int32
	Name, Symbol                                                  string
}

type LineageModule struct {
	Sequence, Thread, Context, ModuleID, Operation uint64
	Bytes                                          uint64
	Payload                                        File
}

type LineageContext struct {
	Primary, Current            uint64
	UID                         uint64
	UUID                        string
	Major, Minor, DriverVersion int
	Environment                 []string
}

type LineageObservation struct {
	Schema, Error                                     string
	PID                                               int
	Context                                           LineageContext
	Input                                             File
	InputCopy                                         File
	InputBytes                                        uint64
	InputAddress, InputOperation                      uint64
	API                                               []LineageAPI
	Modules                                           []LineageModule
	CallbackError, LaunchCallbacks                    uint64
	LoadCalls, LookupCalls, UnloadCalls, ReleaseCalls uint64
	CleanupComplete                                   bool
}

type LineageBinding struct {
	Schema, Status, Architecture                                                          string
	Input, Output                                                                         File
	LoadOperation, ResourceOperation, ModuleHandle, ModuleID, Context, ContextUID, Thread uint64
}

func lineageLaunch(a LineageAPI) bool {
	// Observe the full driver callback domain. In addition to the pinned
	// header's launch IDs, reject a future launch spelling conservatively.
	for _, id := range []uint32{115, 116, 117, 307, 442, 477, 478, 480, 514, 515, 527, 528, 652, 653, 831, 832} {
		if a.Callback == id {
			return true
		}
	}
	for i := 0; i+6 <= len(a.Name); i++ {
		if a.Name[i:i+6] == "Launch" {
			return true
		}
	}
	return false
}

// BindLineage checks a reproduced input-to-output relationship only. It never
// changes capture002 or qualifies its numerical buffers.
func BindLineage(c LineageContract, o LineageObservation) (LineageBinding, error) {
	var result LineageBinding
	if err := c.Validate(); err != nil {
		return result, err
	}
	if o.Schema != "nanopore-decoder-load-observation/v1" || o.Error != "" || o.Input != c.Input.Image || o.InputCopy.SHA256 != c.Input.Image.SHA256 || o.InputBytes != 261192 || o.CallbackError != 0 || o.LaunchCallbacks != 0 || o.LoadCalls != 1 || o.UnloadCalls != 1 || o.ReleaseCalls != 1 || !o.CleanupComplete || len(o.Modules) != 1 {
		return result, fmt.Errorf("lineage observation counts/input/cleanup differ")
	}
	ctx := o.Context
	if o.PID <= 0 {
		return result, fmt.Errorf("lineage process identity absent")
	}
	wantEnvironment := slices.Clone(c.Environment[2:])
	gotEnvironment := slices.Clone(ctx.Environment)
	slices.Sort(wantEnvironment)
	slices.Sort(gotEnvironment)
	if !slices.Equal(wantEnvironment, gotEnvironment) {
		return result, fmt.Errorf("%s", lineageEnvironmentError)
	}
	if ctx.Primary == 0 || ctx.Current != ctx.Primary || ctx.UID == 0 || len(ctx.UUID) != 32 || ctx.Major != 12 || ctx.Minor != 0 || ctx.DriverVersion <= 0 {
		return result, fmt.Errorf("lineage actual primary context/device identity absent")
	}
	for _, b := range []byte(ctx.UUID) {
		if !((b >= '0' && b <= '9') || (b >= 'a' && b <= 'f')) {
			return result, fmt.Errorf("lineage UUID is not raw hexadecimal")
		}
	}
	if ctx.UUID == "00000000000000000000000000000000" {
		return result, fmt.Errorf("lineage UUID is empty")
	}
	pairs := map[uint64][]LineageAPI{}
	last := uint64(0)
	for _, a := range o.API {
		if a.Sequence <= last || a.Thread == 0 || a.Name == "" || a.Site > 1 || lineageLaunch(a) {
			return result, fmt.Errorf("lineage API sequence/launch evidence differs")
		}
		last = a.Sequence
		if a.Callback == 18 || a.Callback == 19 || a.Callback == 21 {
			return result, fmt.Errorf("unexpected additional module load API")
		}
		if a.Callback == 20 || a.Callback == 23 {
			if a.Operation == 0 || a.Context != ctx.Current || a.ContextUID != ctx.UID {
				return result, fmt.Errorf("lineage load/lookup context or operation absent")
			}
			pairs[a.Operation] = append(pairs[a.Operation], a)
		}
	}
	loads, lookups := [][2]LineageAPI{}, [][2]LineageAPI{}
	for token, pair := range pairs {
		if len(pair) != 2 {
			return result, fmt.Errorf("lineage operation entry/exit unavailable")
		}
		a, b := pair[0], pair[1]
		if a.Site != 0 || b.Site != 1 || a.Sequence != token || a.Callback != b.Callback || a.Thread != b.Thread || a.Context != b.Context || a.ContextUID != b.ContextUID || a.Correlation != b.Correlation || a.Correlation == 0 || b.Result != 0 || a.Name != b.Name || a.Symbol != b.Symbol {
			return result, fmt.Errorf("lineage operation pairing differs")
		}
		if a.Callback == 20 {
			if a.Name != "cuModuleLoadDataEx" || b.Module == 0 || a.Module != 0 {
				return result, fmt.Errorf("lineage load handle/API differs")
			}
			loads = append(loads, [2]LineageAPI{a, b})
		} else {
			if a.Name != "cuModuleGetFunction" || a.Module == 0 || a.Module != b.Module || b.Function == 0 {
				return result, fmt.Errorf("lineage lookup handle/API differs")
			}
			lookups = append(lookups, [2]LineageAPI{a, b})
		}
	}
	lookupCount := 0
	if c.Lookup == LineageFourLookups {
		lookupCount = 4
	}
	if len(loads) != 1 || len(lookups) != lookupCount || o.LookupCalls != uint64(lookupCount) {
		return result, fmt.Errorf("lineage declared operation count differs")
	}
	load := loads[0]
	if o.InputAddress == 0 || o.InputOperation != load[0].Operation || load[0].Image != o.InputAddress || load[1].Image != o.InputAddress || load[0].OptionCount != 0 || load[1].OptionCount != 0 || load[0].Options != 0 || load[1].Options != 0 || load[0].OptionValues != 0 || load[1].OptionValues != 0 {
		return result, fmt.Errorf("lineage copied entry bytes do not bind to the zero-option load operation")
	}
	cleanup := map[string][]LineageAPI{"unload": {}, "release": {}}
	for _, a := range o.API {
		if a.Callback == 22 {
			cleanup["unload"] = append(cleanup["unload"], a)
		}
		if a.Callback == 387 || a.Callback == 544 {
			cleanup["release"] = append(cleanup["release"], a)
		}
	}
	for kind, pair := range cleanup {
		if len(pair) != 2 || pair[0].Site != 0 || pair[1].Site != 1 || pair[0].Callback != pair[1].Callback || pair[0].Thread != load[0].Thread || pair[1].Thread != pair[0].Thread || pair[0].Correlation == 0 || pair[0].Correlation != pair[1].Correlation || pair[1].Result != 0 || pair[0].Sequence <= load[1].Sequence {
			return result, fmt.Errorf("lineage %s cleanup callbacks unavailable", kind)
		}
		if kind == "unload" && (pair[0].Module != load[1].Module || pair[1].Module != load[1].Module || pair[0].Context != ctx.Current) {
			return result, fmt.Errorf("lineage cleanup module differs")
		}
	}
	if cleanup["release"][0].Sequence <= cleanup["unload"][1].Sequence {
		return result, fmt.Errorf("lineage release precedes unload")
	}
	seen := map[string]bool{}
	for _, pair := range lookups {
		a := pair[0]
		if a.Sequence <= load[1].Sequence || a.Module != load[1].Module || a.Thread != load[0].Thread || c.Input.Symbols[a.Symbol] == 0 || seen[a.Symbol] {
			return result, fmt.Errorf("lineage lookup does not consume original load handle")
		}
		if pair[1].Sequence >= cleanup["unload"][0].Sequence {
			return result, fmt.Errorf("lineage lookup occurs after unload")
		}
		seen[a.Symbol] = true
	}
	m := o.Modules[0]
	pair := pairs[m.Operation]
	if len(pair) != 2 || m.Sequence <= pair[0].Sequence || m.Sequence >= pair[1].Sequence || m.Thread != pair[0].Thread || m.Context != pair[0].Context || m.ModuleID == 0 || m.Bytes != 266280 || m.Payload.SHA256 != c.ExpectedOutput.SHA256 {
		return result, fmt.Errorf("lineage resource is unbound/ambiguous or full output differs")
	}
	result = LineageBinding{Schema: "nanopore-decoder-load-binding/v1", Status: "reproduced_transformation_only", Architecture: c.Architecture, Input: c.Input.Image, Output: m.Payload, LoadOperation: load[0].Operation, ResourceOperation: m.Operation, ModuleHandle: load[1].Module, ModuleID: m.ModuleID, Context: ctx.Current, ContextUID: ctx.UID, Thread: m.Thread}
	return result, nil
}
