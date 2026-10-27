package decodergpu

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func lineageFixture() (LineageContract, LineageObservation) {
	symbols := map[string]uint32{"back_guide_step": 21, "beam_search_step": 20, "compute_posts_step": 19, "decode_step": 18}
	images := []DeviceImage{{Architecture: "sm_120", Image: File{"/original120", "292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938"}, Symbols: symbols}, {Architecture: "sm_120a", Image: File{"/original120a", "8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40"}, Symbols: symbols}}
	member := File{"/member", "930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f"}
	execEnvironment := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "LD_AUDIT=/audit.so", "LD_LIBRARY_PATH=/providers:/usr/lib", "LD_BIND_NOW=1"}
	c := LineageContract{Schema: "nanopore-decoder-load-lineage/v1", Architecture: "sm_120a", Archive: File{"/archive", koiSHA}, Member: member, Capture: File{"/capture002", lineageCaptureSHA}, ExpectedOutput: File{"/captured-output", lineageOutputSHA}, Policy: ImagePolicy{Schema: "nanopore-decoder-device-images/v1", Member: member, Images: images}, Input: images[1], Lookup: LineageFourLookups, Environment: append([]string{"SUPGPU_COMMAND_CWD=/repo", "SUPGPU_GATE_FDS=3,4,5,6"}, execEnvironment...)}
	o := LineageObservation{Schema: "nanopore-decoder-load-observation/v1", PID: 5, Context: LineageContext{Primary: 123, Current: 123, UID: 4, UUID: strings.Repeat("a", 32), Major: 12, Minor: 0, DriverVersion: 13040, Environment: slices.Clone(execEnvironment)}, Input: c.Input.Image, InputCopy: File{"/case/load-input.bin", c.Input.Image.SHA256}, InputBytes: 261192, InputAddress: 123456, InputOperation: 10, LoadCalls: 1, LookupCalls: 4, UnloadCalls: 1, ReleaseCalls: 1, CleanupComplete: true}
	o.API = []LineageAPI{{Sequence: 10, Thread: 7, Context: 123, ContextUID: 4, Correlation: 3, Operation: 10, Image: 123456, Callback: 20, Site: 0, Name: "cuModuleLoadDataEx"}, {Sequence: 14, Thread: 7, Context: 123, ContextUID: 4, Correlation: 3, Operation: 10, Image: 123456, Callback: 20, Site: 1, Name: "cuModuleLoadDataEx", Module: 99}}
	for i, name := range []string{"back_guide_step", "beam_search_step", "compute_posts_step", "decode_step"} {
		seq := uint64(20 + i*4)
		a := LineageAPI{Sequence: seq, Thread: 7, Context: 123, ContextUID: 4, Correlation: uint64(4 + i), Operation: seq, Module: 99, Callback: 23, Site: 0, Name: "cuModuleGetFunction", Symbol: name}
		b := a
		b.Sequence++
		b.Site = 1
		b.Function = uint64(200 + i)
		o.API = append(o.API, a, b)
	}
	o.API = append(o.API, LineageAPI{Sequence: 50, Thread: 7, Context: 123, ContextUID: 4, Correlation: 9, Module: 99, Callback: 22, Name: "cuModuleUnload"}, LineageAPI{Sequence: 51, Thread: 7, Context: 123, ContextUID: 4, Correlation: 9, Module: 99, Callback: 22, Site: 1, Name: "cuModuleUnload"}, LineageAPI{Sequence: 52, Thread: 7, Context: 123, ContextUID: 4, Correlation: 10, Callback: 544, Name: "cuDevicePrimaryCtxRelease_v2"}, LineageAPI{Sequence: 53, Thread: 7, Correlation: 10, Callback: 544, Site: 1, Name: "cuDevicePrimaryCtxRelease_v2"})
	o.Modules = []LineageModule{{Sequence: 12, Thread: 7, Context: 123, ModuleID: 23, Operation: 10, Bytes: 266280, Payload: File{"/case/module-0.cubin", lineageOutputSHA}}}
	return c, o
}

func TestLineageReceiptAnnotationsAndExecEnvironment(t *testing.T) {
	c, o := lineageFixture()
	before, _ := json.Marshal(c)
	exec, err := lineageExecEnvironment(c.Environment)
	if err != nil || len(c.Environment) != 10 || len(exec) != 8 || !reflect.DeepEqual(exec, o.Context.Environment) {
		t.Fatalf("receipt envelope did not yield exact exec environment: %v %v", exec, err)
	}
	if _, err = BindLineage(c, o); err != nil {
		t.Fatal(err)
	}
	exec[0] = "PATH=changed"
	after, _ := json.Marshal(c)
	if string(before) != string(after) {
		t.Fatal("environment derivation changed immutable contract bytes")
	}
	for _, kind := range []string{"missing-cwd", "relative-cwd", "unclean-cwd", "gate-fds", "swapped", "pid-annotation", "unknown-annotation", "duplicate", "cache-setting"} {
		t.Run(kind, func(t *testing.T) {
			bad, _ := lineageFixture()
			switch kind {
			case "missing-cwd":
				bad.Environment = bad.Environment[1:]
			case "relative-cwd":
				bad.Environment[0] = "SUPGPU_COMMAND_CWD=repo"
			case "unclean-cwd":
				bad.Environment[0] = "SUPGPU_COMMAND_CWD=/repo/../repo"
			case "gate-fds":
				bad.Environment[1] = "SUPGPU_GATE_FDS=7,8,9,10"
			case "swapped":
				bad.Environment[0], bad.Environment[1] = bad.Environment[1], bad.Environment[0]
			case "pid-annotation":
				bad.Environment = append(bad.Environment, "SUPGPU_TARGET_PID=5")
			case "unknown-annotation":
				bad.Environment = append(bad.Environment, "SUPGPU_OTHER=1")
			case "duplicate":
				bad.Environment = append(bad.Environment, "PATH=/usr/bin:/bin")
			case "cache-setting":
				bad.Environment = append(bad.Environment, "CUDA_CACHE_DISABLE=1")
			}
			if err := bad.Validate(); err == nil {
				t.Fatal("malformed receipt environment accepted")
			}
		})
	}
}

func TestLineageExecEnvironmentRejectsChanges(t *testing.T) {
	for _, kind := range []string{"extra", "changed-path", "missing", "duplicate", "library-path", "auditor", "cwd-annotation", "pid-annotation", "cache-disable", "force-ptx", "module-loading"} {
		t.Run(kind, func(t *testing.T) {
			c, bad := lineageFixture()
			switch kind {
			case "extra":
				bad.Context.Environment = append(bad.Context.Environment, "OTHER=1")
			case "changed-path":
				bad.Context.Environment[0] = "PATH=/bin"
			case "missing":
				bad.Context.Environment = bad.Context.Environment[:7]
			case "duplicate":
				bad.Context.Environment = append(bad.Context.Environment, bad.Context.Environment[0])
			case "library-path":
				bad.Context.Environment[6] = "LD_LIBRARY_PATH=/alternate"
			case "auditor":
				bad.Context.Environment[5] = "LD_AUDIT=/alternate.so"
			case "cwd-annotation":
				bad.Context.Environment = append(bad.Context.Environment, "SUPGPU_COMMAND_CWD=/repo")
			case "pid-annotation":
				bad.Context.Environment = append(bad.Context.Environment, "SUPGPU_TARGET_PID=5")
			case "cache-disable":
				bad.Context.Environment = append(bad.Context.Environment, "CUDA_CACHE_DISABLE=1")
			case "force-ptx":
				bad.Context.Environment = append(bad.Context.Environment, "CUDA_FORCE_PTX_JIT=1")
			case "module-loading":
				bad.Context.Environment = append(bad.Context.Environment, "CUDA_MODULE_LOADING=EAGER")
			}
			if _, err := BindLineage(c, bad); err == nil || err.Error() != lineageEnvironmentError {
				t.Fatalf("native environment change did not hit exact environment gate: %v", err)
			}
		})
	}
}

func TestLineageObservedIntervalsRequireFullOutputAndZeroLaunches(t *testing.T) {
	c, o := lineageFixture()
	got, err := BindLineage(c, o)
	if err != nil || got.Status != "reproduced_transformation_only" || got.ModuleHandle != 99 || got.ModuleID != 23 {
		t.Fatalf("positive synthetic metadata: %+v %v", got, err)
	}
	raw, _ := json.Marshal(o)
	for _, kind := range []string{"unbound-resource", "async-thread", "wrong-context", "resource-outside-interval", "ambiguous-module", "output-hash", "input-copy", "zero-launch-count", "launch-callback", "changed-load-handle", "wrong-lookup-handle", "missing-lookup", "missing-cleanup-callback", "callback-error", "UUID", "native-error"} {
		t.Run(kind, func(t *testing.T) {
			var bad LineageObservation
			if err := json.Unmarshal(raw, &bad); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "unbound-resource":
				bad.Modules[0].Operation = 0
			case "async-thread":
				bad.Modules[0].Thread = 8
			case "wrong-context":
				bad.Modules[0].Context = 124
			case "resource-outside-interval":
				bad.Modules[0].Sequence = 15
			case "ambiguous-module":
				bad.Modules = append(bad.Modules, bad.Modules[0])
			case "output-hash":
				bad.Modules[0].Payload.SHA256 = strings.Repeat("0", 64)
			case "input-copy":
				bad.InputCopy.SHA256 = strings.Repeat("0", 64)
			case "zero-launch-count":
				bad.LaunchCallbacks = 1
			case "launch-callback":
				bad.API[2].Callback = 307
				bad.API[2].Name = "cuLaunchKernel"
			case "changed-load-handle":
				bad.API[1].Module = 98
			case "wrong-lookup-handle":
				bad.API[2].Module = 98
			case "missing-lookup":
				bad.API = append(bad.API[:2], bad.API[4:]...)
			case "missing-cleanup-callback":
				bad.API = bad.API[:len(bad.API)-1]
			case "callback-error":
				bad.CallbackError = 1
			case "UUID":
				bad.Context.UUID = strings.Repeat("0", 32)
			case "native-error":
				bad.Error = "retained cleanup failure"
			}
			if _, err := BindLineage(c, bad); err == nil {
				t.Fatal("ambiguous or mismatched synthetic lineage accepted")
			}
		})
	}
}

func TestLineageLookupModeAndOriginalArchitectureAreProspective(t *testing.T) {
	c, o := lineageFixture()
	c.Lookup = LineageNoLookup
	o.LookupCalls = 0
	o.API = append(o.API[:2], o.API[10:]...)
	if _, err := BindLineage(c, o); err != nil {
		t.Fatal(err)
	}
	c.Lookup = LineageFourLookups
	if _, err := BindLineage(c, o); err == nil {
		t.Fatal("observed lookup count selected its own contract")
	}
	c, _ = lineageFixture()
	c.Architecture = "sm_120"
	if err := c.Validate(); err == nil {
		t.Fatal("architecture label replaced the exact selected input")
	}
	phases := lineagePhases()
	phases[0] = "changed"
	if lineagePhases()[0] != "pre-model" || len(lineagePhases()) != 5 {
		t.Fatal("phase list changed")
	}
}
