package supgpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// Import requires independently retained prepared/build manifest hashes. It
// observes saved files only. No native backend is opened and no child runs.
func Import(dir, preparedPath, preparedSHA, buildPath, buildSHA, runtimePath, runtimeSHA string) (Capture, error) {
	var c Capture
	p, err := ReadPrepared(preparedPath, preparedSHA)
	if err != nil {
		return c, err
	}
	var b Build
	if err = readPinnedJSON(buildPath, buildSHA, &b); err != nil {
		return c, err
	}
	if err = verifyBuild(b, preparedSHA); err != nil {
		return c, err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "capture.json"))
	if err != nil {
		return c, err
	}
	if len(raw) > 64<<20 {
		return c, fmt.Errorf("capture manifest exceeds 64 MiB")
	}
	if err = decodeClosed(raw, &c); err != nil {
		return c, err
	}
	if c.Schema != CaptureSchema || c.PreparedSHA256 != preparedSHA || c.BuildSHA256 != buildSHA || c.DeviceUUID != p.DeviceUUID || c.UseI8 != p.UseI8 || c.UseF8 != p.UseF8 {
		return c, fmt.Errorf("capture identity/dispatch changed")
	}
	expectedRuntime, err := pin(runtimePath)
	if err != nil {
		return c, err
	}
	if expectedRuntime.SHA256 != runtimeSHA || c.RuntimePlan != expectedRuntime {
		return c, fmt.Errorf("capture runtime plan differs from external pre-model anchor")
	}
	buildPin, err := pin(buildPath)
	if err != nil {
		return c, err
	}
	runtime, err := readRuntimePlan(c.RuntimePlan, b, buildPin, false)
	if err != nil {
		return c, err
	}
	if err = verifyCapture(dir, p, c); err != nil {
		return c, err
	}
	if err = verifyCaptureCommands(dir, p, b, c, runtime); err != nil {
		return c, err
	}
	if err = requireCaptureRuntime(c.Runtime, runtime.Policy); err != nil {
		return c, err
	}
	return c, nil
}

func verifyCapture(dir string, p Prepared, c Capture) error {
	for _, v := range []Pin{c.Reference, c.Staged} {
		if err := verifyPin(dir, v); err != nil {
			return err
		}
	}
	wantBytes := int64(p.Geometry.Batch) * int64(p.Geometry.EncoderT) * 2 * 4096 * 2
	if c.Reference.Bytes != wantBytes || c.Reference.SHA256 != c.Staged.SHA256 || c.Reference.Bytes != c.Staged.Bytes {
		return fmt.Errorf("untouched and observed scores differ")
	}
	registry := eventRegistry(p.Geometry)
	if len(c.Events) != len(registry) {
		return fmt.Errorf("capture has %d events, want %d", len(c.Events), len(registry))
	}
	for i, e := range c.Events {
		s := registry[i]
		if e.Name != s.Name || e.Layer != s.Layer || len(e.Tensors) != len(s.Tensors) {
			return fmt.Errorf("event %d identity/membership changed", i)
		}
		if !reflect.DeepEqual(e.Descriptors, descriptorRegistry(e.Name, p.Geometry)) {
			return fmt.Errorf("%s native Koi descriptor changed", e.Name)
		}
		for j, t := range e.Tensors {
			if err := verifyTensor(dir, t, s.Tensors[j]); err != nil {
				return fmt.Errorf("%s/%s: %w", e.Name, t.Name, err)
			}
		}
		if strings.HasSuffix(e.Name, ".cache.after_remove_bits") {
			if err := verifyCacheTransition(dir, c.Events[i-1], e); err != nil {
				return err
			}
		}
	}
	byName := map[string]Event{}
	for _, e := range c.Events {
		byName[e.Name] = e
	}
	for layer := 0; layer < 18; layer++ {
		enter := byName[fmt.Sprintf("encoder.%02d.enter", layer)]
		previous := byName["encoder.input"]
		if layer > 0 {
			previous = byName[fmt.Sprintf("encoder.%02d.norm2", layer-1)]
		}
		for j := 0; j < 3; j++ {
			if previous.Tensors[j].Data.SHA256 != enter.Tensors[j].Data.SHA256 || previous.Tensors[j].Data.Bytes != enter.Tensors[j].Data.Bytes {
				return fmt.Errorf("encoder %d activation/reciprocal input handoff changed", layer)
			}
		}
	}
	// Every input bit and every final output bit belongs to the pinned run.
	if c.Events[0].Tensors[0].Data.SHA256 != p.Input.SHA256 || c.Events[len(c.Events)-1].Tensors[0].Data.SHA256 != c.Staged.SHA256 {
		return fmt.Errorf("input/final score handoff changed")
	}
	boundaryNames := []string{"conv4", "encoder.output", "upsample", "scores"}
	if len(c.Boundaries) != len(boundaryNames) {
		return fmt.Errorf("missing independent reference boundaries")
	}
	for i, b := range c.Boundaries {
		if b.Name != boundaryNames[i] {
			return fmt.Errorf("reference boundary order changed")
		}
		if err := verifyPin(dir, b.Data); err != nil {
			return err
		}
		found := false
		for _, e := range c.Events {
			if e.Name == b.Name {
				found = true
				if b.Data.SHA256 != e.Tensors[0].Data.SHA256 || b.Data.Bytes != e.Tensors[0].Data.Bytes {
					return fmt.Errorf("independent %s boundary differs", b.Name)
				}
			}
		}
		if !found {
			return fmt.Errorf("missing observed boundary %s", b.Name)
		}
	}
	// Fresh zero counters are a source invariant, independent of observed kernel
	// results. Mutated counters cannot enter a later consumer as valid operands.
	for _, e := range c.Events {
		if strings.HasSuffix(e.Name, ".enter") {
			body, err := os.ReadFile(filepath.Join(dir, e.Tensors[3].Data.Path))
			if err != nil {
				return err
			}
			if !bytes.Equal(body, make([]byte, 16)) {
				return fmt.Errorf("%s counter is not four fresh I32 zeros", e.Name)
			}
		}
	}
	if len(c.Runtime) == 0 || len(c.RuntimeMaps) != 2 || len(c.Commands) != 2 {
		return fmt.Errorf("missing runtime or two-process execution closure")
	}
	for _, v := range append(append([]Pin{}, c.Runtime...), c.RuntimeMaps...) {
		if err := verifyPin("", v); err != nil {
			return err
		}
	}
	for _, v := range c.RuntimeMaps {
		if err := verifyPin("", v); err != nil {
			return err
		}
	}

	for _, cmd := range c.Commands {
		if err := verifyCommand(cmd); err != nil {
			return err
		}
		raw, err := os.ReadFile(cmd.Stdout)
		if err != nil {
			return err
		}
		if !bytes.Contains(raw, []byte("device_uuid="+p.DeviceUUID+" driver=13040 runtime=13040 compute=12.0 sms=36\n")) {
			return fmt.Errorf("missing actual capture device/runtime observation")
		}
	}
	return nil
}

func verifyCaptureCommands(dir string, p Prepared, b Build, c Capture, runtime RuntimePlan) error {
	if err := verifyCaptureGuardFiles(dir, c, runtime); err != nil {
		return err
	}
	const cwd = "/home/jake/Projects/nanopore_basecaller_go"
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	for i, v := range []struct {
		mode   string
		binary Pin
	}{{"reference", b.Reference}, {"observer", b.Observer}} {
		cmd := c.Commands[i]
		args := append([]string{}, cmd.Args...)
		if len(args) != 8 {
			return fmt.Errorf("capture argv length changed")
		}
		if !filepath.IsAbs(args[4]) {
			args[4] = filepath.Join(cwd, args[4])
		}
		want := []string{v.binary.Path, v.mode, filepath.Dir(p.Config.Path), p.Input.Path, filepath.Join(abs, v.mode), fmt.Sprint(p.Geometry.Batch), fmt.Sprint(p.Geometry.Chunk), "capture"}
		var guard GuardReceipt
		if err := readPinnedJSON(c.Guards[i].Path, c.Guards[i].SHA256, &guard); err != nil {
			return err
		}
		env := append([]string{"SUPGPU_COMMAND_CWD=" + cwd, "SUPGPU_GATE_FDS=3,4,5,6"}, modelEnvironment(runtime.Policy)...)
		env = append(env, fmt.Sprintf("SUPGPU_TARGET_PID=%d", guard.PID))
		if !reflect.DeepEqual(args, want) || !reflect.DeepEqual(cmd.Environment, env) {
			return fmt.Errorf("capture argv/environment/cwd differs from frozen case")
		}
		if c.RuntimeMaps[i].Path != filepath.Join(abs, v.mode, "runtime.maps") {
			return fmt.Errorf("runtime map binding changed")
		}
		found := false
		for _, runtime := range c.Runtime {
			if runtime == v.binary {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("actual executable is absent from runtime closure")
		}
	}
	return nil
}

func verifyCacheTransition(dir string, before, after Event) error {
	if strings.TrimSuffix(before.Name, "before_remove_bits") != strings.TrimSuffix(after.Name, "after_remove_bits") || len(before.Tensors) != len(after.Tensors) {
		return fmt.Errorf("cache before/after identity changed")
	}
	rounded := map[string]bool{"qkv_half": true, "proj_weight": true, "norm1_weight": true, "norm2_weight": true, "fc1_half": true}
	for i, post := range after.Tensors {
		pre := before.Tensors[i]
		if pre.Name != post.Name {
			return fmt.Errorf("cache binding changed")
		}
		a, err := os.ReadFile(filepath.Join(dir, pre.Data.Path))
		if err != nil {
			return err
		}
		b, err := os.ReadFile(filepath.Join(dir, post.Data.Path))
		if err != nil {
			return err
		}
		if len(a) != len(b) {
			return fmt.Errorf("cache size changed")
		}
		if rounded[pre.Name] {
			for j := 0; j < len(a); j += 2 {
				want := (binary.LittleEndian.Uint16(a[j:]) + 8) & 0xfff0
				if binary.LittleEndian.Uint16(b[j:]) != want {
					return fmt.Errorf("%s %s source remove_bits relation failed", after.Name, pre.Name)
				}
			}
		} else if !bytes.Equal(a, b) {
			return fmt.Errorf("unrounded cache %s changed across remove_bits", pre.Name)
		}
	}
	return nil
}

func verifyTensor(dir string, t Tensor, s tensorSpec) error {
	if t.Name != s.Name || t.DType != s.DType || !reflect.DeepEqual(t.Shape, s.Shape) || !reflect.DeepEqual(t.Tags, s.Tags) || t.ScaleTag != s.ScaleTag {
		return fmt.Errorf("name/dtype/shape/tags changed")
	}
	width := int64(0)
	switch t.DType {
	case "Half":
		width = 2
	case "Char", "Float8_e4m3fn":
		width = 1
	case "Int", "Float":
		width = 4
	default:
		return fmt.Errorf("unsupported dtype")
	}
	if len(t.Shape) == 0 || len(t.Shape) != len(t.Strides) {
		return fmt.Errorf("invalid tensor rank")
	}
	elems := int64(1)
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if t.Shape[i] <= 0 || t.Strides[i] != elems || elems > math.MaxInt64/t.Shape[i] {
			return fmt.Errorf("source shape/contiguous stride changed")
		}
		elems *= t.Shape[i]
	}
	if elems > math.MaxInt64/width || t.Data.Bytes != elems*width || t.OffsetBytes < 0 || t.OffsetBytes%width != 0 || t.Backing.Bytes < t.Data.Bytes || t.OffsetBytes > t.Backing.Bytes-t.Data.Bytes {
		return fmt.Errorf("tensor escapes complete allocation")
	}
	if len(t.Regions) != 1 || t.Regions[0] != (Range{0, t.Backing.Bytes, "initialized", "fixed-source-full-allocation/v1"}) {
		return fmt.Errorf("prospective initialized allocation rule changed")
	}
	if err := verifyPin(dir, t.Data); err != nil {
		return err
	}
	if err := verifyPin(dir, t.Backing); err != nil {
		return err
	}
	data, err := os.ReadFile(filepath.Join(dir, t.Data.Path))
	if err != nil {
		return err
	}
	back, err := os.ReadFile(filepath.Join(dir, t.Backing.Path))
	if err != nil {
		return err
	}
	if !bytes.Equal(data, back[t.OffsetBytes:t.OffsetBytes+t.Data.Bytes]) {
		return fmt.Errorf("logical tensor differs from retained allocation")
	}
	if t.DType == "Float" && (t.Name == "scale" || t.Name == "qkv_scale") {
		for i := 0; i < len(data); i += 4 {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[i:]))
			if !(v > 0) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("reciprocal scale is not positive finite")
			}
		}
	}
	return nil
}

func verifyCommand(c runner.CommandReceipt) error {
	start, e1 := time.Parse(time.RFC3339Nano, c.Started)
	end, e2 := time.Parse(time.RFC3339Nano, c.Finished)
	if len(c.Args) == 0 || c.Environment == nil || c.Exit != 0 || e1 != nil || e2 != nil || end.Before(start) {
		return fmt.Errorf("missing command argv/environment/clock/exit")
	}
	for _, v := range []struct{ path, sha string }{{c.Stdout, c.StdoutSHA}, {c.Stderr, c.StderrSHA}} {
		p, err := pin(v.path)
		if err != nil {
			return err
		}
		if p.SHA256 != v.sha {
			return fmt.Errorf("raw command log changed")
		}
	}
	return nil
}

func verifyBuild(b Build, preparedSHA string) error {
	if b.Schema != "nanopore-supgpu-build/v2" || b.Status != "built" || b.PreparedSHA256 != preparedSHA || len(b.Inputs) == 0 || len(b.Depfiles) != nativeCompileCount || len(b.LinkMaps) != nativeLinkCount || len(b.Commands) != nativeCommandCount || len(b.PreDepfiles) != nativeCompileCount || len(b.Dependencies) != nativeCompileCount || len(b.Objects) != nativeCompileCount {
		return fmt.Errorf("incomplete SUP build closure")
	}
	if err := verifyPin("", b.Plan); err != nil {
		return err
	}
	var plan BuildPlan
	if err := readPinnedJSON(b.Plan.Path, b.Plan.SHA256, &plan); err != nil {
		return err
	}
	if err := verifyPlan(plan, false); err != nil {
		return err
	}
	if len(b.PreflightCommands) != len(plan.Queries)+nativeCompileCount {
		return fmt.Errorf("missing precompile command closure")
	}
	for i, cmd := range b.PreflightCommands {
		dir := plan.Commands[0].Directory
		var args []string
		if i < len(plan.Queries) {
			args = plan.Queries[i].Argv
		} else {
			args = plan.Preprocess[i-len(plan.Queries)].Argv
		}
		if !reflect.DeepEqual(cmd.Args, args) || !reflect.DeepEqual(cmd.Environment, append([]string{"SUPGPU_COMMAND_CWD=" + dir}, nativeEnvironment()...)) {
			return fmt.Errorf("precompile argv/environment changed")
		}
		if err := verifyCommand(cmd); err != nil {
			return err
		}
		if i < len(plan.Queries) {
			raw, err := os.ReadFile(cmd.Stdout)
			if err != nil {
				return err
			}
			if err = sameQueryResult(string(raw), plan.Queries[i], dir); err != nil {
				return err
			}
		}
	}
	if plan.Prepared.SHA256 != preparedSHA || !reflect.DeepEqual(plan.Compiler, b.Compiler) || plan.Reference != b.Reference.Path || plan.Observer != b.Observer.Path || plan.Audit != b.Audit.Path || len(plan.Commands) != len(b.Commands) {
		return fmt.Errorf("build differs from frozen plan")
	}
	for i, cmd := range b.Commands {
		if !reflect.DeepEqual(cmd.Args, plan.Commands[i].Argv) || !reflect.DeepEqual(cmd.Environment, append([]string{"SUPGPU_COMMAND_CWD=" + plan.Commands[i].Directory}, nativeEnvironment()...)) {
			return fmt.Errorf("build argv/cwd changed")
		}
	}
	all := append([]Pin{b.Compiler, b.Reference, b.Observer, b.Audit}, b.Inputs...)
	all = append(all, b.Depfiles...)
	all = append(all, b.LinkMaps...)
	all = append(all, b.PreDepfiles...)
	all = append(all, b.Objects...)
	all = append(all, b.BeforeInputs)
	for _, v := range all {
		if err := verifyPin("", v); err != nil {
			return err
		}
	}
	for _, c := range b.Commands {
		if err := verifyCommand(c); err != nil {
			return err
		}
	}
	var frozen []Pin
	if err := readPinnedJSON(b.BeforeInputs.Path, b.BeforeInputs.SHA256, &frozen); err != nil {
		return err
	}
	if !reflect.DeepEqual(frozen, b.Inputs) || b.BeforeInputs.Path != filepath.Join(filepath.Dir(b.Plan.Path), "before-compile-inputs.json") {
		return fmt.Errorf("build inputs differ from before-compile freeze")
	}
	inputs := map[string]Pin{}
	for _, v := range b.Inputs {
		if _, ok := inputs[v.Path]; ok {
			return fmt.Errorf("duplicate build pin")
		}
		inputs[v.Path] = v
	}
	for _, v := range append(append([]Pin{}, plan.Inputs...), plan.Generated, plan.Prepared) {
		if inputs[v.Path] != v {
			return fmt.Errorf("missing plan input %s", v.Path)
		}
	}
	for i, v := range b.Depfiles {
		if v.Path != plan.Commands[i].Depfile {
			return fmt.Errorf("depfile binding changed")
		}
		if b.PreDepfiles[i].Path != plan.Preprocess[i].Depfile || b.Objects[i].Path != plan.Commands[i].Argv[len(plan.Commands[i].Argv)-1] {
			return fmt.Errorf("precompile/object binding changed")
		}
		before, err := sortedDependencies(b.PreDepfiles[i].Path, plan.Commands[i].Directory)
		if err != nil {
			return err
		}
		actual, err := sortedDependencies(v.Path, plan.Commands[i].Directory)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(before, b.Dependencies[i]) || !reflect.DeepEqual(before, actual) {
			return fmt.Errorf("actual compilation used a changed dependency set")
		}
		raw, err := os.ReadFile(v.Path)
		if err != nil {
			return err
		}
		target, _, ok := strings.Cut(strings.ReplaceAll(string(raw), "\\\n", ""), ":")
		if !ok || strings.ReplaceAll(strings.TrimSpace(target), "\\ ", " ") != plan.Commands[i].Argv[len(plan.Commands[i].Argv)-1] {
			return fmt.Errorf("compiler depfile target differs from exact object")
		}
		deps, err := DepfileInputs(v.Path)
		if err != nil {
			return err
		}
		for _, path := range deps {
			if !filepath.IsAbs(path) {
				path = filepath.Join(plan.Commands[i].Directory, path)
			}
			if _, ok := inputs[path]; !ok {
				return fmt.Errorf("unclosed compiler header dependency %s", path)
			}
		}
	}
	for i := nativeCompileCount; i < nativeCommandCount; i++ {
		trace, err := os.ReadFile(b.Commands[i].Stdout)
		if err != nil {
			return err
		}
		if err = verifyLinkInputs(string(trace), plan.Commands[i].Directory, b.Inputs, b.Objects, false); err != nil {
			return err
		}
		body, err := os.ReadFile(b.LinkMaps[i-nativeCompileCount].Path)
		if err != nil {
			return err
		}
		if b.LinkMaps[i-nativeCompileCount].Path != plan.Commands[i].LinkMap {
			return fmt.Errorf("link map binding changed")
		}
		if err = verifyLinkInputs(string(body), plan.Commands[i].Directory, b.Inputs, b.Objects, true); err != nil {
			return err
		}
	}
	if err := verifyAuditELF(b.Audit); err != nil {
		return err
	}
	// The two separate executables must be observable in the actual link maps.
	ref, err := os.ReadFile(b.LinkMaps[0].Path)
	if err != nil {
		return err
	}
	obs, err := os.ReadFile(b.LinkMaps[1].Path)
	if err != nil {
		return err
	}
	if bytes.Contains(ref, []byte("supgpu-observed-TxModules.o")) || !bytes.Contains(ref, []byte("TxModules.cpp.o")) || !bytes.Contains(obs, []byte("supgpu-observed-TxModules.o")) || bytes.Contains(obs, []byte("libdorado_nn.a(TxModules.cpp.o)")) {
		return fmt.Errorf("untouched/observed TxModules link identities are not separate")
	}
	return nil
}
