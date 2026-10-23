package koiquant

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

type RowClassification struct {
	Rows, FiniteInputRows, ZeroRangeRows, NonfiniteInputRows, SubnormalInputRows        int
	PositiveFiniteScaleRows, OverflowScaleRows, ZeroScaleRows, NaNScaleRows             int
	InfiniteScaleRows                                                                   int
	AdmittedRows, GoHalfScaleMismatchRows, GoReciprocalMismatchRows                     int
	NativeRangeMismatchRows, NativeWideningMismatchRows                                 int
	ReciprocalFiniteRows, ReciprocalZeroRows, ReciprocalInfinityRows, ReciprocalNaNRows int
}

type Observation struct {
	Name, Status, Error string
	Metadata            Pin
	Devices             map[string]string
	Stages              map[string]koiweights.Data
	Classification      RowClassification
}

type RangeCoverage struct {
	UniqueRanges, Rows, PaddingRows, PositiveFiniteScales, OverflowScales, OtherScales int
	GoHalfScaleMismatches, GoReciprocalMismatches                                      int
	Table                                                                              Pin
}

type Imported struct {
	Schema, Status, Started, Finished string
	Raw                               Pin
	ConsumerSources                   map[string]string
	Observations                      []Observation
	Ranges                            RangeCoverage
}

func validHash(h string) bool { b, err := hex.DecodeString(h); return err == nil && len(b) == 32 }

func validateRaw(path string, m RawCapture) error {
	if m.Schema != CaptureSchema || (m.Status != "native_completed" && m.Status != "native_observed_errors") || m.Error != "" || (m.Device != "cpu" && m.Device != "cuda") || m.Ordinal < 0 || m.Device == "cpu" && m.Ordinal != 0 || len(m.Invocation) == 0 {
		return fmt.Errorf("invalid native capture identity")
	}
	if err := validateClocks(m.Started, m.Finished); err != nil {
		return err
	}
	if m.Basic.SHA256 != BasicSHA256 || m.Build.SHA256 != BuildSHA256 {
		return fmt.Errorf("missing exact qualification pins")
	}
	build, err := qualification(m.Basic.Path, m.Build.Path)
	if err != nil {
		return err
	}
	if m.Torch != build.Torch || m.Archive != (Pin{build.Archive, build.ArchiveSHA}) || build.Inputs[m.Compiler.Path] != m.Compiler.SHA256 {
		return fmt.Errorf("raw runtime or compiler differs from accepted build")
	}
	for _, p := range []Pin{m.Preparation, m.Basic, m.Build, m.Binary, m.Compiler, m.Archive} {
		if err := validatePin(p); err != nil {
			return err
		}
	}
	var prepared Preparation
	if err := readClosed(m.Preparation.Path, &prepared); err != nil {
		return err
	}
	if !reflect.DeepEqual(prepared, m.Prepared) || prepared.Device != m.Device || prepared.Status != "prepared_native_held" || prepared.NativeExecution || prepared.OriginSHA256 != OriginSHA256 || len(prepared.Fixtures) != len(fixtureNames) || !reflect.DeepEqual(prepared.Stages, StageSpecs()) {
		return fmt.Errorf("saved preparation mismatch")
	}
	if !reflect.DeepEqual(m.ProducerSources, prepared.Sources) || len(m.ProducerSources) == 0 {
		return fmt.Errorf("missing producer source pins")
	}
	root := filepath.Dir(path)
	for name := range m.ProducerSources {
		if err := checkHash(filepath.Join(root, "producer-source", name), m.ProducerSources[name]); err != nil {
			return err
		}
	}
	if len(prepared.Generated) != 2 {
		return fmt.Errorf("missing generated pins")
	}
	for _, name := range []string{"oracle.cpp", "fixture-list.txt"} {
		if err := checkHash(filepath.Join(root, name), prepared.Generated[name]); err != nil {
			return err
		}
	}
	for i, fixture := range prepared.Fixtures {
		if fixture.Name != fixtureNames[i] || fixture.Domain != fixtureDomain(fixture.Name) {
			return fmt.Errorf("missing or reordered fixture")
		}
		if err := checkHash(filepath.Join(root, "inputs", fixture.Name+"-u16le.bin"), fixture.Input.SHA256); err != nil {
			return err
		}
		if fixture.Name == "genuine" && fixture.Input.SHA256 != GenuineSHA256 {
			return fmt.Errorf("genuine input pin changed")
		}
	}
	if len(m.Environment) != 6 {
		return fmt.Errorf("missing effective environment")
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "LD_LIBRARY_PATH", "LD_PRELOAD", "DORADO_MATMUL_FP16_MODE"} {
		v, ok := m.Environment[key]
		if !ok || (!v.Present && v.Value != "") {
			return fmt.Errorf("missing or invalid environment %s", key)
		}
	}
	if m.Environment["LD_PRELOAD"] != (EnvironmentValue{true, ""}) || m.Environment["LD_LIBRARY_PATH"] != (EnvironmentValue{true, ""}) || m.Environment["DORADO_MATMUL_FP16_MODE"] != (EnvironmentValue{true, "CUBLAS"}) {
		return fmt.Errorf("invalid clean effective runtime environment")
	}
	if m.Torch == "" || m.CUDA != "/opt/cuda" || m.GPUIdentity == "" || len(m.Dependencies) == 0 || len(m.ArchiveMembers) == 0 || len(m.Linked) == 0 || len(m.MappedBefore) == 0 || len(m.MappedAfter) == 0 {
		return fmt.Errorf("missing compiler or observed runtime closure")
	}
	seen := map[string]string{}
	for _, pins := range []map[string]string{m.Dependencies, m.Linked, m.MappedBefore, m.MappedAfter} {
		for p, h := range pins {
			if prior, ok := seen[p]; ok && prior != h {
				return fmt.Errorf("conflicting runtime pin %s", p)
			}
			seen[p] = h
			if !validHash(h) {
				return fmt.Errorf("invalid closure pin %s", p)
			}
			if expected, ok := build.Runtime[p]; ok && expected != h {
				return fmt.Errorf("actual runtime differs from qualified build %s", p)
			}
			if strings.HasPrefix(p, m.Torch+"/") {
				if err := archivedPathMatches(m.Torch, p, h, m.ArchiveMembers); err != nil {
					return err
				}
			}
		}
	}
	for p, h := range seen {
		if err := checkHash(p, h); err != nil {
			return err
		}
	}
	if err := validateObservedClosure(path, m); err != nil {
		return err
	}
	for _, basename := range []string{"libtorch_cpu.so", "libc10.so"} {
		if m.MappedAfter[filepath.Join(m.Torch, "lib", basename)] == "" {
			return fmt.Errorf("missing actual mapped %s", basename)
		}
	}
	loader := false
	for p := range m.MappedAfter {
		if strings.Contains(filepath.Base(p), "ld-linux") {
			loader = true
		}
	}
	if !loader {
		return fmt.Errorf("missing mapped dynamic loader")
	}
	for name, hash := range m.ArchiveMembers {
		if !strings.HasPrefix(name, "libtorch/") || !validHash(hash) {
			return fmt.Errorf("invalid archive member")
		}
	}
	if len(m.Commands) < 6 {
		return fmt.Errorf("missing native command evidence")
	}
	native := 0
	labels := map[string]int{}
	expectedEnv := []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}
	for _, c := range m.Commands {
		isNative := len(c.Args) == 4 && c.Args[0] == m.Binary.Path
		if err := validateReceipt(c, isNative && m.Status == "native_observed_errors"); err != nil {
			return err
		}
		if !reflect.DeepEqual(c.Environment, expectedEnv) {
			return fmt.Errorf("native command overrides differ from declared environment")
		}
		label := strings.TrimSuffix(filepath.Base(c.Stdout), ".stdout")
		if c.Stdout != filepath.Join(root, label+".stdout") || c.Stderr != filepath.Join(root, label+".stderr") {
			return fmt.Errorf("command log path differs from capture")
		}
		labels[label]++
		if isNative {
			if c.Args[1] != root || c.Args[2] != m.Device || c.Args[3] != strconv.Itoa(m.Ordinal) {
				return fmt.Errorf("native invocation mismatch")
			}
			native++
		}
	}
	if native != 1 {
		return fmt.Errorf("expected one native producer invocation")
	}
	required := []string{"before-gpu", "before-apps", "after-gpu", "after-apps", "linkage", "native"}
	if m.Device == "cpu" {
		required = append(required, "compiler-version", "compiler-cc1plus", "compiler-as", "compiler-collect2", "compiler-ld", "compiler-linkage-0", "compiler-linkage-1", "compiler-linkage-2", "compiler-linkage-3", "compiler-linkage-4", "compile")
	}
	if len(labels) != len(required) {
		return fmt.Errorf("missing or extra native command")
	}
	for _, label := range required {
		if labels[label] != 1 {
			return fmt.Errorf("missing or repeated native command %s", label)
		}
	}
	actual, err := pinTree(filepath.Join(root, "observations"))
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(actual, m.Outputs) {
		return fmt.Errorf("missing, changed, or additional raw observation")
	}
	return nil
}

func requirePaths(paths []string, pins map[string]string) error {
	seen := map[string]bool{}
	for _, path := range paths {
		seen[path] = true
		if !validHash(pins[path]) {
			return fmt.Errorf("missing observed closure pin %s", path)
		}
	}
	if len(seen) != len(pins) {
		return fmt.Errorf("extra observed closure pin")
	}
	return nil
}

func validateObservedClosure(path string, m RawCapture) error {
	root := filepath.Dir(path)
	for label, pins := range map[string]map[string]string{"before": m.MappedBefore, "after": m.MappedAfter} {
		b, err := os.ReadFile(filepath.Join(root, "observations", label+"-maps.txt"))
		if err != nil {
			return err
		}
		paths, err := mappedPaths(b)
		if err != nil {
			return err
		}
		if err = requirePaths(paths, pins); err != nil {
			return err
		}
	}
	linkage, err := os.ReadFile(filepath.Join(root, "linkage.stdout"))
	if err != nil {
		return err
	}
	paths, err := lddPaths(linkage)
	if err != nil {
		return err
	}
	if err = requirePaths(paths, m.Linked); err != nil {
		return err
	}
	if m.Device == "cuda" {
		if err = validatePin(m.BinaryOrigin); err != nil {
			return err
		}
		var prior RawCapture
		if err = readClosed(m.BinaryOrigin.Path, &prior); err != nil {
			return err
		}
		if prior.Device != "cpu" || prior.Status != "native_completed" || prior.Binary != m.Binary || prior.Compiler != m.Compiler || !reflect.DeepEqual(prior.Prepared.Generated, m.Prepared.Generated) || !reflect.DeepEqual(prior.Dependencies, m.Dependencies) {
			return fmt.Errorf("reused producer closure mismatch")
		}
	} else if m.BinaryOrigin != (Pin{}) {
		return fmt.Errorf("CPU capture has unexpected reused binary")
	}
	depRoot := root
	if m.Device == "cuda" {
		depRoot = filepath.Dir(m.BinaryOrigin.Path)
	}
	depfile := filepath.Join(depRoot, "oracle.d")
	if m.Dependencies[depfile] == "" || m.Dependencies[m.Compiler.Path] != m.Compiler.SHA256 {
		return fmt.Errorf("missing compiler or depfile pin")
	}
	b, err := os.ReadFile(depfile)
	if err != nil {
		return err
	}
	_, text, ok := strings.Cut(strings.ReplaceAll(string(b), "\\\n", ""), ":")
	if !ok || strings.Contains(text, "\\") {
		return fmt.Errorf("unsupported retained depfile")
	}
	for _, p := range strings.Fields(text) {
		absolute, err := filepath.Abs(p)
		if err != nil {
			return err
		}
		if m.Dependencies[absolute] == "" {
			return fmt.Errorf("missing depfile header pin %s", p)
		}
	}
	for _, tool := range []string{"cc1plus", "as", "collect2", "ld"} {
		found := false
		for p := range m.Dependencies {
			if filepath.Base(p) == tool {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("missing compiler tool pin %s", tool)
		}
	}
	for _, label := range []string{"before", "after"} {
		b, err := os.ReadFile(filepath.Join(root, label+"-gpu.stdout"))
		if err != nil {
			return err
		}
		if strings.TrimSpace(string(b)) != m.GPUIdentity {
			return fmt.Errorf("bookend GPU UUID or driver differs")
		}
	}
	before, err := os.ReadFile(filepath.Join(root, "observations/before-runtime.txt"))
	if err != nil {
		return err
	}
	after, err := os.ReadFile(filepath.Join(root, "observations/after-runtime.txt"))
	if err != nil {
		return err
	}
	if string(before) != string(after) || !strings.HasPrefix(string(before), fmt.Sprintf("device=%s ordinal=%d toolkit=13", m.Device, m.Ordinal)) {
		return fmt.Errorf("native device/runtime bookend differs")
	}
	for key, value := range m.Environment {
		presence := "unset"
		if value.Present {
			presence = "present"
		}
		if !strings.Contains(string(before), key+"\t"+presence+"\t"+value.Value+"\n") {
			return fmt.Errorf("native effective environment differs for %s", key)
		}
	}
	if m.Device == "cuda" {
		uuid := regexp.MustCompile(`uuid=([0-9a-f]{32})`).FindSubmatch(before)
		if len(uuid) != 2 || !strings.Contains(strings.ToLower(strings.ReplaceAll(m.GPUIdentity, "-", "")), string(uuid[1])) {
			return fmt.Errorf("native CUDA UUID differs from GPU bookend")
		}
		if !strings.Contains(string(before), fmt.Sprintf("selected=%d ", m.Ordinal)) || !strings.Contains(string(before), " runtime=13") {
			return fmt.Errorf("native CUDA runtime family or selected ordinal differs")
		}
	}
	return nil
}

func parseInts(value string) ([]int, error) {
	out := []int{}
	for _, part := range strings.Split(value, ",") {
		n, err := strconv.Atoi(part)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("invalid shape or stride %q", value)
		}
		out = append(out, n)
	}
	return out, nil
}

func importStages(root, name, device string, ordinal int, input koiweights.Data, partial bool) (Observation, error) {
	o := Observation{Name: name, Devices: map[string]string{}, Stages: map[string]koiweights.Data{}}
	dir := filepath.Join(root, "observations", name)
	metadata := filepath.Join(dir, "stages.tsv")
	var err error
	o.Metadata, err = filePin(metadata)
	if err != nil {
		return o, err
	}
	b, err := os.ReadFile(metadata)
	if err != nil {
		return o, err
	}
	specs := map[string]StageSpec{}
	for _, spec := range StageSpecs() {
		specs[spec.Name] = spec
	}
	for _, line := range strings.Split(strings.TrimSuffix(string(b), "\n"), "\n") {
		if line == "" && partial {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) != 6 {
			return o, fmt.Errorf("invalid stage metadata row")
		}
		spec, exists := specs[f[0]]
		if !exists {
			return o, fmt.Errorf("unknown native stage %s", f[0])
		}
		if _, exists = o.Stages[f[0]]; exists {
			return o, fmt.Errorf("duplicate native stage %s", f[0])
		}
		wantDevice := device
		if device == "cuda" {
			wantDevice += ":" + strconv.Itoa(ordinal)
		}
		if f[0] == "input" {
			wantDevice = "cpu"
		}
		if f[2] != wantDevice {
			return o, fmt.Errorf("observed device mismatch for %s", f[0])
		}
		shape, err := parseInts(f[3])
		if err != nil {
			return o, err
		}
		strides, err := parseInts(f[4])
		if err != nil {
			return o, err
		}
		count, err := strconv.ParseInt(f[5], 10, 64)
		if err != nil {
			return o, err
		}
		pin, err := filePin(filepath.Join(dir, f[0]+".bin"))
		if err != nil {
			return o, err
		}
		d := koiweights.Data{Path: pin.Path, DType: f[1], Endian: "little", Shape: shape, Strides: strides, Bytes: count, SHA256: pin.SHA256}
		if err = validateData(d, spec); err != nil {
			return o, err
		}
		o.Stages[f[0]], o.Devices[f[0]] = d, f[2]
	}
	if !partial {
		if err = ValidateStages(o.Stages, input); err != nil {
			return o, err
		}
	}
	for _, pair := range [][2]string{{"input", "transfer"}, {"i8tile", "untouched-i8tile"}, {"freciprocal", "untouched-freciprocal"}} {
		a, aok := o.Stages[pair[0]]
		b, bok := o.Stages[pair[1]]
		if aok && bok && a.SHA256 != b.SHA256 {
			return o, fmt.Errorf("native parity mismatch %v", pair)
		}
	}
	if d, exists := o.Stages["input"]; exists && d.SHA256 != input.SHA256 {
		return o, fmt.Errorf("native input mismatch")
	}
	if !partial {
		o.Classification, err = classifyRows(o.Stages)
	}
	return o, err
}

func classifyRows(stages map[string]koiweights.Data) (c RowClassification, err error) {
	input, err := os.ReadFile(stages["input"].Path)
	if err != nil {
		return c, err
	}
	hscale, err := os.ReadFile(stages["hscale"].Path)
	if err != nil {
		return c, err
	}
	fscale, err := os.ReadFile(stages["fscale"].Path)
	if err != nil {
		return c, err
	}
	recip, err := os.ReadFile(stages["freciprocal"].Path)
	if err != nil {
		return c, err
	}
	amax, err := os.ReadFile(stages["amax"].Path)
	if err != nil {
		return c, err
	}
	for row := 0; row < 1536; row++ {
		c.Rows++
		finite, subnormal := true, false
		var maximum uint16
		for col := 0; col < 512; col++ {
			bits := binary.LittleEndian.Uint16(input[(row*512+col)*2:]) & 0x7fff
			if bits >= 0x7c00 {
				finite = false
			}
			if bits > 0 && bits < 0x400 {
				subnormal = true
			}
			if bits > maximum {
				maximum = bits
			}
		}
		if finite {
			c.FiniteInputRows++
		} else {
			c.NonfiniteInputRows++
		}
		if subnormal {
			c.SubnormalInputRows++
		}
		if maximum == 0 {
			c.ZeroRangeRows++
		}
		observedRange := binary.LittleEndian.Uint16(amax[row*2:])
		if finite && observedRange != maximum {
			c.NativeRangeMismatchRows++
		}
		scale := binary.LittleEndian.Uint16(hscale[row*2:])
		scaleValue := signal.Float32FromHalf(scale)
		if binary.LittleEndian.Uint32(fscale[row*4:]) != math.Float32bits(scaleValue) {
			c.NativeWideningMismatchRows++
		}
		eligible := scale > 0 && scale < 0x7c00
		switch {
		case eligible:
			c.PositiveFiniteScaleRows++
		case scale&0x7fff == 0:
			c.ZeroScaleRows++
		case scale&0x7fff == 0x7c00:
			c.InfiniteScaleRows++
			if finite && maximum != 0 {
				c.OverflowScaleRows++
			}
		case scale&0x7fff > 0x7c00:
			c.NaNScaleRows++
		default:
			return c, fmt.Errorf("negative finite quantization scale")
		}
		if finite && maximum != 0 && eligible {
			c.AdmittedRows++
		}
		if finite && maximum != 0 && signal.Float16Bits(float32(128)/signal.Float32FromHalf(maximum)) != scale {
			c.GoHalfScaleMismatchRows++
		}
		bits := binary.LittleEndian.Uint32(recip[row*4:])
		if eligible && math.Float32bits(float32(1)/scaleValue) != bits {
			c.GoReciprocalMismatchRows++
		}
		switch {
		case bits&0x7fffffff == 0:
			c.ReciprocalZeroRows++
		case bits&0x7fffffff == 0x7f800000:
			c.ReciprocalInfinityRows++
		case bits&0x7fffffff > 0x7f800000:
			c.ReciprocalNaNRows++
		default:
			c.ReciprocalFiniteRows++
		}
	}
	return c, nil
}

func Import(root, rawPath, out string) (m Imported, err error) {
	root, err = filepath.Abs(root)
	if err != nil {
		return m, err
	}
	rawPath, err = filepath.Abs(rawPath)
	if err != nil {
		return m, err
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return m, err
	}
	if err = admitOutput(root, out); err != nil {
		return m, err
	}
	var raw RawCapture
	if err = readClosed(rawPath, &raw); err != nil {
		return m, err
	}
	if err = validateRaw(rawPath, raw); err != nil {
		return m, err
	}
	m = Imported{Schema: ImportSchema, Status: "observations_validated", Started: clock(), ConsumerSources: map[string]string{}, Observations: []Observation{}}
	m.Raw, err = filePin(rawPath)
	if err != nil {
		return m, err
	}
	m.ConsumerSources, err = currentSources(root)
	if err != nil {
		return m, err
	}
	b, err := os.ReadFile(filepath.Join(filepath.Dir(rawPath), "observations/outcomes.tsv"))
	if err != nil {
		return m, err
	}
	lines := strings.Split(strings.TrimSuffix(string(b), "\n"), "\n")
	if len(lines) != len(raw.Prepared.Fixtures) {
		return m, fmt.Errorf("missing native fixture outcome")
	}
	complete := true
	for i, line := range lines {
		fields := strings.Split(line, "\t")
		fixture := raw.Prepared.Fixtures[i]
		if len(fields) != 2 || fields[0] != fixture.Name || (fields[1] != "passed" && fields[1] != "error") {
			return m, fmt.Errorf("invalid native outcome")
		}
		partial := fields[1] == "error"
		o, e := importStages(filepath.Dir(rawPath), fixture.Name, raw.Device, raw.Ordinal, fixture.Input, partial)
		if e != nil {
			return m, e
		}
		o.Status = fields[1]
		if partial {
			complete = false
			b, e := os.ReadFile(filepath.Join(filepath.Dir(rawPath), "observations", fixture.Name, "error.txt"))
			if e != nil {
				return m, e
			}
			o.Error = string(b)
			if o.Error == "" {
				return m, fmt.Errorf("missing fixture exception")
			}
		}
		m.Observations = append(m.Observations, o)
	}
	if complete != (raw.Status == "native_completed") {
		return m, fmt.Errorf("native exit and fixture outcomes disagree")
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return m, err
	}
	if complete {
		m.Ranges, err = rangeCoverage(m.Observations, filepath.Join(out, "finite-ranges.tsv"))
		if err != nil {
			return m, err
		}
		m.Status = "complete_observations_validated"
	}
	m.Finished = clock()
	if err = checkSources(root, m.ConsumerSources); err != nil {
		return m, err
	}
	err = writeJSON(filepath.Join(out, "manifest.json"), m)
	return m, err
}

func rangeCoverage(observations []Observation, path string) (c RangeCoverage, err error) {
	type values struct {
		nativeRange, scale  uint16
		widened, reciprocal uint32
	}
	seen := map[uint16]values{}
	for _, o := range observations {
		if !strings.HasPrefix(o.Name, "finite-ranges-") {
			continue
		}
		input, e := os.ReadFile(o.Stages["input"].Path)
		if e != nil {
			return c, e
		}
		amax, e := os.ReadFile(o.Stages["amax"].Path)
		if e != nil {
			return c, e
		}
		scale, e := os.ReadFile(o.Stages["hscale"].Path)
		if e != nil {
			return c, e
		}
		wide, e := os.ReadFile(o.Stages["fscale"].Path)
		if e != nil {
			return c, e
		}
		recip, e := os.ReadFile(o.Stages["freciprocal"].Path)
		if e != nil {
			return c, e
		}
		for row := 0; row < 1536; row++ {
			r := binary.LittleEndian.Uint16(input[row*512*2:]) & 0x7fff
			v := values{binary.LittleEndian.Uint16(amax[row*2:]), binary.LittleEndian.Uint16(scale[row*2:]), binary.LittleEndian.Uint32(wide[row*4:]), binary.LittleEndian.Uint32(recip[row*4:])}
			if prior, ok := seen[r]; ok && prior != v {
				return c, fmt.Errorf("inconsistent repeated range %04x", r)
			}
			seen[r] = v
			c.Rows++
		}
	}
	if len(seen) != 31743 || c.Rows != 21*1536 {
		return c, fmt.Errorf("incomplete positive finite range coverage")
	}
	c.UniqueRanges, c.PaddingRows = len(seen), c.Rows-len(seen)
	var table strings.Builder
	table.WriteString("input_range_hbits\tnative_amax_bits\tnative_hscale_bits\tnative_fscale_bits\tnative_reciprocal_bits\tgo_diagnostic_hscale_bits\tgo_diagnostic_reciprocal_bits\tclassification\n")
	for bits := uint16(1); bits <= 0x7bff; bits++ {
		v, ok := seen[bits]
		if !ok {
			return c, fmt.Errorf("missing positive finite range %04x", bits)
		}
		goScale := signal.Float16Bits(float32(128) / signal.Float32FromHalf(bits))
		goRecip := math.Float32bits(float32(1) / signal.Float32FromHalf(v.scale))
		class := "other"
		switch {
		case v.scale > 0 && v.scale < 0x7c00:
			class = "positive-finite"
			c.PositiveFiniteScales++
		case v.scale == 0x7c00:
			class = "overflow"
			c.OverflowScales++
		default:
			c.OtherScales++
		}
		if goScale != v.scale {
			c.GoHalfScaleMismatches++
		}
		if class == "positive-finite" && goRecip != v.reciprocal {
			c.GoReciprocalMismatches++
		}
		fmt.Fprintf(&table, "%04x\t%04x\t%04x\t%08x\t%08x\t%04x\t%08x\t%s\n", bits, v.nativeRange, v.scale, v.widened, v.reciprocal, goScale, goRecip, class)
	}
	if err = os.WriteFile(path, []byte(table.String()), 0644); err != nil {
		return c, err
	}
	c.Table, err = filePin(path)
	return c, err
}
