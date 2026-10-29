package packed

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
)

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func validSHA(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && s == strings.ToLower(s)
}

// canonical rejects lexical and symlink aliases, including symlinked parents.
func canonical(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || strings.ContainsAny(path, "\x00\r\n\t") {
		return fmt.Errorf("absolute clean path required: %q", path)
	}
	real, err := filepath.EvalSymlinks(path)
	if err != nil {
		return err
	}
	if real != path {
		return fmt.Errorf("path alias rejected: %s", path)
	}
	return nil
}

func readPin(p Pin) ([]byte, error) {
	if !validSHA(p.SHA256) || p.Bytes <= 0 || p.Bytes > MaxArtifactBytes {
		return nil, fmt.Errorf("invalid bounded pin: %s", p.Path)
	}
	if err := canonical(p.Path); err != nil {
		return nil, err
	}
	entry, err := os.Lstat(p.Path)
	if err != nil {
		return nil, err
	}
	if !entry.Mode().IsRegular() || entry.Size() != p.Bytes {
		return nil, fmt.Errorf("regular pinned extent required: %s", p.Path)
	}
	// This repository's observer tooling targets Linux. Nonblocking/no-follow
	// prevents a replaced final component from blocking on a FIFO or symlink.
	f, err := os.OpenFile(p.Path, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !st.Mode().IsRegular() || st.Size() != p.Bytes || !os.SameFile(entry, st) {
		return nil, fmt.Errorf("regular pinned extent required: %s", p.Path)
	}
	b, err := io.ReadAll(io.LimitReader(f, p.Bytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) != p.Bytes || digest(b) != p.SHA256 {
		return nil, fmt.Errorf("pin differs: %s", p.Path)
	}
	return b, nil
}

func strictJSON(b []byte, dst any) error {
	if err := exactJSONShape(json.NewDecoder(bytes.NewReader(b)), reflect.TypeOf(dst).Elem()); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}

// Walk the existing wire types, requiring every exact tag once. Only pointer
// fields can carry null; calibration pointers still require semantic validation.
func exactJSONShape(d *json.Decoder, shape reflect.Type) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	if shape.Kind() == reflect.Pointer {
		if tok == nil {
			return nil
		}
		shape = shape.Elem()
	}
	if tok == nil {
		return fmt.Errorf("null ordinary JSON declaration")
	}
	switch shape.Kind() {
	case reflect.Struct:
		if tok != json.Delim('{') {
			return fmt.Errorf("JSON object required")
		}
		fields := map[string]reflect.Type{}
		for i := 0; i < shape.NumField(); i++ {
			field := shape.Field(i)
			fields[field.Tag.Get("json")] = field.Type
		}
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			child, known := fields[name]
			if !ok || !known || seen[name] {
				return fmt.Errorf("unknown or duplicate JSON key: %v", key)
			}
			seen[name] = true
			if err := exactJSONShape(d, child); err != nil {
				return err
			}
		}
		if len(seen) != len(fields) {
			return fmt.Errorf("missing required JSON declaration")
		}
		_, err = d.Token()
		return err
	case reflect.Slice:
		// encoding/json represents []uint8 as base64, and also accepts arrays.
		if shape.Elem().Kind() == reflect.Uint8 {
			if _, ok := tok.(string); ok {
				return nil
			}
		}
		if tok != json.Delim('[') {
			return fmt.Errorf("JSON array required")
		}
		for d.More() {
			if err := exactJSONShape(d, shape.Elem()); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	}
	return nil
}

func beneath(a, b string) bool {
	r, err := filepath.Rel(a, b)
	return err == nil && r != ".." && !strings.HasPrefix(r, ".."+string(os.PathSeparator))
}

// freshOutput requires an existing canonical parent and a previously absent leaf.
// No cleanup/retry: any partial output is retained for inspection.
func freshOutput(out string, pins []Pin) error {
	if !filepath.IsAbs(out) || filepath.Clean(out) != out || strings.ContainsAny(out, "\x00\r\n\t") {
		return fmt.Errorf("absolute clean output required")
	}
	if err := canonical(filepath.Dir(out)); err != nil {
		return err
	}
	if _, err := os.Lstat(out); !os.IsNotExist(err) {
		return fmt.Errorf("output must be fresh: %s", out)
	}
	for _, p := range pins {
		if beneath(out, p.Path) || beneath(p.Path, out) {
			return fmt.Errorf("input/output overlap: %s", p.Path)
		}
	}
	return nil
}

func writeNew(path string, b []byte) (Pin, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0444)
	if err != nil {
		return Pin{}, err
	}
	n, err := f.Write(b)
	closeErr := f.Close()
	if err != nil {
		return Pin{}, err
	}
	if n != len(b) {
		return Pin{}, io.ErrShortWrite
	}
	if closeErr != nil {
		return Pin{}, closeErr
	}
	return Pin{path, digest(b), int64(len(b))}, nil
}

func writeJSON(path string, v any) (Pin, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return Pin{}, err
	}
	return writeNew(path, append(b, '\n'))
}

func forbidden(s string) bool { return strings.Contains(strings.ToUpper(s), "HG005") }

// These unchanged dependencies affect allocation and constructor custody.
// They do not constitute a native build closure.
func sourceAuthorities() []sourceSpec {
	return []sourceSpec{
		{Path: "dorado/nn/WorkingMemory.cpp", SHA: "4b512195a11e418ce35f8a60cf050dc499cec1b3c5504fa6ce8e21d6986e2b0d"},
		{Path: "dorado/nn/include/nn/AuxiliaryData.h", SHA: "4a833be15283fec4e0c732f575f89d1545a91b980e478e4eb4da1cd14a703fec"},
		{Path: "dorado/basecall/CudaModelRunner.cpp", SHA: "ad9d5fcb5c3fb5db00312119498e1113194ab5f6077fbea7b76fc2b9ddb0c3b6"},
		{Path: "dorado/basecall/CudaCaller.cpp", SHA: "e7e4d11dfc33f1f4350fa3e3841f31da882cc4d4b4d072cb5c78c1d9f194c94d"},
	}
}

func validateAuthorities(pins []Pin) error {
	want := sourceAuthorities()
	if len(pins) != len(want) {
		return fmt.Errorf("four unchanged source authorities required")
	}
	for i, s := range want {
		if !strings.HasSuffix(pins[i].Path, "/"+s.Path) || pins[i].SHA256 != s.SHA {
			return fmt.Errorf("source authority order/identity differs")
		}
	}
	return nil
}

// ValidateRequest handles only the first frozen geometry; it does not admit a
// GPU tuple. All arithmetic below is bounded by that equality before use.
func ValidateRequest(r Request) (p Plan, err error) {
	if r.Schema != RequestSchema || r.Case != "first" || (r.Split != "development" && r.Split != "validation") || r.Dataset == "" || forbidden(r.Dataset) {
		return p, fmt.Errorf("first development/validation case required")
	}
	if r.Geometry != FirstGeometry() {
		return p, fmt.Errorf("frozen HAC N64/chunk12288 geometry required")
	}
	if len(r.Members) != 3 || len(r.ExpectedOrder) != 3 {
		return p, fmt.Errorf("three ordered members required")
	}
	if r.Recipe.Workers < 1 || r.Recipe.Workers > 64 || !r.Recipe.CPUWorkspacePinned || r.Recipe.GuardBytes < 64 || r.Recipe.GuardBytes > 1<<20 || len(r.Recipe.ObservationFills) != 2 || r.Recipe.ObservationFills[0] == r.Recipe.ObservationFills[1] || len(r.Recipe.Assertions) == 0 || r.Recipe.UnusedInputFill&0x7c00 == 0x7c00 {
		return p, fmt.Errorf("explicit finite fill, two-fill/guard/pinned-workspace recipe required")
	}
	p.TOut, p.TLSTM, p.InputCapacity = 2048, 2050, 64*12288
	seen := map[string]bool{}
	lengths := []int64{768, 1536, 12288}
	for i, m := range r.Members {
		if m.ID == "" || m.ReadID == "" || seen[m.ID] || m.ID != r.ExpectedOrder[i] || forbidden(m.ID+" "+m.ReadID+" "+m.Dataset+" "+m.Source.Path+" "+m.Normalized.Path) || m.Dataset != r.Dataset || m.Split != r.Split {
			return p, fmt.Errorf("member %d identity/order/split differs", i)
		}
		seen[m.ID] = true
		if m.RawLength <= 0 || m.RawLength > 12288 || m.RoundedLength != lengths[i] || ((m.RawLength+5)/6)*6 != m.RoundedLength {
			return p, fmt.Errorf("member %d raw/rounded lengths differ", i)
		}
		n := m.Normalisation
		if n.Algorithm != "dorado-pa-standardisation-f32-to-f16/v1" || n.CalibrationScale == nil || n.CalibrationOffset == nil || n.Mean == nil || n.Stdev == nil {
			return p, fmt.Errorf("member %d normalization provenance missing", i)
		}
		for _, v := range []*float64{n.CalibrationScale, n.CalibrationOffset, n.Mean, n.Stdev} {
			if math.IsNaN(*v) || math.IsInf(*v, 0) || math.IsInf(float64(float32(*v)), 0) {
				return p, fmt.Errorf("non-finite calibration")
			}
		}
		if *n.CalibrationScale <= 0 || *n.Stdev <= 0 {
			return p, fmt.Errorf("positive scaling required")
		}
		scale := float32(*n.Stdev) / float32(*n.CalibrationScale)
		shift := float32(*n.Mean)/float32(*n.CalibrationScale) - float32(*n.CalibrationOffset)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(shift)) || math.IsInf(float64(shift), 0) {
			return p, fmt.Errorf("degenerate float32 normalization")
		}
		if m.RawParentSamples <= 0 || m.RawParentSamples > MaxArtifactBytes/2 || n.TrimStart < 0 || n.TrimStart >= m.RawParentSamples || m.RawOffset < n.TrimStart || m.RawOffset > m.RawParentSamples-m.RawLength || m.NormalizedSamples != m.RawParentSamples-n.TrimStart || m.NormalizedOffset != m.RawOffset-n.TrimStart || m.Normalized.Bytes != m.NormalizedSamples*2 {
			return p, fmt.Errorf("member %d source/trim/fixture extent differs", i)
		}
		p.Segments = append(p.Segments, Segment{m.ID, m.RawLength, m.RoundedLength, p.ActiveInput, p.ActiveScores, m.RoundedLength / 6})
		p.ActiveInput += m.RoundedLength
		p.ActiveScores += m.RoundedLength / 6
	}
	// CPU21 mapping describes source upload writes, not opaque kernel reads.
	a, b, d := p.ActiveScores+6, int64(64*2050), int64(64*2051)
	p.CPUWorkspaceWords = 6 * (p.TOut + 3) * 64
	p.InitializedWorkspaceWords = 2*a + 2*b + d
	p.Uploads = []UploadSpan{{"OutLayout", b, d, 2*a + b + d}, {"ForwardEncoding", b, b, a}, {"InLayout", p.ActiveScores, p.ActiveScores, a + b}, {"BackwardEncoding", d, d, 2*a + b}}
	if p.ActiveInput > p.InputCapacity || p.InitializedWorkspaceWords > p.CPUWorkspaceWords {
		return p, fmt.Errorf("source capacity overfull")
	}
	for _, s := range p.Uploads {
		if s.InitializedWords > s.CapacityWords || s.WorkspaceOffsetWords+s.InitializedWords > p.CPUWorkspaceWords || s.CapacityWords > math.MaxInt32 {
			return p, fmt.Errorf("signed I32 span overflow")
		}
	}
	p.RecurrentParentBytes = (p.TLSTM + 5) * 64 * 1024
	p.DownScratchBytes = 2 * (p.TLSTM + 1) * 64 * (2 * 128)
	p.ScaleScratchBytes = 4 * (p.TLSTM + 1) * 64
	p.WorkspaceScratchBytes = 4 * (64 / 2)
	return p, nil
}

func requestPins(r Request) []Pin {
	p := []Pin{r.Custody.Model, r.Custody.Config, r.Custody.Weights, r.Custody.Dataset, r.Custody.CPU21, r.Custody.Generation}
	p = append(p, r.Custody.Authorities...)
	for _, m := range r.Members {
		p = append(p, m.Source, m.Normalisation.Provenance, m.Normalized)
	}
	return p
}

// One collector covers the full consumed closure, including generation inputs.
func collectInputs(pins []Pin) (map[string][]byte, []Pin, error) {
	cache := map[string][]byte{}
	declared := map[string]Pin{}
	unique := []Pin{}
	infos := []os.FileInfo{}
	var totalBytes int64
	for _, pin := range pins {
		if prior, ok := declared[pin.Path]; ok {
			if prior != pin {
				return nil, nil, fmt.Errorf("conflicting pin: %s", pin.Path)
			}
			continue
		}
		if pin.Bytes <= 0 || pin.Bytes > MaxArtifactBytes || totalBytes > (256<<20)-pin.Bytes {
			return nil, nil, fmt.Errorf("preparation input budget exceeded")
		}
		totalBytes += pin.Bytes
		b, err := readPin(pin)
		if err != nil {
			return nil, nil, err
		}
		info, err := os.Stat(pin.Path)
		if err != nil {
			return nil, nil, err
		}
		for _, prior := range infos {
			if os.SameFile(info, prior) {
				return nil, nil, fmt.Errorf("hard-link alias rejected: %s", pin.Path)
			}
		}
		infos = append(infos, info)
		unique = append(unique, pin)
		declared[pin.Path], cache[pin.Path] = pin, b
	}
	return cache, unique, nil
}

// Prepare accepts existing, explicitly normalized per-read fixtures. It does
// not decode POD5, load weights, call native code, or invent calibration.
func Prepare(request Pin, out string) (p Prepared, err error) {
	raw, err := readPin(request)
	if err != nil {
		return p, err
	}
	var r Request
	if err = strictJSON(raw, &r); err != nil {
		return p, err
	}
	plan, err := ValidateRequest(r)
	if err != nil {
		return p, err
	}
	if err = validateAuthorities(r.Custody.Authorities); err != nil {
		return p, err
	}
	// Expand the authenticated generation manifest before collecting any closure.
	generationBytes, err := readPin(r.Custody.Generation)
	if err != nil {
		return p, err
	}
	var generation Generated
	if err = strictJSON(generationBytes, &generation); err != nil {
		return p, err
	}
	pins := append([]Pin{request}, requestPins(r)...)
	pins = append(pins, generationPins(generation)...)
	if err = freshOutput(out, pins); err != nil {
		return p, err
	}
	cache, unique, err := collectInputs(pins)
	if err != nil {
		return p, err
	}
	if err = validateGeneration(generation, cache); err != nil {
		return p, err
	}
	for i, spec := range sourceSpecs() {
		sourceTree := filepath.Clean(strings.TrimSuffix(generation.Units[i].Source.Path, filepath.FromSlash(spec.Path)))
		if beneath(sourceTree, out) || beneath(out, sourceTree) {
			return p, fmt.Errorf("preparation output overlaps stock source tree")
		}
	}
	input := make([]byte, int(plan.InputCapacity*2))
	for i := 0; i < len(input); i += 2 {
		binary.LittleEndian.PutUint16(input[i:], r.Recipe.UnusedInputFill)
	}
	for i, m := range r.Members {
		fixture := cache[m.Normalized.Path]
		for j := 0; j < len(fixture); j += 2 {
			if binary.LittleEndian.Uint16(fixture[j:])&0x7c00 == 0x7c00 {
				return p, fmt.Errorf("non-finite normalized fixture")
			}
		}
		// Repeat-prefix padding equals the source's repeated concatenation,
		// including raw lengths shorter than the required suffix.
		for j := int64(0); j < m.RoundedLength; j++ {
			src := 2 * (m.NormalizedOffset + j%m.RawLength)
			dst := 2 * (plan.Segments[i].InputOffset + j)
			copy(input[dst:dst+2], fixture[src:src+2])
		}
	}
	// Recheck declared inputs after preparation, before publishing any file.
	for _, pin := range unique {
		if _, e := readPin(pin); e != nil {
			return p, e
		}
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return p, err
	}
	inputPin, err := writeNew(filepath.Join(out, "input.f16le"), input)
	if err != nil {
		return p, err
	}
	p = Prepared{PreparedSchema, "source-only-uncompiled", request, r, plan, inputPin, unique, nil, []string{
		"Explicit normalized fixtures; normalization/source/model declarations require independent producer review.",
		"CPU21 spans are declared upload geometry, not GPU scatter/repair/recurrence/gather proof.",
		"No native harness/build/runtime tuple, native admission, or biological evidence.",
	}}
	_, err = writeJSON(filepath.Join(out, "prepared.json"), p)
	return p, err
}
