package graph

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type selectedBuildWire struct{ Evidence, Selection nativeproof.Pin }
type requestWire struct {
	Schema     string
	Consumer   selectedBuildWire
	Backend    closedbuild.ArtifactOptions
	CPU13      nativeproof.Pin
	Sources    WitnessAnchors
	Model, Out string
	Ordinal    int
}
type admissionWire struct {
	Schema                              string
	Request, Identity                   nativeproof.Pin
	GuardPlan, PhaseContract            supgpu.Pin
	MetadataGuardPlan, MetadataContract supgpu.Pin
	Command                             supgpu.GuardCommand
}
type checkedRequest struct {
	wire    requestWire
	binary  cgobuild.Artifact
	backend closedbuild.BackendArtifact
	witness Witness
	inputs  []supgpu.Pin
}
type checkedAdmission struct {
	wire         admissionWire
	request      checkedRequest
	plan         supgpu.GuardPlan
	identity     identityWire
	metadataPlan supgpu.GuardPlan
}

// Result names retained evidence, not a runtime capability.
type Result struct {
	Evidence            nativeproof.Pin
	Events, Comparisons int
	ScoreSHA256         string
}
type wrapperWire struct {
	Schema, Status, Error string
	Admission             nativeproof.Pin
	Run                   supgpu.GuardedRun
	Numerical             nativeproof.Pin
	Metadata              []supgpu.GuardedRun
}

func canonical(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return fmt.Errorf("absolute clean path required: %s", path)
	}
	p, err := filepath.EvalSymlinks(path)
	if err != nil || p != path {
		return fmt.Errorf("canonical path required: %s: %v", path, err)
	}
	return nil
}
func openRegular(path string, max int64) (*os.File, os.FileInfo, error) {
	if err := canonical(path); err != nil {
		return nil, nil, err
	}
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, nil, err
	}
	s, err := f.Stat()
	if err != nil || !s.Mode().IsRegular() || s.Size() < 0 || s.Size() > max {
		f.Close()
		return nil, nil, fmt.Errorf("bounded regular input required: %s: %v", path, err)
	}
	return f, s, nil
}
func validSHA(sha string) bool {
	b, e := hex.DecodeString(sha)
	return e == nil && len(b) == 32 && sha == strings.ToLower(sha)
}
func sizedPin(p nativeproof.Pin) (supgpu.Pin, error) {
	if !validSHA(p.SHA256) {
		return supgpu.Pin{}, fmt.Errorf("independent SHA256 required")
	}
	f, s, e := openRegular(p.Path, 8<<30)
	if e != nil {
		return supgpu.Pin{}, e
	}
	defer f.Close()
	h := sha256.New()
	n, e := io.Copy(h, io.LimitReader(f, s.Size()+1))
	after, ae := os.Stat(p.Path)
	if e != nil || ae != nil || !os.SameFile(s, after) || s.Size() != n || !s.ModTime().Equal(after.ModTime()) || hex.EncodeToString(h.Sum(nil)) != p.SHA256 {
		return supgpu.Pin{}, fmt.Errorf("live input pin differs: %s", p.Path)
	}
	return supgpu.Pin{Path: p.Path, SHA256: p.SHA256, Bytes: n}, nil
}
func readSized(path string, p supgpu.Pin, max int64) ([]byte, error) {
	if !validSHA(p.SHA256) || p.Bytes < 0 || p.Bytes > max {
		return nil, fmt.Errorf("invalid sized input")
	}
	f, s, e := openRegular(path, max)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	if s.Size() != p.Bytes {
		return nil, fmt.Errorf("sized input extent differs")
	}
	b, e := io.ReadAll(io.LimitReader(f, max+1))
	after, ae := os.Stat(path)
	if e != nil || ae != nil || !os.SameFile(s, after) || !s.ModTime().Equal(after.ModTime()) || int64(len(b)) != p.Bytes || digest(b) != p.SHA256 {
		return nil, fmt.Errorf("sized input bytes differ: %s", path)
	}
	return b, nil
}
func readJSON(p nativeproof.Pin, out any) error {
	if !validSHA(p.SHA256) {
		return fmt.Errorf("independent JSON SHA256 required")
	}
	f, s, e := openRegular(p.Path, 64<<20)
	if e != nil {
		return e
	}
	defer f.Close()
	b, e := io.ReadAll(io.LimitReader(f, (64<<20)+1))
	if e != nil {
		return e
	}
	if int64(len(b)) != s.Size() || len(b) > 64<<20 || digest(b) != p.SHA256 {
		return fmt.Errorf("JSON extent/digest differs")
	}
	return decodeStrict(b, out)
}

// Validate original wire tokens before decoding; encoding/json alone merges
// duplicate/case-folded keys and accepts null scalar fields.
func decodeStrict(b []byte, out any) error {
	t := reflect.TypeOf(out)
	if t == nil || t.Kind() != reflect.Pointer {
		return fmt.Errorf("typed destination required")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if e := checkJSON(d, t.Elem()); e != nil {
		return e
	}
	if _, e := d.Token(); e != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return json.Unmarshal(b, out)
}
func checkJSON(d *json.Decoder, t reflect.Type) error {
	tok, e := d.Token()
	if e != nil {
		return e
	}
	if tok == nil {
		return fmt.Errorf("null %s", t)
	}
	switch t.Kind() {
	case reflect.Struct:
		if tok != json.Delim('{') {
			return fmt.Errorf("expected object")
		}
		fields := map[string]reflect.Type{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = f.Name
			}
			fields[name] = f.Type
		}
		seen := map[string]bool{}
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return e
			}
			name, ok := k.(string)
			ft, exists := fields[name]
			if !ok || !exists || seen[name] {
				return fmt.Errorf("unknown/duplicate field %v", k)
			}
			seen[name] = true
			if e = checkJSON(d, ft); e != nil {
				return fmt.Errorf("%s: %w", name, e)
			}
		}
		if len(seen) != len(fields) {
			return fmt.Errorf("missing fields in %s", t)
		}
		_, e = d.Token()
		return e
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			if _, ok := tok.(string); !ok {
				return fmt.Errorf("expected byte string")
			}
			return nil
		}
		if tok != json.Delim('[') {
			return fmt.Errorf("expected array")
		}
		for d.More() {
			if e = checkJSON(d, t.Elem()); e != nil {
				return e
			}
		}
		_, e = d.Token()
		return e
	case reflect.Map:
		if tok != json.Delim('{') || t.Key().Kind() != reflect.String {
			return fmt.Errorf("expected string map")
		}
		seen := map[string]bool{}
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return e
			}
			s, ok := k.(string)
			if !ok || seen[s] {
				return fmt.Errorf("duplicate map key")
			}
			seen[s] = true
			if e = checkJSON(d, t.Elem()); e != nil {
				return e
			}
		}
		_, e = d.Token()
		return e
	default:
		if _, ok := tok.(json.Delim); ok {
			return fmt.Errorf("unexpected container")
		}
		return nil
	}
}
func writeJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL|syscall.O_NOFOLLOW, 0600)
	if e != nil {
		return e
	}
	_, e = f.Write(append(b, '\n'))
	return errors.Join(e, f.Close())
}
func unsized(p supgpu.Pin) nativeproof.Pin { return nativeproof.Pin{Path: p.Path, SHA256: p.SHA256} }
func childCommand(binary, request nativeproof.Pin) []string {
	return []string{binary.Path, "-mode", "guarded-child", "-request", request.Path, "-request-sha", request.SHA256}
}
func currentRoot() (string, error) {
	p, e := os.Getwd()
	if e != nil {
		return "", e
	}
	return p, canonical(p)
}
func actualSUP(q checkedRequest, request nativeproof.Pin, child bool) error {
	p, e := os.Executable()
	if e != nil {
		return e
	}
	v, e := sizedPin(q.binary.Output())
	if e != nil {
		return e
	}
	if p != v.Path {
		return fmt.Errorf("executing SUP path differs from selected output")
	}
	actual, e := sizedPin(nativeproof.Pin{Path: p, SHA256: v.SHA256})
	if e != nil || actual != v {
		return fmt.Errorf("executing SUP bytes differ: %v", e)
	}
	if child && !slices.Equal(os.Args, childCommand(q.binary.Output(), request)) {
		return fmt.Errorf("actual child argv differs")
	}
	return nil
}
func readRequest(pin nativeproof.Pin, root string) (q checkedRequest, err error) {
	if err = readJSON(pin, &q.wire); err != nil {
		return
	}
	o := q.wire
	if o.Schema != "nanopore-sup-tiny-request/v1" || o.Ordinal < 0 || o.Ordinal > 255 {
		return q, fmt.Errorf("tiny request schema/ordinal differs")
	}
	if q.binary, err = cgobuild.ReadSUPArtifact(o.Consumer.Evidence, o.Consumer.Selection, root); err != nil {
		return
	}
	// Preflight every selected native input before legacy readers inspect it.
	pins := []nativeproof.Pin{pin, o.Consumer.Evidence, o.Consumer.Selection, o.CPU13, o.Backend.Build, o.Backend.Plan, o.Backend.Producer, o.Backend.Output, q.binary.Output()}
	for _, p := range []supgpu.Pin{o.Sources.Capture, o.Sources.Prepared, o.Sources.SourceBuild, o.Sources.SourceRuntime} {
		v, e := sizedPin(unsized(p))
		if e != nil || v != p {
			return q, fmt.Errorf("source anchor differs: %v", e)
		}
		pins = append(pins, unsized(p))
	}
	for _, p := range pins {
		v, e := sizedPin(p)
		if e != nil {
			return q, e
		}
		q.inputs = append(q.inputs, v)
	}
	var cpu h08.CPU13Receipt
	if err = readJSON(o.CPU13, &cpu); err != nil {
		return
	}
	var cpuAdmission h08.RuntimeAdmission
	if err = readJSON(cpu.Admission, &cpuAdmission); err != nil {
		return
	}
	var cpuRequest h08.CPU13Request
	if err = readJSON(cpuAdmission.Request, &cpuRequest); err != nil {
		return
	}
	if q.backend, err = closedbuild.ReadArtifact(o.Backend); err != nil {
		return
	}
	if len(q.backend.Runtime) != 23 {
		return q, fmt.Errorf("all 23 backend runtime providers required")
	}
	paths := make([]string, 0, len(q.backend.Inputs))
	for p := range q.backend.Inputs {
		paths = append(paths, p)
	}
	slices.Sort(paths)
	for _, p := range paths {
		v, e := sizedPin(nativeproof.Pin{Path: p, SHA256: q.backend.Inputs[p]})
		if e != nil {
			return q, e
		}
		q.inputs = append(q.inputs, v)
	}
	if _, err = h08.ReadCPU13(o.CPU13, o.Backend); err != nil {
		return
	}
	if q.witness, err = ReadWitness(o.Sources); err != nil {
		return
	}
	if err = canonical(o.Model); err != nil {
		return
	}
	w := q.witness.prepared
	if filepath.Dir(w.Config.Path) != o.Model || filepath.Base(w.Config.Path) != "config.toml" || len(w.Weights) != 139 {
		return q, fmt.Errorf("model configuration/139 weight custody differs")
	}
	preparedInputs := append([]supgpu.Pin{w.Config, w.Input, w.Profile, w.Fixture, w.Signal}, w.Weights...)
	preparedInputs = append(preparedInputs, w.Sources...)
	for _, p := range preparedInputs {
		v, e := sizedPin(unsized(p))
		if e != nil || v != p {
			return q, fmt.Errorf("prepared live custody differs: %s: %v", p.Path, e)
		}
		q.inputs = append(q.inputs, v)
	}
	for _, p := range w.Weights {
		if filepath.Dir(p.Path) != o.Model {
			return q, fmt.Errorf("weight outside selected model")
		}
	}
	for _, m := range q.backend.Runtime {
		v, e := sizedPin(m.Pin)
		if e != nil {
			return q, e
		}
		q.inputs = append(q.inputs, v)
	}
	if err = validateOutputs(root, []string{o.Out}, q.inputs); err != nil {
		return
	}
	return
}
func within(parent, child string) bool {
	r, e := filepath.Rel(parent, child)
	return e == nil && (r == "." || r != ".." && !strings.HasPrefix(r, "../") && !filepath.IsAbs(r))
}
func outputPath(root, path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || path == filepath.Join(root, "artifacts") || !within(filepath.Join(root, "artifacts"), path) {
		return fmt.Errorf("output must be clean strict descendant of selected root artifacts")
	}
	return canonical(filepath.Dir(path))
}
func validateOutputs(root string, outputs []string, inputs []supgpu.Pin) error {
	for i, p := range outputs {
		if e := outputPath(root, p); e != nil {
			return e
		}
		for _, other := range outputs[:i] {
			if within(p, other) || within(other, p) {
				return fmt.Errorf("overlapping outputs")
			}
		}
		for _, v := range inputs {
			if within(p, v.Path) || within(v.Path, p) {
				return fmt.Errorf("output/input collision")
			}
		}
	}
	return nil
}
func backendPolicy(b closedbuild.BackendArtifact, p supgpu.GuardPlan) error {
	if len(b.Runtime) != 23 {
		return fmt.Errorf("backend provider set incomplete")
	}
	out, e := nativeproof.InspectELF(b.Output.Path)
	if e != nil {
		return e
	}
	if out.Pin != b.Output || out.Type != elf.ET_DYN || out.Interpreter != "" {
		return fmt.Errorf("backend ELF differs")
	}
	allowed := map[string]supgpu.RuntimeCode{}
	names := map[string]bool{}
	for _, c := range p.Policy.Allowed {
		if _, ok := allowed[c.Object.Path]; ok {
			return fmt.Errorf("duplicate Allowed path")
		}
		allowed[c.Object.Path] = c
		if c.SONAME != "" {
			if names[c.SONAME] {
				return fmt.Errorf("duplicate SONAME")
			}
			names[c.SONAME] = true
		}
	}
	seen := map[string]bool{}
	for _, m := range append(slices.Clone(b.Runtime), out) {
		if seen[m.Pin.Path] {
			return fmt.Errorf("duplicate required provider")
		}
		seen[m.Pin.Path] = true
		c, ok := allowed[m.Pin.Path]
		v, e := sizedPin(m.Pin)
		if e != nil || !ok || c.Object != v || c.SONAME != m.SONAME || !slices.Equal(c.Needed, m.Needed) {
			return fmt.Errorf("actual ELF provider differs from Allowed: %s: %v", m.Pin.Path, e)
		}
		actual, e := nativeproof.InspectELF(m.Pin.Path)
		if e != nil || !reflect.DeepEqual(actual, m) {
			return fmt.Errorf("live provider metadata differs: %v", e)
		}
		for _, n := range m.Needed {
			if !names[n] {
				return fmt.Errorf("provider dependency absent: %s", n)
			}
		}
	}
	return nil
}
func readAdmission(pin nativeproof.Pin) (a checkedAdmission, err error) {
	root, e := currentRoot()
	if e != nil {
		return a, e
	}
	if err = readJSON(pin, &a.wire); err != nil {
		return
	}
	w := a.wire
	if w.Schema != "nanopore-sup-tiny-admission/v1" {
		return a, fmt.Errorf("SUP admission schema differs")
	}
	if a.request, err = readRequest(w.Request, root); err != nil {
		return
	}
	// Strictly validate original policy bytes before public guard validation.
	var raw supgpu.GuardPlan
	if err = readJSON(unsized(w.GuardPlan), &raw); err != nil {
		return
	}
	var contract supgpu.PhaseContract
	if err = readJSON(unsized(w.PhaseContract), &contract); err != nil {
		return
	}
	if a.plan, contract, err = supgpu.ReadGuardPlan(w.GuardPlan, w.PhaseContract); err != nil {
		return
	}
	if !reflect.DeepEqual(raw, a.plan) || !numericalContract(contract) {
		return a, fmt.Errorf("original SUP policy/exact phases differ")
	}
	target, e := sizedPin(a.request.binary.Output())
	if e != nil {
		return a, e
	}
	if a.plan.Policy.Observer != target || a.plan.Policy.Reference != target || w.Command.Directory != root || !slices.Equal(w.Command.Args, childCommand(a.request.binary.Output(), w.Request)) {
		return a, fmt.Errorf("selected root/target/argv differs")
	}
	if err = backendPolicy(a.request.backend, a.plan); err != nil {
		return
	}
	if a.identity, err = readIdentity(w.Identity); err != nil {
		return
	}
	if !slices.Equal(a.identity.Args, identityArgs(a.identity, a.request.wire.Ordinal)) {
		return a, fmt.Errorf("independently selected identity argv differs")
	}
	var metadataRaw supgpu.GuardPlan
	if err = readJSON(unsized(w.MetadataGuardPlan), &metadataRaw); err != nil {
		return
	}
	var metadataContract supgpu.PhaseContract
	if err = readJSON(unsized(w.MetadataContract), &metadataContract); err != nil {
		return
	}
	if a.metadataPlan, metadataContract, err = supgpu.ReadGuardPlan(w.MetadataGuardPlan, w.MetadataContract); err != nil {
		return
	}
	if !reflect.DeepEqual(metadataRaw, a.metadataPlan) || metadataContract.Schema != "nanopore-audited-metadata-contract/v1" || metadataContract.Role != "observer" || metadataContract.Action != "device-identity" || metadataContract.Names == nil || len(metadataContract.Names) != 0 {
		return a, fmt.Errorf("independent metadata guard contract differs")
	}
	if err = identityPolicy(a.identity, a.metadataPlan, a.plan.Policy.Audit); err != nil {
		return
	}
	if _, err = sourceUUID(a.request.witness.prepared.DeviceUUID); err != nil {
		return
	}
	required := append(slices.Clone(a.request.inputs), w.GuardPlan, w.PhaseContract, w.MetadataGuardPlan, w.MetadataContract)
	// Plan/contract cannot include their own hashes; admission itself is likewise
	// an independent outer anchor. Everything they select is explicit custody.
	custody := slices.Clone(a.request.inputs)
	for _, p := range append([]nativeproof.Pin{w.Identity, a.identity.Executable}, a.identity.Closure.Inputs...) {
		v, e := sizedPin(p)
		if e != nil {
			return a, e
		}
		custody = append(custody, v)
		required = append(required, v)
	}
	metadataInputs := map[string]supgpu.Pin{}
	for _, v := range a.metadataPlan.Inputs {
		metadataInputs[v.Path] = v
		required = append(required, v)
	}
	for _, v := range append([]supgpu.Pin{a.plan.Policy.Audit}, custody[len(a.request.inputs):]...) {
		if metadataInputs[v.Path] != v {
			return a, fmt.Errorf("metadata guard omits exact custody: %s", v.Path)
		}
	}
	custody = append(custody, w.MetadataGuardPlan, w.MetadataContract)
	selected := map[string]supgpu.Pin{}
	for _, v := range a.plan.Inputs {
		selected[v.Path] = v
	}
	for _, v := range custody {
		if selected[v.Path] != v {
			return a, fmt.Errorf("guard omits exact custody: %s", v.Path)
		}
	}
	for _, v := range append(slices.Clone(a.plan.Policy.Allowed), a.plan.Policy.Alternates...) {
		required = append(required, v.Object)
	}
	required = append(required, a.plan.Policy.Audit, target)
	ap, e := sizedPin(pin)
	if e != nil {
		return a, e
	}
	required = append(required, ap)
	if err = validateOutputs(root, []string{w.Command.Output, a.request.wire.Out, w.Command.Output + "-identity", identityCommand(w, "before", a.identity).Output, identityCommand(w, "after", a.identity).Output}, required); err != nil {
		return
	}
	return
}
func numericalContract(c supgpu.PhaseContract) bool {
	return c.Schema == "nanopore-audited-phase-contract/v1" && c.Role == "observer" && c.Action == "sup-tiny" && slices.Equal(c.Names, []string{"pre-model", "pre-forward", "post-forward"})
}
func requireFresh(paths ...string) error {
	for _, p := range paths {
		if _, e := os.Lstat(p); !errors.Is(e, os.ErrNotExist) {
			return fmt.Errorf("fresh absent output required: %s: %v", p, e)
		}
	}
	return nil
}
func resultFor(pin nativeproof.Pin, r Receipt) Result {
	return Result{Evidence: pin, Events: r.Events, Comparisons: len(r.Comparisons), ScoreSHA256: r.ScoreSHA256}
}

// Consume requires the external launcher to hold the existing read-only EX|NB
// stock host.lock until watcher, unit and every retained descendant are terminal
// and empty, including parent failure. This function never acquires that lock.
// The independently reviewed outer watcher supplies finite resource/cleanup
// bounds; ExecuteGuarded additionally honors local context cancellation.
func Consume(ctx context.Context, pin nativeproof.Pin) (result Result, err error) {
	if ctx == nil {
		return result, fmt.Errorf("nil context")
	}
	if err = ctx.Err(); err != nil {
		return
	}
	a, err := readAdmission(pin)
	if err != nil {
		return
	}
	if err = actualSUP(a.request, a.wire.Request, false); err != nil {
		return
	}
	if err = runner.CheckRoot(); err != nil {
		return
	}
	if !slices.Equal(os.Args, []string{a.request.binary.Output().Path, "-mode", "consumer", "-admission", pin.Path, "-admission-sha", pin.SHA256}) {
		return result, fmt.Errorf("actual parent argv differs")
	}
	w := a.wire
	meta := w.Command.Output + "-identity"
	if err = requireFresh(w.Command.Output, a.request.wire.Out, meta, identityCommand(w, "before", a.identity).Output, identityCommand(w, "after", a.identity).Output); err != nil {
		return
	}
	if err = os.Mkdir(meta, 0700); err != nil {
		return
	}
	r := wrapperWire{Schema: "nanopore-sup-tiny-wrapper/v1", Status: "failed", Admission: pin, Metadata: []supgpu.GuardedRun{}}
	defer func() {
		if err != nil {
			r.Status = "failed"
			r.Error = err.Error()
		}
		err = errors.Join(err, writeJSON(filepath.Join(meta, "parent.json"), r))
	}()
	before, e := executeIdentity(ctx, w, "before", a.identity, a.request.wire.Ordinal, a.request.witness.prepared.DeviceUUID)
	r.Metadata = append(r.Metadata, before)
	if e != nil {
		return result, e
	}
	r.Run, err = supgpu.ExecuteGuarded(ctx, w.GuardPlan, w.PhaseContract, w.Command)
	// Preserve the raw run and attempt the after bookend even on child failure.
	after, ae := executeIdentity(ctx, w, "after", a.identity, a.request.wire.Ordinal, a.request.witness.prepared.DeviceUUID)
	r.Metadata = append(r.Metadata, after)
	err = errors.Join(err, ae)
	if err != nil {
		return
	}
	p, e := pinFile(filepath.Join(a.request.wire.Out, "consumer.json"))
	if e != nil {
		return result, e
	}
	r.Numerical = p
	numerical, e := validateWrapper(a, r)
	if e != nil {
		return result, e
	}
	r.Status = "tiny_numerical_equal"
	path := filepath.Join(w.Command.Output, "sup-graph.json")
	if err = writeJSON(path, r); err != nil {
		return
	}
	p, err = pinFile(path)
	if err != nil {
		return
	}
	return resultFor(p, numerical), nil
}
func pinFile(path string) (nativeproof.Pin, error) {
	f, s, e := openRegular(path, 64<<20)
	if e != nil {
		return nativeproof.Pin{}, e
	}
	defer f.Close()
	h := sha256.New()
	n, e := io.Copy(h, io.LimitReader(f, s.Size()+1))
	if e != nil || n != s.Size() {
		return nativeproof.Pin{}, fmt.Errorf("receipt pin read differs: %v", e)
	}
	return nativeproof.Pin{Path: path, SHA256: hex.EncodeToString(h.Sum(nil))}, nil
}
func validateNumerical(q checkedRequest, request nativeproof.Pin, r childWire) error {
	if r.Schema != "nanopore-sup-tiny-numerical/v1" || r.Error != "" || r.Status != "tiny_numerical_equal" || r.Request != request || r.Consumer != q.wire.Consumer || r.Backend != q.wire.Backend || r.CPU13 != q.wire.CPU13 || r.Sources != q.wire.Sources || r.Executable != q.binary.Output() || r.ForwardCalls != 1 || !r.Cleanup || !slices.Equal(r.Invocation, childCommand(q.binary.Output(), request)) {
		return fmt.Errorf("child numerical custody/cleanup differs")
	}
	if !reflect.DeepEqual(r.Inputs, q.inputs) {
		return fmt.Errorf("child live input custody differs")
	}
	start, e := time.Parse(time.RFC3339Nano, r.Started)
	end, f := time.Parse(time.RFC3339Nano, r.Finished)
	if e != nil || f != nil || end.Before(start) {
		return fmt.Errorf("child numerical clocks differ")
	}
	return validateComparisons(q.witness, r.Numerical)
}
func validateComparisons(w Witness, r Receipt) error {
	if r.Status != "numerical_equal" || r.Events != 191 || len(w.capture.Events) != 191 || len(r.EventContracts) != 191 || len(r.Comparisons) != 733 || r.ScoreSHA256 != w.capture.Reference.SHA256 {
		return fmt.Errorf("tiny numerical coverage/score differs")
	}
	at := 0
	for i, event := range w.capture.Events {
		contract := eventContract(event)
		if r.EventContracts[i] != contract {
			return fmt.Errorf("event descriptors/order differ")
		}
		for _, t := range event.Tensors {
			if at >= len(r.Comparisons) {
				return fmt.Errorf("missing comparison")
			}
			c := r.Comparisons[at]
			at++
			if c.Event != event.Name || c.Role != t.Name || c.ContractSHA256 != contract || c.NativeSHA256 != t.Data.SHA256 || c.BackingSHA256 != t.Backing.SHA256 || c.GoSHA256 != t.Data.SHA256 || c.GoSHA256 != t.Backing.SHA256 || int64(c.Bytes) != t.Data.Bytes || int64(c.Bytes) != t.Backing.Bytes || !c.Equal {
				return fmt.Errorf("logical/backing/descriptor comparison differs at %s/%s", event.Name, t.Name)
			}
		}
	}
	if at != len(r.Comparisons) {
		return fmt.Errorf("extra comparisons")
	}
	return nil
}
func validateWrapper(a checkedAdmission, r wrapperWire) (Receipt, error) {
	w := a.wire
	if r.Schema != "nanopore-sup-tiny-wrapper/v1" || r.Error != "" || len(r.Metadata) != 2 || r.Numerical.Path != filepath.Join(a.request.wire.Out, "consumer.json") {
		return Receipt{}, fmt.Errorf("wrapper custody differs")
	}
	if _, e := supgpu.ReplayGuarded(w.GuardPlan, w.PhaseContract, w.Command, r.Run); e != nil {
		return Receipt{}, e
	}
	for i, c := range r.Metadata {
		name := "before"
		if i == 1 {
			name = "after"
		}
		if e := validateIdentity(c, w, name, a.identity, a.request.wire.Ordinal, a.request.witness.prepared.DeviceUUID); e != nil {
			return Receipt{}, e
		}
	}
	before, e := time.Parse(time.RFC3339Nano, r.Metadata[0].Command.Finished)
	start, f := time.Parse(time.RFC3339Nano, r.Run.Command.Started)
	end, g := time.Parse(time.RFC3339Nano, r.Run.Command.Finished)
	after, h := time.Parse(time.RFC3339Nano, r.Metadata[1].Command.Started)
	if e != nil || f != nil || g != nil || h != nil || before.After(start) || end.After(after) {
		return Receipt{}, fmt.Errorf("identity bookends do not surround child")
	}
	var child childWire
	if e := readJSON(r.Numerical, &child); e != nil {
		return Receipt{}, e
	}
	if e := validateNumerical(a.request, w.Request, child); e != nil {
		return Receipt{}, e
	}
	childStart, ce := time.Parse(time.RFC3339Nano, child.Started)
	childEnd, cf := time.Parse(time.RFC3339Nano, child.Finished)
	if ce != nil || cf != nil || childStart.Before(start) || childEnd.After(end) {
		return Receipt{}, fmt.Errorf("numerical clocks outside actual guarded child")
	}
	// Recheck the parent's previously authenticated live bytes after execution;
	// equality with a remembered receipt list alone cannot authenticate mutation.
	for _, p := range a.request.inputs {
		actual, e := sizedPin(unsized(p))
		if e != nil || actual != p {
			return Receipt{}, fmt.Errorf("post-child custody changed: %s: %v", p.Path, e)
		}
	}
	var emitted Result
	if e := readJSON(nativeproof.Pin{Path: r.Run.Command.Stdout, SHA256: r.Run.Command.StdoutSHA}, &emitted); e != nil {
		return Receipt{}, e
	}
	if emitted != resultFor(r.Numerical, child.Numerical) {
		return Receipt{}, fmt.Errorf("guard-pinned stdout does not bind numerical receipt")
	}
	return child.Numerical, nil
}

// Replay selects all expectations from admission, never from the wrapper.
func Replay(admission, receipt nativeproof.Pin) (Result, error) {
	a, e := readAdmission(admission)
	if e != nil {
		return Result{}, e
	}
	if receipt.Path != filepath.Join(a.wire.Command.Output, "sup-graph.json") {
		return Result{}, fmt.Errorf("wrapper path differs")
	}
	var r wrapperWire
	if e = readJSON(receipt, &r); e != nil {
		return Result{}, e
	}
	if r.Admission != admission || r.Status != "tiny_numerical_equal" {
		return Result{}, fmt.Errorf("wrapper admission/status differs")
	}
	n, e := validateWrapper(a, r)
	if e != nil {
		return Result{}, e
	}
	return resultFor(receipt, n), nil
}
