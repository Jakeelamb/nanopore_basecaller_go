// Package gpuoracle defines the shared evidence boundary, not a producer.
package gpuoracle

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const Schema = "nanopore-gpu-oracle/v1"

type Stage string

const (
	StagePrepared Stage = "prepared"
	StageFrozen   Stage = "frozen"
	StageImported Stage = "imported"
	StageAccepted Stage = "accepted"
)

type Pin struct {
	Path   string
	SHA256 string
}
type Prepared struct {
	Schema           string
	Stage            Stage
	Source           []Pin
	Inputs           []Pin
	RangeDeclaration Pin
}
type Frozen struct {
	Schema        string
	Stage         Stage
	Preparation   Pin
	Producer      []Pin
	Qualification Pin
	Commands      []runner.CommandReceipt
	Payloads      []Pin
}
type Imported struct {
	Schema        string
	Stage         Stage
	Capture       Pin
	Consumer      []Pin
	RequiredTests []string
	TestReceipt   Pin
}
type Accepted struct {
	Schema               string
	Stage                Stage
	Import               Pin
	RuntimeQualification Pin
	IndependentVerdict   Pin
}

// Ranges partition the complete retained allocation before execution.
type Range struct {
	Offset uint64
	Length uint64
	Kind   string
	Source Pin
}
type Allocation struct {
	Name   string
	Bytes  uint64
	Ranges []Range
}

func (p Pin) Validate() error {
	b, err := hex.DecodeString(p.SHA256)
	if p.Path == "" || err != nil || len(b) != 32 {
		return fmt.Errorf("invalid file pin %q", p.Path)
	}
	return nil
}
func (p Pin) Check() error {
	if err := p.Validate(); err != nil {
		return err
	}
	got, err := runner.SHA(p.Path)
	if err != nil {
		return err
	}
	if got != p.SHA256 {
		return fmt.Errorf("changed payload %s", p.Path)
	}
	return nil
}
func (a Allocation) Validate() error {
	if a.Name == "" || a.Bytes == 0 || len(a.Ranges) == 0 {
		return fmt.Errorf("missing allocation declaration")
	}
	var end uint64
	for _, r := range a.Ranges {
		if r.Offset != end || r.Length == 0 || r.Length > a.Bytes-end {
			return fmt.Errorf("ranges must partition allocation %s", a.Name)
		}
		switch r.Kind {
		case "live", "initialized", "guard", "uninitialized":
		default:
			return fmt.Errorf("unknown range kind %q", r.Kind)
		}
		if err := r.Source.Validate(); err != nil {
			return err
		}
		end += r.Length
	}
	if end != a.Bytes {
		return fmt.Errorf("incomplete allocation %s", a.Name)
	}
	return nil
}

// Decode requires every exact field, rejects nulls and duplicates at every depth.
// It is restricted to the contract's concrete structs and their field types.
func Decode(raw []byte, out any) error {
	switch out.(type) {
	case *Prepared, *Frozen, *Imported, *Accepted, *Allocation:
	default:
		return fmt.Errorf("unsupported contract type")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := unique(d); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	var tree any
	if err := json.Unmarshal(raw, &tree); err != nil {
		return err
	}
	if err := fields(tree, reflect.TypeOf(out).Elem()); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	switch v := out.(type) {
	case *Prepared:
		return v.ValidateIdentity()
	case *Frozen:
		return v.ValidateIdentity()
	case *Imported:
		return v.ValidateIdentity()
	case *Accepted:
		return v.ValidateIdentity()
	}
	return nil
}

// ValidateIdentity checks the wire identity only, not evidence or qualification.
func (p Prepared) ValidateIdentity() error { return identity(p.Schema, p.Stage, StagePrepared) }
func (p Frozen) ValidateIdentity() error   { return identity(p.Schema, p.Stage, StageFrozen) }
func (p Imported) ValidateIdentity() error { return identity(p.Schema, p.Stage, StageImported) }
func (p Accepted) ValidateIdentity() error { return identity(p.Schema, p.Stage, StageAccepted) }
func identity(schema string, stage, want Stage) error {
	if schema != Schema || stage != want {
		return fmt.Errorf("invalid contract identity %q/%q; expected %q/%q", schema, stage, Schema, want)
	}
	return nil
}
func unique(d *json.Decoder) error {
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	seen := map[string]bool{}
	for d.More() {
		if delim == '{' {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			if !ok || seen[key] {
				return fmt.Errorf("duplicate JSON field %v", k)
			}
			seen[key] = true
		}
		if err := unique(d); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}
func fields(v any, t reflect.Type) error {
	if v == nil {
		return fmt.Errorf("null %s", t)
	}
	switch t.Kind() {
	case reflect.Struct:
		m, ok := v.(map[string]any)
		if !ok || len(m) != t.NumField() {
			return fmt.Errorf("incomplete or unknown fields for %s", t)
		}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "" {
				name = f.Name
			}
			x, ok := m[name]
			if !ok {
				return fmt.Errorf("missing field %s", name)
			}
			if err := fields(x, f.Type); err != nil {
				return err
			}
		}
	case reflect.Slice:
		a, ok := v.([]any)
		if !ok {
			return fmt.Errorf("expected array")
		}
		for _, x := range a {
			if err := fields(x, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

func RequireTests(path, pkg string, names []string) error {
	if pkg == "" || len(names) == 0 {
		return fmt.Errorf("explicit required test names needed")
	}
	seen := map[string]bool{}
	for _, name := range names {
		if !strings.HasPrefix(name, "Test") || strings.Contains(name, "/") || seen[name] {
			return fmt.Errorf("invalid or duplicate required test %q", name)
		}
		seen[name] = true
	}
	return runner.RequireGoTests(path, pkg, names)
}

// Admit checks intake and qualification before acquiring the canonical host lock.
// The caller must retain the returned close function until all children exit.
func Admit(intake bool, qualification func() error) (func() error, error) {
	return admit(intake, qualification, runner.HostLock)
}

// admit exposes a private lock-path seam for package rejection controls.
func admit(intake bool, qualification func() error, lockPath string) (func() error, error) {
	if !intake || qualification == nil {
		return nil, runner.ErrAdmissionBlocked
	}
	if err := qualification(); err != nil {
		return nil, fmt.Errorf("%w: qualification: %v", runner.ErrAdmissionBlocked, err)
	}
	lock, err := runner.Acquire(lockPath)
	if err != nil {
		return nil, err
	}
	return lock.Close, nil
}
