// Package fakecapture captures only the two maintained CPU13 fake-loader tests.
// It never acquires the stock lock or installs a watchdog; the admitting root
// launcher must own both before this Go0 binary starts.
package fakecapture

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type Pin = h08.Pin

const RequestSchema = "nanopore-h08-fake-capture-request/v1"
const sessionSchema = "nanopore-h08-fake-capture-session/v1"
const acceptedStatus = "two_fake_loader_tests_passed"
const selector = "^(TestNativeLoaderABIRejections|TestNativeCloseFailureRetainsLibrary)$"

// All pins are independently selected by root, including the exact output of
// the successful CPU13 build. Out's parent must already exist canonically.
type Request struct {
	Schema                                                              string
	ControllerBuild, ControllerBuildSelection, Target, Logger, Compiler Pin
	Root, Out                                                           string
}

func hash(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

func cleanAbsolute(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || strings.ContainsAny(path, "\x00\r\n") {
		return fmt.Errorf("clean absolute path required: %q", path)
	}
	return nil
}

func canonical(path string) error {
	if e := cleanAbsolute(path); e != nil {
		return e
	}
	actual, e := filepath.EvalSymlinks(path)
	if e != nil {
		return e
	}
	if actual != path {
		return fmt.Errorf("noncanonical path: %s", path)
	}
	return nil
}

func within(root, path string) bool {
	rel, e := filepath.Rel(root, path)
	return e == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func pinBytes(p Pin) ([]byte, error) {
	if e := canonical(p.Path); e != nil {
		return nil, e
	}
	b, e := hex.DecodeString(p.SHA256)
	if e != nil || len(b) != 32 || strings.ToLower(p.SHA256) != p.SHA256 {
		return nil, fmt.Errorf("complete lowercase SHA256 required: %s", p.Path)
	}
	actual, e := closedbuild.PinFile(p.Path)
	if e != nil {
		return nil, e
	}
	if actual != p {
		return nil, fmt.Errorf("pin changed: %s", p.Path)
	}
	b, e = os.ReadFile(p.Path)
	if e != nil {
		return nil, e
	}
	if hash(b) != p.SHA256 {
		return nil, fmt.Errorf("pin changed while reading: %s", p.Path)
	}
	return b, nil
}

// exactJSON also rejects duplicate, case-folded, omitted and null fields. JSON
// decoder's DisallowUnknownFields alone accepts several of those spellings.
func exactJSON(body []byte, value any) error {
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if e := d.Decode(value); e != nil {
		return e
	}
	if e := d.Decode(new(any)); e != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	wire, e := json.Marshal(value)
	if e != nil {
		return e
	}
	var shape any
	if e = json.Unmarshal(wire, &shape); e != nil {
		return e
	}
	d = json.NewDecoder(bytes.NewReader(body))
	return exactShape(d, shape)
}

func exactShape(d *json.Decoder, shape any) error {
	t, e := d.Token()
	if e != nil {
		return e
	}
	switch s := shape.(type) {
	case map[string]any:
		if t != json.Delim('{') {
			return fmt.Errorf("object required")
		}
		seen := map[string]bool{}
		for d.More() {
			key, e := d.Token()
			if e != nil {
				return e
			}
			k, ok := key.(string)
			child, known := s[k]
			if !ok || !known || seen[k] {
				return fmt.Errorf("unknown/duplicate JSON field %v", key)
			}
			seen[k] = true
			if e = exactShape(d, child); e != nil {
				return e
			}
		}
		if len(seen) != len(s) {
			return fmt.Errorf("omitted JSON field")
		}
		_, e = d.Token()
		return e
	case []any:
		if t != json.Delim('[') {
			return fmt.Errorf("array required")
		}
		for _, child := range s {
			if e = exactShape(d, child); e != nil {
				return e
			}
		}
		if d.More() {
			return fmt.Errorf("extra array member")
		}
		_, e = d.Token()
		return e
	default:
		if t == nil && shape != nil {
			return fmt.Errorf("null JSON field")
		}
		return nil
	}
}

func readExact(p Pin, value any) error {
	b, e := pinBytes(p)
	if e != nil {
		return e
	}
	return exactJSON(b, value)
}

func outputBoundary(q Request, fresh bool) error {
	if e := canonical(q.Root); e != nil {
		return e
	}
	if e := cleanAbsolute(q.Out); e != nil {
		return e
	}
	base := filepath.Join(q.Root, "artifacts", "tensor")
	if !within(base, q.Out) {
		return fmt.Errorf("output must be below artifacts/tensor")
	}
	if e := canonical(filepath.Dir(q.Out)); e != nil {
		return e
	}
	if fresh {
		if _, e := os.Lstat(q.Out); !os.IsNotExist(e) {
			return fmt.Errorf("fresh output required: %s", q.Out)
		}
	} else if e := canonical(q.Out); e != nil {
		return e
	}
	return nil
}

func readRequest(pin Pin, fresh bool) (q Request, err error) {
	if err = readExact(pin, &q); err != nil {
		return
	}
	if q.Schema != RequestSchema {
		return q, fmt.Errorf("fake capture request schema differs")
	}
	if err = outputBoundary(q, fresh); err != nil {
		return
	}
	paths := map[string]bool{pin.Path: true}
	for _, p := range []Pin{q.ControllerBuild, q.ControllerBuildSelection, q.Target, q.Logger, q.Compiler} {
		if paths[p.Path] {
			return q, fmt.Errorf("request input roles must be distinct")
		}
		paths[p.Path] = true
		if within(q.Out, p.Path) {
			return q, fmt.Errorf("input is inside fresh output")
		}
		if _, err = pinBytes(p); err != nil {
			return
		}
	}
	if within(q.Out, pin.Path) {
		return q, fmt.Errorf("request is inside output")
	}
	return
}

func acceptedPlan(q Request) (p cgobuild.Plan, err error) {
	a, err := cgobuild.ReadCPU13Artifact(q.ControllerBuild, q.ControllerBuildSelection)
	if err != nil {
		return p, err
	}
	if a.Output() != q.Target {
		return p, fmt.Errorf("independently expected target differs")
	}
	// The public reader authenticates the whole tuple. Restrict this producer to
	// its successful (rather than retained-observation) selection profile.
	b, err := pinBytes(q.ControllerBuildSelection)
	if err != nil {
		return p, err
	}
	var header struct {
		Schema string
		Role   cgobuild.BuildRole
	}
	if err = json.Unmarshal(b, &header); err != nil {
		return
	}
	if header.Schema != "nanopore-h08-successful-selection/v1" || header.Role != cgobuild.CPU13Tests {
		return p, fmt.Errorf("successful CPU13 selection required")
	}
	b, err = pinBytes(q.ControllerBuild)
	if err != nil {
		return
	}
	var r cgobuild.Receipt
	if err = json.Unmarshal(b, &r); err != nil {
		return
	}
	if r.Role != cgobuild.CPU13Tests || r.Status != "compiled_unexecuted" || r.Output != q.Target || !r.RuntimeHeld || r.Error != "" {
		return p, fmt.Errorf("successful CPU13 receipt required")
	}
	b, err = pinBytes(r.Plan)
	if err != nil {
		return
	}
	if err = json.Unmarshal(b, &p); err != nil {
		return
	}
	if p.Role != cgobuild.CPU13Tests || p.Options.Root != q.Root || p.CC != q.Compiler {
		return p, fmt.Errorf("accepted plan root/compiler/role differs")
	}
	_, err = compilerEnvironment(p, filepath.Join(q.Out, "tmp"))
	return
}

// Preflight authenticates metadata only. It creates nothing and grants no
// native admission. Capture repeats it inside the externally bounded unit.
func Preflight(pin Pin) (Request, error) {
	q, e := readRequest(pin, true)
	if e != nil {
		return q, e
	}
	_, e = acceptedPlan(q)
	return q, e
}

func compilerEnvironment(p cgobuild.Plan, temp string) ([]string, error) {
	// Fixed subset of the accepted Plan.Environment; exclude all Go-build
	// controls and all loader/framework variables. TMPDIR is session-private.
	env := []string{}
	for _, kv := range []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC"} {
		k, v, _ := strings.Cut(kv, "=")
		if p.Environment[k] != v {
			return nil, fmt.Errorf("accepted compiler environment differs: %s", k)
		}
		env = append(env, kv)
	}
	return append(env, "TMPDIR="+temp), nil
}

func targetArgs(q Request) []string {
	return []string{q.Target.Path, "-h08-profile", "fake", "-test.v", "-test.count=1", "-test.run", selector}
}

func writeNew(path string, value any) (Pin, error) {
	b, e := json.MarshalIndent(value, "", "  ")
	if e != nil {
		return Pin{}, e
	}
	return writeBytes(path, append(b, '\n'))
}

func writeBytes(path string, body []byte) (Pin, error) {
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0400)
	if e != nil {
		return Pin{}, e
	}
	_, e = f.Write(body)
	ce := f.Close()
	if e != nil {
		return Pin{}, e
	}
	if ce != nil {
		return Pin{}, ce
	}
	p, e := closedbuild.PinFile(path)
	if e == nil && p.SHA256 != hash(body) {
		e = fmt.Errorf("retained bytes changed: %s", path)
	}
	return p, e
}
