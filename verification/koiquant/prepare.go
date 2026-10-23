package koiquant

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

type Preparation struct {
	Schema, Status, Device, NativeAdmission string
	NativeExecution                         bool
	Started, Finished, Error                string
	Invocation                              []string
	OriginManifest, OriginSHA256            string
	Grounding, Sources, Generated           map[string]string
	Blocks                                  []Block
	Fixtures                                []Fixture
	Stages                                  []StageSpec
}

type Options struct {
	Root, InputCapture, Out, Device string
	Invocation                      []string
}

func grounding(root string) map[string]string {
	return map[string]string{
		filepath.Join(root, "artifacts/koiweights/int8-design/design.md"): "9115492cd9dd2a5c0b279c8c78ad95625c2bbc9b5a1e8aacd89a53e0040696b6",
		"/tmp/nanopore-go-sup-int8-contract/report.md":                    "8389bca7b02a5350b6f115ad55c0f8a01f56ccf668f0a35411451772aaa698c4",
		"/tmp/nanopore-go-sup-int8-contract/source-pins.sha256":           "494785b3ba70e9f1dabc80b18f13b275f882d661db7386c63dd6b6e9d3a88977",
	}
}

func checkGrounding(root string, saved map[string]string) error {
	if !reflect.DeepEqual(grounding(root), saved) {
		return fmt.Errorf("missing or changed grounding pin")
	}
	for path, hash := range saved {
		if err := checkHash(path, hash); err != nil {
			return err
		}
	}
	return nil
}

// ReuseInput reads the already-interleaved raw half storage. It never loads an archive.
func ReuseInput(root, path string) (koiweights.Data, []uint16, error) {
	if err := checkHash(path, OriginSHA256); err != nil {
		return koiweights.Data{}, nil, err
	}
	m, err := koiweights.ValidateCapture(path, root)
	if err != nil {
		return koiweights.Data{}, nil, err
	}
	var data koiweights.Data
	found := 0
	for _, f := range m.Fixtures {
		if f.Name == "genuine" {
			data = f.Interleaved
			found++
		}
	}
	if found != 1 || data.SHA256 != GenuineSHA256 {
		return data, nil, fmt.Errorf("missing or changed genuine unrounded input")
	}
	if err := validateData(data, StageSpecs()[0]); err != nil {
		return data, nil, err
	}
	values, err := koiweights.ReadHalf(data)
	if err != nil {
		return data, nil, err
	}
	if len(values) != Elements || digest(halfBytes(values)) != GenuineSHA256 {
		return data, nil, fmt.Errorf("genuine input element count or read-buffer pin mismatch")
	}
	if err := checkHash(path, OriginSHA256); err != nil {
		return data, nil, err
	}
	return data, values, nil
}

func admitOutput(root, out string) error {
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil || realRoot != root {
		return fmt.Errorf("module root must be an absolute real directory")
	}
	b, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil || !strings.HasPrefix(string(b), "module github.com/Jakeelamb/nanopore_basecaller_go\n") {
		return fmt.Errorf("run from the canonical Go module root")
	}
	parent := filepath.Join(root, "artifacts", "koiquant")
	if filepath.Dir(out) != parent || !strings.HasPrefix(filepath.Base(out), "attempt-") {
		return fmt.Errorf("output must be a fresh artifacts/koiquant/attempt-* directory")
	}
	for _, path := range []string{root, filepath.Join(root, "artifacts"), parent} {
		if st, err := os.Lstat(path); err == nil {
			if st.Mode()&os.ModeSymlink != 0 || !st.IsDir() {
				return fmt.Errorf("output ancestor must be a real directory: %s", path)
			}
		} else if !os.IsNotExist(err) || path != parent {
			return err
		}
	}
	if _, err := os.Lstat(out); !os.IsNotExist(err) {
		return fmt.Errorf("output already exists or cannot be inspected: %s", out)
	}
	return nil
}

func clock() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// Prepare writes only source and input payloads into a fresh immutable attempt.
func Prepare(o Options) (m Preparation, err error) {
	o.Root, err = filepath.Abs(o.Root)
	if err != nil {
		return m, err
	}
	o.Out, err = filepath.Abs(o.Out)
	if err != nil {
		return m, err
	}
	if o.Device != "cpu" && o.Device != "cuda" {
		return m, fmt.Errorf("device must be cpu or cuda")
	}
	if err = admitOutput(o.Root, o.Out); err != nil {
		return m, err
	}
	m = Preparation{Schema: Schema, Status: "preparing", Device: o.Device, NativeAdmission: "held", Started: clock(), Invocation: append([]string(nil), o.Invocation...), OriginSHA256: OriginSHA256, Grounding: grounding(o.Root), Stages: StageSpecs()}
	m.OriginManifest, err = filepath.Abs(o.InputCapture)
	if err != nil {
		return m, err
	}
	if err = checkGrounding(o.Root, m.Grounding); err != nil {
		return m, err
	}
	m.Blocks, err = Extract(o.Root)
	if err != nil {
		return m, err
	}
	m.Sources, err = currentSources(o.Root)
	if err != nil {
		return m, err
	}
	_, genuine, err := ReuseInput(o.Root, m.OriginManifest)
	if err != nil {
		return m, err
	}
	if err = os.MkdirAll(filepath.Dir(o.Out), 0755); err != nil {
		return m, err
	}
	if err = os.Mkdir(o.Out, 0755); err != nil {
		return m, err
	}
	defer func() {
		m.Finished = clock()
		if err != nil {
			m.Status, m.Error = "prepare_failed", err.Error()
		}
		b, marshalErr := json.MarshalIndent(m, "", "  ")
		if marshalErr == nil {
			marshalErr = os.WriteFile(filepath.Join(o.Out, "manifest.json"), append(b, '\n'), 0644)
		}
		if err == nil {
			err = marshalErr
		}
	}()
	if err = os.Mkdir(filepath.Join(o.Out, "inputs"), 0755); err != nil {
		return m, err
	}
	m.Fixtures, err = writeFixtures(o.Out, genuine)
	if err != nil {
		return m, err
	}
	cpp, err := generated(m.Blocks)
	if err != nil {
		return m, err
	}
	m.Generated = make(map[string]string)
	for name, content := range map[string]string{"oracle.cpp": cpp, "fixture-list.txt": strings.Join(fixtureNames, "\n") + "\n"} {
		path := filepath.Join(o.Out, name)
		if err = os.WriteFile(path, []byte(content), 0644); err != nil {
			return m, err
		}
		m.Generated[name] = digest([]byte(content))
	}
	if err = checkHash(m.OriginManifest, OriginSHA256); err != nil {
		return m, err
	}
	if err = checkSources(o.Root, m.Sources); err != nil {
		return m, err
	}
	m.Status = "prepared_native_held"
	return m, nil
}

// CheckPrepared revalidates current source and all prepared inputs without a producer.
func CheckPrepared(root, path string) (Preparation, error) {
	var m Preparation
	root, err := filepath.Abs(root)
	if err != nil {
		return m, err
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return m, err
	}
	if err = readClosed(path, &m); err != nil {
		return m, err
	}
	if m.Schema != Schema || m.Status != "prepared_native_held" || m.Error != "" || m.NativeExecution || m.NativeAdmission != "held" || (m.Device != "cpu" && m.Device != "cuda") || m.OriginSHA256 != OriginSHA256 {
		return m, fmt.Errorf("invalid source-only preparation identity")
	}
	start, err := time.Parse(time.RFC3339Nano, m.Started)
	if err != nil {
		return m, err
	}
	finish, err := time.Parse(time.RFC3339Nano, m.Finished)
	if err != nil || finish.Before(start) || len(m.Invocation) == 0 {
		return m, fmt.Errorf("missing invocation or invalid preparation clocks")
	}
	if err = checkGrounding(root, m.Grounding); err != nil {
		return m, err
	}
	blocks, err := Extract(root)
	if err != nil {
		return m, err
	}
	if !reflect.DeepEqual(blocks, m.Blocks) || !reflect.DeepEqual(StageSpecs(), m.Stages) {
		return m, fmt.Errorf("source block or stage schema mismatch")
	}
	if err = checkSources(root, m.Sources); err != nil {
		return m, err
	}
	_, _, err = ReuseInput(root, m.OriginManifest)
	if err != nil {
		return m, err
	}
	cpp, err := generated(blocks)
	if err != nil {
		return m, err
	}
	expected := map[string]string{"oracle.cpp": digest([]byte(cpp)), "fixture-list.txt": digest([]byte(strings.Join(fixtureNames, "\n") + "\n"))}
	if !reflect.DeepEqual(expected, m.Generated) || len(m.Fixtures) != len(fixtureNames) {
		return m, fmt.Errorf("missing or changed generated source or fixture set")
	}
	for name, hash := range expected {
		if err = checkHash(filepath.Join(filepath.Dir(path), name), hash); err != nil {
			return m, err
		}
	}
	for i, fixture := range m.Fixtures {
		if fixture.Name != fixtureNames[i] || fixture.Domain != fixtureDomain(fixture.Name) || fixture.Input.Path != filepath.Join(filepath.Dir(path), "inputs", fixture.Name+"-u16le.bin") {
			return m, fmt.Errorf("fixture identity, order, or path mismatch")
		}
		if err = validateData(fixture.Input, StageSpecs()[0]); err != nil {
			return m, err
		}
		if fixture.Name == "genuine" {
			if fixture.Input.SHA256 != GenuineSHA256 {
				return m, fmt.Errorf("genuine fixture pin mismatch")
			}
		} else {
			values, fixtureErr := fixtureHalf(fixture.Name)
			if fixtureErr != nil {
				return m, fixtureErr
			}
			actual, fixtureErr := koiweights.ReadHalf(fixture.Input)
			if fixtureErr != nil || !reflect.DeepEqual(actual, values) {
				return m, fmt.Errorf("deterministic fixture payload mismatch: %s", fixture.Name)
			}
		}
	}
	return m, checkHash(m.OriginManifest, OriginSHA256)
}
