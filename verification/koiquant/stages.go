package koiquant

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

type StageSpec struct {
	Name, DType    string
	Shape, Strides []int
}

// StageSpecs names storage outputs. These expected dtypes still require native observation.
func StageSpecs() []StageSpec {
	tensor := []int{3, 8, 64, 512}
	tensorStrides := []int{262144, 32768, 512, 1}
	row := []int{3, 8, 64}
	rowStrides := []int{512, 64, 1}
	tile := []int{3, 8, 4, 32, 16, 16}
	tileStrides := []int{262144, 32768, 8192, 256, 16, 1}
	return []StageSpec{
		{"input", "binary16", tensor, tensorStrides}, {"transfer", "binary16", tensor, tensorStrides},
		{"abs", "binary16", tensor, tensorStrides}, {"amax", "binary16", row, rowStrides},
		{"hscale", "binary16", row, rowStrides}, {"hproduct", "binary16", tensor, tensorStrides},
		{"hround", "binary16", tensor, tensorStrides}, {"hclip", "binary16", tensor, tensorStrides},
		{"char", "int8", tensor, tensorStrides}, {"fscale", "float32", row, rowStrides},
		{"freciprocal", "float32", row, rowStrides}, {"i8tile", "int8", tile, tileStrides},
		{"untouched-i8tile", "int8", tile, tileStrides}, {"untouched-freciprocal", "float32", row, rowStrides},
	}
}

func width(dtype string) (int64, error) {
	switch dtype {
	case "binary16":
		return 2, nil
	case "int8":
		return 1, nil
	case "float32":
		return 4, nil
	default:
		return 0, fmt.Errorf("unknown dtype %q", dtype)
	}
}

func validateData(d koiweights.Data, spec StageSpec) error {
	w, err := width(d.DType)
	if err != nil {
		return err
	}
	if d.DType != spec.DType || d.Endian != "little" || !reflect.DeepEqual(d.Shape, spec.Shape) || !reflect.DeepEqual(d.Strides, spec.Strides) || len(d.Tags) != 0 {
		return fmt.Errorf("invalid storage contract for %s", spec.Name)
	}
	n := w
	for _, size := range spec.Shape {
		n *= int64(size)
	}
	if d.Bytes != n {
		return fmt.Errorf("invalid byte count for %s", spec.Name)
	}
	st, err := os.Stat(d.Path)
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() || st.Size() != n {
		return fmt.Errorf("invalid payload size for %s", spec.Name)
	}
	return checkHash(d.Path, d.SHA256)
}

// ValidateStages checks saved observations, including transfer and untouched parity.
// It does not qualify the native runtime or prove a Go arithmetic implementation.
func ValidateStages(stages map[string]koiweights.Data, input koiweights.Data) error {
	specs := StageSpecs()
	if err := validateData(input, specs[0]); err != nil {
		return err
	}
	if len(stages) != len(specs) {
		return fmt.Errorf("missing or unknown stage")
	}
	for _, spec := range specs {
		if err := validateData(stages[spec.Name], spec); err != nil {
			return err
		}
	}
	for _, pair := range [][2]string{{"input", "transfer"}, {"i8tile", "untouched-i8tile"}, {"freciprocal", "untouched-freciprocal"}} {
		if stages[pair[0]].SHA256 != stages[pair[1]].SHA256 {
			return fmt.Errorf("native byte parity failed for %s and %s", pair[0], pair[1])
		}
	}
	if stages["input"].SHA256 != input.SHA256 {
		return fmt.Errorf("saved observation input differs from prepared input")
	}
	return nil
}

// RequireTestPass rejects regexes which matched nothing, skipped tests, and exit-only claims.
func RequireTestPass(raw []byte, required []string) error {
	if len(required) == 0 {
		return fmt.Errorf("empty required test set")
	}
	seen := make(map[string]map[string]int)
	dec := json.NewDecoder(bytes.NewReader(raw))
	for {
		var event struct{ Action, Test string }
		if err := dec.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		if event.Test != "" {
			if seen[event.Test] == nil {
				seen[event.Test] = make(map[string]int)
			}
			seen[event.Test][event.Action]++
		}
		if event.Action == "skip" || event.Action == "fail" {
			return fmt.Errorf("consumer %s: %s", event.Action, event.Test)
		}
	}
	for _, name := range required {
		if seen[name]["run"] != 1 || seen[name]["pass"] != 1 {
			return fmt.Errorf("required test did not run and pass exactly once: %s", name)
		}
	}
	return nil
}

// ValidateCommand checks raw receipts before interpreting any command output.
func ValidateCommand(c koiweights.Command) error {
	start, err := time.Parse(time.RFC3339Nano, c.Started)
	if err != nil {
		return err
	}
	finish, err := time.Parse(time.RFC3339Nano, c.Finished)
	if err != nil || finish.Before(start) || len(c.Args) == 0 || c.Phase == "" {
		return fmt.Errorf("invalid command identity or clocks")
	}
	if err := checkHash(c.Stdout, c.StdoutSHA256); err != nil {
		return err
	}
	if err := checkHash(c.Stderr, c.StderrSHA256); err != nil {
		return err
	}
	if c.Exit != 0 {
		return fmt.Errorf("command %s exited %d", c.Phase, c.Exit)
	}
	return nil
}
