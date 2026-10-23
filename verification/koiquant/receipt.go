package koiquant

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Pin struct{ Path, SHA256 string }
type EnvironmentValue struct {
	Present bool
	Value   string
}

type RawCapture struct {
	Schema, Status, Started, Finished, Error                                                  string
	Device                                                                                    string
	Ordinal                                                                                   int
	Preparation                                                                               Pin
	Prepared                                                                                  Preparation
	Basic, Build, Archive, Binary, Compiler, BinaryOrigin                                     Pin
	Torch, CUDA                                                                               string
	Invocation                                                                                []string
	Environment                                                                               map[string]EnvironmentValue
	ProducerSources, Dependencies, ArchiveMembers, Linked, MappedBefore, MappedAfter, Outputs map[string]string
	GPUIdentity                                                                               string
	Commands                                                                                  []runner.CommandReceipt
}

const CaptureSchema = "nanopore-sup-qkv-i8/raw-v1"
const ImportSchema = "nanopore-sup-qkv-i8/import-v1"
const BasicSHA256 = "15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae"
const BuildSHA256 = "f41faaed7ba863f3d16c5a5f08cb373bf588794ec73fe25d1620359863b9d0dd"
const BackendSHA256 = "d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7"

func filePin(path string) (Pin, error) {
	h, err := runner.SHA(path)
	return Pin{path, h}, err
}

func validatePin(p Pin) error { return checkHash(p.Path, p.SHA256) }

func writeJSON(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}

// Every declared field is required, including zero-valued booleans and integers.
func readClosed(path string, out any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err = uniqueJSON(dec); err != nil {
		return err
	}
	if _, err = dec.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON data")
	}
	dec = json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err = dec.Decode(out); err != nil {
		return err
	}
	var tree any
	if err = json.Unmarshal(raw, &tree); err != nil {
		return err
	}
	return requiredFields(tree, reflect.TypeOf(out).Elem(), "receipt")
}

func uniqueJSON(dec *json.Decoder) error {
	token, err := dec.Token()
	if err != nil {
		return err
	}
	d, compound := token.(json.Delim)
	if !compound {
		return nil
	}
	seen := map[string]bool{}
	for dec.More() {
		if d == '{' {
			token, err = dec.Token()
			if err != nil {
				return err
			}
			key, ok := token.(string)
			if !ok || seen[key] {
				return fmt.Errorf("duplicate or invalid JSON field %v", token)
			}
			seen[key] = true
		}
		if err = uniqueJSON(dec); err != nil {
			return err
		}
	}
	_, err = dec.Token()
	return err
}

func requiredFields(value any, typ reflect.Type, path string) error {
	if typ.Kind() == reflect.Pointer {
		return requiredFields(value, typ.Elem(), path)
	}
	switch typ.Kind() {
	case reflect.Struct:
		object, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("required object %s", path)
		}
		allowed := map[string]bool{}
		for i := 0; i < typ.NumField(); i++ {
			f := typ.Field(i)
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "-" || !f.IsExported() {
				continue
			}
			if name == "" {
				name = f.Name
			}
			allowed[name] = true
			v, exists := object[name]
			if !exists {
				return fmt.Errorf("missing required field %s.%s", path, name)
			}
			if err := requiredFields(v, f.Type, path+"."+name); err != nil {
				return err
			}
		}
		for key := range object {
			if !allowed[key] {
				return fmt.Errorf("unknown exact field %s.%s", path, key)
			}
		}
	case reflect.Map:
		if value == nil {
			return nil
		}
		object, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("required map %s", path)
		}
		for key, v := range object {
			if err := requiredFields(v, typ.Elem(), path+"."+key); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if value == nil {
			return nil
		}
		array, ok := value.([]any)
		if !ok {
			return fmt.Errorf("required array %s", path)
		}
		for _, v := range array {
			if err := requiredFields(v, typ.Elem(), path+"[]"); err != nil {
				return err
			}
		}
	default:
		if value == nil {
			return fmt.Errorf("null required value %s", path)
		}
	}
	return nil
}

func validateClocks(started, finished string) error {
	start, err := time.Parse(time.RFC3339Nano, started)
	if err != nil {
		return err
	}
	finish, err := time.Parse(time.RFC3339Nano, finished)
	if err != nil || finish.Before(start) {
		return fmt.Errorf("invalid receipt clocks")
	}
	return nil
}

func validateReceipt(c runner.CommandReceipt, allowFailure bool) error {
	if len(c.Args) == 0 || c.Started == "" || c.Finished == "" || c.Exit < 0 || (!allowFailure && c.Exit != 0) {
		return fmt.Errorf("missing command or failed receipt")
	}
	if err := validateClocks(c.Started, c.Finished); err != nil {
		return err
	}
	if err := checkHash(c.Stdout, c.StdoutSHA); err != nil {
		return err
	}
	return checkHash(c.Stderr, c.StderrSHA)
}
