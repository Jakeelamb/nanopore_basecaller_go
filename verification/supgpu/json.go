package supgpu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
)

// decodeClosed checks exact key spelling, duplicates, missing fields and nulls
// before Go's case-insensitive decoder can merge them.
func decodeClosed(raw []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := checkJSON(d, reflect.TypeOf(out).Elem()); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON: %v", err)
	}
	return json.Unmarshal(raw, out)
}

func checkJSON(d *json.Decoder, t reflect.Type) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		return fmt.Errorf("null %s", t)
	}
	switch t.Kind() {
	case reflect.Struct:
		if tok != json.Delim('{') {
			return fmt.Errorf("expected object for %s", t)
		}
		fields := map[string]reflect.Type{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			fields[f.Tag.Get("json")] = f.Type
		}
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return fmt.Errorf("nonstring key")
			}
			ft, ok := fields[name]
			if !ok || seen[name] {
				return fmt.Errorf("unknown or duplicate key %q", name)
			}
			seen[name] = true
			if err := checkJSON(d, ft); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
		if len(seen) != len(fields) {
			return fmt.Errorf("missing fields in %s", t)
		}
		_, err = d.Token()
		return err
	case reflect.Slice:
		if tok != json.Delim('[') {
			return fmt.Errorf("expected array for %s", t)
		}
		for d.More() {
			if err := checkJSON(d, t.Elem()); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	default:
		if _, ok := tok.(json.Delim); ok {
			return fmt.Errorf("unexpected container for %s", t)
		}
		return nil
	}
}
