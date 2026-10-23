//go:build linux

package gpucaller

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/runledger"
)

const InventoryCaptureSchema = "runledger-events/1"
const InventoryImportSchema = "runledger-import/1"
const inventoryMaxBytes = 64 << 20
const inventoryMaxCaptureBytes = 16 << 20

// InventoryCapture contains retained observations, not instructions to run a
// producer. Import validates accounting only, not the origin of these events.
type InventoryCapture struct {
	Schema   string
	Files    []runledger.File
	Requests []runledger.Request
	Events   []InventoryEvent
}

type InventoryEvent struct {
	Op   string
	Data json.RawMessage
}

type InventoryReceipt struct {
	Schema            string
	Mode              Mode
	Started, Finished string
	InputSHA256       string
	CaptureSHA256     string
	CaptureJSON       []byte
	Report            runledger.Report
}

// InventoryHandlers supplies only host import/replay. P05 owns registration.
func InventoryHandlers() UnitHandlers {
	return UnitHandlers{Import: inventoryImport, Replay: inventoryReplay, Native: nil}
}

func inventoryDecode(b []byte, out any) error {
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return fmt.Errorf("null inventory document")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("expected one JSON document")
	}
	return nil
}

func inventoryHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func inventoryRead(path string) ([]byte, error) {
	// Linux nonblocking open precedes descriptor inspection, including for FIFOs
	// reached through a symlink or a concurrent path replacement.
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	if f == nil {
		syscall.Close(fd)
		return nil, fmt.Errorf("invalid inventory descriptor")
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("inventory input must be a regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, inventoryMaxBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > inventoryMaxBytes {
		return nil, fmt.Errorf("inventory input exceeds %d bytes", inventoryMaxBytes)
	}
	return b, nil
}

func inventoryRequest(r Request, mode Mode) error {
	if r.Unit != Inventory || r.Mode != mode || r.Input == "" || r.Output == "" {
		return fmt.Errorf("inventory requires its exact host unit/mode and input/fresh output")
	}
	return nil
}

func inventoryImport(r Request) error {
	if err := inventoryRequest(r, Import); err != nil {
		return err
	}
	started := time.Now().UTC().Format(time.RFC3339Nano)
	b, err := inventoryRead(r.Input)
	if err != nil {
		return err
	}
	if len(b) > inventoryMaxCaptureBytes {
		return fmt.Errorf("inventory capture exceeds %d bytes", inventoryMaxCaptureBytes)
	}
	report, err := inventoryReduce(b)
	if err != nil {
		return err
	}
	return inventorySave(r, started, b, b, report)
}

func inventoryReplay(r Request) error {
	if err := inventoryRequest(r, Replay); err != nil {
		return err
	}
	started := time.Now().UTC().Format(time.RFC3339Nano)
	b, err := inventoryRead(r.Input)
	if err != nil {
		return err
	}
	var saved InventoryReceipt
	if err := inventoryShape(b, "receipt", true); err != nil {
		return err
	}
	if err := inventoryDecode(b, &saved); err != nil {
		return err
	}
	if saved.Schema != InventoryImportSchema || (saved.Mode != Import && saved.Mode != Replay) || saved.CaptureSHA256 != inventoryHash(saved.CaptureJSON) {
		return fmt.Errorf("invalid imported inventory identity or capture pin")
	}
	report, err := inventoryReduce(saved.CaptureJSON)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(saved.Report, report) {
		return fmt.Errorf("retained inventory report differs from replay")
	}
	return inventorySave(r, started, b, saved.CaptureJSON, report)
}

func inventorySave(r Request, started string, input, capture []byte, report runledger.Report) error {
	receipt := InventoryReceipt{
		Schema: InventoryImportSchema, Mode: r.Mode,
		Started: started, Finished: time.Now().UTC().Format(time.RFC3339Nano),
		InputSHA256: inventoryHash(input), CaptureSHA256: inventoryHash(capture),
		CaptureJSON: capture, Report: report,
	}
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	if len(b)+1 > inventoryMaxBytes {
		return fmt.Errorf("inventory receipt exceeds replay limit")
	}
	// A fresh directory prevents overwrite and retains partial failed attempts.
	if err := os.Mkdir(r.Output, 0755); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(r.Output, "inventory.json"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(append(b, '\n'))
	return errors.Join(writeErr, f.Close())
}

func inventoryReduce(b []byte) (runledger.Report, error) {
	if len(b) > inventoryMaxCaptureBytes {
		return runledger.Report{}, fmt.Errorf("inventory capture too large")
	}
	var capture InventoryCapture
	if err := inventoryShape(b, "capture", false); err != nil {
		return runledger.Report{}, err
	}
	if err := inventoryDecode(b, &capture); err != nil {
		return runledger.Report{}, err
	}
	if capture.Schema != InventoryCaptureSchema {
		return runledger.Report{}, fmt.Errorf("unsupported inventory capture schema %q", capture.Schema)
	}
	l, err := runledger.New(capture.Files, capture.Requests)
	if err != nil {
		return runledger.Report{}, err
	}
	for i, e := range capture.Events {
		if err := inventoryApply(l, e); err != nil {
			return runledger.Report{}, fmt.Errorf("inventory event %d %q: %w", i, e.Op, err)
		}
	}
	return l.Finalize()
}

// Each operation decodes its own closed payload so irrelevant fields cannot
// silently change meaning when the event format grows.
func inventoryApply(l *runledger.Ledger, e InventoryEvent) error {
	required := map[string][]string{
		"load": {"Index", "Raw", "Admitted"}, "chunks": {"Index", "Count"},
		"stop": {"Index", "State", "Reason"}, "children": {"Index", "Children"},
		"open-writer": {"Path"}, "write": {"Index", "Write"},
		"close-writer": {"Path", "Records", "Failure"}, "complete": {"Index"},
	}
	fields, ok := required[e.Op]
	if !ok {
		return fmt.Errorf("unknown inventory operation %q", e.Op)
	}
	payload, err := inventoryObject(e.Data, fields, nil)
	if err != nil {
		return err
	}
	if e.Op == "children" {
		if err := inventoryArray(payload["Children"], "child", false); err != nil {
			return err
		}
	}
	if e.Op == "write" {
		if err := inventoryShape(payload["Write"], "write", false); err != nil {
			return err
		}
	}
	switch e.Op {
	case "load":
		var v struct {
			Index         int
			Raw, Admitted uint64
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.Load(v.Index, v.Raw, v.Admitted)
	case "chunks":
		var v struct {
			Index int
			Count uint64
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.AddChunks(v.Index, v.Count)
	case "stop":
		var v struct {
			Index  int
			State  runledger.State
			Reason string
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.Stop(v.Index, v.State, v.Reason)
	case "children":
		var v struct {
			Index    int
			Children []runledger.Child
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.SetChildren(v.Index, v.Children)
	case "open-writer":
		var v struct{ Path string }
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.OpenWriter(v.Path)
	case "write":
		var v struct {
			Index int
			Write runledger.Write
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.RecordWrite(v.Index, v.Write)
	case "close-writer":
		var v struct {
			Path    string
			Records uint64
			Failure string
		}
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		err := l.CloseWriter(v.Path, v.Records, v.Failure)
		if err == nil {
			return nil
		}
		// Only a retained physical closure error is an accounted event. Invalid
		// opens, duplicate closure and other protocol errors remain rejections.
		var closure *runledger.WriterClosureError
		if errors.As(err, &closure) {
			return nil
		}
		return err
	case "complete":
		var v struct{ Index int }
		if err := inventoryDecode(e.Data, &v); err != nil {
			return err
		}
		return l.Complete(v.Index)
	default:
		return fmt.Errorf("unknown inventory operation %q", e.Op)
	}
}

// This is the inventory wire schema only. The maintained koiquant readClosed
// helper is private and file-based; its null-slice policy also differs from
// capture observations. No shared verifier or registry contract is changed.
func inventoryObject(raw []byte, fields, nullable []string) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	if token != json.Delim('{') {
		return nil, fmt.Errorf("required inventory object")
	}
	values := make(map[string]json.RawMessage, len(fields))
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("invalid inventory field")
		}
		if _, exists := values[name]; exists {
			return nil, fmt.Errorf("duplicate inventory field %q", name)
		}
		allowed := false
		for _, field := range fields {
			if name == field {
				allowed = true
				break
			}
		}
		if !allowed {
			return nil, fmt.Errorf("unknown exact inventory field %q", name)
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return nil, err
		}
		canNull := false
		for _, field := range nullable {
			if name == field {
				canNull = true
				break
			}
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) && !canNull {
			return nil, fmt.Errorf("null inventory field %q", name)
		}
		values[name] = value
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing inventory JSON")
	}
	for _, name := range fields {
		if _, exists := values[name]; !exists {
			return nil, fmt.Errorf("missing inventory field %q", name)
		}
	}
	return values, nil
}

func inventoryArray(raw []byte, element string, report bool) error {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if report {
			return nil
		}
		return fmt.Errorf("null capture array")
	}
	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		return err
	}
	for _, value := range values {
		if element == "string" {
			if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return fmt.Errorf("null parent ID")
			}
			var id string
			if err := json.Unmarshal(value, &id); err != nil {
				return err
			}
		} else if err := inventoryShape(value, element, report); err != nil {
			return err
		}
	}
	return nil
}

func inventoryShape(raw []byte, shape string, report bool) error {
	var names string
	arrays := map[string]string{}
	objects := map[string]string{}
	switch shape {
	case "capture":
		names = "Schema Files Requests Events"
		arrays = map[string]string{"Files": "file", "Requests": "request", "Events": "event"}
	case "event":
		names = "Op Data"
	case "file":
		names = "UUID ParentIDs"
		arrays["ParentIDs"] = "string"
	case "request":
		names = "FileUUID ParentID"
	case "child":
		names = "ID ParentID Keep FilterReason"
	case "write":
		names = "ChildID ParentID Path"
	case "receipt":
		names = "Schema Mode Started Finished InputSHA256 CaptureSHA256 CaptureJSON Report"
		objects["Report"] = "report"
	case "report":
		names = "Files Parents Writers RawSamples AdmittedSamples Chunks Children Kept Written"
		arrays = map[string]string{"Files": "file", "Parents": "parent", "Writers": "writer"}
	case "parent":
		names = "Request State Reason RawSamples AdmittedSamples Chunks Children Writes"
		objects["Request"] = "request"
		arrays = map[string]string{"Children": "child", "Writes": "write"}
	case "writer":
		names = "Path Records Recorded Mismatch Closed Failure"
	default:
		return fmt.Errorf("unknown inventory wire shape %q", shape)
	}
	var nullable []string
	if report {
		for name := range arrays {
			nullable = append(nullable, name)
		}
	}
	values, err := inventoryObject(raw, strings.Fields(names), nullable)
	if err != nil {
		return fmt.Errorf("%s: %w", shape, err)
	}
	for name, element := range arrays {
		if err := inventoryArray(values[name], element, report); err != nil {
			return fmt.Errorf("%s.%s: %w", shape, name, err)
		}
	}
	for name, child := range objects {
		if err := inventoryShape(values[name], child, report); err != nil {
			return fmt.Errorf("%s.%s: %w", shape, name, err)
		}
	}
	return nil
}
