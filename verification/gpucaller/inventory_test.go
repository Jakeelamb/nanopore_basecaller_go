//go:build linux

package gpucaller

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/runledger"
)

// These observations are synthetic accounting inputs, not production identity
// or evidence of POD5/output provenance, native parity or biological accuracy.
const inventoryTestCapture = `{
 "Schema":"runledger-events/1",
 "Files":[{"UUID":"synthetic-file","ParentIDs":["synthetic-parent"]}],
 "Requests":[{"FileUUID":"synthetic-file","ParentID":"synthetic-parent"}],
 "Events":[
  {"Op":"load","Data":{"Index":0,"Raw":20,"Admitted":10}},
  {"Op":"chunks","Data":{"Index":0,"Count":2}},
  {"Op":"children","Data":{"Index":0,"Children":[{"ID":"synthetic-parent","ParentID":"","Keep":true,"FilterReason":""}]}},
  {"Op":"open-writer","Data":{"Path":"synthetic-output.fastq"}},
  {"Op":"write","Data":{"Index":0,"Write":{"ChildID":"synthetic-parent","ParentID":"","Path":"synthetic-output.fastq"}}},
  {"Op":"close-writer","Data":{"Path":"synthetic-output.fastq","Records":1,"Failure":""}},
  {"Op":"complete","Data":{"Index":0}}
 ]
}`

func inventoryTestWrite(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

func inventoryTestRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func inventoryTestJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func inventoryTestReceipt(t *testing.T, dir string) InventoryReceipt {
	t.Helper()
	var receipt InventoryReceipt
	if err := json.Unmarshal(inventoryTestRead(t, filepath.Join(dir, "inventory.json")), &receipt); err != nil {
		t.Fatal(err)
	}
	return receipt
}

func inventoryTestImported(t *testing.T) (string, string, InventoryReceipt) {
	t.Helper()
	root := t.TempDir()
	input := filepath.Join(root, "synthetic-capture.json")
	out := filepath.Join(root, "import")
	inventoryTestWrite(t, input, []byte(inventoryTestCapture))
	if err := Dispatch(Request{Unit: Inventory, Mode: Import, Input: input, Output: out}, Handlers{Inventory: InventoryHandlers()}); err != nil {
		t.Fatal(err)
	}
	return root, out, inventoryTestReceipt(t, out)
}

func inventoryTestNoOutput(t *testing.T, output string) {
	t.Helper()
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("rejected request created output or stat failed: %v", err)
	}
}

func TestInventoryHostImportReplay(t *testing.T) {
	root, imported, first := inventoryTestImported(t)
	h := sha256.Sum256([]byte(inventoryTestCapture))
	wantHash := hex.EncodeToString(h[:])
	if first.Schema != InventoryImportSchema || first.Mode != Import || first.InputSHA256 != wantHash || first.CaptureSHA256 != wantHash || !bytes.Equal(first.CaptureJSON, []byte(inventoryTestCapture)) {
		t.Fatalf("capture identity changed %+v", first)
	}
	r := first.Report
	if len(r.Parents) != 1 || len(r.Files) != 1 || len(r.Writers) != 1 || r.RawSamples != 20 || r.AdmittedSamples != 10 || r.Chunks != 2 || r.Children != 1 || r.Kept != 1 || r.Written != 1 {
		t.Fatalf("wrong accounting %+v", r)
	}
	if r.Parents[0].State != runledger.ChildBearing || r.Writers[0].Records != 1 || r.Writers[0].Recorded != 1 || r.Writers[0].Mismatch || !r.Writers[0].Closed || r.Writers[0].Failure != "" {
		t.Fatalf("wrong dispositions %+v", r)
	}
	start, err := time.Parse(time.RFC3339Nano, first.Started)
	if err != nil {
		t.Fatal(err)
	}
	end, err := time.Parse(time.RFC3339Nano, first.Finished)
	if err != nil || end.Before(start) {
		t.Fatalf("invalid receipt clocks %v", err)
	}
	input := filepath.Join(imported, "inventory.json")
	before := inventoryTestRead(t, input)
	for _, name := range []string{"replay", "replay-again"} {
		out := filepath.Join(root, name)
		if err := Dispatch(Request{Unit: Inventory, Mode: Replay, Input: input, Output: out}, Handlers{Inventory: InventoryHandlers()}); err != nil {
			t.Fatal(err)
		}
		next := inventoryTestReceipt(t, out)
		h := sha256.Sum256(inventoryTestRead(t, input))
		if next.Mode != Replay || next.InputSHA256 != hex.EncodeToString(h[:]) || next.CaptureSHA256 != wantHash || !reflect.DeepEqual(first.Report, next.Report) {
			t.Fatal("replay changed retained accounting")
		}
		input = filepath.Join(out, "inventory.json")
	}
	if !bytes.Equal(before, inventoryTestRead(t, filepath.Join(imported, "inventory.json"))) {
		t.Fatal("replay modified imported receipt")
	}
	if _, err := os.Stat(filepath.Join(root, "synthetic-output.fastq")); !os.IsNotExist(err) {
		t.Fatal("adapter created a declared writer output")
	}
}

func TestInventoryReplayRejectsCorruptReportCaptureAndPin(t *testing.T) {
	mutations := map[string]func(*InventoryReceipt){
		"report-count":  func(r *InventoryReceipt) { r.Report.RawSamples++ },
		"report-parent": func(r *InventoryReceipt) { r.Report.Parents[0].State = runledger.ZeroSurvivor },
		"report-writer": func(r *InventoryReceipt) { r.Report.Writers[0].Records = 0 },
		"capture-bytes": func(r *InventoryReceipt) { r.CaptureJSON = append(r.CaptureJSON, ' ') },
		"capture-pin":   func(r *InventoryReceipt) { r.CaptureSHA256 = strings.Repeat("0", 64) },
		"valid-capture-repinned": func(r *InventoryReceipt) {
			r.CaptureJSON = bytes.Replace(r.CaptureJSON, []byte(`"Raw":20`), []byte(`"Raw":21`), 1)
			h := sha256.Sum256(r.CaptureJSON)
			r.CaptureSHA256 = hex.EncodeToString(h[:])
		},
		"schema":            func(r *InventoryReceipt) { r.Schema = "unknown" },
		"saved-native-mode": func(r *InventoryReceipt) { r.Mode = Native },
		"invalid-capture-repinned": func(r *InventoryReceipt) {
			r.CaptureJSON = []byte(`{"Schema":"unknown"}`)
			h := sha256.Sum256(r.CaptureJSON)
			r.CaptureSHA256 = hex.EncodeToString(h[:])
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			root, _, receipt := inventoryTestImported(t)
			mutate(&receipt)
			input, output := filepath.Join(root, "corrupt.json"), filepath.Join(root, "rejected")
			inventoryTestWrite(t, input, inventoryTestJSON(t, receipt))
			if err := inventoryReplay(Request{Unit: Inventory, Mode: Replay, Input: input, Output: output}); err == nil {
				t.Fatal("corruption accepted")
			}
			inventoryTestNoOutput(t, output)
		})
	}
}

func TestInventoryRejectsModesAndOtherUnits(t *testing.T) {
	handlers := InventoryHandlers()
	if handlers.Import == nil || handlers.Replay == nil || handlers.Native != nil {
		t.Fatal("host handlers missing or native enabled")
	}
	for _, pair := range []struct {
		handler Handler
		wrong   Mode
	}{{handlers.Import, Replay}, {handlers.Replay, Import}} {
		root := t.TempDir()
		output := filepath.Join(root, "rejected")
		if err := pair.handler(Request{Unit: Inventory, Mode: pair.wrong, Input: "unused", Output: output}); err == nil {
			t.Fatal("wrong host mode accepted")
		}
		inventoryTestNoOutput(t, output)
	}
	for _, handler := range []Handler{handlers.Import, handlers.Replay} {
		for _, r := range []Request{
			{Unit: Inventory, Mode: Native, Admitted: true},
			{Unit: Inventory, Mode: Mode("unknown")},
			{Unit: Unit("schedule-check"), Mode: Import},
			{Unit: Unit("postcall-check"), Mode: Replay},
		} {
			root := t.TempDir()
			r.Input, r.Output = filepath.Join(root, "unread-input"), filepath.Join(root, "rejected")
			if err := handler(r); err == nil {
				t.Fatal("wrong unit or mode accepted")
			}
			inventoryTestNoOutput(t, r.Output)
		}
	}
	for _, admitted := range []bool{false, true} {
		root := t.TempDir()
		r := Request{Unit: Inventory, Mode: Native, Input: "unused", Output: filepath.Join(root, "rejected"), Admitted: admitted}
		if err := Dispatch(r, Handlers{Inventory: handlers}); err == nil {
			t.Fatal("native inventory dispatched")
		}
		inventoryTestNoOutput(t, r.Output)
	}
}

func TestInventoryImportRejectsMalformedEvents(t *testing.T) {
	var unfinished InventoryCapture
	if err := json.Unmarshal([]byte(inventoryTestCapture), &unfinished); err != nil {
		t.Fatal(err)
	}
	unfinished.Events = unfinished.Events[:len(unfinished.Events)-1]
	cases := map[string]string{
		"unknown-operation": strings.Replace(inventoryTestCapture, `"Op":"chunks"`, `"Op":"unknown"`, 1),
		"missing-index":     strings.Replace(inventoryTestCapture, `"Index":0,"Count":2`, `"Count":2`, 1),
		"null-index":        strings.Replace(inventoryTestCapture, `"Index":0,"Count":2`, `"Index":null,"Count":2`, 1),
		"unknown-field":     strings.Replace(inventoryTestCapture, `"Count":2`, `"Count":2,"Ignored":1`, 1),
		"late-chunks":       strings.Replace(inventoryTestCapture, `{"Op":"complete","Data":{"Index":0}}`, `{"Op":"chunks","Data":{"Index":0,"Count":1}}`, 1),
		"unfinished":        string(inventoryTestJSON(t, unfinished)),
		"trailing-json":     inventoryTestCapture + `{}`,
		"null-document":     `null`,
	}
	for name, capture := range cases {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			input, output := filepath.Join(root, "invalid.json"), filepath.Join(root, "rejected")
			inventoryTestWrite(t, input, []byte(capture))
			if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: output}); err == nil {
				t.Fatal("invalid capture accepted")
			}
			inventoryTestNoOutput(t, output)
		})
	}
}

func TestInventoryFailedClosureRemainsFailedOnReplay(t *testing.T) {
	capture := strings.Replace(inventoryTestCapture, `"Records":1,"Failure":""`, `"Records":0,"Failure":"synthetic flush failure"`, 1)
	capture = strings.Replace(capture, `{"Op":"complete","Data":{"Index":0}}`, `{"Op":"stop","Data":{"Index":0,"State":"failed","Reason":"cancelled after closure"}}`, 1)
	root := t.TempDir()
	input, imported := filepath.Join(root, "failed-capture.json"), filepath.Join(root, "import")
	inventoryTestWrite(t, input, []byte(capture))
	if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: imported}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "replay")
	if err := inventoryReplay(Request{Unit: Inventory, Mode: Replay, Input: filepath.Join(imported, "inventory.json"), Output: out}); err != nil {
		t.Fatal(err)
	}
	r := inventoryTestReceipt(t, out).Report
	if r.Parents[0].State != runledger.Failed || r.Written != 1 || r.Writers[0].Records != 0 || r.Writers[0].Recorded != 1 || !r.Writers[0].Mismatch || !r.Writers[0].Closed || r.Writers[0].Failure != "synthetic flush failure" {
		t.Fatalf("failed closure lost %+v", r)
	}
	for name, replacement := range map[string]string{
		"false-completion":  `{"Op":"complete","Data":{"Index":0}}`,
		"duplicate-closure": `{"Op":"close-writer","Data":{"Path":"synthetic-output.fastq","Records":1,"Failure":""}}`,
	} {
		bad := strings.Replace(capture, `{"Op":"stop","Data":{"Index":0,"State":"failed","Reason":"cancelled after closure"}}`, replacement, 1)
		path, output := filepath.Join(root, name+".json"), filepath.Join(root, name)
		inventoryTestWrite(t, path, []byte(bad))
		if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: path, Output: output}); err == nil {
			t.Fatal("physical failure became completion or was replaced")
		}
		inventoryTestNoOutput(t, output)
	}
}

func TestInventoryFreshOutputRefusesOverwrite(t *testing.T) {
	_, imported, _ := inventoryTestImported(t)
	input := filepath.Join(imported, "inventory.json")
	before := inventoryTestRead(t, input)
	if err := inventoryReplay(Request{Unit: Inventory, Mode: Replay, Input: input, Output: imported}); err == nil {
		t.Fatal("existing output accepted")
	}
	if !bytes.Equal(before, inventoryTestRead(t, input)) {
		t.Fatal("existing receipt overwritten")
	}
}

func TestInventoryExactUniqueCaptureAndNestedFields(t *testing.T) {
	changes := map[string][2]string{
		"capture-case":           {`"Schema":`, `"schema":`},
		"capture-duplicate":      {`"Schema":"runledger-events/1",`, `"Schema":"runledger-events/1","Schema":"runledger-events/1",`},
		"capture-missing-files":  {`"Files":[{"UUID":"synthetic-file","ParentIDs":["synthetic-parent"]}],`, ``},
		"capture-null-requests":  {`"Requests":[{"FileUUID":"synthetic-file","ParentID":"synthetic-parent"}]`, `"Requests":null`},
		"event-case":             {`"Op":"load"`, `"op":"load"`},
		"event-duplicate-op":     {`"Op":"load"`, `"Op":"load","Op":"load"`},
		"event-duplicate-data":   {`"Data":{"Index":0,"Raw":20,"Admitted":10}`, `"Data":{"Index":0,"Raw":20,"Admitted":10},"Data":{"Index":0,"Raw":20,"Admitted":10}`},
		"payload-case-ambiguity": {`"Raw":20,"Admitted":10`, `"Raw":20,"Admitted":10,"raw":999`},
		"payload-case-only":      {`"Raw":20`, `"raw":20`},
		"payload-duplicate":      {`"Raw":20`, `"Raw":20,"Raw":999`},
		"file-missing-id":        {`"UUID":"synthetic-file",`, ``},
		"file-duplicate":         {`"UUID":"synthetic-file"`, `"UUID":"synthetic-file","UUID":"synthetic-file"`},
		"file-case":              {`"ParentIDs":`, `"parentIDs":`},
		"file-null-parent-ids":   {`"ParentIDs":["synthetic-parent"]`, `"ParentIDs":null`},
		"request-missing-parent": {`"FileUUID":"synthetic-file","ParentID":"synthetic-parent"`, `"FileUUID":"synthetic-file"`},
		"request-duplicate":      {`"FileUUID":"synthetic-file"`, `"FileUUID":"synthetic-file","FileUUID":"synthetic-file"`},
		"child-missing-filter":   {`"Keep":true,"FilterReason":""`, `"Keep":true`},
		"child-case":             {`"Keep":true`, `"keep":true`},
		"child-duplicate":        {`"Keep":true`, `"Keep":true,"Keep":false`},
		"write-missing-lineage":  {`"ChildID":"synthetic-parent","ParentID":"","Path":`, `"ChildID":"synthetic-parent","Path":`},
		"write-case":             {`"ChildID":`, `"childID":`},
		"write-duplicate":        {`"ChildID":"synthetic-parent"`, `"ChildID":"synthetic-parent","ChildID":"renamed"`},
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			raw := strings.Replace(inventoryTestCapture, change[0], change[1], 1)
			if raw == inventoryTestCapture {
				t.Fatal("mutation did not change fixture")
			}
			root := t.TempDir()
			input, output := filepath.Join(root, "bad.json"), filepath.Join(root, "rejected")
			inventoryTestWrite(t, input, []byte(raw))
			if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: output}); err == nil {
				t.Fatal("ambiguous or incomplete observation accepted")
			}
			inventoryTestNoOutput(t, output)
		})
	}
}

func TestInventoryCaptureRequiresExplicitArrays(t *testing.T) {
	for name, raw := range map[string]string{
		"schema-only":      `{"Schema":"runledger-events/1"}`,
		"missing-files":    `{"Schema":"runledger-events/1","Requests":[],"Events":[]}`,
		"missing-requests": `{"Schema":"runledger-events/1","Files":[],"Events":[]}`,
		"missing-events":   `{"Schema":"runledger-events/1","Files":[],"Requests":[]}`,
		"null-files":       `{"Schema":"runledger-events/1","Files":null,"Requests":[],"Events":[]}`,
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			input, output := filepath.Join(root, "capture.json"), filepath.Join(root, "rejected")
			inventoryTestWrite(t, input, []byte(raw))
			if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: output}); err == nil {
				t.Fatal("undeclared empty run accepted")
			}
			inventoryTestNoOutput(t, output)
		})
	}
	root := t.TempDir()
	input, output := filepath.Join(root, "empty.json"), filepath.Join(root, "import")
	inventoryTestWrite(t, input, []byte(`{"Schema":"runledger-events/1","Files":[],"Requests":[],"Events":[]}`))
	if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: output}); err != nil {
		t.Fatal(err)
	}
	first := inventoryTestReceipt(t, output)
	// Finalize currently declares Files nil for no files, while Parents and
	// Writers are explicit empty slices. Replay must preserve these distinctions.
	if first.Report.Files != nil || first.Report.Parents == nil || first.Report.Writers == nil {
		t.Fatalf("source-defined empty report shape changed %+v", first.Report)
	}
	replayed := filepath.Join(root, "replay")
	if err := inventoryReplay(Request{Unit: Inventory, Mode: Replay, Input: filepath.Join(output, "inventory.json"), Output: replayed}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.Report, inventoryTestReceipt(t, replayed).Report) {
		t.Fatal("report null arrays were normalized")
	}
}

func TestInventoryExactUniqueReceiptAndReportFields(t *testing.T) {
	changes := map[string][2]string{
		"receipt-case":                  {`"Mode":"import"`, `"mode":"import"`},
		"receipt-duplicate":             {`"Mode":"import"`, `"Mode":"import","Mode":"replay"`},
		"receipt-missing-mode":          {`"Mode":"import",`, ``},
		"receipt-missing-clock":         {`"Started":`, `"RemovedStarted":`},
		"receipt-missing-pin":           {`"InputSHA256":`, `"inputSHA256":`},
		"report-missing-total":          {`"Kept":1,`, ``},
		"report-case-ambiguity":         {`"Kept":1`, `"Kept":1,"kept":999`},
		"report-duplicate-total":        {`"Kept":1`, `"Kept":1,"Kept":999`},
		"report-parent-missing-state":   {`"State":"child-bearing",`, ``},
		"report-parent-case":            {`"Reason":""`, `"reason":""`},
		"report-child-missing-filter":   {`"Keep":true,"FilterReason":""`, `"Keep":true`},
		"report-write-missing-path":     {`"ChildID":"synthetic-parent","ParentID":"","Path":"synthetic-output.fastq"`, `"ChildID":"synthetic-parent","ParentID":""`},
		"report-writer-missing-failure": {`"Closed":true,"Failure":""`, `"Closed":true`},
		"report-writer-case":            {`"Mismatch":false`, `"mismatch":false`},
		"report-writer-duplicate":       {`"Closed":true`, `"Closed":true,"Closed":false`},
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			root, _, receipt := inventoryTestImported(t)
			original := string(inventoryTestJSON(t, receipt))
			raw := strings.Replace(original, change[0], change[1], 1)
			if original == raw {
				t.Fatal("mutation did not change receipt")
			}
			input, output := filepath.Join(root, "bad-receipt.json"), filepath.Join(root, "rejected")
			inventoryTestWrite(t, input, []byte(raw))
			if err := inventoryReplay(Request{Unit: Inventory, Mode: Replay, Input: input, Output: output}); err == nil {
				t.Fatal("ambiguous or incomplete receipt accepted")
			}
			inventoryTestNoOutput(t, output)
		})
	}
}

func TestInventoryZeroObservationsRemainExplicit(t *testing.T) {
	raw := strings.Replace(inventoryTestCapture, `"Raw":20,"Admitted":10`, `"Raw":0,"Admitted":0`, 1)
	root := t.TempDir()
	input, output := filepath.Join(root, "zero.json"), filepath.Join(root, "import")
	inventoryTestWrite(t, input, []byte(raw))
	if err := inventoryImport(Request{Unit: Inventory, Mode: Import, Input: input, Output: output}); err != nil {
		t.Fatal(err)
	}
	r := inventoryTestReceipt(t, output).Report
	if r.RawSamples != 0 || r.AdmittedSamples != 0 || r.Parents[0].Reason != "" || r.Parents[0].Writes[0].ParentID != "" {
		t.Fatal("explicit zero or empty observation changed")
	}
}

func TestInventoryRejectsFIFOWithoutWriter(t *testing.T) {
	const childPath = "NANOPORE_I04_FIFO_CONTROL_PATH"
	if path := os.Getenv(childPath); path != "" {
		_, err := inventoryRead(path)
		if err == nil || !strings.Contains(err.Error(), "regular file") {
			t.Fatalf("FIFO not rejected by descriptor type: %v", err)
		}
		return
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	fifo := filepath.Join(root, "no-writer.fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "fifo-link")
	if err := os.Symlink(fifo, link); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{fifo, link} {
		// A delayed or blocked regression stays in this child. CombinedOutput
		// waits for termination before TempDir cleanup can remove the FIFO.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestInventoryRejectsFIFOWithoutWriter$", "-test.count=1")
		for _, value := range os.Environ() {
			if !strings.HasPrefix(value, childPath+"=") {
				cmd.Env = append(cmd.Env, value)
			}
		}
		cmd.Env = append(cmd.Env, childPath+"="+path)
		cmd.WaitDelay = time.Second
		output, runErr := cmd.CombinedOutput()
		contextErr := ctx.Err()
		cancel()
		if contextErr != nil {
			t.Fatalf("FIFO control child exceeded deadline and was joined: %v; %s", contextErr, output)
		}
		if runErr != nil {
			t.Fatalf("FIFO control child failed and was joined: %v; %s", runErr, output)
		}
	}
}
