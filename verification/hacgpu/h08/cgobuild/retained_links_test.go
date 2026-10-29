package cgobuild

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestGenuineRetainedLinkSubset(t *testing.T) {
	path, sha := os.Getenv("H08_RETAINED_LINK_RECEIPT"), os.Getenv("H08_RETAINED_LINK_RECEIPT_SHA256")
	if path == "" && sha == "" {
		t.Skip("explicit hash-bound retained-link selection required")
	}
	hash, e := hex.DecodeString(sha)
	if path == "" || !filepath.IsAbs(path) || e != nil || len(hash) != 32 || strings.ToLower(sha) != sha {
		t.Fatal("complete absolute receipt path and independent SHA256 required")
	}
	selected := Pin{Path: path, SHA256: sha}
	var r Receipt
	if e = read(selected, &r); e != nil {
		t.Fatal(e)
	}
	var p Plan
	if e = read(r.Plan, &p); e != nil {
		t.Fatal(e)
	}
	if r.Status != "failed" || r.Error != "link used input outside before-child snapshots" {
		t.Fatal("not the selected failed005 profile")
	}
	if e = ValidateReceipt(p, r); e == nil || !strings.Contains(e.Error(), "native phase receipt incomplete") {
		t.Fatalf("strict failed receipt rejection lost: %v", e)
	}
	var s Session
	if e = read(r.Session, &s); e != nil {
		t.Fatal(e)
	}
	if s.Phase != "build" || s.Plan != r.Plan || s.Controller != r.Controller {
		t.Fatal("selected build session differs")
	}
	resources, e := deriveQueryResources(p, r)
	if e != nil {
		t.Fatal(e)
	}
	special := []string{"crt1.o", "Scrt1.o", "crti.o", "crtn.o", "libc.so", "libdl.a", "libpthread.a"}
	for _, name := range special {
		var expected Pin
		for _, q := range p.Queries {
			if q.Name == name {
				expected = q.Expected
			}
		}
		if expected.Path == "" || resources.exactAnswerBindings["/lib/../lib/"+name] != expected {
			t.Fatalf("special per-file observation omitted: %s", name)
		}
	}
	var qr Receipt
	if e = read(r.Queries, &qr); e != nil {
		t.Fatal(e)
	}
	additional := 0
	for _, pin := range qr.Operations {
		var op operation
		if e = read(pin, &op); e != nil {
			t.Fatal(e)
		}
		for _, q := range p.Queries {
			if len(q.Argv) != 2 || q.Argv[1] != "-print-file-name="+q.Name || !slices.Equal(q.Argv[1:], op.OriginalArgs) {
				continue
			}
			body, err := os.ReadFile(op.Commands[0].Stdout)
			if err != nil {
				t.Fatal(err)
			}
			raw := strings.TrimSpace(string(body))
			if raw == q.Expected.Path {
				continue
			}
			if resources.exactAnswerBindings[raw] != q.Expected {
				t.Fatalf("exact query answer missing: %s", q.Name)
			}
			if !slices.Contains(special, q.Name) {
				additional++
			}
		}
	}
	if additional != 9 || len(resources.exactAnswerBindings) != len(special)+additional {
		t.Fatalf("want 7 special + 9 additional exact bindings, got %d total/%d additional", len(resources.exactAnswerBindings), additional)
	}
	operations := map[string]operation{}
	for _, pin := range r.Operations {
		var op operation
		if e = read(pin, &op); e != nil {
			t.Fatal(e)
		}
		dir := filepath.Dir(pin.Path)
		if filepath.Base(pin.Path) != "operation.json" || !under(filepath.Join(s.Directory, "commands"), pin.Path) {
			t.Fatal("operation outside selected session")
		}
		if _, ok := operations[dir]; ok {
			t.Fatal("duplicate operation")
		}
		operations[dir] = op
	}
	var generated []GeneratedSet
	for _, pin := range r.Generated {
		var g GeneratedSet
		if e = read(pin, &g); e != nil {
			t.Fatal(e)
		}
		pkg, e := packageByName(p, g.Package)
		if e != nil {
			t.Fatal(e)
		}
		if e = CheckGenerated(pkg, g.Stage, g.Files); e != nil {
			t.Fatal(e)
		}
		if len(g.Files) != len(g.Snapshots) || len(g.Files) != len(g.Bindings) {
			t.Fatal("generated closure incomplete")
		}
		for name, q := range g.Files {
			snap := g.Snapshots[name]
			if snap.SHA256 != q.SHA256 || g.Bindings[q.Path] != q.Path {
				t.Fatal("generated snapshot/binding differs")
			}
			if e = verifyPin(snap); e != nil {
				t.Fatal(e)
			}
		}
		if e = generatedProducer(g, operations); e != nil {
			t.Fatal(e)
		}
		generated = append(generated, g)
	}
	if len(generated) != 4 {
		t.Fatal("actual generated stages omitted")
	}
	count, maps, unsupported := 0, 0, 0
	for dir, op := range operations {
		if op.Request == nil || !op.Request.Link {
			continue
		}
		if op.Schema != "nanopore-h08-cgo-operation/v1" || op.Error != "" || op.Status != "accepted" && op.Status != "accepted_expected_exit" || len(op.Commands) == 0 || len(op.Expected) == 0 || len(op.Inputs) == 0 || len(op.Snapshots) != len(op.Inputs) {
			t.Fatal("native operation envelope incomplete")
		}
		authenticateRetainedLink(t, op)
		if e = verifyNativeOperation(p, s, op, dir, generated, resources); e != nil {
			t.Fatalf("%s: %v", dir, e)
		}
		count++
		if op.LinkMap.Path != "" {
			maps++
		} else {
			if op.Role != "go-link-probe" || op.Status != "accepted_expected_exit" || len(op.Commands) != 2 || op.Commands[1].Exit != 1 || !strings.Contains(strings.Join(op.OriginalArgs, " "), "-Qunused-arguments") {
				t.Fatal("unexpected nonlinking request")
			}
			unsupported++
		}
		t.Logf("retained native operation=%s role=%s commands=%d map=%t", dir, op.Role, len(op.Commands), op.LinkMap.Path != "")
	}
	if count != 8 || maps != 7 || unsupported != 1 {
		t.Fatalf("want 8 operations/7 maps/1 unsupported, got %d/%d/%d", count, maps, unsupported)
	}
	if e = finalOutputBindings(p, r, operations); e != nil {
		t.Fatal(e)
	}
	proof, e := finalOutputTransition(p, r, operations)
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("8operations/7maps selected=%+v plan=%+v controller=%+v queries=%+v producerNativeSHA=%s aliases=%v", selected, r.Plan, r.Controller, r.Queries, queryProducerNativeSHA, resources.exactAnswerBindings)
	t.Logf("Go1.27.3 linux/amd64 linkSHA=%s finalization offset=%d checksum=%x ranges=%+v", go1273LinuxAMD64LinkSHA, proof.checksumOffset, proof.checksum, proof.ranges)
	if e = verifyPin(selected); e != nil {
		t.Fatal(e)
	}
	if e = ValidateReceipt(p, r); e == nil {
		t.Fatal("failed receipt admitted after subset")
	}
}

func authenticateRetainedLink(t *testing.T, op operation) {
	t.Helper()
	for path, sha := range op.Inputs {
		q, ok := op.Snapshots[path]
		if !ok || q.SHA256 != sha {
			t.Fatal("input snapshot binding omitted")
		}
		if e := verifyPin(q); e != nil {
			t.Fatal(e)
		}
	}
	for i, c := range op.Commands {
		expected := op.Expected
		if i < len(op.Commands)-1 {
			expected = []int{0, 1}
		}
		if e := checkCommand(c.Args, expected, c.Started, c.Finished, c.Stdout, c.StdoutSHA, c.Stderr, c.StderrSHA, c.Exit); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(filepath.Join(filepath.Dir(c.Stdout), "before.json"))
		if e != nil {
			t.Fatal(e)
		}
		var before map[string]string
		if e = read(q, &before); e != nil || len(before) == 0 {
			t.Fatal("before manifest omitted", e)
		}
		for path, sha := range before {
			if op.Inputs[path] != sha || op.Snapshots[path].SHA256 != sha {
				t.Fatal("before snapshot differs")
			}
		}
		if i == len(op.Commands)-1 && q != op.Before {
			t.Fatal("final before manifest differs")
		}
	}
	for _, q := range []Pin{op.PreflightDep, op.ActualDep, op.LinkMap, op.Stdin} {
		if q.Path != "" {
			if e := verifyPin(q); e != nil {
				t.Fatal(e)
			}
		}
	}
	if op.Output.Path != "" {
		if op.Output.SHA256 != op.OutputSnapshot.SHA256 {
			t.Fatal("output snapshot differs")
		}
		if e := verifyPin(op.OutputSnapshot); e != nil {
			t.Fatal(e)
		}
	}
}
