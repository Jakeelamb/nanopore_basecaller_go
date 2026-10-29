package cgobuild

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"
)

const queryProducerNativeSHA = "7e8d7e1a705d93c4d734d987bc90d3440e7a07c2406172e6eab4f044dd192c47"

type attestedQueryResources struct{ exactAnswerBindings map[string]Pin }

func deriveQueryResources(p Plan, build Receipt) (attestedQueryResources, error) {
	out := attestedQueryResources{exactAnswerBindings: map[string]Pin{}}
	var r Receipt
	if e := read(build.Queries, &r); e != nil {
		return out, e
	}
	if r.Schema != "nanopore-h08-cgo-execution/v1" || r.Status != "metadata_queries_passed" || r.Error != "" || !r.RuntimeHeld || r.Plan != build.Plan || r.Controller != build.Controller || r.Role != p.Role || len(r.Commands) != 0 || len(r.Generated) != 0 || r.Output.Path != "" || len(r.Operations) != len(p.Queries) {
		return out, fmt.Errorf("selected prior query receipt differs")
	}
	var s Session
	if e := read(r.Session, &s); e != nil {
		return out, e
	}
	if s.Schema != "nanopore-h08-cgo-session/v1" || s.Phase != "queries" || !filepath.IsAbs(s.Directory) || s.Plan != r.Plan || s.Controller != r.Controller {
		return out, fmt.Errorf("prior query session binding differs")
	}
	if e := noRejections(s); e != nil {
		return out, e
	}
	start, e := time.Parse(time.RFC3339Nano, build.Started)
	if e != nil {
		return out, fmt.Errorf("query-before-build clock missing")
	}
	qstart, e := time.Parse(time.RFC3339Nano, r.Started)
	if e != nil {
		return out, fmt.Errorf("prior query receipt clock missing")
	}
	finish, e := time.Parse(time.RFC3339Nano, r.Finished)
	if e != nil || finish.Before(qstart) || !finish.Before(start) {
		return out, fmt.Errorf("prior query clock is not before build")
	}
	for _, c := range build.Commands {
		t, e := time.Parse(time.RFC3339Nano, c.Started)
		if e != nil || !finish.Before(t) {
			return out, fmt.Errorf("prior query clock is not before Go command")
		}
	}
	frozenProducer, e := controllerRead(r.Controller)
	if e != nil {
		return out, e
	}
	source := filepath.Join(p.Options.Root, "verification/hacgpu/h08/cgobuild/native.go")
	frozenProducerSnapshot := frozenProducer.Snapshots[source]
	if p.Inputs[source] != queryProducerNativeSHA || frozenProducer.Inputs[source] != queryProducerNativeSHA || frozenProducerSnapshot.SHA256 != queryProducerNativeSHA || s.Driver != frozenProducer.Executable {
		return out, fmt.Errorf("prior query producer source/driver differs")
	}
	if e := verifyPin(frozenProducerSnapshot); e != nil {
		return out, e
	}
	seen := map[string]bool{}
	for _, pin := range r.Operations {
		var op operation
		if e := read(pin, &op); e != nil {
			return out, e
		}
		dir := filepath.Dir(pin.Path)
		if !under(filepath.Join(s.Directory, "commands"), pin.Path) || filepath.Base(pin.Path) != "operation.json" || op.Schema != "nanopore-h08-cgo-operation/v1" || op.Role != "metadata" || op.Status != "accepted" || op.Error != "" || op.Directory != p.Options.Root || op.Lease != nil || op.Stdin.Path != "" || op.Wrapper.Path != "" || len(op.Commands) != 1 || !slices.Equal(op.Expected, []int{0}) {
			return out, fmt.Errorf("prior query operation boundary differs")
		}
		var selected *Query
		for i := range p.Queries {
			q := &p.Queries[i]
			if len(q.Argv) > 1 && q.Argv[0] == p.CC.Path && slices.Equal(op.OriginalArgs, q.Argv[1:]) {
				if selected != nil {
					return out, fmt.Errorf("conflicting selected queries")
				}
				selected = q
			}
		}
		if selected == nil || seen[selected.Name] {
			return out, fmt.Errorf("wrong or duplicate selected prior query")
		}
		seen[selected.Name] = true
		expected, e := classifyRequest(p, op.Directory, op.OriginalArgs, nil, nil, nil, &op)
		if e != nil || op.Request == nil || !reflect.DeepEqual(expected, *op.Request) || !maps.Equal(op.Inputs, operationInputs(p, s)) {
			return out, fmt.Errorf("prior query request/input binding differs")
		}
		c := op.Commands[0]
		if filepath.Dir(c.Stdout) != filepath.Join(dir, "actual") {
			return out, fmt.Errorf("prior query command directory differs")
		}
		if e := checkCommand(c.Args, []int{0}, c.Started, c.Finished, c.Stdout, c.StdoutSHA, c.Stderr, c.StderrSHA, c.Exit); e != nil {
			return out, e
		}
		if e := beforeCommand(c, selected.Argv, append(environment(p), "TERM=dumb"), op.Inputs, op); e != nil {
			return out, e
		}
		bp, e := closedPin(filepath.Join(dir, "actual/before.json"))
		if e != nil || bp != op.Before {
			return out, fmt.Errorf("prior query before manifest differs")
		}
		if len(op.Snapshots) != len(op.Inputs) {
			return out, fmt.Errorf("prior query before snapshot omitted")
		}
		for path, hash := range op.Inputs {
			snap, ok := op.Snapshots[path]
			if !ok || snap.SHA256 != hash {
				return out, fmt.Errorf("prior query before snapshot omitted")
			}
			if e := verifyPin(snap); e != nil {
				return out, e
			}
		}
		cs, e := time.Parse(time.RFC3339Nano, c.Started)
		cf, f := time.Parse(time.RFC3339Nano, c.Finished)
		if e != nil || f != nil || cs.Before(qstart) || cf.After(finish) || !cf.Before(start) {
			return out, fmt.Errorf("prior query command clock differs")
		}
		if len(selected.Argv) != 2 || selected.Argv[1] != "-print-file-name="+selected.Name {
			continue
		}
		q := *selected
		if !slices.Equal(q.Argv, []string{p.CC.Path, "-print-file-name=" + q.Name}) || q.Expected.Path == "" || p.Inputs[q.Expected.Path] != q.Expected.SHA256 {
			return out, fmt.Errorf("selected query resource differs")
		}
		matches := 0
		for _, resource := range p.Resources {
			if resource.Name == q.Name {
				if resource.Input != q.Expected || !slices.Equal(resource.Query, q.Argv) {
					return out, fmt.Errorf("prospective query resource differs")
				}
				matches++
			}
		}
		if matches != 1 {
			return out, fmt.Errorf("prospective query resource missing or conflicting")
		}
		raw, e := os.ReadFile(c.Stdout)
		if e != nil {
			return out, e
		}
		answer := strings.TrimSpace(string(raw))
		if answer == q.Expected.Path {
			continue
		}
		if !filepath.IsAbs(answer) || strings.ContainsAny(answer, "\r\n\t ") {
			return out, fmt.Errorf("unselected raw query spelling")
		}
		if old, ok := out.exactAnswerBindings[answer]; ok && old != q.Expected {
			return out, fmt.Errorf("conflicting raw query resource")
		}
		out.exactAnswerBindings[answer] = q.Expected
	}
	if len(seen) != len(p.Queries) {
		return out, fmt.Errorf("selected prior queries omitted")
	}
	return out, nil
}
