// Package runledger accounts for requested work independently of progress text.
// A single owner serializes events. It must supply observations from the reader,
// postprocessor and writer; the ledger does not inspect files or prove provenance.
package runledger

import (
	"fmt"
	"math"
)

type State string

const (
	Pending      State = "pending"
	Absent       State = "absent"
	Rejected     State = "rejected"
	Loaded       State = "loaded"
	Failed       State = "failed"
	ZeroSurvivor State = "zero-survivor"
	ChildBearing State = "child-bearing"
)

type File struct {
	UUID      string
	ParentIDs []string
}

// Request preserves selection order and duplicates through its slice index.
type Request struct{ FileUUID, ParentID string }

type Child struct {
	ID string
	// ParentID is the expected output pi lineage, empty for an unsplit parent.
	ParentID     string
	Keep         bool
	FilterReason string
}

type Write struct{ ChildID, ParentID, Path string }

type Parent struct {
	Request                             Request
	State                               State
	Reason                              string
	RawSamples, AdmittedSamples, Chunks uint64
	Children                            []Child
	Writes                              []Write
}

type Writer struct {
	Path     string
	Records  uint64
	Recorded uint64
	Mismatch bool
	Closed   bool
	Failure  string
}

// WriterClosureError means the physical closure was retained despite failure.
// Protocol errors, including a repeated close, do not have this type.
type WriterClosureError struct {
	Path               string
	Observed, Recorded uint64
	Failure            string
}

func (e *WriterClosureError) Error() string {
	return fmt.Sprintf("output %q closed with %d observed records, %d recorded writes, failure %q", e.Path, e.Observed, e.Recorded, e.Failure)
}

type Report struct {
	Files                                                        []File
	Parents                                                      []Parent
	Writers                                                      []Writer
	RawSamples, AdmittedSamples, Chunks, Children, Kept, Written uint64
}

type Ledger struct {
	files       []File
	parents     []Parent
	present     []bool
	children    map[string]Request
	writers     []Writer
	writerIndex map[string]int
	sealed      bool
}

func New(files []File, requests []Request) (*Ledger, error) {
	inventory := make(map[string]map[string]bool, len(files))
	for _, f := range files {
		if f.UUID == "" || inventory[f.UUID] != nil {
			return nil, fmt.Errorf("empty or duplicate file UUID %q", f.UUID)
		}
		ids := make(map[string]bool, len(f.ParentIDs))
		for _, id := range f.ParentIDs {
			if id == "" || ids[id] {
				return nil, fmt.Errorf("empty or duplicate parent %q in %q", id, f.UUID)
			}
			ids[id] = true
		}
		inventory[f.UUID] = ids
	}
	l := &Ledger{children: make(map[string]Request), writerIndex: make(map[string]int)}
	for _, f := range files {
		l.files = append(l.files, File{UUID: f.UUID, ParentIDs: append([]string{}, f.ParentIDs...)})
	}
	for _, r := range requests {
		if r.ParentID == "" || inventory[r.FileUUID] == nil {
			return nil, fmt.Errorf("invalid request %+v", r)
		}
		l.parents = append(l.parents, Parent{Request: r, State: Pending})
		l.present = append(l.present, inventory[r.FileUUID][r.ParentID])
	}
	return l, nil
}

func (l *Ledger) parent(index int) (*Parent, error) {
	if l.sealed {
		return nil, fmt.Errorf("ledger finalized")
	}
	if index < 0 || index >= len(l.parents) {
		return nil, fmt.Errorf("unknown request %d", index)
	}
	return &l.parents[index], nil
}

// Load records decoded and admitted counts before scheduling any chunks.
func (l *Ledger) Load(index int, raw, admitted uint64) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	if p.State != Pending || !l.present[index] || admitted > raw {
		return fmt.Errorf("invalid load for request %d", index)
	}
	p.State, p.RawSamples, p.AdmittedSamples = Loaded, raw, admitted
	return nil
}

func (l *Ledger) AddChunks(index int, count uint64) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	if p.State != Loaded || p.Children != nil || count == 0 || count > math.MaxUint64-p.Chunks {
		return fmt.Errorf("invalid chunk count for request %d", index)
	}
	p.Chunks += count
	return nil
}

// Stop records an explicit terminal error, including corrupt rows, unsupported
// metadata, cancellation and writer failures. Failure retains partial work.
func (l *Ledger) Stop(index int, state State, reason string) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	if reason == "" {
		return fmt.Errorf("terminal reason required")
	}
	switch state {
	case Absent:
		if p.State != Pending || l.present[index] {
			return fmt.Errorf("absent request exists or already started")
		}
	case Rejected:
		if p.State != Pending || !l.present[index] {
			return fmt.Errorf("rejection requires present pending parent")
		}
	case Failed:
		if p.State != Pending && p.State != Loaded {
			return fmt.Errorf("parent already terminal")
		}
	default:
		return fmt.Errorf("invalid stop state %q", state)
	}
	p.State, p.Reason = state, reason
	return nil
}

// SetChildren atomically records the postprocessor's complete child inventory.
// An unsplit read retains its parent ID and has no pi; split children require pi.
func (l *Ledger) SetChildren(index int, children []Child) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	if p.State != Loaded || p.Children != nil {
		return fmt.Errorf("children require loaded parent and one inventory")
	}
	seen := make(map[string]bool, len(children))
	for _, c := range children {
		owner, exists := l.children[c.ID]
		if c.ID == "" || seen[c.ID] || (exists && owner != p.Request) {
			return fmt.Errorf("duplicate or empty child %q", c.ID)
		}
		if (c.ID == p.Request.ParentID && c.ParentID != "") || (c.ID != p.Request.ParentID && c.ParentID != p.Request.ParentID) {
			return fmt.Errorf("invalid lineage for %q", c.ID)
		}
		if (c.Keep && c.FilterReason != "") || (!c.Keep && c.FilterReason == "") {
			return fmt.Errorf("invalid filter decision for %q", c.ID)
		}
		seen[c.ID] = true
	}
	p.Children = append([]Child{}, children...)
	for _, c := range children {
		l.children[c.ID] = p.Request
	}
	return nil
}

func (l *Ledger) OpenWriter(path string) error {
	if l.sealed || path == "" {
		return fmt.Errorf("invalid writer open")
	}
	if _, exists := l.writerIndex[path]; exists {
		return fmt.Errorf("duplicate output path %q", path)
	}
	l.writerIndex[path] = len(l.writers)
	l.writers = append(l.writers, Writer{Path: path})
	return nil
}

// RecordWrite consumes the actual successfully written QNAME and pi, not an
// expected record count. CloseWriter separately reconciles the output inventory.
func (l *Ledger) RecordWrite(index int, w Write) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	wi, writerOK := l.writerIndex[w.Path]
	if !writerOK || l.writers[wi].Closed {
		return fmt.Errorf("unknown child or unavailable writer")
	}
	if p.State != Loaded {
		return fmt.Errorf("write requires loaded parent")
	}
	for _, old := range p.Writes {
		if old.ChildID == w.ChildID {
			return fmt.Errorf("duplicate write %q", w.ChildID)
		}
	}
	for _, c := range p.Children {
		if c.ID == w.ChildID {
			if !c.Keep || c.ParentID != w.ParentID {
				return fmt.Errorf("filtered child or changed lineage")
			}
			p.Writes = append(p.Writes, w)
			return nil
		}
	}
	return fmt.Errorf("unknown child")
}

func (l *Ledger) CloseWriter(path string, observedRecords uint64, failure string) error {
	if l.sealed {
		return fmt.Errorf("ledger finalized")
	}
	i, ok := l.writerIndex[path]
	if !ok || l.writers[i].Closed {
		return fmt.Errorf("unknown or closed writer")
	}
	var recorded uint64
	for _, p := range l.parents {
		for _, w := range p.Writes {
			if w.Path == path {
				recorded++
			}
		}
	}
	l.writers[i].Records, l.writers[i].Recorded = observedRecords, recorded
	l.writers[i].Failure, l.writers[i].Closed = failure, true
	l.writers[i].Mismatch = observedRecords != recorded
	if l.writers[i].Mismatch || failure != "" {
		return &WriterClosureError{Path: path, Observed: observedRecords, Recorded: recorded, Failure: failure}
	}
	return nil
}

func (l *Ledger) Complete(index int) error {
	p, err := l.parent(index)
	if err != nil {
		return err
	}
	if p.State != Loaded || p.Children == nil {
		return fmt.Errorf("completion requires child inventory")
	}
	var kept int
	for _, c := range p.Children {
		if c.Keep {
			kept++
		}
	}
	if kept != len(p.Writes) {
		return fmt.Errorf("unwritten surviving children")
	}
	for _, w := range p.Writes {
		writer := l.writers[l.writerIndex[w.Path]]
		if !writer.Closed || writer.Failure != "" || writer.Mismatch {
			return fmt.Errorf("writer is unfinished or failed")
		}
	}
	p.State = ZeroSurvivor
	if kept > 0 {
		p.State = ChildBearing
	}
	return nil
}

// Finalize succeeds for fully accounted failed runs too. Consumers must inspect
// parent states and writer failures before claiming successful basecalling.
func (l *Ledger) Finalize() (Report, error) {
	if l.sealed {
		return Report{}, fmt.Errorf("ledger finalized")
	}
	for i, p := range l.parents {
		if p.State == Pending || p.State == Loaded {
			return Report{}, fmt.Errorf("request %d has no final disposition", i)
		}
	}
	for _, w := range l.writers {
		if !w.Closed {
			return Report{}, fmt.Errorf("output %q not finalized", w.Path)
		}
	}
	r := Report{Parents: make([]Parent, len(l.parents)), Writers: append([]Writer{}, l.writers...)}
	for _, f := range l.files {
		r.Files = append(r.Files, File{UUID: f.UUID, ParentIDs: append([]string{}, f.ParentIDs...)})
	}
	for i, p := range l.parents {
		r.Parents[i] = p
		r.Parents[i].Children = append([]Child{}, p.Children...)
		r.Parents[i].Writes = append([]Write{}, p.Writes...)
		counts := []struct {
			dst *uint64
			n   uint64
		}{
			{&r.RawSamples, p.RawSamples}, {&r.AdmittedSamples, p.AdmittedSamples}, {&r.Chunks, p.Chunks},
			{&r.Children, uint64(len(p.Children))}, {&r.Written, uint64(len(p.Writes))},
		}
		for _, c := range p.Children {
			if c.Keep {
				counts = append(counts, struct {
					dst *uint64
					n   uint64
				}{&r.Kept, 1})
			}
		}
		for _, c := range counts {
			if c.n > math.MaxUint64-*c.dst {
				return Report{}, fmt.Errorf("run count overflow")
			}
			*c.dst += c.n
		}
	}
	l.sealed = true
	return r, nil
}
