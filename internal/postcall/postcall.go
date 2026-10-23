// Package postcall composes the pinned plain DNA post-MUX CPU pipeline.
package postcall

import (
	"fmt"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/ends"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

const PolicyVersion = "plain-dna-sqk-lsk114/1"

// Policy selects independent native trim and split stages. MinQ is fixed at the
// stock CLI default zero until additional CLI quality policies are qualified.
type Policy struct {
	Version     string
	Trim, Split bool
	MinQ        uint32
}

func DefaultPolicy() Policy { return Policy{Version: PolicyVersion, Trim: true, Split: true} }

func (p Policy) Validate(kit string) error {
	if p.Version != PolicyVersion || p.MinQ != 0 {
		return fmt.Errorf("unsupported postcall policy")
	}
	if (p.Trim || p.Split) && !strings.EqualFold(kit, ends.SupportedKit) {
		return fmt.Errorf("postprocessing supports only plain DNA nonrapid %s, got kit %q", ends.SupportedKit, kit)
	}
	return nil
}

// Record retains rejected children too, so a successful empty output is explicit.
type Record struct {
	Read      subread.Read
	Detection *ends.Result
	Filter    readtrim.FilterResult
}
type Result struct {
	Policy        Policy
	SplitDecision *splitter.Decision
	Children      []Record
}

// Process borrows an immutable normalized read after core MUX trimming. Its
// floor-sized move contract matches core Stitch and excludes native ceil-child
// outputs whose final sequence base has no corresponding move.
func Process(in subread.Read, p Policy) (Result, error) {
	if err := p.Validate(in.Inherited.SequencingKit); err != nil {
		return Result{}, err
	}
	if in.Stride == 0 || uint64(len(in.Moves)) != uint64(len(in.Signal))/in.Stride {
		return Result{}, fmt.Errorf("postcall requires core floor-sized moves")
	}
	out := Result{Policy: p, Children: []Record{}}
	reads := []subread.Read{in}
	if p.Split {
		var spacers []subread.Interval
		if len(in.Sequence) != 0 && len(in.Moves) != 0 {
			d, err := splitter.FindSimplexPA(splitter.Input{Signal: in.Signal, Moves: in.Moves, Sequence: in.Sequence, Qstring: in.Qstring, Stride: in.Stride})
			if err != nil {
				return Result{}, err
			}
			out.SplitDecision = &d
			for _, s := range d.Spacers {
				spacers = append(spacers, subread.Interval{Start: s.Start, End: s.End})
			}
		}
		var err error
		reads, err = subread.Materialize(in, spacers)
		if err != nil {
			return Result{}, err
		}
	}
	for _, r := range reads {
		record := Record{Read: r}
		if p.Trim {
			d, err := ends.Detect(r.Sequence, r.Inherited.SequencingKit, ends.DefaultOptions())
			if err != nil {
				return Result{}, err
			}
			record.Detection = &d
			if !d.Conflict && d.Retained.Start < d.Retained.End {
				trimmed, err := readtrim.ApplyInterval(readtrim.MuxRead{Sequence: r.Sequence, Qstring: r.Qstring, Moves: r.Moves, Signal: r.Signal, Stride: uint32(r.Stride), IsMuxChange: r.IsMuxChange, NumMinknowEvents: r.NumEvents, PreTrimSequenceLength: r.PreTrimSeqLen}, readtrim.Interval{Start: d.Retained.Start, End: d.Retained.End}, r.LeadingTrim)
				if err != nil {
					return Result{}, err
				}
				record.Read.Sequence, record.Read.Qstring = trimmed.Read.Sequence, trimmed.Read.Qstring
				record.Read.Signal, record.Read.Moves = trimmed.Read.Signal, trimmed.Read.Moves
				record.Read.LeadingTrim = trimmed.LeadingTrimSamples
			}
		}
		filter, err := readtrim.Filter(record.Read.Sequence, record.Read.Qstring, int(record.Read.MeanQStart), float32(p.MinQ))
		if err != nil {
			return Result{}, err
		}
		record.Filter = filter
		out.Children = append(out.Children, record)
	}
	return out, nil
}
