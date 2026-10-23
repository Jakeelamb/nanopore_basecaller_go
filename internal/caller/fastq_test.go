package caller

import (
	"bytes"
	"math"
	"testing"
)

func TestFASTQClockValidation(t *testing.T) {
	for _, change := range []string{"zero-rate", "sample-overflow", "time-overflow"} {
		r := testRead()
		switch change {
		case "zero-rate":
			r.SampleRate = 0
		case "sample-overflow":
			r.StartSample = math.MaxUint64
		case "time-overflow":
			r.AcquisitionStartTimeMS = math.MaxInt64
			r.StartSample = 5000
		}
		if _, err := fastqMetadata(r, testSpec()); err == nil {
			t.Fatalf("accepted %s", change)
		}
	}
	if _, err := timestamp(math.MaxInt64); err == nil {
		t.Fatal("accepted unsupported timestamp year")
	}
}

func TestFASTQRejectsMalformedRecordBeforeWriting(t *testing.T) {
	for _, change := range []string{"id", "empty", "length", "sequence", "quality-low", "quality-high", "metadata", "mean-start", "tag", "time"} {
		r := CalledRead{ID: "01000000-0000-0000-0000-000000000000", Sequence: "A", Qstring: "I", FASTQ: &FASTQMetadata{}}
		switch change {
		case "id":
			r.ID = "bad"
		case "empty":
			r.Sequence = ""
			r.Qstring = ""
		case "length":
			r.Qstring = "II"
		case "sequence":
			r.Sequence = "\n"
		case "quality-low":
			r.Qstring = " "
		case "quality-high":
			r.Qstring = "\x7f"
		case "metadata":
			r.FASTQ = nil
		case "mean-start":
			r.FASTQ.MeanQStart = -1
		case "tag":
			r.FASTQ.RunID = "run\tRG:Z:injected"
		case "time":
			r.FASTQ.StartTimeMS = math.MaxInt64
		}
		var out bytes.Buffer
		if err := r.WriteFASTQ(&out); err == nil {
			t.Fatalf("accepted %s", change)
		}
		if out.Len() != 0 {
			t.Fatalf("%s rejection emitted partial FASTQ", change)
		}
	}
}
