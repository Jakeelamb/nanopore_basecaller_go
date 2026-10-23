package caller

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
)

// FASTQMetadata describes the admitted unbarcoded, unsplit CPU DNA writer path.
// Times are Unix milliseconds before Dorado's six-digit UTC formatting.
type FASTQMetadata struct {
	Channel             uint16    `json:"channel"`
	StartTimeMS         int64     `json:"start_time_ms"`
	ProtocolStartTimeMS int64     `json:"protocol_start_time_ms"`
	RunID               string    `json:"run_id"`
	ModelName           string    `json:"model_name"`
	FlowCellID          string    `json:"flowcell_id"`
	MeanQStart          int       `json:"mean_q_start"`
	TrimFlags           TrimFlags `json:"trim_flags,omitempty"`
}

type TrimFlags uint8

const (
	TrimAdapter TrimFlags = 1 << iota
	TrimPrimer
)

func (t TrimFlags) text() (string, error) {
	if t & ^(TrimAdapter|TrimPrimer) != 0 {
		return "", errors.New("unsupported FASTQ trim flags")
	}
	var values []string
	if t&TrimAdapter != 0 {
		values = append(values, "adapter")
	}
	if t&TrimPrimer != 0 {
		values = append(values, "primer")
	}
	if len(values) == 0 {
		return "none", nil
	}
	return strings.Join(values, ","), nil
}

func fastqMetadata(r pod5.Read, spec model.Spec) (*FASTQMetadata, error) {
	if r.SampleRate == 0 || r.StartSample > math.MaxUint64/1000 {
		return nil, errors.New("invalid FASTQ sample clock")
	}
	delta := r.StartSample * 1000 / r.SampleRate
	if delta > math.MaxInt64 || r.AcquisitionStartTimeMS > math.MaxInt64-int64(delta) {
		return nil, errors.New("FASTQ read timestamp overflows int64")
	}
	return &FASTQMetadata{Channel: r.Channel, StartTimeMS: r.AcquisitionStartTimeMS + int64(delta), ProtocolStartTimeMS: r.ProtocolStartTimeMS, RunID: r.RunID, ModelName: spec.Name, FlowCellID: r.FlowCellID, MeanQStart: spec.MeanQStart}, nil
}

func timestamp(ms int64) (string, error) {
	// The pinned libstdc++ system_clock represents its intermediate time in
	// int64 nanoseconds before Dorado converts it to microseconds.
	if ms < math.MinInt64/1000000 || ms > math.MaxInt64/1000000 {
		return "", errors.New("FASTQ timestamp exceeds native system_clock range")
	}
	t := time.UnixMilli(ms).UTC()
	return t.Format("2006-01-02T15:04:05.000000+00:00"), nil
}

// WriteFASTQ emits the pinned stock FASTQ tag selection and BAM insertion order.
func (r CalledRead) WriteFASTQ(w io.Writer) error {
	if _, err := pod5.ParseID(r.ID); err != nil {
		return err
	}
	if len(r.Sequence) == 0 || len(r.Sequence) != len(r.Qstring) || strings.IndexFunc(r.Sequence, func(c rune) bool { return !strings.ContainsRune("ACGTN", c) }) >= 0 {
		return errors.New("invalid FASTQ sequence or sequence/quality lengths")
	}
	for i := range r.Qstring {
		if r.Qstring[i] < 33 || r.Qstring[i] > 126 {
			return errors.New("invalid printable FASTQ quality")
		}
	}
	m := r.FASTQ
	if m == nil || m.MeanQStart < 0 {
		return errors.New("missing or invalid FASTQ metadata")
	}
	trim, err := m.TrimFlags.text()
	if err != nil {
		return err
	}
	for _, value := range []string{m.RunID, m.ModelName, m.FlowCellID} {
		if strings.IndexFunc(value, func(c rune) bool { return c < 33 || c > 126 }) >= 0 {
			return errors.New("invalid FASTQ metadata text")
		}
	}
	start, err := timestamp(m.StartTimeMS)
	if err != nil {
		return err
	}
	protocol, err := timestamp(m.ProtocolStartTimeMS)
	if err != nil {
		return err
	}
	q := r.Qstring
	if len(q) > m.MeanQStart {
		q = q[m.MeanQStart:]
	}
	mean, err := decode.MeanQScore(q)
	if err != nil {
		return err
	}
	header := "@" + r.ID + "\tqs:f:" + strconv.FormatFloat(float64(mean), 'g', 6, 32) + "\tch:i:" + strconv.Itoa(int(m.Channel)) + "\tst:Z:" + start
	if m.RunID != "" {
		name := m.ModelName
		if name == "" {
			name = "unknown"
		}
		header += "\tRG:Z:" + m.RunID + "_" + name
	}
	if m.FlowCellID != "" {
		header += "\tPU:Z:" + m.FlowCellID
	}
	header += "\tDT:Z:" + protocol + "\ttm:Z:" + trim
	_, err = fmt.Fprintf(w, "%s\n%s\n+\n%s\n", header, r.Sequence, r.Qstring)
	return err
}
