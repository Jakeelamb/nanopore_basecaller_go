// Package subread materializes pinned Dorado plain simplex DNA children.
package subread

import (
	"crypto/sha256"
	"fmt"
	"math"
	"strconv"
	"time"
)

// The pinned Linux native time conversion uses a signed 64-bit nanosecond clock.
const maxTimestampMS = uint64(math.MaxInt64 / 1_000_000)

// Interval is a half-open base interval occupied by a spacer.
type Interval struct{ Start, End uint64 }

// Inherited contains the scalar and string state copied by shallow_copy_read.
// Float values remain binary32 bits. ClientToken represents an opaque inherited
// client identity; this package does not implement client behavior.
type Inherited struct {
	DigitisationBits, RangeBits, OffsetBits, OpenPoreLevelBits uint32
	ShiftBits, ScaleBits, ScalingBits                          uint32
	ScalingMethod                                              string
	RunID, FlowCellProductCode, SequencingKit                  string
	FlowcellID, PositionID, AcquisitionID, ExperimentID        string
	ModelName, SampleID                                        string
	ProtocolStartMS                                            int64
	ReadTag, ClientToken                                       uint64
	Mux                                                        uint32
	Channel                                                    int32
	Filename, PoreType                                         string
}

// Read holds only plain simplex DNA state. RNA, duplex and modified-base state
// have no representation here. Signal contains normalized binary16 bits.
// Inputs and returned signal views are borrowed and must remain immutable while
// in use. Actual children own their move tables; unchanged reads retain theirs.
type Read struct {
	Signal                               []uint16
	Sequence, Qstring                    string
	Moves                                []uint8
	Stride                               uint64
	ID, ParentID, PrevID, NextID         string
	StartSample, EndSample               uint64
	AcquisitionStartMS, SampleRate       uint64
	StartTimeMS                          uint64
	StartTime                            string
	LeadingTrim                          uint64
	SplitPoint                           uint32
	SubreadID, SplitCount                uint64
	ReadNumber                           int64
	NumSamples, NumEvents, PreTrimSeqLen uint64
	MeanQStart                           uint32
	EndReason                            string
	IsMuxChange                          bool
	Inherited                            Inherited
}

// DeriveUUID matches derive_uuid(parent, decimal(index)), including its version
// 4 bits despite deterministic SHA256 derivation. Parent is hashed verbatim.
func DeriveUUID(parent string, index uint64) string {
	h := sha256.Sum256([]byte(parent + strconv.FormatUint(index, 10)))
	h[6] = h[6]&0x0f | 0x40
	h[8] = h[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", h[:4], h[4:6], h[6:8], h[8:10], h[10:16])
}

// Materialize applies one supplied split-finder result and native finalization.
// It does not detect spacers or change its inputs. Spacers must be sorted and
// nonoverlapping. Zero-width spacers are admitted as in the native slice helper.
func Materialize(in Read, spacers []Interval) ([]Read, error) {
	if err := validate(in, spacers); err != nil {
		return nil, err
	}
	if len(in.Sequence) == 0 || len(in.Moves) == 0 {
		return []Read{in}, nil
	}
	var out []Read
	if len(spacers) == 0 {
		out = []Read{in}
	} else {
		mapping := make([]uint64, 0, len(in.Sequence)+1)
		for i, move := range in.Moves {
			if move == 1 {
				mapping = append(mapping, uint64(i)*in.Stride)
			}
		}
		mapping = append(mapping, uint64(len(in.Signal)))
		start, signalStart := uint64(0), mapping[0]
		for _, spacer := range spacers {
			if start < spacer.Start && signalStart/in.Stride < mapping[spacer.Start]/in.Stride {
				r, err := child(in, start, spacer.Start, signalStart, mapping[spacer.Start])
				if err != nil {
					return nil, err
				}
				out = append(out, r)
			}
			start, signalStart = spacer.End, mapping[spacer.End]
		}
		if start < uint64(len(in.Sequence)) && signalStart/in.Stride < uint64(len(in.Signal))/in.Stride {
			r, err := child(in, start, uint64(len(in.Sequence)), signalStart, uint64(len(in.Signal)))
			if err != nil {
				return nil, err
			}
			out = append(out, r)
		}
		if len(out) > 0 {
			out[len(out)-1].EndReason = in.EndReason
			out[len(out)-1].IsMuxChange = in.IsMuxChange
		}
	}
	var id uint64
	for i := range out {
		if out[i].ParentID != "" {
			out[i].SubreadID, out[i].SplitCount = id, uint64(len(out))
			out[i].ID = DeriveUUID(out[i].ParentID, id)
			id++
		}
	}
	if len(out) > 1 {
		for i := range out {
			if i > 0 {
				out[i].PrevID = out[i-1].ID
			}
			if i+1 < len(out) {
				out[i].NextID = out[i+1].ID
			}
		}
	}
	return out, nil
}

func child(in Read, baseStart, baseEnd, signalStart, signalEnd uint64) (Read, error) {
	startSample := in.StartSample + in.LeadingTrim + signalStart
	roundedMS := math.Round(float64(startSample) * 1000.0 / float64(in.SampleRate))
	if roundedMS > float64(maxTimestampMS) || uint64(roundedMS) > maxTimestampMS-in.AcquisitionStartMS {
		return Read{}, fmt.Errorf("child timestamp outside native signed-nanosecond clock domain")
	}
	out := in
	out.Signal = in.Signal[signalStart:signalEnd]
	out.Sequence, out.Qstring = in.Sequence[baseStart:baseEnd], in.Qstring[baseStart:baseEnd]
	out.Moves = append([]uint8{}, in.Moves[signalStart/in.Stride:signalEnd/in.Stride]...)
	out.NumEvents, out.ReadNumber = 0, -1
	out.SplitPoint = uint32(uint64(in.SplitPoint) + in.LeadingTrim + signalStart)
	out.NumSamples, out.LeadingTrim = signalEnd-signalStart, 0
	out.StartSample = startSample
	out.EndSample = out.StartSample + out.NumSamples
	out.StartTimeMS = uint64(roundedMS)
	out.StartTime = time.UnixMilli(int64(in.AcquisitionStartMS + out.StartTimeMS)).UTC().Format("2006-01-02T15:04:05.000000+00:00")
	out.EndReason, out.IsMuxChange = "unknown", false
	out.PreTrimSeqLen, out.MeanQStart = uint64(len(out.Sequence)), 0
	out.ParentID = in.ParentID
	if out.ParentID == "" {
		out.ParentID = in.ID
	}
	return out, nil
}

func validate(in Read, spacers []Interval) error {
	if in.ID == "" || len(in.Sequence) != len(in.Qstring) {
		return fmt.Errorf("missing read ID or sequence/quality length mismatch")
	}
	if in.Stride == 0 || in.Stride > math.MaxInt32 || in.SampleRate == 0 || in.SampleRate > math.MaxUint32 {
		return fmt.Errorf("stride or sample rate outside admitted native domain")
	}
	if len(in.Sequence) > math.MaxInt32 || len(in.Moves) > math.MaxInt32 || len(in.Signal) > math.MaxInt32 {
		return fmt.Errorf("read counts outside admitted native domain")
	}
	if in.ReadNumber < math.MinInt32 || in.ReadNumber > math.MaxInt32 {
		return fmt.Errorf("read number outside native int32 domain")
	}
	for i := range in.Sequence {
		switch in.Sequence[i] {
		case 'A', 'C', 'G', 'T', 'N':
		default:
			return fmt.Errorf("invalid DNA base at %d", i)
		}
		if in.Qstring[i] < 33 || in.Qstring[i] > 126 {
			return fmt.Errorf("invalid Phred quality at %d", i)
		}
	}
	var bases uint64
	for i, move := range in.Moves {
		if move > 1 {
			return fmt.Errorf("nonbinary move at %d", i)
		}
		bases += uint64(move)
	}
	if len(in.Sequence) != 0 && len(in.Moves) != 0 {
		if bases != uint64(len(in.Sequence)) {
			return fmt.Errorf("move base count differs from sequence length")
		}
		samples, moves := uint64(len(in.Signal)), uint64(len(in.Moves))
		floor := samples / in.Stride
		ceil := floor
		if samples%in.Stride != 0 {
			ceil++
		}
		if moves != floor && moves != ceil {
			return fmt.Errorf("signal/move stride mismatch")
		}
	}
	var previous uint64
	for i, r := range spacers {
		if r.Start > r.End || r.End > uint64(len(in.Sequence)) || i > 0 && r.Start < previous {
			return fmt.Errorf("spacers must be sorted nonoverlapping base intervals")
		}
		previous = r.End
	}
	if in.StartSample > math.MaxUint64-in.LeadingTrim || in.StartSample+in.LeadingTrim > math.MaxUint64-uint64(len(in.Signal)) {
		return fmt.Errorf("sample coordinate overflow")
	}
	if in.AcquisitionStartMS > maxTimestampMS {
		return fmt.Errorf("acquisition timestamp outside native signed-nanosecond clock domain")
	}
	return nil
}
