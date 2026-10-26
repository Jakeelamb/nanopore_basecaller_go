package gpudecode

import "fmt"

// BeamHistoryRecord describes the four-byte source record without relying on
// Go memory layout. StayByte is preserved because the source tests equality
// with zero rather than a particular nonzero encoding.
type BeamHistoryRecord struct {
	State    uint16
	Previous uint8
	StayByte uint8
}

type BeamTrace struct {
	// Path contains full state IDs before the quality pass replaces them with
	// base IDs. It is not the final decode_step path allocation.
	Path  []int32
	Moves []byte
}

// TraceBeamHistory walks only initialized records. Rows are source history
// frames 1 through T, each containing its retained beam, after terminal slot
// zero has been selected. The source's frame-zero history is never read.
func TraceBeamHistory(rows [][]BeamHistoryRecord, states int) (BeamTrace, error) {
	if len(rows) == 0 || (states != 256 && states != 1024) {
		return BeamTrace{}, fmt.Errorf("positive history length and HAC/SUP state count required")
	}
	for _, row := range rows {
		if len(row) == 0 || len(row) > 32 {
			return BeamTrace{}, fmt.Errorf("history rows require one to 32 initialized records")
		}
	}
	out := BeamTrace{Path: make([]int32, len(rows)), Moves: make([]byte, len(rows))}
	slot := 0
	for frame := len(rows) - 1; frame >= 0; frame-- {
		if slot >= len(rows[frame]) {
			return BeamTrace{}, fmt.Errorf("history frame %d predecessor selects uninitialized slot %d", frame+1, slot)
		}
		record := rows[frame][slot]
		if int(record.State) >= states {
			return BeamTrace{}, fmt.Errorf("history frame %d state exceeds model domain", frame+1)
		}
		out.Path[frame] = int32(int16(record.State))
		if record.StayByte == 0 {
			out.Moves[frame] = 1
		}
		slot = int(record.Previous)
	}
	out.Moves[0] = 1
	return out, nil
}
