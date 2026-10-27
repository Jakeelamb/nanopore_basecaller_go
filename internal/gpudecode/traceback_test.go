package gpudecode

import (
	"reflect"
	"testing"
)

func TestTraceBeamHistory(t *testing.T) {
	rows := [][]BeamHistoryRecord{
		{{State: 5, Previous: 255, StayByte: 7}, {State: 6, Previous: 255, StayByte: 1}},
		{{State: 20, Previous: 0}, {State: 21, Previous: 1, StayByte: 2}},
		{{State: 30, Previous: 1}, {State: 65535, Previous: 255}},
		{{State: 40, Previous: 0, StayByte: 255}},
	}
	before := make([][]BeamHistoryRecord, len(rows))
	for i := range rows {
		before[i] = append([]BeamHistoryRecord(nil), rows[i]...)
	}
	got, err := TraceBeamHistory(rows, 256)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Path, []int32{6, 21, 30, 40}) || !reflect.DeepEqual(got.Moves, []byte{1, 0, 1, 0}) {
		t.Fatalf("trace = %+v", got)
	}
	if !reflect.DeepEqual(rows, before) {
		t.Fatal("history mutated")
	}
	got.Path[0] = 99
	if rows[0][1].State != 6 {
		t.Fatal("trace aliases history")
	}
}

func TestTraceBeamHistorySingleFrame(t *testing.T) {
	got, err := TraceBeamHistory([][]BeamHistoryRecord{{{State: 1023, Previous: 255, StayByte: 1}}}, 1024)
	if err != nil || !reflect.DeepEqual(got.Path, []int32{1023}) || !reflect.DeepEqual(got.Moves, []byte{1}) {
		t.Fatalf("%+v %v", got, err)
	}
}

func TestTraceBeamHistoryRejectsInvalidSelectedRecords(t *testing.T) {
	cases := []struct {
		rows   [][]BeamHistoryRecord
		states int
	}{
		{nil, 256},
		{[][]BeamHistoryRecord{{{}}}, 16},
		{[][]BeamHistoryRecord{nil}, 256},
		{[][]BeamHistoryRecord{make([]BeamHistoryRecord, 33)}, 256},
		{[][]BeamHistoryRecord{{{State: 256}}}, 256},
		{[][]BeamHistoryRecord{{{State: 65535}}}, 1024},
		{[][]BeamHistoryRecord{{{}}, {{Previous: 1}}}, 256},
	}
	for i, tc := range cases {
		got, err := TraceBeamHistory(tc.rows, tc.states)
		if err == nil || got.Path != nil || got.Moves != nil {
			t.Fatalf("case%d returned partial or successful trace: %+v %v", i, got, err)
		}
	}
}
