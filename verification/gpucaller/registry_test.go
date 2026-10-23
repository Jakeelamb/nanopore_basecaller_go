package gpucaller

import (
	"errors"
	"testing"
)

func TestFailClosedDispatch(t *testing.T) {
	calls := 0
	handler := func(Request) error { calls++; return nil }
	h := Handlers{ScheduleCheck: UnitHandlers{Import: handler, Replay: handler, Native: handler}}
	for _, r := range []Request{
		{Unit: ScheduleCheck, Mode: Native, Input: "in", Output: "out"},
		{Unit: ScheduleCheck, Mode: "unknown", Input: "in", Output: "out"},
		{Unit: "unknown", Mode: Import, Input: "in", Output: "out"},
		{Mode: Import, Input: "in", Output: "out"},
		{Unit: ScheduleCheck, Mode: Import, Output: "out"},
		{Unit: Inventory, Mode: Import, Input: "in", Output: "out"},
	} {
		if err := Dispatch(r, h); err == nil {
			t.Fatal("invalid request admitted")
		}
	}
	if calls != 0 {
		t.Fatalf("launched %d handlers", calls)
	}
}
func TestUnitIsolation(t *testing.T) {
	units := []Unit{ScheduleCheck, PostcallCheck, Qualify, Biological, Benchmark, Inventory}
	for _, mode := range []Mode{Import, Replay, Native} {
		seen := []Unit{}
		registration := func(want Unit) UnitHandlers {
			handler := func(r Request) error {
				if r.Unit != want || r.Mode != mode {
					t.Fatalf("wrong handler for %+v", r)
				}
				seen = append(seen, want)
				return nil
			}
			return UnitHandlers{Import: handler, Replay: handler, Native: handler}
		}
		h := Handlers{ScheduleCheck: registration(ScheduleCheck), PostcallCheck: registration(PostcallCheck), Qualify: registration(Qualify), Biological: registration(Biological), Benchmark: registration(Benchmark), Inventory: registration(Inventory)}
		for _, unit := range units {
			if err := Dispatch(Request{Unit: unit, Mode: mode, Input: "saved", Output: "fresh", Admitted: mode == Native}, h); err != nil {
				t.Fatal(err)
			}
		}
		if len(seen) != len(units) {
			t.Fatal("missing unit dispatch")
		}
	}
}

func TestInventoryHostOnlyRegistration(t *testing.T) {
	for _, mode := range []Mode{Import, Replay} {
		for _, admitted := range []bool{false, true} {
			calls := 0
			want := Request{Unit: Inventory, Mode: mode, Input: "retained", Output: "fresh", Admitted: admitted}
			handler := func(got Request) error {
				calls++
				if got != want {
					t.Fatalf("request changed: got %+v want %+v", got, want)
				}
				return nil
			}
			h := Handlers{Inventory: UnitHandlers{Import: handler, Replay: handler}}
			if err := Dispatch(want, h); err != nil || calls != 1 {
				t.Fatalf("host inventory dispatch: calls=%d err=%v", calls, err)
			}
		}
	}
	for _, admitted := range []bool{false, true} {
		calls := 0
		handler := func(Request) error { calls++; return nil }
		h := Handlers{Inventory: UnitHandlers{Import: handler, Replay: handler}}
		if err := Dispatch(Request{Unit: Inventory, Mode: Native, Input: "retained", Output: "fresh", Admitted: admitted}, h); err == nil || calls != 0 {
			t.Fatalf("host-only inventory reached a handler: calls=%d err=%v", calls, err)
		}
	}
}

func TestInventoryRegistrationPreservesOtherUnits(t *testing.T) {
	calls := 0
	wantErr := errors.New("retained adapter rejection")
	h := Handlers{ScheduleCheck: UnitHandlers{Replay: func(Request) error { calls++; return nil }}}
	h.Inventory = UnitHandlers{Import: func(Request) error { return wantErr }}
	if err := Dispatch(Request{Unit: ScheduleCheck, Mode: Replay, Input: "in", Output: "out"}, h); err != nil || calls != 1 {
		t.Fatalf("inventory registration changed another unit: calls=%d err=%v", calls, err)
	}
	if err := Dispatch(Request{Unit: Inventory, Mode: Import, Input: "in", Output: "out"}, h); !errors.Is(err, wantErr) {
		t.Fatalf("adapter rejection lost: %v", err)
	}
}
