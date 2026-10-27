package gpudecode

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestGuidePlanAcceptsFixedProfilesAndRejectsUnsafeGeometry(t *testing.T) {
	for _, v := range [][3]int64{{64, 32, 256}, {32, 64, 1024}} {
		if _, err := PlanGuides(v[0], v[1], v[2], 0, 2); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range [][3]int64{{0, 32, 256}, {63, 32, 256}, {32, 32, 256}, {64, 0, 256}, {64, 32, 255}, {64, math.MaxInt32, 256}, {math.MaxInt64, 32, 1024}, {64, 1 << 25, 256}} {
		if _, err := PlanGuides(v[0], v[1], v[2], 0, 2); err == nil {
			t.Fatalf("unsafe geometry accepted %v", v)
		}
	}
	for _, v := range [][2]float32{{-1, 2}, {float32(math.Inf(1)), 2}, {float32(math.NaN()), 2}, {0, float32(math.Inf(-1))}} {
		if _, err := PlanGuides(64, 32, 256, v[0], v[1]); err == nil {
			t.Fatal("nonfinite or negative option accepted")
		}
	}
}

func TestGuideScheduleInitializesTerminalBeforeEveryReverseFrame(t *testing.T) {
	p, err := PlanGuides(64, 32, 256, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	var events []int64
	err = guideSchedule(context.Background(), p, func() error { events = append(events, 32); return nil }, func(t int64) error { events = append(events, t); return nil })
	var want []int64
	for i := int64(32); i >= 0; i-- {
		want = append(want, i)
	}
	if err != nil || !reflect.DeepEqual(events, want) {
		t.Fatalf("schedule %v err=%v", events, err)
	}
}

func TestGuideScheduleStopsAfterCancellationOrNativeFailure(t *testing.T) {
	p, _ := PlanGuides(64, 32, 256, 0, 2)
	sentinel := errors.New("native frame failure")
	calls := 0
	err := guideSchedule(context.Background(), p, func() error { return nil }, func(t int64) error {
		calls++
		if t == 27 {
			return sentinel
		}
		return nil
	})
	if !errors.Is(err, sentinel) || calls != 5 {
		t.Fatalf("failed frame continued: %d %v", calls, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	calls = 0
	err = guideSchedule(ctx, p, func() error { cancel(); return nil }, func(t int64) error { calls++; return nil })
	if !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("canceled scan ran frame: %d %v", calls, err)
	}
	err = guideSchedule(nil, p, func() error { t.Fatal("nil context initialized output"); return nil }, func(int64) error { return nil })
	if err == nil {
		t.Fatal("nil context accepted")
	}
}
