package gpudecode

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestPosteriorSchedulePreservesOldAlphaAndFinalRow(t *testing.T) {
	for _, frames := range []int64{1, 2, 3, 32, 64} {
		p, err := PlanGuides(64, frames, 256, 0, 2)
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		err = posteriorSchedule(context.Background(), p, func(frame int64, previous, next int) error {
			got = append(got, fmt.Sprintf("alpha%d:%d>%d", frame+1, previous, next))
			return nil
		}, func(frame int64, alpha int) error {
			got = append(got, fmt.Sprintf("post%d:alpha%d", frame, alpha))
			return nil
		})
		var want []string
		for frame := int64(0); frame < frames; frame++ {
			want = append(want, fmt.Sprintf("alpha%d:%d>%d", frame+1, frame%2, (frame+1)%2), fmt.Sprintf("post%d:alpha%d", frame, frame%2))
		}
		want = append(want, fmt.Sprintf("post%d:alpha%d", frames, frames%2))
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("T=%d got=%v want=%v err=%v", frames, got, want, err)
		}
	}
}

func TestPosteriorScheduleStopsBeforeDestructiveStore(t *testing.T) {
	p, _ := PlanGuides(64, 32, 256, 0, 2)
	stop := errors.New("failed forward")
	for _, failRow := range []bool{false, true} {
		forward, rows := 0, 0
		err := posteriorSchedule(context.Background(), p, func(frame int64, _, _ int) error {
			forward++
			if frame == 2 && !failRow {
				return stop
			}
			return nil
		}, func(frame int64, _ int) error {
			rows++
			if frame == 2 && failRow {
				return stop
			}
			return nil
		})
		wantRows := 2
		if failRow {
			wantRows = 3
		}
		if !errors.Is(err, stop) || forward != 3 || rows != wantRows {
			t.Fatalf("failure continued forward=%d rows=%d err=%v", forward, rows, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	rows := 0
	err := posteriorSchedule(ctx, p, func(int64, int, int) error { cancel(); return nil }, func(int64, int) error { rows++; return nil })
	if !errors.Is(err, context.Canceled) || rows != 0 {
		t.Fatalf("destructive write after cancellation rows=%d err=%v", rows, err)
	}
	if err = posteriorSchedule(nil, p, func(int64, int, int) error { t.Fatal("nil context launched"); return nil }, func(int64, int) error { return nil }); err == nil {
		t.Fatal("nil context accepted")
	}
}

func TestPosteriorsRejectsZeroPlanAndCancelledContextBeforeDevice(t *testing.T) {
	if _, err := Posteriors(context.Background(), nil, nil, nil, nil, GuidePlan{}); err == nil {
		t.Fatal("zero plan accepted")
	}
	p, _ := PlanGuides(64, 32, 256, 0, 2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Posteriors(ctx, nil, nil, nil, nil, p); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled call reached device: %v", err)
	}
}
