//go:build linux

package gpucaller

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
)

func planChunks(from, to, length int) []ScheduleChunk {
	out := make([]ScheduleChunk, 0, to-from)
	for i := from; i < to; i++ {
		out = append(out, ScheduleChunk{ID: fmt.Sprintf("c%03d", i), Length: length})
	}
	return out
}

func planIDs(from, to int) []string {
	chunks := planChunks(from, to, 768)
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.ID
	}
	return out
}

func basePlan() SchedulePlan {
	return SchedulePlan{
		Schema: SchedulePlanSchema,
		Geometry: ScheduleGeometry{
			Lane: "hac", ChunkSize: 768, BatchSize: 64,
			Stride: 6, InnerStride: 6, Granularity: 6,
		},
		Selection:  "explicit",
		MaxPending: 256,
		Steps: []ScheduleStep{
			{Op: "submit", Chunks: planChunks(0, 70, 768)},
			{Op: "flush"},
			{Op: "shutdown"},
		},
	}
}

func writePlan(t *testing.T, plan SchedulePlan) string {
	t.Helper()
	b, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeRawPlan(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func scheduleHandlers() Handlers {
	return Handlers{ScheduleCheck: ScheduleHandlers()}
}

func runSchedule(t *testing.T, mode Mode, input string) (string, error) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "attempt")
	err := Dispatch(Request{Unit: ScheduleCheck, Mode: mode, Input: input, Output: out}, scheduleHandlers())
	return out, err
}

func readReceipt(t *testing.T, dir string) ScheduleReceipt {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "schedule-receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt ScheduleReceipt
	if err := scheduleDecode(b, &receipt); err != nil {
		t.Fatal(err)
	}
	return receipt
}

func TestScheduleImportAndReplay(t *testing.T) {
	plan := basePlan()
	path := writePlan(t, plan)
	dir, err := runSchedule(t, Import, path)
	if err != nil {
		t.Fatal(err)
	}
	receipt := readReceipt(t, dir)
	if receipt.Schema != ScheduleReceiptSchema || !receipt.Deterministic {
		t.Fatalf("receipt header %+v", receipt)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.PlanSHA256 != scheduleHash(body) {
		t.Fatal("plan digest does not pin the input bytes")
	}
	if receipt.Result.Stats.Dispatched != 2 || receipt.Result.Stats.RealChunks != 70 {
		t.Fatalf("stats %+v", receipt.Result.Stats)
	}
	if len(receipt.Result.Batches) != 2 || receipt.Result.Batches[1].Real != 6 {
		t.Fatalf("batches %+v", receipt.Result.Batches)
	}
	if receipt.Result.Drains != 1 || !receipt.Result.Stats.Drained {
		t.Fatalf("drain accounting %+v", receipt.Result)
	}
	if !reflect.DeepEqual(receipt.Result.Geometry, plan.Geometry) {
		t.Fatalf("geometry pin %+v", receipt.Result.Geometry)
	}
	if receipt.Result.Batches[1].PaddedSlots != 64 {
		t.Fatalf("padded capacity lost: %+v", receipt.Result.Batches[1])
	}

	replayDir, err := runSchedule(t, Replay, filepath.Join(dir, "schedule-receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	replayed := readReceipt(t, replayDir)
	if replayed.ResultSHA256 != receipt.ResultSHA256 {
		t.Fatalf("replay digest %s, want %s", replayed.ResultSHA256, receipt.ResultSHA256)
	}
	if !reflect.DeepEqual(replayed.Result, receipt.Result) {
		t.Fatal("replay result differs from the retained receipt")
	}
}

func TestScheduleRejectsAutomaticSelection(t *testing.T) {
	for name, mutate := range map[string]func(*SchedulePlan){
		"automatic": func(p *SchedulePlan) { p.Selection = "automatic" },
		"memory":    func(p *SchedulePlan) { p.Selection = "memory" },
		"zero":      func(p *SchedulePlan) { p.Geometry.BatchSize = 0 },
	} {
		plan := basePlan()
		mutate(&plan)
		_, err := runSchedule(t, Import, writePlan(t, plan))
		if !errors.Is(err, scheduler.ErrAutomaticSelection) {
			t.Fatalf("%s: got %v, want ErrAutomaticSelection", name, err)
		}
		var automatic *scheduler.AutomaticSelectionError
		if !errors.As(err, &automatic) {
			t.Fatalf("%s: error is not typed: %v", name, err)
		}
	}
}

func TestScheduleRejectsInvalidPlans(t *testing.T) {
	plan := basePlan()
	unknownOp := plan
	unknownOp.Steps = []ScheduleStep{{Op: "submit"}, {Op: "drain"}}
	noShutdown := plan
	noShutdown.Steps = []ScheduleStep{{Op: "submit"}, {Op: "flush"}}
	noSteps := plan
	noSteps.Steps = nil
	badLane := plan
	badLane.Geometry.Lane = "koi"
	badSelection := plan
	badSelection.Selection = "guess"
	negativeOrdinal := plan
	negativeOrdinal.FailAtBatch = -1
	for name, p := range map[string]SchedulePlan{
		"unknown-op": unknownOp, "no-shutdown": noShutdown, "no-steps": noSteps,
		"lane": badLane, "selection": badSelection, "negative-ordinal": negativeOrdinal,
	} {
		if _, err := runSchedule(t, Import, writePlan(t, p)); err == nil {
			t.Fatalf("%s plan accepted", name)
		}
	}
	unknownField := `{"Schema":"gpu-schedule-plan/1","Extra":1}`
	if _, err := runSchedule(t, Import, writeRawPlan(t, unknownField)); err == nil {
		t.Fatal("unknown field accepted")
	}
	trailing := string(mustJSON(t, basePlan())) + "\n{}\n"
	if _, err := runSchedule(t, Import, writeRawPlan(t, trailing)); err == nil {
		t.Fatal("trailing JSON accepted")
	}
	nullPlan := `null`
	if _, err := runSchedule(t, Import, writeRawPlan(t, nullPlan)); err == nil {
		t.Fatal("null plan accepted")
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestScheduleRecordsQueueAndShutdownRejections(t *testing.T) {
	plan := basePlan()
	plan.MaxPending = 2
	plan.Steps = []ScheduleStep{
		{Op: "submit", Chunks: planChunks(0, 2, 768)},
		{Op: "submit", Chunks: planChunks(2, 5, 768)},
		{Op: "flush"},
		{Op: "shutdown"},
		{Op: "shutdown"},
	}
	dir, err := runSchedule(t, Import, writePlan(t, plan))
	if err != nil {
		t.Fatal(err)
	}
	receipt := readReceipt(t, dir)
	steps := receipt.Result.Steps
	if len(steps) != 5 {
		t.Fatalf("steps %+v", steps)
	}
	if steps[0].Error != "" {
		t.Fatalf("first submit: %+v", steps[0])
	}
	if !strings.Contains(steps[1].Error, scheduler.ErrQueueFull.Error()) {
		t.Fatalf("over-capacity submit not recorded: %+v", steps[1])
	}
	for i := 2; i < len(steps); i++ {
		if steps[i].Error != "" {
			t.Fatalf("step %d: %+v", i, steps[i])
		}
	}
	st := receipt.Result.Stats
	if st.Submitted != 2 || st.Dispatched != 1 || st.RealChunks != 2 || st.Pending != 0 {
		t.Fatalf("over-capacity work leaked into the queue: %+v", st)
	}
	if receipt.Result.Drains != 1 {
		t.Fatalf("double shutdown drained %d times", receipt.Result.Drains)
	}
}

func TestScheduleRecordsInjectedFailure(t *testing.T) {
	plan := basePlan()
	plan.Steps = []ScheduleStep{
		{Op: "submit", Chunks: planChunks(0, 130, 768)},
		{Op: "flush"},
		{Op: "shutdown"},
	}
	plan.FailAtBatch = 2
	dir, err := runSchedule(t, Import, writePlan(t, plan))
	if err != nil {
		t.Fatal(err)
	}
	receipt := readReceipt(t, dir)
	if !strings.Contains(receipt.Result.Steps[1].Error, "injected invocation failure at batch 2") {
		t.Fatalf("failure not propagated: %+v", receipt.Result.Steps)
	}
	// The failure is sticky: shutdown must refuse to drain and report it rather
	// than dispatch the remaining chunks and return nil.
	shutdownErr := receipt.Result.Steps[2].Error
	if !strings.Contains(shutdownErr, scheduler.ErrInvocationFailed.Error()) || !strings.Contains(shutdownErr, "injected invocation failure at batch 2") {
		t.Fatalf("shutdown hid the sticky failure: %+v", receipt.Result.Steps)
	}
	st := receipt.Result.Stats
	if st.FailedBatches != 1 || st.Dispatched != 1 {
		t.Fatalf("failed batch accounting %+v", st)
	}
	if st.RealChunks != 64 || st.PaddedSlots != 64 {
		t.Fatalf("failed batch reported as successful work: %+v", st)
	}
	if st.FailedChunks != 64 || !reflect.DeepEqual(st.FailedChunkIDs, planIDs(64, 128)) {
		t.Fatalf("failed chunk accounting %+v", st)
	}
	if st.Abandoned != 2 || st.Pending != 0 {
		t.Fatalf("work behind the failure was not abandoned: %+v", st)
	}
	if len(receipt.Result.Batches) != 2 {
		t.Fatalf("shutdown dispatched past the failure: %+v", receipt.Result.Batches)
	}
	failed := 0
	for _, batch := range receipt.Result.Batches {
		if batch.Failed {
			failed++
			if batch.Real != 64 || !reflect.DeepEqual(batch.Chunks, planIDs(64, 128)) {
				t.Fatalf("failed batch record %+v", batch)
			}
		}
	}
	if failed != 1 {
		t.Fatalf("failed batches %d, want 1", failed)
	}
	if receipt.Result.Drains != 1 {
		t.Fatalf("shutdown after failure drained %d times", receipt.Result.Drains)
	}
}

func TestScheduleResumeAfterFailureDrainsRemainder(t *testing.T) {
	plan := basePlan()
	plan.Steps = []ScheduleStep{
		{Op: "submit", Chunks: planChunks(0, 130, 768)},
		{Op: "flush"},
		{Op: "resume"},
		{Op: "flush"},
		{Op: "shutdown"},
	}
	plan.FailAtBatch = 2
	dir, err := runSchedule(t, Import, writePlan(t, plan))
	if err != nil {
		t.Fatal(err)
	}
	receipt := readReceipt(t, dir)
	steps := receipt.Result.Steps
	if len(steps) != 5 {
		t.Fatalf("steps %+v", steps)
	}
	if !strings.Contains(steps[1].Error, "injected invocation failure at batch 2") {
		t.Fatalf("failure not propagated: %+v", steps)
	}
	for _, i := range []int{2, 3, 4} {
		if steps[i].Error != "" {
			t.Fatalf("step %d after resume: %+v", i, steps[i])
		}
	}
	st := receipt.Result.Stats
	if st.Dispatched != 2 || st.RealChunks != 66 || st.Pending != 0 || st.Abandoned != 0 {
		t.Fatalf("resumed drain stats %+v", st)
	}
	if st.FailedBatches != 1 || st.FailedChunks != 64 || !reflect.DeepEqual(st.FailedChunkIDs, planIDs(64, 128)) {
		t.Fatalf("resume cleared the failure record: %+v", st)
	}
	if !st.Drained || !st.Closed {
		t.Fatalf("shutdown state %+v", st)
	}
	batches := receipt.Result.Batches
	if len(batches) != 3 || !reflect.DeepEqual(batches[2].Chunks, planIDs(128, 130)) {
		t.Fatalf("resumed batches %+v", batches)
	}
	if receipt.Result.Drains != 1 {
		t.Fatalf("resumed run drained %d times", receipt.Result.Drains)
	}

	replayDir, err := runSchedule(t, Replay, filepath.Join(dir, "schedule-receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	replayed := readReceipt(t, replayDir)
	if replayed.ResultSHA256 != receipt.ResultSHA256 || !reflect.DeepEqual(replayed.Result, receipt.Result) {
		t.Fatalf("replay of the resumed plan differs:\n%+v\n%+v", replayed.Result, receipt.Result)
	}
}

func TestScheduleRecordsCancelMidBatch(t *testing.T) {
	plan := basePlan()
	plan.Steps = []ScheduleStep{
		{Op: "submit", Chunks: planChunks(0, 130, 768)},
		{Op: "flush"},
		{Op: "shutdown"},
	}
	plan.CancelAtBatch = 1
	dir, err := runSchedule(t, Import, writePlan(t, plan))
	if err != nil {
		t.Fatal(err)
	}
	receipt := readReceipt(t, dir)
	for _, index := range []int{1, 2} {
		if !strings.Contains(receipt.Result.Steps[index].Error, "context canceled") {
			t.Fatalf("step %d: %+v", index, receipt.Result.Steps[index])
		}
	}
	st := receipt.Result.Stats
	if st.Dispatched != 1 || st.RealChunks != 64 || st.FailedBatches != 0 {
		t.Fatalf("cancelled drain reported partial success: %+v", st)
	}
	if st.Abandoned != 66 || st.Pending != 0 {
		t.Fatalf("abandoned accounting %+v", st)
	}
	if len(receipt.Result.Batches) != 1 || !receipt.Result.Batches[0].Cancelled {
		t.Fatalf("batch records %+v", receipt.Result.Batches)
	}
}

func TestScheduleReplayRejectsTampering(t *testing.T) {
	path := writePlan(t, basePlan())
	dir, err := runSchedule(t, Import, path)
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(dir, "schedule-receipt.json")
	saved := readReceipt(t, dir)

	changedResult := saved
	changedResult.Result.Stats.Dispatched++
	if _, err := runSchedule(t, Replay, writeRawPlan(t, string(mustJSON(t, changedResult)))); err == nil {
		t.Fatal("tampered result accepted")
	}

	changedDigest := saved
	changedDigest.ResultSHA256 = strings.Repeat("0", 64)
	if _, err := runSchedule(t, Replay, writeRawPlan(t, string(mustJSON(t, changedDigest)))); err == nil {
		t.Fatal("tampered digest accepted")
	}

	changedPlan := saved
	var tamperedPlan SchedulePlan
	if err := scheduleDecode(saved.Plan, &tamperedPlan); err != nil {
		t.Fatal(err)
	}
	tamperedPlan.MaxPending = 1
	changedPlan.Plan = json.RawMessage(mustJSON(t, tamperedPlan))
	if _, err := runSchedule(t, Replay, writeRawPlan(t, string(mustJSON(t, changedPlan)))); err == nil {
		t.Fatal("tampered plan accepted")
	}

	if _, err := runSchedule(t, Replay, receiptPath); err != nil {
		t.Fatalf("untampered replay failed: %v", err)
	}
}

func TestScheduleRequiresFreshOutputAndRegularInput(t *testing.T) {
	path := writePlan(t, basePlan())
	out := filepath.Join(t.TempDir(), "attempt")
	req := Request{Unit: ScheduleCheck, Mode: Import, Input: path, Output: out}
	if err := Dispatch(req, scheduleHandlers()); err != nil {
		t.Fatal(err)
	}
	if err := Dispatch(req, scheduleHandlers()); err == nil {
		t.Fatal("existing output accepted")
	}
	if _, err := runSchedule(t, Import, t.TempDir()); err == nil {
		t.Fatal("directory input accepted")
	}
}

func TestScheduleHandlersAreHostOnly(t *testing.T) {
	h := ScheduleHandlers()
	if h.Import == nil || h.Replay == nil || h.Native != nil {
		t.Fatalf("handlers %+v", h)
	}
	req := Request{Unit: ScheduleCheck, Mode: Native, Input: "plan", Output: "out", Admitted: true}
	if err := Dispatch(req, scheduleHandlers()); err == nil {
		t.Fatal("native schedule mode is wired")
	}
}
