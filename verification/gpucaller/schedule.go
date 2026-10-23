//go:build linux

package gpucaller

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
)

const SchedulePlanSchema = "gpu-schedule-plan/1"
const ScheduleReceiptSchema = "gpu-schedule-receipt/2"
const scheduleMaxPlanBytes = 4 << 20
const scheduleMaxReceiptBytes = 16 << 20

// SchedulePlan is a host-only script over the scheduler. It carries no signal,
// weights or device state and never launches native code. FailAtBatch and
// CancelAtBatch are 1-based dispatch ordinals; zero disables them. Steps run in
// order and may be submit, flush, resume or shutdown; resume clears a sticky
// invocation failure so a later step can drain again.
type SchedulePlan struct {
	Schema        string
	Geometry      ScheduleGeometry
	Selection     string
	MaxPending    int
	Steps         []ScheduleStep
	FailAtBatch   int
	CancelAtBatch int
}

// ScheduleGeometry is the explicit geometry the plan pins. Lane is "hac" or
// "sup"; every other field maps to the matching internal/gpubatch field.
type ScheduleGeometry struct {
	Lane             string
	ChunkSize        int
	BatchSize        int
	Stride           int
	InnerStride      int
	Granularity      int
	FirstConvPadding int
	MaxConvPadding   int
	VariableChunks   bool
}

type ScheduleStep struct {
	Op     string
	Chunks []ScheduleChunk
}

type ScheduleChunk struct {
	ID     string
	Length int
}

// ScheduleBatchRecord is one invocation the plan actually dispatched. A batch
// that returned an error is recorded as failed and contributes no real work to
// the stats.
type ScheduleBatchRecord struct {
	Sequence      int
	Real          int
	PaddedSlots   int
	PaddedSamples int
	Chunks        []string
	Failed        bool
	Cancelled     bool
}

type ScheduleStepResult struct {
	Op    string
	Error string
}

// ScheduleResult is the deterministic part of the receipt.
type ScheduleResult struct {
	Selection  string
	Geometry   ScheduleGeometry
	MaxPending int
	Steps      []ScheduleStepResult
	Batches    []ScheduleBatchRecord
	Drains     int
	Stats      scheduler.Stats
}

// ScheduleReceipt pins the plan bytes, its digest and the replayable result.
// Started and Finished are the only non-deterministic fields.
type ScheduleReceipt struct {
	Schema        string
	PlanSHA256    string
	Plan          json.RawMessage
	Result        ScheduleResult
	ResultSHA256  string
	Deterministic bool
	Started       string
	Finished      string
}

// ScheduleHandlers supplies host import and replay only. P05 owns registration;
// the native path stays unwired.
func ScheduleHandlers() UnitHandlers {
	return UnitHandlers{Import: scheduleImport, Replay: scheduleReplay, Native: nil}
}

func scheduleRequest(r Request, mode Mode) error {
	if r.Unit != ScheduleCheck || r.Mode != mode || r.Input == "" || r.Output == "" {
		return fmt.Errorf("schedule-check requires its exact host unit/mode and input/fresh output")
	}
	return nil
}

func scheduleRead(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > limit {
		return nil, errors.New("schedule input is not a bounded regular file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != info.Size() {
		return nil, errors.New("schedule plan changed while reading")
	}
	return data, nil
}

func scheduleDecode(b []byte, out any) error {
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return fmt.Errorf("null schedule document")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("expected one JSON document")
	}
	return nil
}

// scheduleCanonical compacts plan bytes so a receipt pins one token stream
// regardless of the indentation MarshalIndent applies to embedded JSON.
func scheduleCanonical(b []byte) ([]byte, error) {
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func scheduleHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func (g ScheduleGeometry) config() (gpubatch.Config, error) {
	var lane gpubatch.Lane
	switch g.Lane {
	case "hac":
		lane = gpubatch.HAC
	case "sup":
		lane = gpubatch.SUP
	default:
		return gpubatch.Config{}, fmt.Errorf("unsupported schedule lane %q", g.Lane)
	}
	return gpubatch.Config{
		Lane: lane, ChunkSize: g.ChunkSize, BatchSize: g.BatchSize,
		Stride: g.Stride, InnerStride: g.InnerStride, Granularity: g.Granularity,
		FirstConvPadding: g.FirstConvPadding, MaxConvPadding: g.MaxConvPadding,
		VariableChunks: g.VariableChunks,
	}, nil
}

func scheduleSelection(s string) (scheduler.Selection, error) {
	switch s {
	case string(scheduler.SelectionExplicit):
		return scheduler.SelectionExplicit, nil
	case string(scheduler.SelectionAutomatic):
		return scheduler.SelectionAutomatic, nil
	case string(scheduler.SelectionMemory):
		return scheduler.SelectionMemory, nil
	case "":
		return "", fmt.Errorf("schedule plan must declare an explicit batch selection")
	default:
		return "", fmt.Errorf("unknown batch selection %q", s)
	}
}

func scheduleDecodePlan(b []byte) (SchedulePlan, error) {
	var plan SchedulePlan
	if err := scheduleDecode(b, &plan); err != nil {
		return SchedulePlan{}, err
	}
	if plan.Schema != SchedulePlanSchema {
		return SchedulePlan{}, fmt.Errorf("unsupported schedule plan schema %q", plan.Schema)
	}
	if _, err := plan.Geometry.config(); err != nil {
		return SchedulePlan{}, err
	}
	if _, err := scheduleSelection(plan.Selection); err != nil {
		return SchedulePlan{}, err
	}
	if len(plan.Steps) == 0 {
		return SchedulePlan{}, fmt.Errorf("schedule plan has no steps")
	}
	if last := plan.Steps[len(plan.Steps)-1]; last.Op != "shutdown" {
		return SchedulePlan{}, fmt.Errorf("schedule plan must end with shutdown, got %q", last.Op)
	}
	for i, step := range plan.Steps {
		switch step.Op {
		case "submit", "flush", "resume", "shutdown":
		default:
			return SchedulePlan{}, fmt.Errorf("schedule step %d: unsupported operation %q", i, step.Op)
		}
	}
	if plan.FailAtBatch < 0 || plan.CancelAtBatch < 0 {
		return SchedulePlan{}, fmt.Errorf("schedule batch ordinals must be non-negative")
	}
	return plan, nil
}

type scheduleInvocation struct {
	cancel   context.CancelFunc
	cancelAt int
	failAt   int
	mu       sync.Mutex
	calls    []ScheduleBatchRecord
	drains   int
}

func (i *scheduleInvocation) Call(ctx context.Context, b scheduler.Batch) error {
	rec := ScheduleBatchRecord{
		Sequence: b.Sequence, Real: b.Real,
		PaddedSlots: b.PaddedSlots, PaddedSamples: b.PaddedSamples,
	}
	for _, p := range b.Placements {
		rec.Chunks = append(rec.Chunks, p.Chunk.ID)
	}
	if i.cancel != nil && b.Sequence == i.cancelAt {
		i.cancel()
		rec.Cancelled = true
	}
	var err error
	if b.Sequence == i.failAt {
		rec.Failed = true
		err = fmt.Errorf("injected invocation failure at batch %d", b.Sequence)
	}
	i.mu.Lock()
	i.calls = append(i.calls, rec)
	i.mu.Unlock()
	return err
}

func (i *scheduleInvocation) Drain(ctx context.Context) error {
	i.mu.Lock()
	i.drains++
	i.mu.Unlock()
	return nil
}

func (i *scheduleInvocation) records() ([]ScheduleBatchRecord, int) {
	i.mu.Lock()
	defer i.mu.Unlock()
	return append([]ScheduleBatchRecord(nil), i.calls...), i.drains
}

// scheduleExecute runs one plan against one scheduler and records what the
// invocation owner actually received. It touches no device.
func scheduleExecute(plan SchedulePlan) (ScheduleResult, error) {
	geom, err := plan.Geometry.config()
	if err != nil {
		return ScheduleResult{}, err
	}
	selection, err := scheduleSelection(plan.Selection)
	if err != nil {
		return ScheduleResult{}, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	inv := &scheduleInvocation{cancel: cancel, cancelAt: plan.CancelAtBatch, failAt: plan.FailAtBatch}
	sched, err := scheduler.New(scheduler.Config{
		Geometry: geom, MaxPending: plan.MaxPending, Selection: selection,
	}, inv)
	if err != nil {
		return ScheduleResult{}, err
	}
	result := ScheduleResult{
		Selection: plan.Selection, Geometry: plan.Geometry, MaxPending: plan.MaxPending,
	}
	for i, step := range plan.Steps {
		var stepErr error
		switch step.Op {
		case "submit":
			chunks := make([]gpubatch.Chunk, len(step.Chunks))
			for j, c := range step.Chunks {
				chunks[j] = gpubatch.Chunk{ID: c.ID, Length: c.Length}
			}
			stepErr = sched.Submit(ctx, chunks...)
		case "flush":
			stepErr = sched.Flush(ctx)
		case "resume":
			stepErr = sched.Resume()
		case "shutdown":
			stepErr = sched.Shutdown(ctx)
		default:
			return ScheduleResult{}, fmt.Errorf("schedule step %d: unsupported operation %q", i, step.Op)
		}
		result.Steps = append(result.Steps, ScheduleStepResult{Op: step.Op, Error: scheduleError(stepErr)})
	}
	result.Stats = sched.Stats()
	result.Batches, result.Drains = inv.records()
	return result, nil
}

func scheduleError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func scheduleDigest(result ScheduleResult) (string, error) {
	b, err := json.Marshal(result)
	if err != nil {
		return "", err
	}
	return scheduleHash(b), nil
}

func scheduleImport(r Request) error {
	if err := scheduleRequest(r, Import); err != nil {
		return err
	}
	started := time.Now().UTC().Format(time.RFC3339)
	data, err := scheduleRead(r.Input, scheduleMaxPlanBytes)
	if err != nil {
		return err
	}
	plan, err := scheduleDecodePlan(data)
	if err != nil {
		return err
	}
	result, err := scheduleExecute(plan)
	if err != nil {
		return err
	}
	again, err := scheduleExecute(plan)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(result, again) {
		return errors.New("schedule import is not deterministic")
	}
	return scheduleSave(r, started, data, result)
}

func scheduleReplay(r Request) error {
	if err := scheduleRequest(r, Replay); err != nil {
		return err
	}
	started := time.Now().UTC().Format(time.RFC3339)
	data, err := scheduleRead(r.Input, scheduleMaxReceiptBytes)
	if err != nil {
		return err
	}
	var saved ScheduleReceipt
	if err := scheduleDecode(data, &saved); err != nil {
		return err
	}
	if saved.Schema != ScheduleReceiptSchema {
		return fmt.Errorf("unsupported schedule receipt schema %q", saved.Schema)
	}
	planBytes, err := scheduleCanonical(saved.Plan)
	if err != nil {
		return err
	}
	if saved.PlanSHA256 != scheduleHash(planBytes) {
		return fmt.Errorf("retained schedule plan does not match its digest")
	}
	plan, err := scheduleDecodePlan(planBytes)
	if err != nil {
		return err
	}
	result, err := scheduleExecute(plan)
	if err != nil {
		return err
	}
	digest, err := scheduleDigest(result)
	if err != nil {
		return err
	}
	if digest != saved.ResultSHA256 || !reflect.DeepEqual(result, saved.Result) {
		return fmt.Errorf("retained schedule result differs from replay")
	}
	return scheduleSave(r, started, planBytes, result)
}

func scheduleSave(r Request, started string, planBytes []byte, result ScheduleResult) error {
	digest, err := scheduleDigest(result)
	if err != nil {
		return err
	}
	canonical, err := scheduleCanonical(planBytes)
	if err != nil {
		return err
	}
	receipt := ScheduleReceipt{
		Schema: ScheduleReceiptSchema, PlanSHA256: scheduleHash(canonical),
		Plan: json.RawMessage(canonical), Result: result, ResultSHA256: digest,
		Deterministic: true, Started: started,
		Finished: time.Now().UTC().Format(time.RFC3339),
	}
	body, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	if len(body) > scheduleMaxReceiptBytes {
		return fmt.Errorf("schedule receipt exceeds %d bytes", scheduleMaxReceiptBytes)
	}
	if err := os.Mkdir(r.Output, 0o755); err != nil {
		return fmt.Errorf("fresh output required: %w", err)
	}
	return publishSchedule(filepath.Join(r.Output, "schedule-receipt.json"), body)
}

// publishSchedule writes finality last and syncs before linking the name.
func publishSchedule(path string, body []byte) (err error) {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".schedule-receipt-*.tmp")
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, os.Remove(f.Name())) }()
	n, writeErr := f.Write(body)
	if writeErr == nil && n != len(body) {
		writeErr = errors.New("short receipt write")
	}
	if writeErr == nil {
		writeErr = f.Sync()
	}
	if err := errors.Join(writeErr, f.Close()); err != nil {
		return err
	}
	return os.Link(f.Name(), path)
}
