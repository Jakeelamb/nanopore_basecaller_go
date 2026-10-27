package h05

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type ScratchResult struct {
	Layer               int    `json:"layer"`
	Bytes               int64  `json:"compared_bytes"`
	CandidateOffset     int64  `json:"candidate_offset"`
	NativeOffset        int64  `json:"native_offset"`
	CandidateFile       string `json:"candidate_backing_file"`
	CandidateFileSHA    string `json:"candidate_backing_sha256"`
	NativeFile          string `json:"native_backing_file"`
	NativeFileSHA       string `json:"native_backing_sha256"`
	CandidateScratchSHA string `json:"candidate_scratch_sha256"`
	NativeScratchSHA    string `json:"native_scratch_sha256"`
	Equal               bool   `json:"equal"`
}

func progress(dir, event string, layer int) error {
	body, err := json.Marshal(struct {
		Time, Event string
		Layer       int
	}{time.Now().UTC().Format(time.RFC3339Nano), event, layer})
	if err != nil {
		return err
	}
	file, err := os.OpenFile(filepath.Join(dir, "progress.jsonl"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(append(body, '\n'))
	return errors.Join(writeErr, file.Close())
}
func observeScratch(p prepared, captureDir, attempt string, layer int, backing []byte) (rec ScratchResult, err error) {
	stage, ok := p.capture.Stage(fmt.Sprintf("flstm%d", layer))
	if !ok {
		return rec, fmt.Errorf("missing retained recurrent stage")
	}
	nativePath := filepath.Join(captureDir, stage.BackingFile)
	native, err := os.ReadFile(nativePath)
	if err != nil {
		return rec, err
	}
	if sum(native) != stage.BackingSHA {
		return rec, fmt.Errorf("native backing changed after import")
	}
	size, err := model.HACFusedScratchBytes(int64(p.capture.Batch), int64(p.capture.TOut))
	if err != nil {
		return rec, err
	}
	// FLSTMStack.cpp temp is placed opposite conv2 current. The imported fixed
	// H01 arena puts conv2 at its back and the initialized U8 temp at byte zero.
	if p.conv.OffsetBytes == 0 || p.conv.OffsetBytes != p.capture.Reservation-p.conv.Bytes || int64(len(native)) < size || p.conv.OffsetBytes < size {
		return rec, fmt.Errorf("retained native scratch placement is not the declared front temp")
	}
	if p.scratch.Offset < 0 || int64(len(backing))-p.scratch.Offset < size {
		return rec, fmt.Errorf("candidate backing lacks declared scratch span")
	}
	candidatePath := filepath.Join(attempt, fmt.Sprintf("flstm%d-backing.bin", layer))
	if err := os.WriteFile(candidatePath, backing, 0600); err != nil {
		return rec, err
	}
	candidate := backing[p.scratch.Offset : p.scratch.Offset+size]
	expected := native[:size]
	rec = ScratchResult{Layer: layer, Bytes: size, CandidateOffset: p.scratch.Offset, NativeOffset: 0, CandidateFile: candidatePath, CandidateFileSHA: sum(backing), NativeFile: nativePath, NativeFileSHA: stage.BackingSHA, CandidateScratchSHA: sum(candidate), NativeScratchSHA: sum(expected), Equal: bytes.Equal(candidate, expected)}
	if got, e := runner.SHA(candidatePath); e != nil || got != rec.CandidateFileSHA {
		return rec, fmt.Errorf("retained candidate backing pin failed: %v", e)
	}
	return rec, nil
}

func retainCacheOrderEvidence(p prepared, captureDir, attempt string, payload []byte) (StageResult, error) {
	path := filepath.Join(attempt, "control-cache-order-inout.bin")
	if err := os.WriteFile(path, payload, 0600); err != nil {
		return StageResult{}, err
	}
	rec, err := compare(p, captureDir, "flstm0", payload)
	if err != nil {
		return rec, err
	}
	rec.CandidateFile = path
	rec.FullSHA = sum(payload)
	return rec, verifyControlEvidence(p, captureDir, rec)
}

func verifyControlEvidence(p prepared, captureDir string, rec StageResult) error {
	payload, err := os.ReadFile(rec.CandidateFile)
	if err != nil {
		return err
	}
	if sum(payload) != rec.FullSHA {
		return fmt.Errorf("retained control payload digest mismatch")
	}
	actual, err := compare(p, captureDir, rec.Name, payload)
	if err != nil {
		return err
	}
	if actual.Equal != rec.Equal || actual.ComparedBytes != rec.ComparedBytes || actual.GotSHA != rec.GotSHA || actual.NativeSHA != rec.NativeSHA || !reflect.DeepEqual(actual.ExcludedRows, rec.ExcludedRows) {
		return fmt.Errorf("retained control comparison differs from declared evidence")
	}
	return nil
}
