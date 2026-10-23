//go:build linux

package gpucaller

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

const PostcallReceiptSchema = "gpu-postcall-replay/1"
const postcallMaxCaptureBytes = 64 << 20

// PostcallReceipt records a producer-free replay. It is not a native run and
// does not claim GPU execution.
type PostcallReceipt struct {
	Schema            string
	CaptureSHA256     string
	CaptureDigest     string
	Lane              string
	Model             string
	ModelConfigSHA256 string
	Policy            postcall.Policy
	Children          int
	Reads             int
	FastqSHA256       string
	Deterministic     bool
	Started, Finished string
}

// PostcallHandlers supplies host replay only. Native execution stays unwired.
func PostcallHandlers() UnitHandlers {
	return UnitHandlers{Import: nil, Replay: postcallReplay, Native: nil}
}

func postcallRead(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > postcallMaxCaptureBytes {
		return nil, errors.New("capture is not a bounded regular file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != info.Size() {
		return nil, errors.New("capture changed while reading")
	}
	return data, nil
}

func postcallReplay(r Request) error {
	started := time.Now().UTC().Format(time.RFC3339)
	data, err := postcallRead(r.Input)
	if err != nil {
		return err
	}
	capture, err := caller.DecodeGPUCapture(data)
	if err != nil {
		return err
	}
	out, err := caller.ReplayGPUCapture(capture, capture.Policy)
	if err != nil {
		return err
	}
	fastq, err := out.FASTQBytes()
	if err != nil {
		return err
	}
	again, err := caller.ReplayGPUCapture(capture, capture.Policy)
	if err != nil {
		return err
	}
	repeat, err := again.FASTQBytes()
	if err != nil {
		return err
	}
	if !bytes.Equal(fastq, repeat) {
		return errors.New("replay is not deterministic")
	}
	if err := os.Mkdir(r.Output, 0o755); err != nil {
		return fmt.Errorf("fresh output required: %w", err)
	}
	if err := os.WriteFile(filepath.Join(r.Output, "reads.fastq"), fastq, 0o644); err != nil {
		return err
	}
	spec, err := caller.CaptureSpec(capture)
	if err != nil {
		return err
	}
	receipt := PostcallReceipt{
		Schema: PostcallReceiptSchema, CaptureSHA256: postcallHash(data), CaptureDigest: capture.Digest,
		Lane: capture.Lane, Model: spec.Name, ModelConfigSHA256: spec.ConfigSHA256,
		Policy:   capture.Policy,
		Children: len(out.Postcall.Children), Reads: len(out.Reads), FastqSHA256: postcallHash(fastq),
		Deterministic: true, Started: started, Finished: time.Now().UTC().Format(time.RFC3339),
	}
	body, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	return publishPostcall(filepath.Join(r.Output, "postcall-receipt.json"), body)
}

// publishPostcall writes finality last and syncs before linking the name.
func publishPostcall(path string, body []byte) (err error) {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".postcall-receipt-*.tmp")
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

func postcallHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
