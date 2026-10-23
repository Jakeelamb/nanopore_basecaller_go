package caller

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

const processedSchema = "postcall/1"
const postcallSource = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

// A core reference is explicit and content pinned. Absolute paths permit new
// sidecars without copying or changing large immutable core captures.
type coreReference struct {
	Path, SHA256 string
	Bytes        int64
}
type processedManifest struct {
	Schema, SourceCommit    string
	Core                    coreReference
	Policy, Output, FASTQ   payload
	InputReads, OutputReads int
}

func manifestBytes(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > 16<<20 {
		return nil, errors.New("invalid manifest file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != info.Size() {
		return nil, errors.New("manifest changed while reading")
	}
	return data, nil
}
func strictJSON(data []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return errors.New("trailing capture JSON")
	}
	return nil
}
func checkCore(ref coreReference) error {
	if !filepath.IsAbs(ref.Path) || filepath.Clean(ref.Path) != ref.Path || filepath.Base(ref.Path) != "manifest.json" || len(ref.SHA256) != 64 || ref.Bytes < 0 || ref.Bytes > 16<<20 {
		return errors.New("invalid processed core reference")
	}
	data, err := manifestBytes(ref.Path)
	if err != nil {
		return err
	}
	if int64(len(data)) != ref.Bytes || digest(data) != ref.SHA256 {
		return errors.New("processed core manifest hash differs")
	}
	return nil
}

func publishProcessedManifest(dir string, m processedManifest) (err error) {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	f, err := os.CreateTemp(dir, ".processed-manifest-*.tmp")
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, os.Remove(f.Name())) }()
	n, writeErr := f.Write(data)
	if writeErr == nil && n != len(data) {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		writeErr = f.Sync()
	}
	if err := errors.Join(writeErr, f.Close()); err != nil {
		return err
	}
	return os.Link(f.Name(), filepath.Join(dir, "processed-manifest.json"))
}

// captureProcessed writes finality last. The directory already exists for a
// producer capture. Replay callers create a new directory before invoking it.
func captureProcessed(dir, core string, result Processed) (string, error) {
	absolute, err := filepath.Abs(core)
	if err != nil {
		return "", err
	}
	data, err := manifestBytes(absolute)
	if err != nil {
		return "", err
	}
	ref := coreReference{Path: absolute, SHA256: digest(data), Bytes: int64(len(data))}
	if err := checkCore(ref); err != nil {
		return "", err
	}
	fastq, err := result.FASTQBytes()
	if err != nil {
		return "", err
	}
	p, err := saveJSON(dir, "processed-policy.json", result.Postcall.Policy)
	if err != nil {
		return "", err
	}
	output, err := saveJSON(dir, "processed-output.json", result)
	if err != nil {
		return "", err
	}
	fq, err := save(dir, "processed.fastq", fastq)
	if err != nil {
		return "", err
	}
	m := processedManifest{Schema: processedSchema, SourceCommit: postcallSource, Core: ref, Policy: p, Output: output, FASTQ: fq, InputReads: 1, OutputReads: len(result.Reads)}
	if err := publishProcessedManifest(dir, m); err != nil {
		return "", err
	}
	return filepath.Join(dir, "processed-manifest.json"), nil
}

// ReplayProcessed composes postprocessing over an original core capture. It
// never loads model or POD5 libraries. An optional capture requires a new dir.
func ReplayProcessed(ctx context.Context, core string, policy postcall.Policy, capture string) (Processed, error) {
	s, err := replayStage(ctx, core)
	if err != nil {
		return Processed{}, err
	}
	out, err := s.process(policy)
	if err != nil {
		return Processed{}, err
	}
	if err := ctx.Err(); err != nil {
		return Processed{}, err
	}
	if capture != "" {
		if err := os.Mkdir(capture, 0700); err != nil {
			return Processed{}, fmt.Errorf("processed capture requires a new directory: %w", err)
		}
		if _, err := captureProcessed(capture, core, out); err != nil {
			return Processed{}, err
		}
	}
	return out, nil
}

func replayFinal(ctx context.Context, path string, data []byte) (Processed, error) {
	var m processedManifest
	if err := strictJSON(data, &m); err != nil {
		return Processed{}, err
	}
	if m.Schema != processedSchema || m.SourceCommit != postcallSource || m.InputReads != 1 || m.OutputReads < 0 {
		return Processed{}, errors.New("unsupported processed manifest")
	}
	if err := checkCore(m.Core); err != nil {
		return Processed{}, err
	}
	dir := filepath.Dir(path)
	var policy postcall.Policy
	if err := readJSON(dir, m.Policy, &policy); err != nil {
		return Processed{}, err
	}
	var expected Processed
	if err := readJSON(dir, m.Output, &expected); err != nil {
		return Processed{}, err
	}
	fastq, err := readPayload(dir, m.FASTQ)
	if err != nil {
		return Processed{}, err
	}
	actual, err := ReplayProcessed(ctx, m.Core.Path, policy, "")
	if err != nil {
		return Processed{}, err
	}
	if !reflect.DeepEqual(actual, expected) || len(actual.Reads) != m.OutputReads {
		return Processed{}, errors.New("processed endpoint differs from capture")
	}
	actualFASTQ, err := actual.FASTQBytes()
	if err != nil {
		return Processed{}, err
	}
	if !bytes.Equal(actualFASTQ, fastq) {
		return Processed{}, errors.New("processed FASTQ endpoint differs from capture")
	}
	if err := checkCore(m.Core); err != nil {
		return Processed{}, err
	}
	return actual, ctx.Err()
}

// ReplayRecords recognizes a final processed manifest or preserves schema1 core
// replay. It checks every endpoint before returning any writer records.
func ReplayRecords(ctx context.Context, path string) ([]CalledRead, error) {
	if ctx == nil {
		return nil, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := manifestBytes(path)
	if err != nil {
		return nil, err
	}
	var probe struct{ Schema json.RawMessage }
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	if string(probe.Schema) == `"postcall/1"` {
		result, err := replayFinal(ctx, path, data)
		return result.Reads, err
	}
	r, err := Replay(ctx, path)
	if err != nil {
		return nil, err
	}
	return []CalledRead{r}, nil
}
