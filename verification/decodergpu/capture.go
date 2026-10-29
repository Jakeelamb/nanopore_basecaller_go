package decodergpu

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const originalScoresSHA = "56b26176b4272906612eb75690087182a72527f94a9171d9937e5dd80b0d12db"

type CapturedBuffer struct {
	Stage, Role    string
	Raw            File
	Bytes          int64
	Compare        []Span
	ComparedBytes  int64
	ComparedSHA256 string
}
type CaptureReceipt struct {
	Schema, Status, Error, Started, Finished            string
	ControllerPID                                       int
	Build, Consumer, Runtime                            File
	SourceCapture, SupervisorCommand, SupervisorReceipt File
	QualifiedUTC                                        string
	Contract                                            Contract
	Guarded                                             supgpu.GuardedRun
	Code                                                []supgpu.Pin
	Images                                              ImageReport
	Identity                                            []NumericIdentity `json:",omitempty"`
	Buffers                                             []CapturedBuffer
	Files                                               map[string]string
}

func captureInputs(p ClosedBuildPlan) error {
	dir := filepath.Dir(p.Preparation.Path)
	params, err := os.ReadFile(filepath.Join(dir, "params.bin"))
	if err != nil {
		return err
	}
	chunks, err := os.ReadFile(filepath.Join(dir, "chunks.bin"))
	if err != nil {
		return err
	}
	if err = ValidateWire(params, chunks); err != nil {
		return err
	}
	scores, err := os.ReadFile(filepath.Join(dir, "scores-original.f16"))
	if err != nil {
		return err
	}
	return ValidateScores(scores, originalScoresSHA)
}

func readGuardedCapture(o CaptureOptions, current bool) (r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, err error) {
	return readGuardedCaptureScoped(o, current, nil)
}

func readGuardedCaptureScoped(o CaptureOptions, current bool, proofs *numericProofs) (r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, err error) {
	for _, file := range []*File{&o.Build, &o.Runtime, &o.Consumer} {
		*file, err = normalizeAnchor(*file)
		if err != nil {
			return
		}
	}
	b, p, err = readBuiltPhaseScoped(o.Build, false, proofs)
	if err != nil {
		return
	}
	if err = readJSON(o.Runtime, &r); err != nil {
		return
	}
	if r.Build != o.Build || r.ConsumerPreparation != o.Consumer {
		err = fmt.Errorf("capture build/current-consumer/runtime anchors differ")
		return
	}
	err = verifyGuardRuntimeScoped(r, b, p, current, proofs)
	if err == nil {
		err = captureInputs(p)
	}
	return
}

func CheckGuardedRuntime(build, consumer, runtime File) error {
	proofs := newNumericProofs()
	_, _, _, err := readGuardedCaptureScoped(CaptureOptions{Build: build, Consumer: consumer, Runtime: runtime}, true, proofs)
	return errors.Join(err, proofs.finish())
}

func Capture(o CaptureOptions) (receipt CaptureReceipt, err error) {
	if !o.Admitted {
		return receipt, fmt.Errorf("D01 decoder capture requires separate coordinator admission")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return receipt, err
	}
	defer lock.Close()
	proofs := newNumericProofs()
	r, b, p, err := readGuardedCaptureScoped(o, true, proofs)
	if err != nil {
		return receipt, err
	}
	out, err := filepath.Abs(o.Out)
	if err != nil {
		return receipt, err
	}
	if out != r.CaptureDirectory {
		return receipt, fmt.Errorf("capture output differs from prospective exact command")
	}
	if err = requireNativeLimits(); err != nil {
		return receipt, err
	}
	out, err = runner.NewAttempt(out)
	if err != nil {
		return receipt, err
	}
	runtimePin, err := normalizeAnchor(o.Runtime)
	if err != nil {
		return receipt, err
	}
	receipt = CaptureReceipt{Schema: "nanopore-decoder-native-capture/v1", Status: "failed", ControllerPID: os.Getpid(), Started: time.Now().UTC().Format(time.RFC3339Nano), Build: r.Build, Consumer: r.ConsumerPreparation, Runtime: runtimePin, Contract: FixedContract(), Files: map[string]string{}}
	defer func() {
		receipt.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			receipt.Error = err.Error()
		}
		save := writeJSON(filepath.Join(out, "capture.json"), receipt)
		err = errors.Join(err, save)
	}()
	if numericBuild(p) {
		receipt.Schema = "nanopore-decoder-native-capture/v2"
	}
	guard, err := guardPin(r.Guard)
	if err != nil {
		return receipt, err
	}
	phases, err := guardPin(r.PhaseContract)
	if err != nil {
		return receipt, err
	}
	command := decoderCommand(r, b, p)
	receipt.Guarded, err = proofs.native(func() (supgpu.GuardedRun, error) {
		return supgpu.ExecuteGuarded(context.Background(), guard, phases, command)
	}, func() error {
		return verifyGuardRuntimeScoped(r, b, p, true, proofs)
	})
	if err != nil {
		failure := filepath.Join(out, "native/images-failure")
		if _, e := os.Stat(failure); e == nil {
			var old Preparation
			old, e = readPreparation(p.Preparation, false, proofs)
			if e == nil {
				receipt.Images, _ = ReadImageReport(failure, old.Images)
			}
		}
		return receipt, err
	}
	receipt.Code, err = supgpu.ReplayGuarded(guard, phases, command, receipt.Guarded)
	if err != nil {
		return receipt, err
	}
	old, err := readPreparation(p.Preparation, false, proofs)
	if err != nil {
		return receipt, err
	}
	receipt.Buffers, receipt.Files, err = readNativeBuffers(filepath.Join(out, "native"), receipt.Contract)
	if err != nil {
		return receipt, err
	}
	receipt.Images, err = readCaptureImagesScoped(r, filepath.Join(out, "native/images-decode"), old.Images, proofs)
	if err != nil {
		return receipt, err
	}
	if numericBuild(p) {
		receipt.Identity, err = readNumericIdentityScoped(filepath.Join(out, "native"), r, receipt.Images, receipt.Guarded, proofs)
		if err != nil {
			return receipt, err
		}
		for _, identity := range receipt.Identity {
			for path, sha := range identity.Files {
				receipt.Files[path] = sha
			}
		}
	}
	if err = verifyGuardRuntimeScoped(r, b, p, true, proofs); err != nil {
		return receipt, err
	}
	receipt.Status = "native_captured_pending_supervisor"
	return receipt, nil
}

func readNativeBuffers(dir string, c Contract) ([]CapturedBuffer, map[string]string, error) {
	files := map[string]string{}
	if err := c.Validate(); err != nil {
		return nil, files, err
	}
	for _, name := range []string{"params.bin", "chunks-original.bin", "scores-original.f16", "events.jsonl", "buffers.tsv", "device-name.txt"} {
		if err := filePin(files, filepath.Join(dir, name)); err != nil {
			return nil, files, err
		}
	}
	params, err := os.ReadFile(filepath.Join(dir, "params.bin"))
	if err != nil {
		return nil, files, err
	}
	chunks, err := os.ReadFile(filepath.Join(dir, "chunks-original.bin"))
	if err != nil {
		return nil, files, err
	}
	if err = ValidateWire(params, chunks); err != nil {
		return nil, files, err
	}
	scores, err := os.ReadFile(filepath.Join(dir, "scores-original.f16"))
	if err != nil {
		return nil, files, err
	}
	if err = ValidateScores(scores, originalScoresSHA); err != nil {
		return nil, files, err
	}
	if err = readNativeEvents(dir, c); err != nil {
		return nil, files, err
	}
	return readNativeBufferRows(dir, c, files)
}

func readNativeEvents(dir string, c Contract) error {
	events, err := os.Open(filepath.Join(dir, "events.jsonl"))
	if err != nil {
		return err
	}
	defer events.Close()
	decoder := json.NewDecoder(events)
	for _, stage := range c.Stages {
		for _, state := range []string{"before-call", "captured"} {
			var e struct{ Stage, State, UTC string }
			if err = decoder.Decode(&e); err != nil {
				return err
			}
			if e.Stage != stage.Name || e.State != state {
				return fmt.Errorf("original four-call event order differs")
			}
			if _, err = time.Parse(time.RFC3339, e.UTC); err != nil {
				return err
			}
		}
	}
	var done struct{ Stage, State, UTC string }
	if err = decoder.Decode(&done); err != nil {
		return err
	}
	if done.Stage != "all" || done.State != "completed" {
		return fmt.Errorf("native capture did not complete")
	}
	if _, err = time.Parse(time.RFC3339, done.UTC); err != nil {
		return err
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF {
		return fmt.Errorf("extra native stage event")
	}
	return nil
}

func readNativeBufferRows(dir string, c Contract, files map[string]string) ([]CapturedBuffer, map[string]string, error) {
	manifest, err := os.Open(filepath.Join(dir, "buffers.tsv"))
	if err != nil {
		return nil, files, err
	}
	defer manifest.Close()
	scanner := bufio.NewScanner(manifest)
	result := []CapturedBuffer{}
	for _, stage := range c.Stages {
		for _, buffer := range c.Buffers {
			if !scanner.Scan() {
				return nil, files, fmt.Errorf("incomplete native buffer manifest")
			}
			row := strings.Split(scanner.Text(), "\t")
			expectedName := stage.Name + "-" + buffer.Name + ".bin"
			if len(row) != 5 || row[0] != stage.Name || row[1] != buffer.Name || row[4] != expectedName {
				return nil, files, fmt.Errorf("native buffer role/order/path differs")
			}
			n, e := strconv.ParseInt(row[2], 10, 64)
			if e != nil || n != buffer.Bytes {
				return nil, files, fmt.Errorf("native buffer extent differs")
			}
			file, e := artifactPin(filepath.Join(dir, expectedName))
			if e != nil {
				return nil, files, e
			}
			info, e := os.Stat(file.Path)
			if e != nil || info.Size() != n || file.SHA256 != row[3] {
				return nil, files, fmt.Errorf("native raw buffer hash/size differs")
			}
			files[file.Path] = file.SHA256
			observation := CapturedBuffer{Stage: stage.Name, Role: buffer.Name, Raw: file, Bytes: n, Compare: stage.Compare[buffer.Name]}
			f, e := os.Open(file.Path)
			if e != nil {
				return nil, files, e
			}
			h := sha256.New()
			for _, span := range observation.Compare {
				if span.Offset < 0 || span.Bytes < 0 || span.Offset > n || span.Bytes > n-span.Offset {
					f.Close()
					return nil, files, fmt.Errorf("prospective comparison span exceeds buffer")
				}
				if _, e = io.Copy(h, io.NewSectionReader(f, span.Offset, span.Bytes)); e != nil {
					f.Close()
					return nil, files, e
				}
				observation.ComparedBytes += span.Bytes
			}
			if e = f.Close(); e != nil {
				return nil, files, e
			}
			observation.ComparedSHA256 = hex.EncodeToString(h.Sum(nil))
			result = append(result, observation)
		}
	}
	if scanner.Scan() || scanner.Err() != nil {
		return nil, files, fmt.Errorf("extra/invalid native buffer rows")
	}
	return result, files, nil
}
