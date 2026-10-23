//go:build cgo

package caller

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	calleroracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/caller"
)

func TestPinnedNativeFASTQMetadata(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE for extracted source FASTQ metadata differential; no inference")
	}
	output := os.Getenv("CALLER_METADATA_ARTIFACT_DIR")
	if output == "" {
		output = t.TempDir()
	}
	executable, err := calleroracle.Build(root, output)
	if err != nil {
		t.Fatal(err)
	}
	type metadataCase struct {
		called CalledRead
		input  pod5.Read
	}
	var cases []metadataCase
	for _, n := range []int{1, 59, 60, 61, 127, 1653} {
		for _, quality := range []byte{'!', '"', 'I', 'S', '~', 0} {
			q := make([]byte, n)
			for i := range q {
				q[i] = quality
				if quality == 0 {
					q[i] = byte(33 + (i*37+i*i)%94)
				}
			}
			for _, mode := range []string{"normal", "empty-run", "empty-model", "empty-flow"} {
				r := testRead()
				r.StartSample = 160723601
				r.AcquisitionStartTimeMS = 1718800934753
				r.ProtocolStartTimeMS = 1718800565543
				r.RunID = "different-protocol-run"
				r.FlowCellID = "flow-42"
				r.Channel = 2479
				spec := testSpec()
				switch mode {
				case "empty-run":
					r.RunID = ""
				case "empty-model":
					spec.Name = ""
				case "empty-flow":
					r.FlowCellID = ""
				}
				m, err := fastqMetadata(r, spec)
				if err != nil {
					t.Fatal(err)
				}
				cases = append(cases, metadataCase{CalledRead{ID: r.ID.String(), Sequence: strings.Repeat("A", n), Qstring: string(q), FASTQ: m}, r})
			}
		}
	}
	for _, ms := range []int64{-1, 0, 1, 999, 1000, 951782399999, 951782400000, math.MinInt64 / 1000000, math.MaxInt64 / 1000000} {
		r := testRead()
		r.AcquisitionStartTimeMS = ms
		r.ProtocolStartTimeMS = ms
		r.StartSample = 4
		m, err := fastqMetadata(r, testSpec())
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, metadataCase{CalledRead{ID: r.ID.String(), Sequence: "A", Qstring: "I", FASTQ: m}, r})
	}
	var input, goOutput bytes.Buffer
	hex := func(s string) string {
		if s == "" {
			return "-"
		}
		return fmt.Sprintf("%x", s)
	}
	for _, c := range cases {
		r := c.called
		fmt.Fprintf(&input, "%s %s %s %s %s %s %d %d %d %d %d %d\n", r.ID, hex(r.Sequence), hex(r.Qstring), hex(r.FASTQ.RunID), hex(r.FASTQ.ModelName), hex(r.FASTQ.FlowCellID), r.FASTQ.Channel, c.input.StartSample, c.input.SampleRate, c.input.AcquisitionStartTimeMS, r.FASTQ.ProtocolStartTimeMS, r.FASTQ.MeanQStart)
		if err := r.WriteFASTQ(&goOutput); err != nil {
			t.Fatal(err)
		}
	}
	for name, data := range map[string][]byte{"inputs.txt": input.Bytes(), "go.fastq": goOutput.Bytes()} {
		if err := os.WriteFile(filepath.Join(output, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	command := exec.Command(executable)
	command.Stdin = &input
	var stderr bytes.Buffer
	command.Stderr = &stderr
	started := time.Now().UTC()
	native, runErr := command.Output()
	exitCode := -1
	if command.ProcessState != nil {
		exitCode = command.ProcessState.ExitCode()
	}
	for name, data := range map[string][]byte{"native.fastq": native, "native.stderr": stderr.Bytes()} {
		if err := os.WriteFile(filepath.Join(output, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	var nativeBits []uint32
	scanner := bufio.NewScanner(bytes.NewReader(stderr.Bytes()))
	for scanner.Scan() {
		var bits uint32
		if _, err := fmt.Sscanf(scanner.Text(), "mean_bits=%d", &bits); err == nil {
			nativeBits = append(nativeBits, bits)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	bitsExact := len(nativeBits) == len(cases)
	var goBits bytes.Buffer
	for i, c := range cases {
		q := c.called.Qstring
		if len(q) > c.called.FASTQ.MeanQStart {
			q = q[c.called.FASTQ.MeanQStart:]
		}
		mean, err := decode.MeanQScore(q)
		if err != nil {
			t.Fatal(err)
		}
		bits := math.Float32bits(mean)
		fmt.Fprintf(&goBits, "mean_bits=%d\n", bits)
		if i >= len(nativeBits) || nativeBits[i] != bits {
			bitsExact = false
		}
	}
	if err := os.WriteFile(filepath.Join(output, "go-mean-bits.txt"), goBits.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	receipt := map[string]any{"started": started, "finished": time.Now().UTC(), "exit_code": exitCode, "cases": len(cases), "native_sha256": fmt.Sprintf("%x", sha256.Sum256(native)), "go_sha256": fmt.Sprintf("%x", sha256.Sum256(goOutput.Bytes())), "raw_fastq_exact": bytes.Equal(native, goOutput.Bytes()), "mean_float32_bits_exact": bitsExact, "source_commit": calleroracle.SourceCommit}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "run-receipt.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	if runErr != nil {
		t.Fatalf("native metadata oracle failed: %v\n%s", runErr, stderr.Bytes())
	}
	if !bytes.Equal(native, goOutput.Bytes()) {
		t.Fatalf("native raw FASTQ differs; evidence %s", output)
	}
	if !bitsExact {
		t.Fatalf("mean quality float32 bits differ; evidence %s", output)
	}
	t.Logf("%d source-oracle raw FASTQ cases exact; evidence %s", len(cases), output)
}

func TestNativeFASTQSourceMutationRejected(t *testing.T) {
	root := os.Getenv("DORADO_SOURCE")
	if root == "" {
		t.Skip("set DORADO_SOURCE for pinned-source mutation rejection")
	}
	if err := calleroracle.VerifySources(root); err != nil {
		t.Fatal(err)
	}
	copyRoot := t.TempDir()
	for name := range calleroracle.SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(copyRoot, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(name, "sequence_utils.cpp") {
			data = append(data, '\n')
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := calleroracle.VerifySources(copyRoot); err == nil {
		t.Fatal("changed source accepted")
	}
}
