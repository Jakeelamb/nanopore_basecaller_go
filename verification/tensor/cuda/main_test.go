package main

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/cudafixture"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestBatchCaptureCommandCountAndRawComparisons(t *testing.T) {
	dir := t.TempDir()
	buildDir := filepath.Join(dir, "build")
	if err := os.Mkdir(buildDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := cudafixture.Generate(filepath.Join(buildDir, "cuda-inputs")); err != nil {
		t.Fatal(err)
	}
	buildInputs := map[string]string{}
	err := filepath.WalkDir(buildDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		buildInputs[path], err = runner.SHA(path)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	m := evidence{Inputs: map[string]string{}}
	pin := func(path, want string) error {
		got, err := runner.SHA(path)
		if err != nil {
			return err
		}
		if want != "" && got != want {
			t.Fatalf("pin differs %s", path)
		}
		m.Inputs[path] = got
		return nil
	}
	commands := 0
	execute := func(label string, env []string, args ...string) (runner.CommandReceipt, error) {
		commands++
		if len(args) != 5 || args[1] != "--batch" {
			t.Fatalf("not a batch: %v", args)
		}
		f, err := os.Open(args[2])
		if err != nil {
			return runner.CommandReceipt{}, err
		}
		defer f.Close()
		reader := csv.NewReader(f)
		reader.Comma = ' '
		reader.FieldsPerRecord = 15
		rows, err := reader.ReadAll()
		if err != nil {
			return runner.CommandReceipt{}, err
		}
		want := 37
		if label == "mutant-f32" {
			want = 32
		}
		if len(rows) != want {
			t.Fatalf("batch cases %d want %d", len(rows), want)
		}
		for _, row := range rows {
			for i, role := range []string{"a", "b", "c"} {
				input := row[2+i]
				if role == "c" && row[5] == "1" {
					input = row[2]
				}
				raw, err := os.ReadFile(input)
				if err != nil {
					return runner.CommandReceipt{}, err
				}
				if role == "c" && label == "mutant-f32" {
					off, err := strconv.Atoi(row[14])
					if err != nil {
						return runner.CommandReceipt{}, err
					}
					raw[off*2] ^= 1
				}
				if err = os.WriteFile(row[1]+"-"+role+".u16le", raw, 0644); err != nil {
					return runner.CommandReceipt{}, err
				}
			}
		}
		return runner.CommandReceipt{}, nil
	}
	if err = captureBatches(&m, dir, filepath.Join(buildDir, "lib.so"), "oracle", 0, buildInputs, nil, pin, execute, func(string) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if commands != 3 || len(m.Cases) != 37 || m.MutantDifferences != 32 {
		t.Fatalf("commands=%d cases=%d synthetic differences=%d", commands, len(m.Cases), m.MutantDifferences)
	}
}

func TestConsumerIdentityIsCurrentAndStable(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(cwd)
	c, err := currentConsumer(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if c.Head == "" || c.Sources["internal/tensor/cuda.go"] == "" {
		t.Fatal("missing current consumer pins")
	}
	if err = consumerStable(c); err != nil {
		t.Fatal(err)
	}
	c.Sources["go.mod"] = "changed"
	if err = consumerStable(c); err == nil {
		t.Fatal("changed consumer accepted")
	}
	c, err = currentConsumer(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(c.TestBinary, []byte("compiled consumer"), 0600); err != nil {
		t.Fatal(err)
	}
	c.TestSHA, err = runner.SHA(c.TestBinary)
	if err != nil {
		t.Fatal(err)
	}
	if err = consumerStable(c); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(c.TestBinary, []byte("different consumer"), 0600)
	if err = consumerStable(c); err == nil {
		t.Fatal("changed executable accepted")
	}
}
func TestCaptureIdentityAndConsumerModes(t *testing.T) {
	prior := evidence{ProducerIdentity: "driver=13040 runtime=13040 cublas=130400 ordinal=0 uuid=abc stream_matches=1 pointer_mode=0 math_mode=0 atomics_mode=1", DeviceIdentity: "0, GPU-abc, 615.78.08", Ordinal: 0, Environment: map[string]string{"CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES.present": "false", "CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_DEVICE_ORDER.present": "true"}}
	current := prior
	current.ProducerIdentity = ""
	if err := matchCaptureIdentity(prior, current); err != nil {
		t.Fatal(err)
	}
	current.DeviceIdentity = "0, GPU-other, 615.78.08"
	if matchCaptureIdentity(prior, current) == nil {
		t.Fatal("different GPU accepted")
	}
	current = prior
	current.Environment = map[string]string{"CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES.present": "true"}
	if matchCaptureIdentity(prior, current) == nil {
		t.Fatal("unset versus empty visible-device environment accepted")
	}
	current = prior
	current.Environment = map[string]string{}
	for key, value := range prior.Environment {
		current.Environment[key] = value
	}
	current.Environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	if matchCaptureIdentity(prior, current) == nil {
		t.Fatal("CUDA_DEVICE_ORDER remapping accepted")
	}
	path := filepath.Join(t.TempDir(), "consumer.json")
	write := func(mode string) {
		t.Helper()
		raw := `{"Output":"nt_cuda_gemm_f16 driver=13040 runtime=13040 cublas=130400 ordinal=0 uuid=abc stream_matches=1 pointer_mode=0 math_mode=` + mode + ` atomics_mode=1\n"}`
		if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("0")
	if err := matchConsumerModes(path, prior.ProducerIdentity, 1); err != nil {
		t.Fatal(err)
	}
	write("1")
	if matchConsumerModes(path, prior.ProducerIdentity, 1) == nil {
		t.Fatal("different cuBLAS mode accepted")
	}
	os.WriteFile(path, []byte(`{"Action":"pass"}`), 0600)
	if matchConsumerModes(path, prior.ProducerIdentity, 1) == nil {
		t.Fatal("missing cuBLAS diagnostics accepted")
	}
}
