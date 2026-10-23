package koiquant

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type ReplayReceipt struct {
	Schema, Status, Started, Finished, Error string
	Raw, Imported, TestBinary                Pin
	ConsumerSources                          map[string]string
	RequiredTests                            []string
	Commands                                 []runner.CommandReceipt
}

// Replay imports retained output and runs pure host consumer checks. It cannot invoke the native producer.
func Replay(root, rawPath, out string) (m ReplayReceipt, err error) {
	root, err = filepath.Abs(root)
	if err != nil {
		return m, err
	}
	rawPath, err = filepath.Abs(rawPath)
	if err != nil {
		return m, err
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return m, err
	}
	if _, err = Import(root, rawPath, out); err != nil {
		return m, err
	}
	m = ReplayReceipt{Schema: "nanopore-sup-qkv-i8/replay-v1", Status: "replay_failed", Started: clock(), RequiredTests: []string{"TestCaptureMetadataBridge", "TestCapturedRawRejection"}, Commands: []runner.CommandReceipt{}}
	defer func() {
		m.Finished = clock()
		if err != nil {
			m.Error = err.Error()
		}
		writeErr := writeJSON(filepath.Join(out, "replay-manifest.json"), m)
		if err == nil {
			err = writeErr
		}
	}()
	m.Raw, err = filePin(rawPath)
	if err != nil {
		return m, err
	}
	m.Imported, err = filePin(filepath.Join(out, "manifest.json"))
	if err != nil {
		return m, err
	}
	m.ConsumerSources, err = currentSources(root)
	if err != nil {
		return m, err
	}
	env := []string{"CGO_ENABLED=0", "NANOPORE_KOIQUANT_REPLAY=" + rawPath}
	binary := filepath.Join(out, "consumer.test")
	r, err := runner.Execute(out, "compile-consumer", env, "go", "test", "-c", "-o", binary, "./verification/koiquant")
	m.Commands = append(m.Commands, r)
	if err != nil {
		return m, err
	}
	if err = validateReceipt(r, false); err != nil {
		return m, err
	}
	m.TestBinary, err = filePin(binary)
	if err != nil {
		return m, err
	}
	r, err = runner.Execute(out, "consumer", env, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant", binary, "-test.v=test2json", "-test.count=1", "-test.run", "^(TestCaptureMetadataBridge|TestCapturedRawRejection)$")
	m.Commands = append(m.Commands, r)
	if err != nil {
		return m, err
	}
	if err = validateReceipt(r, false); err != nil {
		return m, err
	}
	data, err := os.ReadFile(r.Stdout)
	if err != nil {
		return m, err
	}
	if err = RequireTestPass(data, m.RequiredTests); err != nil {
		return m, err
	}
	if err = runner.RequireGoTests(r.Stdout, "github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant", m.RequiredTests); err != nil {
		return m, err
	}
	if err = validatePin(m.Raw); err != nil {
		return m, err
	}
	if err = validatePin(m.Imported); err != nil {
		return m, err
	}
	if err = validatePin(m.TestBinary); err != nil {
		return m, err
	}
	if err = checkSources(root, m.ConsumerSources); err != nil {
		return m, err
	}
	if len(m.Commands) != 2 {
		return m, fmt.Errorf("unexpected replay commands")
	}
	m.Status = "pure_host_replay_passed"
	return m, nil
}
