package decodergpu

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This opt-in compiles no genuine CUDA/CUPTI header or library. Saved cubins
// are byte fixtures; the fake does not establish a driver transformation.
func TestLineageNativeFakeHost(t *testing.T) {
	if os.Getenv("D01_LINEAGE_FAKE_HOST") != "1" {
		t.Skip("explicit fake-header compiler opt-in")
	}
	out := os.Getenv("D01_LINEAGE_FAKE_ARTIFACTS")
	if !filepath.IsAbs(out) {
		t.Fatal("fresh absolute retained fake artifact path required")
	}
	if err := os.Mkdir(out, 0700); err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repo := filepath.Clean(filepath.Join(cwd, "../.."))
	binaryPin, compileCommands, sourcePins := lineageBuildFake(t, repo, out)
	binary := binaryPin.Path
	original := filepath.Join(repo, "artifacts/decoder-static-20261025-sm120a-001/decoder_lib.sm_120a.cubin")
	module := filepath.Join(repo, "artifacts/tensor/d01-native-capture-002/native/images-decode/module-0.cubin")
	originalPin, err := artifactPin(original)
	if err != nil || originalPin.SHA256 != "8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40" {
		t.Fatalf("original byte fixture changed: %v", err)
	}
	modulePin, err := artifactPin(module)
	if err != nil || modulePin.SHA256 != lineageOutputSHA {
		t.Fatalf("retained observed byte fixture changed: %v", err)
	}
	records := []map[string]any{}
	for _, kind := range []string{"positive", "lookup-resource", "copy-lifetime", "async-resource", "launch", "missing-resource", "cleanup-failure", "write-close-failure", "existing-output", "api-overflow", "module-overflow"} {
		t.Run(kind, func(t *testing.T) {
			dir := filepath.Join(out, kind)
			if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(dir, "native")
			if kind == "existing-output" {
				if err := os.Mkdir(target, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(target, "sentinel"), []byte("do not replace"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			destructor := filepath.Join(dir, "destructor.txt")
			argv := []string{binary, "--decoder-load-lineage", "sm_120a", "four-original-functions", original, target}
			env := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "D01_LINEAGE_FAKE_CASE=" + kind, "D01_LINEAGE_FAKE_MODULE=" + module, "D01_LINEAGE_FAKE_OUT=" + target, "D01_LINEAGE_FAKE_DESTRUCTOR=" + destructor}
			started := time.Now().UTC().Format(time.RFC3339Nano)
			exit, phases, stdout, stderr := lineageFakeRun(t, argv, env)
			finished := time.Now().UTC().Format(time.RFC3339Nano)
			os.WriteFile(filepath.Join(dir, "stdout"), stdout, 0600)
			os.WriteFile(filepath.Join(dir, "stderr"), stderr, 0600)
			expectedExit := 0
			if kind == "async-resource" || kind == "launch" || kind == "write-close-failure" || kind == "existing-output" || kind == "api-overflow" || kind == "module-overflow" {
				expectedExit = 1
			}
			if kind == "cleanup-failure" {
				expectedExit = 2
			}
			if exit != expectedExit {
				t.Fatalf("fake %s exit=%d want%d: %s", kind, exit, expectedExit, stderr)
			}
			count := 5
			if kind == "cleanup-failure" {
				count = 4
			}
			if kind == "existing-output" {
				count = 0
			}
			if len(phases) != count {
				t.Fatalf("fake phase count %v want%d", phases, count)
			}
			record := map[string]any{"case": kind, "command": argv, "environment": env, "started_utc": started, "finished_utc": finished, "exit_code": exit, "phases": phases, "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr), "synthetic_no_GPU_or_guard_qualification": true}
			if kind == "write-close-failure" {
				if !strings.Contains(string(stderr), "observation close failed") {
					t.Fatal("buffered close failure did not fail")
				}
				link, e := os.Readlink(filepath.Join(target, "observation.json"))
				if e != nil || link != "/dev/full" {
					t.Fatal("effective close-failure fixture missing")
				}
			} else if kind == "existing-output" {
				body, e := os.ReadFile(filepath.Join(target, "sentinel"))
				if e != nil || string(body) != "do not replace" {
					t.Fatal("existing output was overwritten")
				}
				if _, e = os.Stat(filepath.Join(target, "observation.json")); !os.IsNotExist(e) {
					t.Fatal("existing output gained observation")
				}
			} else {
				observationFile, e := artifactPin(filepath.Join(target, "observation.json"))
				if e != nil {
					t.Fatal(e)
				}
				var observed LineageObservation
				if e = readJSON(observationFile, &observed); e != nil {
					t.Fatal(e)
				}
				record["observation"] = observationFile
				c, _ := lineageFixture()
				c.Input.Image = originalPin
				c.Policy.Images[1].Image = originalPin
				c.Environment = []string{"SUPGPU_COMMAND_CWD=" + repo, "SUPGPU_GATE_FDS=3,4,5,6"}
				for _, v := range observed.Context.Environment {
					if !strings.HasPrefix(v, "SUPGPU_TARGET_PID=") {
						c.Environment = append(c.Environment, v)
					}
				}
				_, binding, e := ReadLineageObservation(observationFile, c)
				positive := kind == "positive" || kind == "lookup-resource" || kind == "copy-lifetime"
				if positive && e != nil {
					t.Fatal(e)
				}
				if !positive && e == nil {
					t.Fatal("failed synthetic callback observation accepted")
				}
				record["synthetic_binding"] = binding
				if e != nil {
					record["rejection"] = e.Error()
				}
				if kind == "copy-lifetime" {
					if e = observed.InputCopy.verify(); e != nil || observed.InputCopy.SHA256 != originalPin.SHA256 || len(observed.Modules) != 1 || observed.Modules[0].Payload.SHA256 != modulePin.SHA256 {
						t.Fatal("callback-lifetime copy was not retained")
					}
				}
			}
			_, destructorErr := os.Stat(destructor)
			if kind == "cleanup-failure" && !os.IsNotExist(destructorErr) {
				t.Fatal("failed final cleanup ran CPU owner destructors")
			}
			records = append(records, record)
		})
	}
	receipt := map[string]any{"scope": "fake CUDA/CUPTI only; synthetic retained byte fixtures; no native lineage acceptance", "compiler_commands": compileCommands, "binary": binaryPin, "source_pins": sourcePins, "original_fixture": originalPin, "module_fixture": modulePin, "cases": records}
	if err = writeJSON(filepath.Join(out, "receipt.json"), receipt); err != nil {
		t.Fatal(err)
	}
}
func slicesClone(s []string) []string { return append([]string{}, s...) }

func lineageFakeRun(t *testing.T, args, environment []string) (int, []string, []byte, []byte) {
	t.Helper()
	requestR, requestW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer requestR.Close()
	replyR, replyW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer replyW.Close()
	null, err := os.OpenFile(os.DevNull, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer null.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Env = environment
	cmd.ExtraFiles = []*os.File{null, null, requestW, replyR}
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	requestW.Close()
	replyR.Close()
	type gateResult struct {
		names []string
		err   error
	}
	done := make(chan gateResult, 1)
	go func() {
		result := gateResult{}
		scan := bufio.NewScanner(requestR)
		for scan.Scan() {
			var request struct {
				Schema, Kind, Path string
				Sequence, PID      int
			}
			if e := json.Unmarshal(scan.Bytes(), &request); e != nil {
				result.err = e
				break
			}
			if request.Schema != "nanopore-supgpu-loader-gate/v1" || request.Kind != "phase" || request.PID != cmd.Process.Pid || request.Sequence != len(result.names)+1 || len(result.names) >= 5 || request.Path != lineagePhases()[len(result.names)] {
				result.err = fmt.Errorf("fake phase protocol differs")
				break
			}
			result.names = append(result.names, request.Path)
			if _, e := fmt.Fprintf(replyW, "allow phase %d\n", request.Sequence); e != nil {
				result.err = e
				break
			}
		}
		if e := scan.Err(); result.err == nil {
			result.err = e
		}
		done <- result
	}()
	err = cmd.Wait()
	replyW.Close()
	result := <-done
	if result.err != nil {
		t.Fatal(result.err)
	}
	if ctx.Err() != nil {
		t.Fatal(ctx.Err())
	}
	exit := 0
	if err != nil {
		var ok bool
		var e *exec.ExitError
		e, ok = err.(*exec.ExitError)
		if !ok {
			t.Fatal(err)
		}
		exit = e.ExitCode()
	}
	return exit, result.names, []byte(stdout.String()), []byte(stderr.String())
}
