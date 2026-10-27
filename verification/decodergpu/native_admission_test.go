package decodergpu

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestCompilerBusyDefersBeforePlanRead(t *testing.T) {
	if os.Getenv("D01_HOST_LOCK_REGRESSION") != "1" {
		t.Skip("set D01_HOST_LOCK_REGRESSION=1 for the canonical read-only lock regression")
	}
	lock, err := os.Open(runner.HostLock)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil && !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EAGAIN) {
		t.Fatal(err)
	}
	dir := t.TempDir()
	out := filepath.Join(dir, "must-not-exist")
	result, err := RunCompilerPhase(CompilerPhaseOptions{
		Admitted: true, Phase: "query", Out: out,
		Plan: File{Path: filepath.Join(dir, "nonexistent-plan.json"), SHA256: strings.Repeat("0", 64)},
	})
	if !errors.Is(err, runner.ErrAdmissionBlocked) || runner.ExitCode(err) != 75 {
		t.Fatalf("busy compiler must defer before plan read: %v", err)
	}
	if result.Directory != "" || len(result.Commands) != 0 {
		t.Fatalf("busy compiler created phase/commands: %+v", result)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("busy compiler created output attempt: %v", err)
	}
}

func TestCaptureBusyDefersBeforeProof(t *testing.T) {
	if os.Getenv("D01_HOST_LOCK_REGRESSION") != "1" {
		t.Skip("set D01_HOST_LOCK_REGRESSION=1 for the canonical read-only lock regression")
	}
	lock, err := os.Open(runner.HostLock)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil && !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EAGAIN) {
		t.Fatal(err)
	}
	dir := t.TempDir()
	missing := func(name string) File { return File{Path: filepath.Join(dir, name), SHA256: strings.Repeat("0", 64)} }
	out := filepath.Join(dir, "must-not-exist")
	receipt, err := Capture(CaptureOptions{Admitted: true, Out: out,
		Build: missing("nonexistent-build.json"), Runtime: missing("nonexistent-runtime.json"), Consumer: missing("nonexistent-consumer.json")})
	if !errors.Is(err, runner.ErrAdmissionBlocked) || runner.ExitCode(err) != 75 {
		t.Fatalf("busy capture must defer before proof read: %v", err)
	}
	if receipt.Schema != "" || len(receipt.Guarded.Command.Args) != 0 {
		t.Fatalf("busy capture started receipt/child: %+v", receipt)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("busy capture created output: %v", err)
	}
}

func TestNumericWatchBudgetPreservesLegacy(t *testing.T) {
	r := RuntimePlan{WorkingDirectory: "/source", CaptureDirectory: "/capture", Consumer: File{Path: "/current/consumer"}, ConsumerPreparation: File{Path: "/current/preparation.json"}}
	b := CompilerPhase{Output: File{Path: "/immutable/oracle", SHA256: "native"}}
	runtime := File{Path: "/current/runtime.json", SHA256: "runtime"}
	p := ClosedBuildPlan{Schema: "nanopore-decoder-closed-build/v2", Preparation: File{Path: "/immutable/preparation.json"}}
	legacy := expectedWatchPlan(r, runtime, b, p)
	t.Run("legacy240-native180", func(t *testing.T) {
		if legacy.Limit != 240 || len(legacy.Roles) != 1 || legacy.Roles[0].Limit != 180 {
			t.Fatalf("legacy orchestration changed: %+v", legacy)
		}
	})
	t.Run("numeric540-native180", func(t *testing.T) {
		p.Schema = "nanopore-decoder-closed-build/v3"
		numeric := expectedWatchPlan(r, runtime, b, p)
		raw, err := json.Marshal(numeric)
		if err != nil {
			t.Fatal(err)
		}
		var observed WatchPlan
		if err = json.Unmarshal(raw, &observed); err != nil {
			t.Fatal(err)
		}
		want := legacy
		want.Limit = 540
		if !reflect.DeepEqual(observed, want) {
			t.Fatalf("numeric proof budget must preserve native180 and exact producer command: got %+v want %+v", observed, want)
		}
	})
}
