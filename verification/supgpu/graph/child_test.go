package graph

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

func TestChildRejectsInvalidContextBeforePhaseOrInput(t *testing.T) {
	if _, e := RunGuardedChild(nil, nativeproof.Pin{}); e == nil {
		t.Fatal("nil context accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := RunGuardedChild(ctx, nativeproof.Pin{Path: "/must/not/open"}); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}

// The helper runs in an isolated test process so its real inherited FD5/6
// cannot overwrite descriptors owned by the test runner. Malformed admission
// must close the pipes without sending even the first phase request.
func TestChildMalformedRequestSendsNoPhase(t *testing.T) {
	if os.Getenv("SUP_GRAPH_PHASE_HELPER") == "1" {
		if _, e := RunGuardedChild(context.Background(), nativeproof.Pin{}); e == nil {
			t.Fatal("malformed request accepted")
		}
		return
	}
	requestRead, requestWrite, e := os.Pipe()
	if e != nil {
		t.Fatal(e)
	}
	defer requestRead.Close()
	defer requestWrite.Close()
	approvalRead, approvalWrite, e := os.Pipe()
	if e != nil {
		t.Fatal(e)
	}
	defer approvalRead.Close()
	defer approvalWrite.Close()
	spareRead, spareWrite, e := os.Pipe()
	if e != nil {
		t.Fatal(e)
	}
	defer spareRead.Close()
	defer spareWrite.Close()
	executable, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestChildMalformedRequestSendsNoPhase$")
	cmd.Env = append(os.Environ(), "SUP_GRAPH_PHASE_HELPER=1")
	cmd.ExtraFiles = []*os.File{spareRead, spareWrite, requestWrite, approvalRead}
	if output, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("child boundary: %v: %s", e, output)
	}
	if e = requestWrite.Close(); e != nil {
		t.Fatal(e)
	}
	body, e := io.ReadAll(io.LimitReader(requestRead, 1024))
	if e != nil || len(body) != 0 {
		t.Fatalf("malformed request emitted phase: %q: %v", body, e)
	}
}
func TestNumericalRejectsStatusOnlySummary(t *testing.T) {
	if e := validateComparisons(Witness{}, Receipt{Status: "numerical_equal", Events: 191, Comparisons: make([]Comparison, 733)}); e == nil {
		t.Fatal("status/totals without authenticated records accepted")
	}
}
