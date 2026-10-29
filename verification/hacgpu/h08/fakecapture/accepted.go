package fakecapture

import (
	"fmt"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
)

// ReadAccepted checks a retained completed capture without executing anything.
// Root still separately proves that its outer unit/cgroup is strictly dead.
func ReadAccepted(request, summary Pin) (Summary, error) {
	q, e := readRequest(request, false)
	if e != nil {
		return Summary{}, e
	}
	if _, e = acceptedPlan(q); e != nil {
		return Summary{}, e
	}
	var r Summary
	if e = readExact(summary, &r); e != nil {
		return r, e
	}
	if summary.Path != filepath.Join(q.Out, "summary.json") || r.Schema != "nanopore-h08-fake-capture/v1" || r.Status != acceptedStatus || r.Error != "" || !r.RuntimeHeld || r.Request != request || r.Cwd != q.Root || r.Session.Path != filepath.Join(q.Out, "session.json") || r.Proof.Path != filepath.Join(q.Out, "fake-proof.json") {
		return r, fmt.Errorf("accepted capture binding differs")
	}
	var s Session
	if e = readExact(r.Session, &s); e != nil {
		return r, e
	}
	if s.Schema != sessionSchema || s.Request != request || s.Logger != q.Logger || s.Compiler != q.Compiler || s.Target != q.Target || s.Root != q.Root || s.Out != q.Out || s.Parent != r.Process || r.TargetProcess.PPID != r.Process.PID || r.TargetProcess.Cgroup != r.Process.Cgroup {
		return r, fmt.Errorf("retained private session differs")
	}
	if _, e = sessionPlan(q, s); e != nil {
		return r, e
	}
	if !slices.Equal(r.Command.Args, targetArgs(q)) || !slices.Equal(r.Command.Environment, targetEnvironment(q, r.Session)) {
		return r, fmt.Errorf("retained target invocation differs")
	}
	if e = readLogged(filepath.Join(q.Out, "logs"), q.Root, q.Target, LoggedCommand{Command: r.Command, Process: r.TargetProcess, Observation: r.TargetObservation, Receipts: r.TargetReceipts}); e != nil {
		return r, e
	}
	compiles, pins, e := collect(q, r.Session, r.TargetProcess)
	if e != nil {
		return r, e
	}
	if len(compiles) != 39 || !slices.Equal(pins, r.Compiles) {
		return r, fmt.Errorf("retained leaf inventory differs")
	}
	proof, e := h08.ReadFakeLoaderProof(r.Proof, q.Target, q.ControllerBuild, q.ControllerBuildSelection)
	if e != nil {
		return r, e
	}
	if proof.Logger != q.Logger || proof.Compiler != q.Compiler || proof.Inputs[request.Path] != request.SHA256 || proof.Inputs[r.Session.Path] != r.Session.SHA256 || !reflect.DeepEqual(proof.Command, r.Command) || !reflect.DeepEqual(proof.Compiles, compiles) {
		return r, fmt.Errorf("proof/capture custody differs")
	}
	return r, nil
}
