package h07

import (
	"encoding/json"
	"fmt"
	"maps"
)

type ControllerBuild struct {
	Schema, Status, Output, OutputSHA256                                         string
	SourceSHA256, AfterSourceSHA256, SourceSnapshots, SnapshotSHA256, ToolSHA256 map[string]string
}

// ReadControllerBuild authenticates the actual Go build and immutable source
// snapshots. It does not compare producer source to a later consumer checkout.
func ReadControllerBuild(path, pin, executable, executableSHA string) (r ControllerBuild, err error) {
	raw, err := readPinned(path, pin)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(raw, &r); err != nil {
		return r, err
	}
	if r.Schema != "nanopore-hac-output-controller-build/v1" || r.Status != "built" || r.Output != executable || r.OutputSHA256 != executableSHA ||
		len(r.SourceSHA256) == 0 || len(r.SourceSnapshots) != len(r.SourceSHA256) || len(r.SnapshotSHA256) != len(r.SourceSHA256) || len(r.ToolSHA256) < 5 || !maps.Equal(r.SourceSHA256, r.AfterSourceSHA256) {
		return r, fmt.Errorf("controller build lacks exact source, tool, or executable custody")
	}
	for original, pin := range r.SourceSHA256 {
		snapshot, ok := r.SourceSnapshots[original]
		if !ok || r.SnapshotSHA256[snapshot] != pin {
			return r, fmt.Errorf("producer source snapshot differs: %s", original)
		}
	}
	if err = checkPins(r.SnapshotSHA256); err != nil {
		return r, err
	}
	if err = checkPins(r.ToolSHA256); err != nil {
		return r, err
	}
	if err = requirePin(executable, executableSHA); err != nil {
		return r, err
	}
	return r, nil
}
