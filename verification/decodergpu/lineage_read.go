package decodergpu

import (
	"fmt"
	"os"
	"path/filepath"
)

// ReadLineageObservation rehashes callback-lifetime copies, without executing
// or importing the historical numerical capture.
func ReadLineageObservation(anchor File, c LineageContract) (LineageObservation, LineageBinding, error) {
	var observation LineageObservation
	var binding LineageBinding
	if err := readJSON(anchor, &observation); err != nil {
		return observation, binding, err
	}
	dir := filepath.Dir(anchor.Path)
	if anchor.Path != filepath.Join(dir, "observation.json") || observation.InputCopy.Path != filepath.Join(dir, "load-input.bin") {
		return observation, binding, fmt.Errorf("lineage observation/input copy path differs")
	}
	if err := observation.InputCopy.verify(); err != nil {
		return observation, binding, err
	}
	info, err := os.Stat(observation.InputCopy.Path)
	if err != nil || info.Size() != 261192 {
		return observation, binding, fmt.Errorf("lineage copied API input extent differs: %v", err)
	}
	for i, module := range observation.Modules {
		if module.Payload.Path != filepath.Join(dir, fmt.Sprintf("module-%d.cubin", i)) {
			return observation, binding, fmt.Errorf("lineage module payload path differs")
		}
		if err := module.Payload.verify(); err != nil {
			return observation, binding, err
		}
		info, err := os.Stat(module.Payload.Path)
		if err != nil || info.Size() != int64(module.Bytes) || module.Bytes > 1<<20 {
			return observation, binding, fmt.Errorf("lineage output copy extent differs: %v", err)
		}
	}
	binding, err = BindLineage(c, observation)
	return observation, binding, err
}
