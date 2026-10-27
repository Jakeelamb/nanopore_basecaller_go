package decodergpu

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

type NumericContext struct {
	Pointer            uint64
	UID, DeviceOrdinal uint32
}
type NumericIdentity struct {
	Stage                               string
	PID                                 int
	DeviceOrdinal                       uint32
	UUID                                string
	Major, Minor, DriverAPI, RuntimeAPI int
	KernelVersion                       string
	Environment                         []string
	Contexts                            []NumericContext
	Files                               map[string]string `json:",omitempty"`
}

func validateNumericIdentity(row NumericIdentity, stage int, images ImageReport, reference LineageContext, kernelSHA string, environment []string, pid int) error {
	stages := []string{"inputs", "back-guide", "beam-search", "posterior", "decode"}
	if stage < 0 || stage >= len(stages) || row.Stage != stages[stage] || pid <= 0 || row.PID != pid || row.DeviceOrdinal != 0 || row.UUID != reference.UUID || len(row.UUID) != 32 || row.UUID == "00000000000000000000000000000000" || row.Major != 12 || row.Minor != 0 || row.DriverAPI != reference.DriverVersion || row.RuntimeAPI <= 0 || row.RuntimeAPI > 999999 || len(row.KernelVersion) == 0 || len(row.KernelVersion) > 16384 || digest([]byte(row.KernelVersion)) != kernelSHA {
		return fmt.Errorf("numeric contemporaneous device/kernel identity differs at stage %d", stage)
	}
	for _, c := range row.UUID {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return fmt.Errorf("numeric device UUID malformed")
		}
	}
	if !equalEnvironment(row.Environment, environment) {
		return fmt.Errorf("numeric complete real environment differs at stage %d", stage)
	}
	wanted := map[uint64]uint32{}
	allUID := map[uint64]uint32{}
	for _, l := range images.Launches {
		if l.Context == 0 || l.UID == 0 || l.UID > 1<<32-1 {
			return fmt.Errorf("numeric launch context absent")
		}
		if old, ok := allUID[l.Context]; ok && old != uint32(l.UID) {
			return fmt.Errorf("numeric context UID ambiguous")
		}
		allUID[l.Context] = uint32(l.UID)
		if l.Stage <= uint64(stage) {
			wanted[l.Context] = uint32(l.UID)
		}
	}
	for _, m := range images.Modules {
		if m.DeliveredStage <= uint64(stage) {
			uid := allUID[m.Context]
			if uid == 0 {
				return fmt.Errorf("numeric module context unbound")
			}
			wanted[m.Context] = uid
		}
	}
	if len(wanted) != len(row.Contexts) {
		return fmt.Errorf("numeric contemporaneous context count differs at stage %d", stage)
	}
	last := uint64(0)
	for _, c := range row.Contexts {
		if c.Pointer == 0 || c.Pointer <= last || c.UID == 0 || wanted[c.Pointer] != c.UID || c.DeviceOrdinal != row.DeviceOrdinal {
			return fmt.Errorf("numeric contemporaneous context/device binding differs at stage %d", stage)
		}
		last = c.Pointer
	}
	return nil
}

func readNumericIdentity(dir string, r RuntimePlan, images ImageReport, guarded supgpu.GuardedRun) ([]NumericIdentity, error) {
	return readNumericIdentityScoped(dir, r, images, guarded, nil)
}

func readNumericIdentityScoped(dir string, r RuntimePlan, images ImageReport, guarded supgpu.GuardedRun, proofs *numericProofs) ([]NumericIdentity, error) {
	proof, err := proofs.historical(r.NumericLineage)
	if err != nil {
		return nil, err
	}
	if err = verifyNumericConditionsScoped(r, false, proofs); err != nil {
		return nil, err
	}
	var transcript supgpu.GuardReceipt
	if err = readJSON(File{guarded.Transcript.Path, guarded.Transcript.SHA256}, &transcript); err != nil {
		return nil, err
	}
	envelope := append([]string{"SUPGPU_COMMAND_CWD=" + r.WorkingDirectory, "SUPGPU_GATE_FDS=3,4,5,6"}, numericEnvironment(r.Policy)...)
	body, err := lineageExecEnvironment(envelope)
	if err != nil {
		return nil, err
	}
	expectedReceipt := append(slices.Clone(envelope), fmt.Sprintf("SUPGPU_TARGET_PID=%d", transcript.PID))
	if !reflect.DeepEqual(guarded.Command.Environment, expectedReceipt) {
		return nil, fmt.Errorf("numeric command receipt annotation/body contract differs")
	}
	result := []NumericIdentity{}
	for i, stage := range []string{"inputs", "back-guide", "beam-search", "posterior", "decode"} {
		path := filepath.Join(dir, "identity-"+stage+".json")
		info, e := os.Stat(path)
		if e != nil {
			return nil, e
		}
		if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > 64*1024 {
			return nil, fmt.Errorf("numeric identity raw extent differs")
		}
		pin, e := artifactPin(path)
		if e != nil {
			return nil, e
		}
		var row NumericIdentity
		if e = readJSON(pin, &row); e != nil {
			return nil, e
		}
		if row.Files != nil {
			return nil, fmt.Errorf("numeric native identity contains consumer-only custody")
		}
		if e = validateNumericIdentity(row, i, images, proof.Context, r.KernelVersion.SHA256, body, transcript.PID); e != nil {
			return nil, e
		}
		if len(result) > 0 && row.RuntimeAPI != result[0].RuntimeAPI {
			return nil, fmt.Errorf("numeric runtime API changed between stages")
		}
		row.Files = map[string]string{pin.Path: pin.SHA256}
		result = append(result, row)
	}
	return result, nil
}
