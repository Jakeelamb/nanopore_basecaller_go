package fakecapture

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type CommandPins struct {
	Launch, Born, Process, Command Pin
}

// ChildBirth records the fact known immediately after successful Start. It
// carries no claim about live identity, executable bytes or kernel start tick.
type ChildBirth struct {
	PID         int
	ObservedUTC string
}

type ProcessObservation struct {
	Born          ChildBirth
	Process       Process
	Authenticated bool
	Error         string
}

type LoggedCommand struct {
	Command     runner.CommandReceipt
	Process     Process
	Observation ProcessObservation
	Receipts    CommandPins
}

type commandLaunch struct {
	Cwd        string
	Executable Pin
	Command    runner.CommandReceipt
}

func launchCommand(c runner.CommandReceipt) runner.CommandReceipt {
	c.Finished, c.Exit, c.StdoutSHA, c.StderrSHA = "", -1, "", ""
	return c
}

// readLogged authenticates only this package's finite retained receipt shape.
// Expected argv/environment/process come from the enclosing target or compile
// record, whose selection/session binding is checked by its caller.
func readLogged(dir, cwd string, executable Pin, want LoggedCommand) error {
	p := want.Receipts
	for name, pin := range map[string]Pin{"launch": p.Launch, "born": p.Born, "process": p.Process, "command": p.Command} {
		if pin.Path != filepath.Join(dir, name+".json") {
			return fmt.Errorf("%s receipt omitted/rebound: %s", name, dir)
		}
	}
	var launch commandLaunch
	if e := readExact(p.Launch, &launch); e != nil {
		return e
	}
	if !reflect.DeepEqual(launch, commandLaunch{cwd, executable, launchCommand(want.Command)}) {
		return fmt.Errorf("launch receipt differs: %s", dir)
	}
	var born ChildBirth
	if e := readExact(p.Born, &born); e != nil {
		return e
	}
	var process ProcessObservation
	if e := readExact(p.Process, &process); e != nil {
		return e
	}
	if born != want.Observation.Born || process != want.Observation || process.Born != born || process.Process != want.Process || !process.Authenticated || process.Error != "" || born.PID <= 1 || born.PID != want.Process.PID || want.Process.PPID <= 1 || want.Process.Start == "" || want.Process.Cgroup == "" {
		return fmt.Errorf("process/birth receipt differs or unqualified: %s", dir)
	}
	var command runner.CommandReceipt
	if e := readExact(p.Command, &command); e != nil {
		return e
	}
	if !reflect.DeepEqual(command, want.Command) || len(command.Args) == 0 || command.Args[0] != executable.Path || command.Exit != 0 || command.Stdout != filepath.Join(dir, "stdout") || command.Stderr != filepath.Join(dir, "stderr") {
		return fmt.Errorf("command receipt differs: %s", dir)
	}
	start, a := time.Parse(time.RFC3339Nano, command.Started)
	birth, b := time.Parse(time.RFC3339Nano, born.ObservedUTC)
	finish, c := time.Parse(time.RFC3339Nano, command.Finished)
	if a != nil || b != nil || c != nil || birth.Before(start) || finish.Before(birth) || !strings.HasSuffix(command.Started, "Z") || !strings.HasSuffix(born.ObservedUTC, "Z") || !strings.HasSuffix(command.Finished, "Z") {
		return fmt.Errorf("command/birth UTC clocks differ: %s", dir)
	}
	for _, pin := range []Pin{{Path: command.Stdout, SHA256: command.StdoutSHA}, {Path: command.Stderr, SHA256: command.StderrSHA}} {
		if _, e := pinBytes(pin); e != nil {
			return e
		}
	}
	return nil
}
