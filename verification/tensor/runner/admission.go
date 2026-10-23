package runner

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

const HostLock = "/home/jake/Projects/nanopore_basecaller/results/stock/host.lock"

var ErrAdmissionBlocked = errors.New("admission_blocked")

func Acquire(path string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%w: open existing lock: %v", ErrAdmissionBlocked, err)
	}
	if err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("%w: %v", ErrAdmissionBlocked, err)
	}
	return f, nil
}
func ExitCode(err error) int {
	if errors.Is(err, ErrAdmissionBlocked) {
		return 75
	}
	return 1
}
func CheckRoot() error {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		return err
	}
	const module = "module github.com/Jakeelamb/nanopore_basecaller_go\n"
	if len(raw) < len(module) || string(raw[:len(module)]) != module {
		return errors.New("run from the Go port repository root")
	}
	for _, path := range []string{"internal/tensor/abi.h", "verification/tensor/backend.cpp"} {
		if _, err := os.Stat(path); err != nil {
			return err
		}
	}
	return nil
}
