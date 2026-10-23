package runner

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestAdmissionExistingLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host.lock")
	if _, err := Acquire(path); !errors.Is(err, ErrAdmissionBlocked) {
		t.Fatalf("missing lock: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("Acquire created lock")
	}
	if err := os.WriteFile(path, []byte("preserved"), 0600); err != nil {
		t.Fatal(err)
	}
	f, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	if other, err := Acquire(path); !errors.Is(err, ErrAdmissionBlocked) {
		if other != nil {
			other.Close()
		}
		t.Fatalf("held lock: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	other, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	other.Close()
	b, err := os.ReadFile(path)
	if err != nil || string(b) != "preserved" {
		t.Fatal("lock changed")
	}
	if ExitCode(ErrAdmissionBlocked) != 75 {
		t.Fatal("wrong admission exit")
	}
}
