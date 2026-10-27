package cgobuild

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func TestFixedGoBuildBudgetKeepsIndividualCommandsBounded(t *testing.T) {
	p := targetFixture(t)
	for _, role := range []BuildRole{H08Command, CPU13Tests} {
		t.Run(string(role), func(t *testing.T) {
			q := p
			q.Role, q.Options.Role = role, role
			if role == H08Command {
				q.CPU13 = nil
			}
			q.BuildArgv = plannedBuildArgs(q)
			budget, e := fixedGoBuildBudget(q, Session{Phase: "build"})
			if e != nil || budget != goBuildOrchestration {
				t.Fatalf("fixed orchestration selection: %v, %v", budget, e)
			}
			ctx, cancel, e := budget.context(context.Background())
			if e != nil {
				t.Fatal(e)
			}
			defer cancel()
			deadline, ok := ctx.Deadline()
			if left := time.Until(deadline); !ok || left < 1799*time.Second || left > 1800*time.Second {
				t.Fatalf("Go orchestration deadline: %v, %v", left, ok)
			}
		})
	}
	ctx, cancel, e := individualProgram.context(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	defer cancel()
	deadline, ok := ctx.Deadline()
	if left := time.Until(deadline); !ok || left < 179*time.Second || left > 180*time.Second {
		t.Fatalf("individual Go/compiler/link command deadline: %v, %v", left, ok)
	}
	parent, end := context.WithTimeout(context.Background(), time.Second)
	defer end()
	child, stop, e := goBuildOrchestration.context(parent)
	if e != nil {
		t.Fatal(e)
	}
	defer stop()
	want, _ := parent.Deadline()
	got, _ := child.Deadline()
	if !got.Equal(want) {
		t.Fatal("orchestration budget extended its parent deadline")
	}
}

func TestGoBuildBudgetRejectsOtherCallersBeforeOpeningOutputs(t *testing.T) {
	p := targetFixture(t)
	for _, kind := range []string{"query phase", "foreign role", "go list", "foreign package", "extra budget flag"} {
		t.Run(kind, func(t *testing.T) {
			q := p
			q.BuildArgv = slices.Clone(p.BuildArgv)
			s := Session{Phase: "build"}
			switch kind {
			case "query phase":
				s.Phase = "queries"
			case "foreign role":
				q.Role, q.Options.Role = "foreign", "foreign"
			case "go list":
				q.BuildArgv = listArgs(q)
			case "foreign package":
				q.BuildArgv[len(q.BuildArgv)-1] = "./cmd/foreign"
			case "extra budget flag":
				q.BuildArgv = append(q.BuildArgv, "-timeout=24h")
			}
			dir := filepath.Join(t.TempDir(), "must-not-be-opened")
			if _, _, e := runFixedGoBuild(context.Background(), dir, q, s, Pin{}); e == nil {
				t.Fatal("foreign orchestration caller accepted")
			}
			if _, e := os.Lstat(dir); !os.IsNotExist(e) {
				t.Fatalf("rejected caller opened its output: %v", e)
			}
		})
	}
	if _, _, e := programBudget(255).context(context.Background()); e == nil {
		t.Fatal("unknown program budget accepted")
	}
}
