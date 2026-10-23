package gpucaller

import "fmt"

type Unit string

const (
	ScheduleCheck Unit = "schedule-check"
	PostcallCheck Unit = "postcall-check"
	Qualify       Unit = "qualify"
	Biological    Unit = "biological"
	Benchmark     Unit = "benchmark"
	Inventory     Unit = "inventory"
)

type Mode string

const (
	Import Mode = "import"
	Replay Mode = "replay"
	Native Mode = "native"
)

type Request struct {
	Unit     Unit
	Mode     Mode
	Input    string
	Output   string
	Admitted bool
}
type Handler func(Request) error

// Import and Replay handlers must remain host-only. Native handlers must call Admit.
type UnitHandlers struct {
	Import Handler
	Replay Handler
	Native Handler
}

// P05 serializes these explicit registrations; unknown units cannot register.
type Handlers struct {
	ScheduleCheck UnitHandlers
	PostcallCheck UnitHandlers
	Qualify       UnitHandlers
	Biological    UnitHandlers
	Benchmark     UnitHandlers
	Inventory     UnitHandlers
}

func Dispatch(r Request, h Handlers) error {
	if r.Input == "" || r.Output == "" {
		return fmt.Errorf("input and fresh output required")
	}
	var unit UnitHandlers
	switch r.Unit {
	case ScheduleCheck:
		unit = h.ScheduleCheck
	case PostcallCheck:
		unit = h.PostcallCheck
	case Qualify:
		unit = h.Qualify
	case Biological:
		unit = h.Biological
	case Benchmark:
		unit = h.Benchmark
	case Inventory:
		unit = h.Inventory
	default:
		return fmt.Errorf("unsupported caller verification unit %q", r.Unit)
	}
	var handler Handler
	switch r.Mode {
	case Import:
		handler = unit.Import
	case Replay:
		handler = unit.Replay
	case Native:
		if !r.Admitted {
			return fmt.Errorf("native resource admission required")
		}
		handler = unit.Native
	default:
		return fmt.Errorf("unsupported caller verification mode %q", r.Mode)
	}
	if handler == nil {
		return fmt.Errorf("caller verification unit %q mode %q is not wired or qualified", r.Unit, r.Mode)
	}
	return handler(r)
}
