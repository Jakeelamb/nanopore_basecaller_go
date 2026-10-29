//go:build linux

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/gpucaller"
)

func main() {
	unit := flag.String("unit", "", "registered: schedule-check and inventory (import/replay), postcall-check (replay); qualify, biological, benchmark unimplemented")
	mode := flag.String("mode", "", "import or replay (unit-dependent); native is unwired")
	input := flag.String("input", "", "unit-specific plan, capture, or retained receipt")
	output := flag.String("out", "", "fresh attempt directory")
	admitted := flag.Bool("admitted", false, "coordinator resource intake granted; does not enable native handlers")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "unexpected positional arguments")
		os.Exit(2)
	}
	handlers := gpucaller.Handlers{}
	handlers.Inventory = gpucaller.InventoryHandlers()
	handlers.PostcallCheck = gpucaller.PostcallHandlers()
	handlers.ScheduleCheck = gpucaller.ScheduleHandlers()
	err := gpucaller.Dispatch(gpucaller.Request{Unit: gpucaller.Unit(*unit), Mode: gpucaller.Mode(*mode), Input: *input, Output: *output, Admitted: *admitted}, handlers)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
