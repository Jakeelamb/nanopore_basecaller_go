package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	d "github.com/Jakeelamb/nanopore_basecaller_go/verification/decodergpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	if err := runner.CheckRoot(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	unit := flag.String("unit", "prepare", "finite lineage unit")
	out := flag.String("out", "", "fresh ignored attempt")
	admitted := flag.Bool("admitted", false, "separate coordinator admission")
	producer := flag.String("producer", "", "immutable accepted decoder build")
	producerSHA := flag.String("producer-sha256", "", "independent build hash")
	originalRuntime := flag.String("producer-runtime", "", "immutable capture002 runtime")
	originalRuntimeSHA := flag.String("producer-runtime-sha256", "", "independent runtime hash")
	capture := flag.String("capture", "", "unchanged capture002")
	captureSHA := flag.String("capture-sha256", "", "independent capture hash")
	plan := flag.String("plan", "", "lineage plan")
	planSHA := flag.String("plan-sha256", "", "independent plan hash")
	prior := flag.String("prior", "", "prior compiler phase")
	priorSHA := flag.String("prior-sha256", "", "independent prior hash")
	build := flag.String("build", "", "lineage native build")
	buildSHA := flag.String("build-sha256", "", "independent native build hash")
	proof := flag.String("auditor-proof", "", "accepted auditor proof")
	proofSHA := flag.String("auditor-proof-sha256", "", "independent auditor proof hash")
	index := flag.Int("case", 0, "prospective original input case")
	output := flag.String("capture-out", "", "prospective load output")
	runtime := flag.String("runtime", "", "lineage runtime")
	runtimeSHA := flag.String("runtime-sha256", "", "independent runtime hash")
	load := flag.String("load", "", "retained load receipt")
	loadSHA := flag.String("load-sha256", "", "independent load hash")
	command := flag.String("supervisor-command", "", "actual supervisor command receipt")
	commandSHA := flag.String("supervisor-command-sha256", "", "independent command hash")
	watch := flag.String("supervisor-watch", "", "actual watcher019 receipt")
	watchSHA := flag.String("supervisor-watch-sha256", "", "independent watcher hash")
	flag.Parse()
	var result any
	var err error
	switch *unit {
	case "prepare":
		result, err = d.PrepareLineage(d.LineagePrepareOptions{Producer: d.File{Path: *producer, SHA256: *producerSHA}, Runtime: d.File{Path: *originalRuntime, SHA256: *originalRuntimeSHA}, Capture: d.File{Path: *capture, SHA256: *captureSHA}, Out: *out})
	case "check-plan":
		err = d.CheckLineagePlan(d.File{Path: *plan, SHA256: *planSHA})
		result = map[string]bool{"passed": err == nil}
	case "query", "headers", "build":
		result, err = d.RunLineageCompiler(d.CompilerPhaseOptions{Plan: d.File{Path: *plan, SHA256: *planSHA}, Prior: d.File{Path: *prior, SHA256: *priorSHA}, Phase: *unit, Out: *out, Admitted: *admitted})
	case "plan-runtime":
		result, err = d.PlanLineageRuntime(d.LineageRuntimeOptions{Build: d.File{Path: *build, SHA256: *buildSHA}, ConsumerPlan: d.File{Path: *plan, SHA256: *planSHA}, AuditorProof: d.File{Path: *proof, SHA256: *proofSHA}, CaseIndex: *index, Out: *out, Output: *output})
	case "check-runtime":
		err = d.CheckLineageRuntime(d.File{Path: *runtime, SHA256: *runtimeSHA})
		result = map[string]bool{"passed": err == nil}
	case "load":
		result, err = d.RunLineageLoad(d.LineageLoadOptions{Runtime: d.File{Path: *runtime, SHA256: *runtimeSHA}, Out: *out, Admitted: *admitted})
	case "check-load":
		result, err = d.CheckLineageLoad(d.File{Path: *runtime, SHA256: *runtimeSHA}, d.File{Path: *load, SHA256: *loadSHA}, d.File{Path: *command, SHA256: *commandSHA}, d.File{Path: *watch, SHA256: *watchSHA})
	case "replay-load":
		result, err = d.ReplayLineageLoad(d.LineageReplayOptions{ConsumerPlan: d.File{Path: *plan, SHA256: *planSHA}, Runtime: d.File{Path: *runtime, SHA256: *runtimeSHA}, Load: d.File{Path: *load, SHA256: *loadSHA}, SupervisorCommand: d.File{Path: *command, SHA256: *commandSHA}, SupervisorWatch: d.File{Path: *watch, SHA256: *watchSHA}, Out: *out})
	default:
		err = fmt.Errorf("unknown finite lineage unit")
	}
	if result != nil {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if e := enc.Encode(result); err == nil {
			err = e
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}
