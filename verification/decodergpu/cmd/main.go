package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/decodergpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}
func run() error {
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	o := decodergpu.PrepareOptions{}
	unit := flag.String("unit", "prepare", "prepare, plan-native, check-native, query, headers, build, plan-runtime, or capture")
	sm120a := flag.String("sm120a", "artifacts/decoder-static-20261025-sm120a-001/decoder_lib.sm_120a.cubin", "pinned selected-member SM120a image")
	imagePolicy := flag.String("image-policy", "", "prospective fixed decoder image policy")
	imagePolicySHA := flag.String("image-policy-sha256", "", "external image policy SHA256")
	observation := flag.String("observation", "", "retained image/module/function/launch observations")
	auditorProof := flag.String("auditor-proof", "", "accepted immutable auditor producer proof")
	auditorProofSHA := flag.String("auditor-proof-sha256", "", "external accepted auditor proof SHA256")
	captureOut := flag.String("capture-out", "", "prospective fresh native capture directory")
	captureReceipt := flag.String("capture-receipt", "", "retained native capture.json")
	captureReceiptSHA := flag.String("capture-receipt-sha256", "", "external native capture receipt SHA256")
	supervisorCommand := flag.String("supervisor-command", "", "actual external watcher command.json")
	supervisorCommandSHA := flag.String("supervisor-command-sha256", "", "external actual watcher command SHA256")
	supervisorReceipt := flag.String("supervisor-receipt", "", "actual watcher019 receipt")
	supervisorReceiptSHA := flag.String("supervisor-receipt-sha256", "", "external watcher019 receipt SHA256")
	preparation := flag.String("preparation", "", "pinned immutable preparation.json")
	preparationSHA := flag.String("preparation-sha256", "", "external preparation SHA256")
	plan := flag.String("plan", "", "closed native plan.json")
	planSHA := flag.String("plan-sha256", "", "external native plan SHA256")
	prior := flag.String("prior", "", "prior compiler phase.json")
	priorSHA := flag.String("prior-sha256", "", "external prior compiler phase SHA256")
	build := flag.String("build", "", "closed successful build phase.json")
	buildSHA := flag.String("build-sha256", "", "external closed build SHA256")
	runtime := flag.String("runtime-plan", "", "prospective runtime.json")
	runtimeSHA := flag.String("runtime-plan-sha256", "", "external runtime plan SHA256")
	numericLineage := flag.String("numeric-lineage", "", "accepted sm120a load-only replay proof for fresh numeric contract")
	numericLineageSHA := flag.String("numeric-lineage-sha256", "", "independent accepted lineage replay SHA256")
	admitted := flag.Bool("admitted", false, "coordinator release for this exact phase only; never implies capture admission")
	flag.StringVar(&o.Capture, "capture", h07.DefaultCapture, "retained H01 stages")
	flag.StringVar(&o.Model, "model", "/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0", "read-only exact HAC config")
	flag.StringVar(&o.BuildDir, "build-dir", "/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/build-gcc15", "retained qualified CMake metadata root")
	flag.StringVar(&o.Dorado, "dorado", "external/dorado-source", "pinned read-only Dorado source")
	flag.StringVar(&o.Static, "static", "artifacts/decoder-static-20261025-001", "retained selected decoder object/PTX evidence")
	flag.StringVar(&o.Out, "out", "", "fresh artifacts/tensor preparation directory")
	flag.Parse()
	o.SM120a = *sm120a
	o.NumericLineage = decodergpu.File{Path: *numericLineage, SHA256: *numericLineageSHA}
	if o.Out == "" && *unit != "check-native" && *unit != "check-producer" && *unit != "check-images" && *unit != "check-capture" && *unit != "check-runtime" {
		return fmt.Errorf("fresh -out required")
	}
	var r any
	var err error
	switch *unit {
	case "prepare":
		r, err = decodergpu.Prepare(o)
	case "plan-native":
		r, err = decodergpu.PlanClosedBuild(decodergpu.File{Path: *preparation, SHA256: *preparationSHA}, o.BuildDir, o.Out)
	case "check-native":
		err = decodergpu.CheckClosedBuildPlan(decodergpu.File{Path: *plan, SHA256: *planSHA})
		if err == nil {
			r = map[string]string{"status": "closed_plan_checked"}
		} else {
			r = map[string]string{"status": "failed", "error": err.Error()}
		}
	case "query", "headers", "build":
		r, err = decodergpu.RunCompilerPhase(decodergpu.CompilerPhaseOptions{Plan: decodergpu.File{Path: *plan, SHA256: *planSHA}, Prior: decodergpu.File{Path: *prior, SHA256: *priorSHA}, Phase: *unit, Out: o.Out, Admitted: *admitted})
	case "check-producer":
		r, err = decodergpu.CheckProducerConsumer(decodergpu.File{Path: *build, SHA256: *buildSHA}, decodergpu.File{Path: *preparation, SHA256: *preparationSHA})
	case "plan-images":
		var policy decodergpu.ImagePolicy
		policy, err = decodergpu.PlanDeviceImages(o.Static, *sm120a)
		r = policy
		if err == nil {
			err = decodergpu.WriteImagePolicy(o.Out, policy)
		}
	case "check-images":
		r, err = decodergpu.CheckImages(decodergpu.File{Path: *imagePolicy, SHA256: *imagePolicySHA}, *observation)
	case "check-runtime":
		err = decodergpu.CheckGuardedRuntime(decodergpu.File{Path: *build, SHA256: *buildSHA}, decodergpu.File{Path: *preparation, SHA256: *preparationSHA}, decodergpu.File{Path: *runtime, SHA256: *runtimeSHA})
		if err == nil {
			r = map[string]string{"status": "guarded_runtime_checked"}
		} else {
			r = map[string]string{"status": "failed", "error": err.Error()}
		}
	case "plan-runtime":
		if *auditorProof != "" {
			r, err = decodergpu.PlanGuardedRuntime(decodergpu.GuardRuntimeOptions{Build: decodergpu.File{Path: *build, SHA256: *buildSHA}, Consumer: decodergpu.File{Path: *preparation, SHA256: *preparationSHA}, AuditorProof: decodergpu.File{Path: *auditorProof, SHA256: *auditorProofSHA}, Out: o.Out, Capture: *captureOut})
		} else if *preparation != "" {
			r, err = decodergpu.PlanRuntimeConsumer(decodergpu.File{Path: *build, SHA256: *buildSHA}, decodergpu.File{Path: *preparation, SHA256: *preparationSHA}, o.Out)
		} else {
			r, err = decodergpu.PlanRuntime(decodergpu.File{Path: *build, SHA256: *buildSHA}, o.Out)
		}
	case "capture":
		r, err = decodergpu.Capture(decodergpu.CaptureOptions{Build: decodergpu.File{Path: *build, SHA256: *buildSHA}, Runtime: decodergpu.File{Path: *runtime, SHA256: *runtimeSHA}, Consumer: decodergpu.File{Path: *preparation, SHA256: *preparationSHA}, Out: o.Out, Admitted: *admitted})
	case "check-capture":
		r, err = decodergpu.CheckCapture(decodergpu.CaptureOptions{Build: decodergpu.File{Path: *build, SHA256: *buildSHA}, Runtime: decodergpu.File{Path: *runtime, SHA256: *runtimeSHA}, Consumer: decodergpu.File{Path: *preparation, SHA256: *preparationSHA}}, decodergpu.File{Path: *captureReceipt, SHA256: *captureReceiptSHA}, decodergpu.File{Path: *supervisorCommand, SHA256: *supervisorCommandSHA}, decodergpu.File{Path: *supervisorReceipt, SHA256: *supervisorReceiptSHA})
	default:
		return fmt.Errorf("unknown D01 unit %q", *unit)
	}
	b, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		return e
	}
	fmt.Println(string(b))
	return err
}
