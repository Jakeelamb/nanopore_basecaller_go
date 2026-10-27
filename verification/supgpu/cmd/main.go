package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	unit := flag.String("unit", "", "prepare, preflight, freeze-controller, generate, plan-build, check-plan, build, check-runtime-consumer, plan-runtime, attest-runtime, source-graph, seal, import")
	dorado := flag.String("dorado-source", "external/dorado-source", "read-only pinned source")
	model := flag.String("model", "", "read-only pinned SUP model")
	fixture := flag.String("fixture", "artifacts/pod5/fixtures-v3/manifest.json", "retained genuine HG002 fixture")
	profile := flag.String("profile", "artifacts/tensor/gpu-profile-native-20261023T204747Z/profile.json", "retained P01 observation")
	target := flag.String("target", "provenance/gpu-target.json", "frozen target")
	prepared := flag.String("prepared", "", "prepared.json")
	preparedSHA := flag.String("prepared-sha256", "", "external frozen prepared SHA256")
	generated := flag.String("generated", "", "generated source directory")
	buildDir := flag.String("build-dir", "", "read-only qualified Dorado build")
	plan := flag.String("plan", "", "build plan.json")
	planSHA := flag.String("plan-sha256", "", "external frozen plan SHA256")
	build := flag.String("build", "", "build.json")
	buildSHA := flag.String("build-sha256", "", "external frozen build SHA256")
	controller := flag.String("controller", "", "frozen Go controller manifest")
	controllerSHA := flag.String("controller-sha256", "", "external controller manifest SHA256")
	runtimePlan := flag.String("runtime-plan", "", "prospective post-build ELF runtime plan")
	runtimeSHA := flag.String("runtime-plan-sha256", "", "external frozen runtime plan SHA256")
	input := flag.String("input", "", "retained capture directory")
	out := flag.String("out", "", "fresh output directory")
	admitted := flag.Bool("runtime-admitted", false, "separate coordinator admission for this exact native phase")
	flag.Parse()
	if flag.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	var v any
	var err error
	switch *unit {
	case "freeze-controller":
		v, err = supgpu.FreezeController(*input)
	case "check-plan":
		v, err = supgpu.CheckPlan(*plan, *planSHA)
	case "plan-runtime":
		v, err = supgpu.PlanRuntime(*build, *buildSHA, *out, *controller, *controllerSHA)
	case "check-runtime-consumer":
		v, err = supgpu.CheckRuntimeConsumer(*build, *buildSHA, *controller, *controllerSHA)
	case "prepare":
		v, err = supgpu.Prepare(supgpu.PrepareOptions{Dorado: *dorado, Model: *model, Fixture: *fixture, Profile: *profile, Target: *target, Out: *out})
	case "preflight":
		v, err = supgpu.ReadPrepared(*prepared, *preparedSHA)
	case "generate":
		v, err = supgpu.Generate(*prepared, *preparedSHA, *out)
	case "plan-build":
		v, err = supgpu.PlanBuild(*prepared, *preparedSHA, *generated, *buildDir, *out, *controller, *controllerSHA)
	case "build":
		v, err = supgpu.RunBuild(*plan, *planSHA, *admitted)
	case "source-graph":
		v, err = supgpu.RunCapture(*prepared, *preparedSHA, *build, *buildSHA, *runtimePlan, *runtimeSHA, *out, *admitted)
	case "attest-runtime":
		v, err = supgpu.RunRuntimeAttest(*prepared, *preparedSHA, *build, *buildSHA, *runtimePlan, *runtimeSHA, *out, *admitted)
	case "seal":
		v, err = supgpu.Seal(*input, *prepared, *preparedSHA, *build, *buildSHA, *runtimePlan, *runtimeSHA)
	case "import":
		v, err = supgpu.Import(*input, *prepared, *preparedSHA, *build, *buildSHA, *runtimePlan, *runtimeSHA)
	default:
		return fmt.Errorf("unknown finite SUP unit %q", *unit)
	}
	if err != nil {
		return err
	}
	body, err := supgpu.Marshal(v)
	if err != nil {
		return err
	}
	fmt.Println(string(body))
	return nil
}
