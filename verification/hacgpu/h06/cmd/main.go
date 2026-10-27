package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h06"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
}
func run() error {
	unit := flag.String("unit", "prepare", "prepare, controller-build, build-plan, or oracle-build; target execution remains held")
	capture := flag.String("capture", h07.DefaultCapture, "authenticated retained H01 stage directory")
	source := flag.String("dorado-source", "external/dorado-source", "read-only pinned source root")
	out := flag.String("out", "", "fresh ignored output directory")
	prepared := flag.String("prepared", "", "immutable H06 prepared manifest")
	preparedSHA := flag.String("prepared-sha", "", "independent prepared manifest SHA256")
	compiler := flag.String("compiler-contract", "", "accepted D01 selected compiler contract")
	compilerSHA := flag.String("compiler-contract-sha", "", "independent selected compiler contract SHA256")
	link := flag.String("link-contract", "", "accepted S00 closed Torch link contract")
	linkSHA := flag.String("link-contract-sha", "", "independent closed Torch link contract SHA256")
	controller := flag.String("controller-build", "", "sealed CGO0 H06 controller receipt")
	controllerSHA := flag.String("controller-build-sha", "", "independent controller receipt SHA256")
	buildDir := flag.String("build-dir", "", "qualified read-only CMake build directory")
	plan := flag.String("plan", "", "reviewed H06 build plan")
	planSHA := flag.String("plan-sha", "", "independent H06 build plan SHA256")
	hostAdmitted := flag.Bool("host-admitted", false, "explicit Go-only controller build admission")
	compilerAdmitted := flag.Bool("compiler-admitted", false, "separate root-reviewed native compiler admission")
	flag.Parse()
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	if *out == "" {
		return fmt.Errorf("explicit fresh -out required")
	}
	var result any
	var err error
	switch *unit {
	case "prepare":
		result, err = h06.Prepare(*capture, *source, *out)
	case "controller-build":
		result, err = h06.BuildController(h06.ControllerBuildOptions{Out: *out, Admitted: *hostAdmitted})
	case "build-plan":
		var p h06.BuildPlan
		p, err = h06.PlanBuild(h06.BuildPlanOptions{Prepared: nativeproof.Pin{Path: *prepared, SHA256: *preparedSHA}, CompilerContract: nativeproof.Pin{Path: *compiler, SHA256: *compilerSHA}, LinkContract: nativeproof.Pin{Path: *link, SHA256: *linkSHA}, Controller: nativeproof.Pin{Path: *controller, SHA256: *controllerSHA}, BuildDir: *buildDir})
		if err == nil {
			var dir string
			dir, err = runner.NewAttempt(*out)
			if err == nil {
				var body []byte
				body, err = json.MarshalIndent(p, "", "  ")
				if err == nil {
					err = os.WriteFile(dir+"/plan.json", append(body, '\n'), 0600)
				}
			}
		}
		result = p
	case "oracle-build":
		result, err = h06.BuildOracle(h06.BuildOptions{Plan: nativeproof.Pin{Path: *plan, SHA256: *planSHA}, Out: *out, Admitted: *compilerAdmitted})
	default:
		return fmt.Errorf("unknown or held H06 unit %q; target execution is unavailable", *unit)
	}
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}
