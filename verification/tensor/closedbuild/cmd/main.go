package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(runner.ExitCode(e))
	}
}
func run() error {
	unit := flag.String("unit", "plan", "controller, plan or compile; no native output-execution operation")
	out := flag.String("out", "", "fresh ignored output directory")
	goRoot := flag.String("go-root", "/usr/lib/go", "explicit selected Go root for CGO0 controller rebuild")
	host := flag.Bool("host-admitted", false, "explicit Go-only rebuild admission")
	native := flag.Bool("compiler-admitted", false, "separate reviewed native compiler admission")
	profiles := flag.String("profiles", "", "explicit comma-separated backend, fixtures, failure-test compiler profiles")
	paths := map[string]*string{}
	hashes := map[string]*string{}
	for _, name := range []string{"recipe", "compiler-contract", "link-contract", "asset-contract", "controller", "driver", "pthread-archive", "plan"} {
		paths[name] = flag.String(name, "", "independently pinned "+name)
		hashes[name] = flag.String(name+"-sha", "", "independent SHA256 of "+name)
	}
	flag.Parse()
	if e := runner.CheckRoot(); e != nil {
		return e
	}
	if *out == "" {
		return fmt.Errorf("fresh -out required")
	}
	pin := func(name string) closedbuild.Pin { return closedbuild.Pin{Path: *paths[name], SHA256: *hashes[name]} }
	var result any
	var e error
	switch *unit {
	case "controller":
		result, e = closedbuild.FreezeControllers(closedbuild.ControllerOptions{Out: *out, GoRoot: *goRoot, Admitted: *host})
	case "plan":
		var p closedbuild.Plan
		p, e = closedbuild.PlanBuild(closedbuild.PlanOptions{Recipe: pin("recipe"), CompilerContract: pin("compiler-contract"), LinkContract: pin("link-contract"), AssetContract: pin("asset-contract"), Controller: pin("controller"), Driver: pin("driver"), PthreadArchive: pin("pthread-archive"), Profiles: strings.Split(*profiles, ",")})
		if e == nil {
			var dir string
			dir, e = runner.NewAttempt(*out)
			if e == nil {
				e = closedbuild.Write(filepath.Join(dir, "plan.json"), p)
			}
		}
		result = p
	case "compile":
		result, e = closedbuild.Compile(closedbuild.BuildOptions{Plan: pin("plan"), Out: *out, Admitted: *native})
	default:
		return fmt.Errorf("unknown or held closed-builder operation %q", *unit)
	}
	if e != nil {
		return e
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}
