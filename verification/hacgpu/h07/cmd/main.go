package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
)

func main() {
	unit := flag.String("unit", "witness", "witness, oracle-build, oracle-capture, cache-verify, or projection")
	capture := flag.String("capture", h07.DefaultCapture, "pinned H01 stage directory")
	out := flag.String("out", "", "fresh artifacts/tensor attempt")
	build := flag.String("build-dir", "", "qualified source build for read-only CMake toolchain metadata")
	source := flag.String("dorado-source", "external/dorado-source", "pinned untouched source checkout")
	runtime := flag.String("runtime-manifest", "", "prospectively frozen complete runtime path-to-SHA256 map")
	runtimeSHA := flag.String("runtime-manifest-sha", "", "independent prospective consumer runtime manifest SHA256")
	compilerAdmitted := flag.Bool("compiler-admitted", false, "coordinator released this genuine C++ build only")
	controllerBuild := flag.String("controller-build", "", "immutable actual Go controller build manifest")
	controllerBuildSHA := flag.String("controller-build-sha", "", "independent controller build manifest SHA256")
	oracleBuild := flag.String("oracle-build", "", "closed oracle build.json")
	oracleBuildSHA := flag.String("oracle-build-sha", "", "independently retained oracle build manifest SHA256")
	backend := flag.String("backend", "", "qualified ABI4 CUDA13 backend for the native CPU loader and typed projection")
	model := flag.String("model", "", "pinned HAC model directory")
	ordinal := flag.Int("ordinal", 0, "visible GPU ordinal")
	runtimeAdmitted := flag.Bool("runtime-admitted", false, "coordinator released this native producer or consumer phase")
	nativeReceipt := flag.String("native-receipt", "", "successful source cache capture receipt")
	nativeSHA := flag.String("native-sha", "", "independently retained native capture receipt SHA256")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "-out fresh-attempt required")
		os.Exit(1)
	}
	var r any
	var err error
	switch *unit {
	case "witness":
		r, err = h07.InspectWitness(*capture, *out)
	case "oracle-build":
		r, err = h07.BuildOracle(h07.OracleBuildOptions{BuildDir: *build, DoradoSource: *source, RuntimeManifest: *runtime, Out: *out, Admitted: *compilerAdmitted, ControllerBuild: *controllerBuild, ControllerBuildSHA: *controllerBuildSHA})
	case "oracle-capture":
		r, err = h07.CaptureOracle(h07.OracleCaptureOptions{Build: *oracleBuild, BuildSHA: *oracleBuildSHA, Capture: *capture, Model: *model, Out: *out, Ordinal: *ordinal, Admitted: *runtimeAdmitted})
	case "cache-verify", "projection":
		r, err = h07.Consumer(context.Background(), h07.ConsumerOptions{NativeReceipt: *nativeReceipt, NativeSHA: *nativeSHA, Backend: *backend, Model: *model, Capture: *capture, Out: *out, Ordinal: *ordinal, Admitted: *runtimeAdmitted, Execute: *unit == "projection", ControllerBuild: *controllerBuild, ControllerBuildSHA: *controllerBuildSHA, RuntimeManifest: *runtime, RuntimeManifestSHA: *runtimeSHA})
	default:
		err = fmt.Errorf("unknown H07 unit %q", *unit)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err = json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
