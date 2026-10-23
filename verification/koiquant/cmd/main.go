package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiquant"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	root := flag.String("root", ".", "canonical Go module root")
	input := flag.String("input-capture", "artifacts/koiweights/attempt-qkv-f16-002/manifest.json", "pinned existing QKV half capture")
	out := flag.String("out", "", "fresh artifacts/koiquant/attempt-* output")
	device := flag.String("device", "cpu", "planned capture tier, cpu or cuda; does not initialize a device")
	check := flag.String("check", "", "validate a prepared source-only manifest without native execution")
	capture := flag.String("capture", "", "prepared manifest to execute only after coordinator release")
	basic := flag.String("basic-runtime", "", "exact accepted ABI4 runtime manifest")
	build := flag.String("build-manifest", "", "matching pinned CUDA13 build manifest")
	reuse := flag.String("reuse-binary", "", "CPU raw manifest whose exact producer binary is reused for CUDA")
	ordinal := flag.Int("ordinal", 0, "CUDA ordinal; CPU requires zero")
	admitted := flag.Bool("runtime-admitted", false, "coordinator released this bounded native invocation")
	importRaw := flag.String("import", "", "import a retained raw manifest without native execution")
	replay := flag.String("replay", "", "import retained raw output and run pure host replay checks")
	cpu := flag.String("compare-cpu", "", "imported CPU manifest to compare")
	cuda := flag.String("compare-cuda", "", "imported CUDA manifest to compare")
	flag.Parse()
	if flag.NArg() != 0 {
		fail(fmt.Errorf("unexpected positional arguments"))
	}
	absRoot, err := filepath.Abs(*root)
	if err != nil {
		fail(err)
	}
	modes := 0
	for _, value := range []string{*check, *capture, *importRaw, *replay, *cpu} {
		if value != "" {
			modes++
		}
	}
	if modes > 1 || (*cuda != "" && *cpu == "") {
		fail(fmt.Errorf("select exactly one check, capture, import, replay, or comparison mode"))
	}
	if *capture == "" && (*admitted || *basic != "" || *build != "" || *reuse != "" || *ordinal != 0) {
		fail(fmt.Errorf("native flags require -capture"))
	}
	if *capture != "" {
		_, err = koiquant.Capture(koiquant.CaptureOptions{Root: absRoot, Prepared: *capture, Basic: *basic, Build: *build, Out: *out, Ordinal: *ordinal, Admitted: *admitted, ReuseBinary: *reuse, Invocation: os.Args})
		if err != nil {
			fail(err)
		}
		fmt.Println(filepath.Join(*out, "raw-manifest.json"))
		return
	}
	if *importRaw != "" || *replay != "" || *cpu != "" {
		if *out == "" {
			fail(fmt.Errorf("-out is required"))
		}
		switch {
		case *importRaw != "":
			_, err = koiquant.Import(absRoot, *importRaw, *out)
		case *replay != "":
			_, err = koiquant.Replay(absRoot, *replay, *out)
		default:
			_, err = koiquant.Compare(absRoot, *cpu, *cuda, *out)
		}
		if err != nil {
			fail(err)
		}
		fmt.Println(filepath.Join(*out, "manifest.json"))
		return
	}
	if *check != "" {
		if *out != "" {
			fail(fmt.Errorf("-check and -out are mutually exclusive"))
		}
		_, err = koiquant.CheckPrepared(absRoot, *check)
		if err != nil {
			fail(err)
		}
		fmt.Println("prepared source and inputs verified; native execution remains held")
		return
	}
	if *out == "" {
		fail(fmt.Errorf("-out is required to prepare a fresh attempt"))
	}
	_, err = koiquant.Prepare(koiquant.Options{Root: absRoot, InputCapture: *input, Out: *out, Device: *device, Invocation: os.Args})
	if err != nil {
		fail(err)
	}
	fmt.Println(filepath.Join(*out, "manifest.json"))
	fmt.Println("source-only preparation complete; native execution remains held")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(runner.ExitCode(err))
}
