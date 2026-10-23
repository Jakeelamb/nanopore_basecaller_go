package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/gpuprofile"
)

func run() error {
	plan := flag.String("plan", "provenance/gpu-target.json", "frozen target")
	out := flag.String("out", "", "new attempt directory")
	admitted := flag.Bool("runtime-admitted", false, "requires coordinator admission and native runner integration")
	basic := flag.String("basic", "", "accepted ABI4 runtime manifest")
	build := flag.String("build", "", "accepted ABI4 build manifest")
	koi := flag.String("koi-root", "", "qualified CUDA13 Koi root")
	models := flag.String("model-root", "", "read-only pinned HAC/SUP model root")
	cuda := flag.String("cuda-root", "/opt/cuda", "qualified CUDA toolkit root")
	replay := flag.String("replay", "", "retained profile.json to import without native execution")
	frozen := flag.String("frozen-sha256", "", "trusted frozen manifest SHA256 for import")
	model := flag.String("model", gpuprofile.HAC, "exact model name")
	config := flag.String("config-sha256", "", "actual model config SHA256")
	batch := flag.Int("batch", 0, "explicit padded batch, HAC multiple of 64 or SUP multiple of 32")
	fixed := flag.Bool("disable-variable-chunk-sizes", false, "use fixed chunks")
	trim := flag.Bool("no-trim", false, "disable trimming")
	split := flag.Bool("disable-read-splitting", false, "disable splitting")
	output := flag.String("output", "fastq", "fastq or unaligned bam")
	kit := flag.String("kit", "SQK-LSK114", "sequencing kit")
	flag.Parse()
	if flag.NArg() != 0 {
		return fmt.Errorf("unsupported positional arguments")
	}
	f, err := os.Open(*plan)
	if err != nil {
		return err
	}
	t, err := gpuprofile.DecodeTarget(f)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if *replay != "" {
		if *admitted {
			return fmt.Errorf("replay and runtime admission are separate modes")
		}
		m, err := gpuprofile.ImportProfile(*replay, *frozen)
		if err != nil {
			return err
		}
		if m.Target != t {
			return fmt.Errorf("replayed target differs")
		}
		return json.NewEncoder(os.Stdout).Encode(m)
	}
	r := gpuprofile.Request{Model: *model, ConfigSHA256: *config, Batch: *batch, FixedChunks: *fixed, DisableTrim: *trim, DisableSplit: *split, Output: *output, Kit: *kit, DevOptions: gpuprofile.DefaultDevOptions()}
	if err := gpuprofile.CheckRequest(t, r); err != nil {
		return err
	}
	value, present := os.LookupEnv("CUDA_VISIBLE_DEVICES")
	if (gpuprofile.Environment{Present: present, Value: value}) != t.CUDAVisibleDevices {
		return fmt.Errorf("CUDA_VISIBLE_DEVICES differs, including presence")
	}
	if *admitted {
		_, err := gpuprofile.CaptureProfile(gpuprofile.ProbeOptions{Target: t, Basic: *basic, Build: *build, KoiRoot: *koi, CUDARoot: *cuda, ModelRoot: *models, Out: *out, Admitted: true, Invocation: os.Args})
		return err
	}
	if *out == "" {
		return fmt.Errorf("-out must name a new attempt directory")
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return err
	}
	payload := struct {
		Status      string                  `json:"status"`
		Target      gpuprofile.Target       `json:"target"`
		Request     gpuprofile.Request      `json:"request"`
		Observation *gpuprofile.Observation `json:"observation"`
	}{"source-prepared-runtime-unobserved", t, r, nil}
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*out, "preparation.json"), append(b, '\n'), 0644)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
