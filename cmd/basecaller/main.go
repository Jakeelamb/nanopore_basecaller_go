package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

type idsFlag []pod5.ID

func (f *idsFlag) String() string { return fmt.Sprint([]pod5.ID(*f)) }
func (f *idsFlag) Set(value string) error {
	id, err := pod5.ParseID(value)
	if err != nil {
		return err
	}
	for _, existing := range *f {
		if existing == id {
			return errors.New("duplicate --read-id")
		}
	}
	*f = append(*f, id)
	return nil
}
func fastq(w io.Writer, r caller.CalledRead) error {
	return r.WriteFASTQ(w)
}
func run(ctx context.Context, args []string, stdout, stderr io.Writer) (err error) {
	fs := flag.NewFlagSet("basecaller", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var modelDir, pod5Path, tensorLib, pod5Lib, device, capture, replay, processing string
	var nativeBundle, chunking, format string
	var batchSize, liveParents, pendingChunks int
	var liveSamples, maxReadSamples uint64
	var emitMoves bool
	var size, overlap uint64
	var all, noTrim, noSplit bool
	var ids idsFlag
	fs.StringVar(&modelDir, "model", "", "pinned HAC or SUP model directory")
	fs.StringVar(&pod5Path, "pod5", "", "one POD5 input file")
	fs.StringVar(&tensorLib, "tensor-library", "", "CPU tensor ABI library")
	fs.StringVar(&pod5Lib, "pod5-library", "", "POD5 0.3.39 library")
	fs.StringVar(&device, "device", "cpu", "cpu or gated cuda:0; no production GPU bundle is approved")
	fs.StringVar(&nativeBundle, "native-bundle", "", "GPU installation requiring independent release approval; none approved")
	fs.StringVar(&chunking, "chunking", "variable", "GPU source chunk policy: variable or fixed; variable invocation is unfinished")
	fs.StringVar(&format, "format", "fastq", "gated GPU output: fastq or bam; CPU and replay emit FASTQ")
	fs.IntVar(&batchSize, "batchsize", 64, "GPU batch capacity; fixed HAC requires 64, fixed SUP requires explicit 32")
	fs.IntVar(&liveParents, "live-parents", 64, "maximum retained GPU parents")
	fs.IntVar(&pendingChunks, "pending-chunks", 256, "maximum queued GPU chunks across profiles")
	fs.Uint64Var(&liveSamples, "live-samples", 6000000, "maximum retained normalized GPU samples")
	fs.Uint64Var(&maxReadSamples, "max-read-samples", 3000000, "maximum decoded samples in one GPU input parent")
	fs.BoolVar(&emitMoves, "emit-moves", false, "emit GPU BAM move tables")
	fs.StringVar(&capture, "capture", "", "new immutable CPU capture directory; rejected for GPU")
	fs.StringVar(&replay, "replay", "", "replay one captured manifest without model or POD5 libraries")
	fs.StringVar(&replay, "manifest", "", "manifest for the replay subcommand")
	fs.StringVar(&processing, "postprocess", "", "apply default processing to an original core replay")
	fs.Uint64Var(&size, "chunksize", 12288, "CPU fixed chunk size or requested GPU maximum profile size in samples")
	fs.Uint64Var(&overlap, "overlap", 600, "overlap in samples")
	fs.Var(&ids, "read-id", "selected UUID; repeat for multiple reads")
	fs.BoolVar(&all, "all", false, "explicitly process every read in the input file")
	fs.BoolVar(&noTrim, "no-trim", false, "disable adapter and primer trimming")
	fs.BoolVar(&noSplit, "disable-read-splitting", false, "disable pore-adapter read splitting")
	replayCommand := len(args) > 0 && args[0] == "replay"
	if replayCommand {
		args = args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}
	policy := postcall.DefaultPolicy()
	policy.Trim, policy.Split = !noTrim, !noSplit
	if processing != "" && processing != "default" {
		return errors.New("--postprocess supports only default")
	}
	policyFlags := false
	gpuFlags := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "no-trim" || f.Name == "disable-read-splitting" {
			policyFlags = true
		}
		switch f.Name {
		case "native-bundle", "chunking", "format", "batchsize", "live-parents", "pending-chunks", "live-samples", "max-read-samples", "emit-moves":
			gpuFlags = true
		}
	})
	if replay != "" || replayCommand {
		out := bufio.NewWriter(stdout)
		defer func() { err = errors.Join(err, out.Flush()) }()
		if replay == "" {
			return errors.New("replay requires --manifest PATH")
		}
		if modelDir != "" || pod5Path != "" || tensorLib != "" || pod5Lib != "" || all || len(ids) > 0 || device != "cpu" || gpuFlags {
			return errors.New("replay accepts no model, input, library or selection flags and requires cpu")
		}
		if processing == "" && (capture != "" || policyFlags) {
			return errors.New("core replay processing flags and a new capture require --postprocess default")
		}
		var reads []caller.CalledRead
		if processing == "default" {
			result, err := caller.ReplayProcessed(ctx, replay, policy, capture)
			if err != nil {
				return err
			}
			reads = result.Reads
		} else {
			var err error
			reads, err = caller.ReplayRecords(ctx, replay)
			if err != nil {
				return err
			}
		}
		data, err := (caller.Processed{Reads: reads}).FASTQBytes()
		if err != nil {
			return err
		}
		if _, err := out.Write(data); err != nil {
			return err
		}
		if capture != "" {
			if _, err := fmt.Fprintln(stderr, "capture", filepath.Join(capture, "processed-manifest.json")); err != nil {
				return err
			}
		}
		_, err = fmt.Fprintf(stderr, "replayed input_reads=1 output_reads=%d\n", len(reads))
		return err
	}
	if device == "cuda:0" {
		if processing != "" || capture != "" || tensorLib != "" || pod5Lib != "" {
			return errors.New("GPU input uses --native-bundle and does not support replay processing, capture, or separate library overrides")
		}
		if all == (len(ids) > 0) {
			return errors.New("choose explicit --read-id values or --all")
		}
		mode := caller.VariableChunks
		switch chunking {
		case "variable":
		case "fixed":
			mode = caller.FixedChunks
		default:
			return errors.New("--chunking must be variable or fixed")
		}
		outputFormat := caller.FASTQ
		switch format {
		case "fastq":
		case "bam":
			outputFormat = caller.BAM
		default:
			return errors.New("--format must be fastq or bam")
		}
		var selection []pod5.ID
		if !all {
			selection = []pod5.ID(ids)
		}
		quoted := make([]string, len(args)+1)
		quoted[0] = "basecaller"
		for i, arg := range args {
			quoted[i+1] = strconv.Quote(arg)
		}
		report, runErr := caller.RunGPU(ctx, caller.GPUConfig{
			NativeBundleDir: nativeBundle, ModelDir: modelDir, POD5Path: pod5Path,
			ReadIDs: selection, BatchSize: batchSize, ChunkSize: size, Overlap: overlap,
			Chunking: mode, Format: outputFormat, Policy: policy, EmitMoves: emitMoves,
			Limits:      caller.GPULimits{LiveParents: liveParents, PendingChunks: pendingChunks, LiveSamples: liveSamples, MaxReadSamples: maxReadSamples},
			CommandLine: strings.Join(quoted, " "),
		}, stdout)
		if runErr != nil {
			return runErr
		}
		_, err = fmt.Fprintf(stderr, "processed input_reads=%d output_reads=%d original_samples=%d\n", report.InputParents, report.OutputRecords, report.OriginalSamples)
		return err
	}
	if device != "cpu" {
		return fmt.Errorf("device %q is unsupported; use cpu", device)
	}
	if gpuFlags {
		return errors.New("GPU options require --device cuda:0")
	}
	if processing != "" {
		return errors.New("--postprocess is only for an original core replay")
	}
	if modelDir == "" || pod5Path == "" || tensorLib == "" || pod5Lib == "" {
		return errors.New("--model, --pod5, --tensor-library and --pod5-library are required")
	}
	if all == (len(ids) > 0) {
		return errors.New("choose explicit --read-id values or --all")
	}
	out := bufio.NewWriter(stdout)
	defer func() { err = errors.Join(err, out.Flush()) }()
	reader, err := pod5.OpenContext(ctx, pod5Lib, pod5Path)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, reader.Close()) }()
	c, err := caller.Open(ctx, caller.Config{TensorLibrary: tensorLib, ModelDir: modelDir, ChunkSize: size, Overlap: overlap, CaptureDir: capture})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, c.Close()) }()
	var selection []pod5.ID
	if !all {
		selection = []pod5.ID(ids)
	}
	count, outputCount := 0, 0
	if err := reader.Iterate(ctx, selection, func(r pod5.Read) error {
		called, err := c.Process(ctx, r, policy)
		if err != nil {
			return err
		}
		data, err := called.FASTQBytes()
		if err != nil {
			return err
		}
		if _, err := out.Write(data); err != nil {
			return err
		}
		count++
		outputCount += len(called.Reads)
		if capture != "" {
			_, err = fmt.Fprintf(stderr, "capture %s\n", filepath.Join(capture, r.ID.String(), "processed-manifest.json"))
			return err
		}
		return nil
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(stderr, "processed input_reads=%d output_reads=%d\n", count, outputCount)
	return err
}
func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx, os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "basecaller:", err)
		os.Exit(1)
	}
}
