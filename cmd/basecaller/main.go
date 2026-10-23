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
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
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
	var modelDir, pod5Path, tensorLib, pod5Lib, device, capture, replay string
	var size, overlap uint64
	var all, noTrim, noSplit bool
	var ids idsFlag
	fs.StringVar(&modelDir, "model", "", "pinned HAC model directory")
	fs.StringVar(&pod5Path, "pod5", "", "one POD5 input file")
	fs.StringVar(&tensorLib, "tensor-library", "", "CPU tensor ABI library")
	fs.StringVar(&pod5Lib, "pod5-library", "", "POD5 0.3.39 library")
	fs.StringVar(&device, "device", "cpu", "supported device is cpu")
	fs.StringVar(&capture, "capture", "", "new immutable capture directory")
	fs.StringVar(&replay, "replay", "", "replay one captured manifest without model or POD5 libraries")
	fs.StringVar(&replay, "manifest", "", "manifest for the replay subcommand")
	fs.Uint64Var(&size, "chunksize", 12288, "fixed chunk size in samples")
	fs.Uint64Var(&overlap, "overlap", 600, "overlap in samples")
	fs.Var(&ids, "read-id", "selected UUID; repeat for multiple reads")
	fs.BoolVar(&all, "all", false, "explicitly process every read in the input file")
	fs.BoolVar(&noTrim, "no-trim", false, "required; adapter trimming is unsupported")
	fs.BoolVar(&noSplit, "disable-read-splitting", false, "required; read splitting is unsupported")
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
	out := bufio.NewWriter(stdout)
	defer func() { err = errors.Join(err, out.Flush()) }()
	if replay != "" || replayCommand {
		if replay == "" {
			return errors.New("replay requires --manifest PATH")
		}
		if modelDir != "" || pod5Path != "" || tensorLib != "" || pod5Lib != "" || capture != "" || all || len(ids) > 0 {
			return errors.New("replay accepts no model, input, library, capture or selection flags")
		}
		r, err := caller.Replay(ctx, replay)
		if err != nil {
			return err
		}
		if err := fastq(out, r); err != nil {
			return err
		}
		_, err = fmt.Fprintln(stderr, "replayed reads=1")
		return err
	}
	if device != "cpu" {
		return fmt.Errorf("device %q is unsupported; use cpu", device)
	}
	if !noTrim || !noSplit {
		return errors.New("this caller requires --no-trim --disable-read-splitting")
	}
	if modelDir == "" || pod5Path == "" || tensorLib == "" || pod5Lib == "" {
		return errors.New("--model, --pod5, --tensor-library and --pod5-library are required")
	}
	if all == (len(ids) > 0) {
		return errors.New("choose explicit --read-id values or --all")
	}
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
	count := 0
	if err := reader.Iterate(ctx, selection, func(r pod5.Read) error {
		called, err := c.Call(ctx, r)
		if err != nil {
			return err
		}
		if err := fastq(out, called); err != nil {
			return err
		}
		count++
		if capture != "" {
			_, err = fmt.Fprintf(stderr, "capture %s\n", filepath.Join(capture, r.ID.String(), "manifest.json"))
			return err
		}
		return nil
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(stderr, "processed reads=%d\n", count)
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
