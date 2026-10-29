package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/fakecapture"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if filepath.Base(os.Args[0]) == "cc" {
		err := fakecapture.CC(ctx, fakecapture.Pin{Path: os.Getenv("H08_FAKE_SESSION"), SHA256: os.Getenv("H08_FAKE_SESSION_SHA")}, os.Args[1:])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	flags := flag.NewFlagSet("cpu13-fake-capture", flag.ContinueOnError)
	request := flags.String("request", "", "independently pinned request JSON")
	sha := flags.String("request-sha", "", "independent request SHA256")
	admitted := flags.Bool("admit-native-phase", false, "root has installed envelope, watchdog and stock lock")
	if err := flags.Parse(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "no positional arguments admitted")
		os.Exit(1)
	}
	r, err := fakecapture.Capture(ctx, fakecapture.Pin{Path: *request, SHA256: *sha}, *admitted)
	if err != nil {
		r.Error = err.Error()
	}
	if e := json.NewEncoder(os.Stdout).Encode(r); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
