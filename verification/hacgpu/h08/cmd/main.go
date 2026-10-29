package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	defaults := h08.DefaultSources()
	mode := flag.String("mode", "replay", "replay, backend-proof, admission-proof, consumer, cpu13, or guarded-child")
	capture := flag.String("capture", defaults.Capture, "retained H01 stage directory")
	conv := flag.String("conv-receipt", defaults.Convolution.Path, "accepted H03 packing receipt")
	convSHA := flag.String("conv-sha", defaults.Convolution.SHA256, "independent H03 receipt SHA256")
	rnn := flag.String("recurrent-receipt", defaults.Recurrence.Path, "accepted H04 cache consumer receipt")
	rnnSHA := flag.String("recurrent-sha", defaults.Recurrence.SHA256, "independent H04 receipt SHA256")
	projection := flag.String("projection-receipt", defaults.Projection.Path, "accepted H07 source-cache receipt")
	projectionSHA := flag.String("projection-sha", defaults.Projection.SHA256, "independent H07 receipt SHA256")
	backend := flag.String("backend", "", "closed typed CUDA backend")
	backendSHA := flag.String("backend-sha", "", "independently admitted selected backend SHA256")
	backendBuild := flag.String("backend-build", "", "actual closedbuild build.json")
	backendBuildSHA := flag.String("backend-build-sha", "", "independently admitted closed build SHA256")
	backendPlan := flag.String("backend-plan", "", "immutable producer plan.json")
	backendPlanSHA := flag.String("backend-plan-sha", "", "independently admitted producer plan SHA256")
	backendProducer := flag.String("backend-producer", "", "historical producer controller.json")
	backendProducerSHA := flag.String("backend-producer-sha", "", "independently admitted historical producer SHA256")
	admission := flag.String("admission", "", "independently admitted source-specific launch JSON")
	admissionSHA := flag.String("admission-sha", "", "independent launch SHA256")
	request := flag.String("request", "", "parent-selected private child request")
	requestSHA := flag.String("request-sha", "", "parent-selected request SHA256")
	out := flag.String("out", "", "fresh replay attempt path")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var r any
	var err error
	pin := h08.Pin{Path: *admission, SHA256: *admissionSHA}
	switch *mode {
	case "replay":
		if *out == "" {
			err = fmt.Errorf("-out fresh-attempt required")
			break
		}
		r, err = h08.Replay(h08.Sources{Capture: *capture, Convolution: h08.Pin{Path: *conv, SHA256: *convSHA}, Recurrence: h08.Pin{Path: *rnn, SHA256: *rnnSHA}, Projection: h08.Pin{Path: *projection, SHA256: *projectionSHA}}, *out)
	case "consumer":
		r, err = h08.Consume(ctx, h08.ConsumerOptions{Admission: pin})
	case "cpu13":
		r, err = h08.RunCPU13(pin)
	case "guarded-child":
		r, err = h08.RunGuardedChild(ctx, h08.Pin{Path: *request, SHA256: *requestSHA})
	case "admission-proof":
		var a h08.RuntimeAdmission
		var q h08.FixedRequest
		a, q, err = h08.ReadAdmission(pin)
		r = struct {
			Admission   h08.RuntimeAdmission
			Request     h08.FixedRequest
			RuntimeHeld bool
		}{a, q, true}
	case "backend-proof":
		var a closedbuild.BackendArtifact
		a, err = h08.AuthenticateBackend(closedbuild.ArtifactOptions{Build: h08.Pin{Path: *backendBuild, SHA256: *backendBuildSHA}, Plan: h08.Pin{Path: *backendPlan, SHA256: *backendPlanSHA}, Producer: h08.Pin{Path: *backendProducer, SHA256: *backendProducerSHA}, Output: h08.Pin{Path: *backend, SHA256: *backendSHA}}, *backend)
		if err == nil {
			var exe string
			exe, err = os.Executable()
			if err == nil {
				var self h08.Pin
				self, err = closedbuild.PinFile(exe)
				r = struct {
					Schema             string
					Backend            closedbuild.BackendArtifact
					ConsumerExecutable h08.Pin
					RuntimeHeld        bool
				}{"nanopore-h08-backend-proof/v1", a, self, true}
			}
		}
	default:
		err = fmt.Errorf("unknown H08 mode")
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(runner.ExitCode(err))
	}
	if err = json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
