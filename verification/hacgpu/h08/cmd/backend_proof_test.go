package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/buildinfo"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func TestBackendProofProducerPlanner(t *testing.T) {
	path, digest := os.Getenv("H08_BACKEND_PROOF_FIXTURE"), os.Getenv("H08_BACKEND_PROOF_FIXTURE_SHA256")
	if path == "" && digest == "" {
		t.Skip("genuine producer/planner fixture absent; no native acceptance claimed")
	}
	if !filepath.IsAbs(path) || len(digest) != 64 {
		t.Fatal("absolute H08_BACKEND_PROOF_FIXTURE and independent SHA256 required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != digest {
		t.Fatal("fixture SHA256 differs")
	}
	var f struct {
		CLI                    closedbuild.Pin
		Backend                closedbuild.ArtifactOptions
		Root, GoRoot, WorkRoot string
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&f); err != nil {
		t.Fatal(err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		t.Fatal("fixture must contain one JSON object")
	}
	for _, dir := range []string{f.Root, f.GoRoot, f.WorkRoot} {
		if !filepath.IsAbs(dir) {
			t.Fatal("absolute Root, GoRoot and allowed WorkRoot required")
		}
	}
	for _, p := range []closedbuild.Pin{f.CLI, f.Backend.Build, f.Backend.Plan, f.Backend.Producer, f.Backend.Output} {
		requireBackendProofPin(t, p)
	}
	info, err := buildinfo.ReadFile(f.CLI.Path)
	if err != nil {
		t.Fatal(err)
	}
	cgo0 := false
	for _, setting := range info.Settings {
		if setting.Key == "CGO_ENABLED" {
			cgo0 = setting.Value == "0"
		}
	}
	if info.Path != cgobuild.Target || !cgo0 {
		t.Fatal("supplied executable must be the independently built CGO0 H08 CLI")
	}
	attempt, err := os.MkdirTemp(f.WorkRoot, "h08-backend-proof-wire-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained producer/planner attempt: %s", attempt)
	stdoutPath := filepath.Join(attempt, "backend-proof.stdout")
	stdout, err := os.OpenFile(stdoutPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer stdout.Close()
	stderr, err := os.OpenFile(filepath.Join(attempt, "backend-proof.stderr"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer stderr.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, f.CLI.Path,
		"-mode", "backend-proof",
		"-backend", f.Backend.Output.Path, "-backend-sha", f.Backend.Output.SHA256,
		"-backend-build", f.Backend.Build.Path, "-backend-build-sha", f.Backend.Build.SHA256,
		"-backend-plan", f.Backend.Plan.Path, "-backend-plan-sha", f.Backend.Plan.SHA256,
		"-backend-producer", f.Backend.Producer.Path, "-backend-producer-sha", f.Backend.Producer.SHA256)
	cmd.Dir = f.Root
	cmd.Env = []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB"}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.WaitDelay = 5 * time.Second
	requireBackendProofPin(t, f.CLI)
	if err := cmd.Run(); err != nil {
		t.Fatalf("backend-proof producer: %v (context: %v); retained stderr: %s", err, ctx.Err(), stderr.Name())
	}
	if err := stdout.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stderr.Close(); err != nil {
		t.Fatal(err)
	}
	requireBackendProofPin(t, f.CLI)
	produced, err := os.ReadFile(stdoutPath)
	if err != nil {
		t.Fatal(err)
	}
	proofPin, err := closedbuild.PinFile(stdoutPath)
	if err != nil {
		t.Fatal(err)
	}
	o := cgobuild.Options{Role: cgobuild.H08Command, Root: f.Root, GoRoot: f.GoRoot,
		WorkRoot: filepath.Join(attempt, "planner-work"), CompilerPlan: f.Backend.Plan, BackendProof: proofPin}
	plan, err := cgobuild.Prepare(o)
	if err != nil {
		t.Fatalf("cgobuild.Prepare rejected verbatim backend-proof stdout: %v; retained proof: %s (SHA256 %s)", err, proofPin.Path, proofPin.SHA256)
	}
	if plan.Backend != f.Backend {
		t.Fatal("planner selected different backend anchors")
	}
	var proof struct {
		Schema             string
		Backend            closedbuild.BackendArtifact
		ConsumerExecutable closedbuild.Pin
		RuntimeHeld        bool
	}
	d = json.NewDecoder(bytes.NewReader(produced))
	d.DisallowUnknownFields()
	if err := d.Decode(&proof); err != nil {
		t.Fatal(err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		t.Fatal("producer must emit one JSON object")
	}
	selected := closedbuild.ArtifactOptions{Build: proof.Backend.Build, Plan: proof.Backend.Plan, Producer: proof.Backend.Producer, Output: proof.Backend.Output}
	if proof.Schema != "nanopore-h08-backend-proof/v1" || proof.Backend.Schema != "nanopore-tensor-backend-artifact/v1" ||
		selected != f.Backend || proof.ConsumerExecutable != f.CLI || !proof.RuntimeHeld || !proof.Backend.RuntimeHeld {
		t.Fatal("producer envelope schema, backend anchors, self executable or runtime hold differs")
	}
	var changed map[string]json.RawMessage
	if err := json.Unmarshal(produced, &changed); err != nil {
		t.Fatal(err)
	}
	changed["Schema"] = json.RawMessage(`"nanopore-h08-backend-proof/invalid"`)
	control, err := json.Marshal(changed)
	if err != nil {
		t.Fatal(err)
	}
	controlPath := filepath.Join(attempt, "changed-schema.json")
	if err := os.WriteFile(controlPath, control, 0600); err != nil {
		t.Fatal(err)
	}
	o.BackendProof, err = closedbuild.PinFile(controlPath)
	if err != nil {
		t.Fatal(err)
	}
	o.WorkRoot = filepath.Join(attempt, "changed-schema-work")
	if _, err := cgobuild.Prepare(o); err == nil || err.Error() != "authentic typed backend proof required" {
		t.Fatalf("changed schema control: want authentic typed backend proof required, got %v", err)
	}
}

func requireBackendProofPin(t *testing.T, want closedbuild.Pin) {
	t.Helper()
	got, err := closedbuild.PinFile(want.Path)
	if err != nil || !filepath.IsAbs(want.Path) || len(want.SHA256) != 64 || got != want {
		t.Fatalf("independent fixture pin differs: %s: got %+v: %v", want.Path, got, err)
	}
}
