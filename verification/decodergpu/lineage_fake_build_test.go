package decodergpu

import (
	"context"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func lineageBuildFake(t *testing.T, repo, out string) (File, []runner.CommandReceipt, map[string]string) {
	t.Helper()
	anchor := File{filepath.Join(repo, "artifacts/tensor/d01-lineage-preparation-002/plan.json"), "5b4406eb6563cae7393dc8dd4a8fcb3fd2a5328499d94dde50e43b3324307987"}
	var selected LineagePlan
	if err := readJSON(anchor, &selected); err != nil {
		t.Fatal(err)
	}
	compiler := selected.Recipe.Compiler.Path
	component := lineageCompiler(selected)
	sources := map[string]string{}
	for _, name := range lineageSourcePaths {
		if err := filePin(sources, filepath.Join(repo, name)); err != nil {
			t.Fatal(err)
		}
	}
	fake := filepath.Join(repo, "verification/decodergpu/native/lineage_fake")
	entries, err := os.ReadDir(fake)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			if err = filePin(sources, filepath.Join(fake, entry.Name())); err != nil {
				t.Fatal(err)
			}
		}
	}
	nativeCopy := filepath.Join(out, "native-inputs")
	fakeCopy := filepath.Join(out, "fake-inputs")
	for _, dir := range []string{nativeCopy, fakeCopy} {
		if err = os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	snapshots := map[string]string{}
	for source, want := range sources {
		body, e := os.ReadFile(source)
		if e != nil || digest(body) != want {
			t.Fatal("fake source changed during snapshot", source, e)
		}
		dir := nativeCopy
		if filepath.Dir(source) == fake {
			dir = fakeCopy
		}
		path := filepath.Join(dir, filepath.Base(source))
		if err = os.WriteFile(path, body, 0400); err != nil {
			t.Fatal(err)
		}
		snapshots[path] = want
	}
	oldProbe := filepath.Join(selected.Directory, "native-inputs/lineage_probe.cpp")
	oldInputs := map[string]string{}
	for _, name := range []string{"lineage_probe.cpp", "lineage_observer.inc", "sha256.inc", "runtime_gate.inc"} {
		path := filepath.Join(selected.Directory, "native-inputs", name)
		want, ok := selected.SnapshotPins[path]
		if !ok {
			t.Fatal("old immutable native alias omitted")
		}
		oldInputs[path] = want
	}
	oldObserver, err := os.ReadFile(filepath.Join(selected.Directory, "native-inputs/lineage_observer.inc"))
	if err != nil || !strings.Contains(string(oldObserver), "#include <generated_cuda_meta.h>") {
		t.Fatal("old duplicate-include control no longer selects old source")
	}
	inputs := maps.Clone(selected.Compiler.Inputs)
	maps.Copy(inputs, snapshots)
	maps.Copy(inputs, oldInputs)
	pthread, err := artifactPin("/usr/lib/libpthread.a")
	if err != nil || pthread.Path != "/usr/lib/libpthread.a" || pthread.SHA256 != "f0a17a43c74d2fe5474fa2fd29c8f14799e777d7d75a2cc4d11c20a6e7b161c5" {
		t.Fatal("fake pthread compatibility archive changed", err)
	}
	if err = addPin(inputs, pthread); err != nil {
		t.Fatal(err)
	}
	records := []runner.CommandReceipt{}
	run := func(label string, args []string, wantExit int) runner.CommandReceipt {
		t.Helper()
		if err := verifyCompilerBoundary(component); err != nil {
			t.Fatal(err)
		}
		if err := checkPins(inputs); err != nil {
			t.Fatal(err)
		}
		if err := checkPins(sources); err != nil {
			t.Fatal(err)
		}
		if err := writeJSON(filepath.Join(out, label+"-before-inputs.json"), inputs); err != nil {
			t.Fatal(err)
		}
		rec := runner.CommandReceipt{Args: append([]string{compiler}, args...), Environment: []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC"}, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
		so, e := os.OpenFile(rec.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			t.Fatal(e)
		}
		se, e := os.OpenFile(rec.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			t.Fatal(e)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
		cmd := exec.CommandContext(ctx, compiler, args...)
		cmd.Dir = repo
		cmd.Env = rec.Environment
		cmd.Stdout = so
		cmd.Stderr = se
		e = cmd.Run()
		rec.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if cmd.ProcessState != nil {
			rec.Exit = cmd.ProcessState.ExitCode()
		}
		cancel()
		closeOut, closeErr := so.Close(), se.Close()
		rec.StdoutSHA, _ = runner.SHA(rec.Stdout)
		rec.StderrSHA, _ = runner.SHA(rec.Stderr)
		if err := writeJSON(filepath.Join(out, label+"-command.json"), rec); err != nil {
			t.Fatal(err)
		}
		records = append(records, rec)
		if err := writeJSON(filepath.Join(out, "compiler-progress-"+label+".json"), records); err != nil {
			t.Fatal(err)
		}
		if rec.Exit != wantExit || closeOut != nil || closeErr != nil {
			body, _ := os.ReadFile(rec.Stderr)
			t.Fatalf("fake CXX %s exit%d want%d: %v %s", label, rec.Exit, wantExit, e, body)
		}
		if err := verifyCompilerBoundary(component); err != nil {
			t.Fatal(err)
		}
		if err := checkPins(inputs); err != nil {
			t.Fatal(err)
		}
		return rec
	}
	flags := []string{"-std=c++23", "-O0", "-fno-lto", "-pthread", "-I", fakeCopy}
	preflight := func(label, source string) []string {
		dep := filepath.Join(out, label+"-prospective.d")
		args := append(slices.Clone(flags), "-M", "-MT", "fake-probe", "-MF", dep, source)
		run(label+"-headers", args, 0)
		deps, err := dependencySet(dep, repo)
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range deps {
			if strings.HasPrefix(path, "/opt/cuda/") {
				t.Fatal("fake preflight selected genuine CUDA header", path)
			}
			if err = filePin(inputs, path); err != nil {
				t.Fatal(err)
			}
		}
		if err = writeJSON(filepath.Join(out, label+"-frozen-headers.json"), inputs); err != nil {
			t.Fatal(err)
		}
		return deps
	}
	redDeps := preflight("past-duplicate", oldProbe)
	redDep := filepath.Join(out, "past-duplicate-actual.d")
	redObject := filepath.Join(out, "past-duplicate.o")
	red := run("past-duplicate-compile", append(slices.Clone(flags), "-MD", "-MF", redDep, "-c", oldProbe, "-o", redObject), 1)
	body, err := os.ReadFile(red.Stderr)
	if err != nil || !strings.Contains(string(body), "redefinition of 'struct cuModuleLoadDataEx_params'") {
		t.Fatalf("old duplicate did not fail intended boundary: %v %s", err, body)
	}
	actual, err := dependencySet(redDep, repo)
	if err != nil || !slices.Equal(redDeps, actual) {
		t.Fatal("red fake M/MD dependencies differ", err)
	}
	if _, err = os.Stat(redObject); !os.IsNotExist(err) {
		t.Fatal("red fake compile produced an object")
	}
	objects := map[string]string{}
	objectPaths := []string{}
	for _, unit := range []struct{ name, source string }{{"probe", filepath.Join(nativeCopy, "lineage_probe.cpp")}, {"fake", filepath.Join(fakeCopy, "fake.cpp")}} {
		deps := preflight(unit.name, unit.source)
		dep := filepath.Join(out, unit.name+"-actual.d")
		object := filepath.Join(out, unit.name+".o")
		run(unit.name+"-compile", append(slices.Clone(flags), "-MD", "-MF", dep, "-c", unit.source, "-o", object), 0)
		actual, err := dependencySet(dep, repo)
		if err != nil || !slices.Equal(deps, actual) {
			t.Fatal("fake M/MD dependency equality failed", unit.name, err)
		}
		if err = filePin(objects, object); err != nil {
			t.Fatal(err)
		}
		objectPaths = append(objectPaths, object)
	}
	linkInputs := map[string]string{}
	if err = addPin(linkInputs, pthread); err != nil {
		t.Fatal(err)
	}
	for _, resource := range selected.Recipe.Resources {
		if err = addPin(linkInputs, resource.Expected); err != nil {
			t.Fatal(err)
		}
	}
	resolve := func(name, from string) (string, error) {
		if filepath.IsAbs(name) {
			return name, nil
		}
		if !strings.HasPrefix(name, "-l") {
			path := filepath.Join(filepath.Dir(from), name)
			if info, e := os.Stat(path); e == nil && info.Mode().IsRegular() {
				return path, nil
			}
		}
		p, e := resolveLibrary(name, selected.Recipe.SearchDirs)
		return p.Path, e
	}
	if err = ExpandLinkerScripts(linkInputs, resolve); err != nil {
		t.Fatal(err)
	}
	maps.Copy(linkInputs, selected.Compiler.Inputs)
	maps.Copy(inputs, linkInputs)
	maps.Copy(inputs, objects)
	binary := filepath.Join(out, "lineage-fake-probe")
	linkMap := filepath.Join(out, "fake-link.map")
	args := append(slices.Clone(objectPaths), "-fno-lto", "-pthread", "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
	link := run("fake-link", args, 0)
	trace, err := os.ReadFile(link.Stdout)
	if err != nil {
		t.Fatal(err)
	}
	if err = h07.VerifyLinkTrace(string(trace), repo, linkInputs, objects); err != nil {
		t.Fatal(err)
	}
	mapBody, err := os.ReadFile(linkMap)
	if err != nil {
		t.Fatal(err)
	}
	if err = lineageLinkMap(string(mapBody), repo, linkInputs, objects); err != nil {
		t.Fatal(err)
	}
	metadata, err := nativeproof.InspectELF(binary)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range metadata.Needed {
		for _, bad := range []string{"libcuda", "libcupti", "libkoi", "libtorch", "libcudnn"} {
			if strings.HasPrefix(name, bad) {
				t.Fatal("CPU fake executable needs forbidden GPU/provider", name)
			}
		}
	}
	if err = writeJSON(filepath.Join(out, "compiler-proof.json"), map[string]any{"compiler": selected.Recipe.Compiler, "compiler_family": selected.Compiler, "source_pins": sources, "snapshot_pins": snapshots, "current_input_pins": inputs, "objects": objects, "commands": records, "negative_control": "unchanged old duplicate include rejected by actual fake-header compiler", "fake_only_header_preprocessing": true, "produced_CPU_ELF": metadata}); err != nil {
		t.Fatal(err)
	}
	binaryPin, err := artifactPin(binary)
	if err != nil {
		t.Fatal(err)
	}
	return binaryPin, records, sources
}
