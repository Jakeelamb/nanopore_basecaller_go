package decodergpu

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
)

func TestNumericIdentityFakeHost(t *testing.T) {
	if os.Getenv("D01_NUMERIC_FAKE_HOST") != "1" {
		t.Skip("explicit numeric fake compiler release required")
	}
	root, e := os.Getwd()
	if e != nil {
		t.Fatal(e)
	}
	root = filepath.Clean(filepath.Join(root, "../.."))
	out := os.Getenv("D01_NUMERIC_FAKE_ARTIFACTS")
	if !filepath.IsAbs(out) {
		t.Fatal("retained fresh absolute fake output required")
	}
	if e = os.Mkdir(out, 0700); e != nil {
		t.Fatal(e)
	}
	anchor := File{os.Getenv("D01_NUMERIC_COMPILER_PLAN"), os.Getenv("D01_NUMERIC_COMPILER_PLAN_SHA256")}
	if !filepath.IsAbs(anchor.Path) || anchor.SHA256 != "2f0af46b5a1ab633606af332cf06d8c18c363c660d2e5bd0e79e381051452a42" {
		t.Fatal("accepted compiler custody anchor required")
	}
	var selected LineagePlan
	if e = readJSON(anchor, &selected); e != nil {
		t.Fatal(e)
	}
	component := lineageCompiler(selected)
	sources := map[string]string{}
	for _, name := range []string{"native/fake/numeric_identity.cpp", "native/numeric_identity.inc", "native/sha256.inc"} {
		if e = filePin(sources, filepath.Join(root, "verification/decodergpu", name)); e != nil {
			t.Fatal(e)
		}
	}
	if e = filePin(sources, "/proc/driver/nvidia/version"); e != nil {
		t.Fatal(e)
	}
	inputs := maps.Clone(selected.Compiler.Inputs)
	for source, sha := range sources {
		if strings.HasPrefix(source, root+"/") {
			rel, _ := filepath.Rel(filepath.Join(root, "verification/decodergpu"), source)
			target := filepath.Join(out, rel)
			if e = os.MkdirAll(filepath.Dir(target), 0700); e != nil {
				t.Fatal(e)
			}
			raw, err := os.ReadFile(source)
			if err != nil || digest(raw) != sha {
				t.Fatal("fake source drift", err)
			}
			if e = os.WriteFile(target, raw, 0400); e != nil {
				t.Fatal(e)
			}
			inputs[target] = sha
		} else {
			inputs[source] = sha
		}
	}
	links := map[string]string{}
	pthread := File{Path: "/usr/lib/libpthread.a", SHA256: "f0a17a43c74d2fe5474fa2fd29c8f14799e777d7d75a2cc4d11c20a6e7b161c5"}
	if e = addPin(links, pthread); e != nil {
		t.Fatal(e)
	}
	for _, resource := range selected.Recipe.Resources {
		if e = addPin(links, resource.Expected); e != nil {
			t.Fatal(e)
		}
	}
	resolve := func(name, from string) (string, error) {
		if filepath.IsAbs(name) {
			return name, nil
		}
		if !strings.HasPrefix(name, "-l") {
			p := filepath.Join(filepath.Dir(from), name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
		p, err := resolveLibrary(name, selected.Recipe.SearchDirs)
		return p.Path, err
	}
	if e = ExpandLinkerScripts(links, resolve); e != nil {
		t.Fatal(e)
	}
	maps.Copy(links, selected.Compiler.Inputs)
	maps.Copy(inputs, links)
	if e = writeJSON(filepath.Join(out, "source-before.json"), sources); e != nil {
		t.Fatal(e)
	}
	run := func(t *testing.T, label string, args []string, want int) {
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
		if err := writeJSON(filepath.Join(out, label+"-inputs.json"), inputs); err != nil {
			t.Fatal(err)
		}
		receipt, err := executeNative(root, out, label, args, []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC"})
		if save := writeJSON(filepath.Join(out, label+"-command.json"), receipt); save != nil {
			t.Fatal(save)
		}
		if receipt.Exit != want {
			t.Fatal(label, receipt.Exit, err)
		}
		if err := checkPins(inputs); err != nil {
			t.Fatal(err)
		}
		if err := checkPins(sources); err != nil {
			t.Fatal(err)
		}
		if err := verifyCompilerBoundary(component); err != nil {
			t.Fatal(err)
		}
	}
	compiler := selected.Recipe.Compiler.Path
	source := filepath.Join(out, "native/fake/numeric_identity.cpp")
	before := filepath.Join(out, "before.d")
	actual := filepath.Join(out, "actual.d")
	object := filepath.Join(out, "identity.o")
	binary := filepath.Join(out, "numeric-fake")
	linkMap := filepath.Join(out, "link.map")
	flags := []string{compiler, "-std=c++20", "-O0", "-fno-lto", "-pthread"}
	run(t, "headers", append(slices.Clone(flags), "-M", "-MT", "numeric-fake", "-MF", before, source), 0)
	deps, err := dependencySet(before, root)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range deps {
		if strings.HasPrefix(path, "/opt/cuda/") {
			t.Fatal("genuine CUDA header selected", path)
		}
		if e = filePin(inputs, path); e != nil {
			t.Fatal(e)
		}
	}
	if e = writeJSON(filepath.Join(out, "before-compile.json"), inputs); e != nil {
		t.Fatal(e)
	}
	run(t, "compile", append(slices.Clone(flags), "-MD", "-MF", actual, "-c", source, "-o", object), 0)
	got, err := dependencySet(actual, root)
	if err != nil || !slices.Equal(deps, got) {
		t.Fatal("fake M/MD differ", err)
	}
	objects := map[string]string{}
	if e = filePin(objects, object); e != nil {
		t.Fatal(e)
	}
	maps.Copy(inputs, objects)
	run(t, "link", []string{compiler, object, "-fno-lto", "-pthread", "-Wl,-t", "-Wl,-Map=" + linkMap, "-o", binary}, 0)
	trace, err := os.ReadFile(filepath.Join(out, "link.stdout"))
	if err != nil {
		t.Fatal(err)
	}
	if e = h07.VerifyLinkTrace(string(trace), root, links, objects); e != nil {
		t.Fatal(e)
	}
	rawMap, err := os.ReadFile(linkMap)
	if err != nil {
		t.Fatal(err)
	}
	if e = lineageLinkMap(string(rawMap), root, links, objects); e != nil {
		t.Fatal(e)
	}
	metadata, err := nativeproof.InspectELF(binary)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range metadata.Needed {
		for _, bad := range []string{"libcuda", "libcupti", "libkoi", "libtorch"} {
			if strings.HasPrefix(name, bad) {
				t.Fatal("fake links GPU provider", name)
			}
		}
	}
	if e = filePin(inputs, binary); e != nil {
		t.Fatal(e)
	}
	for _, mode := range []string{"positive", "context-uid", "context-device", "driver-query", "close-failure"} {
		t.Run(mode, func(t *testing.T) {
			dir := filepath.Join(out, mode)
			want := 1
			if mode == "positive" {
				want = 0
			}
			run(t, "case-"+mode, []string{binary, mode, dir}, want)
			path := filepath.Join(dir, "identity-decode.json")
			if want == 0 {
				var row NumericIdentity
				pin, err := artifactPin(path)
				if err != nil {
					t.Fatal(err)
				}
				if err = readJSON(pin, &row); err != nil || row.PID <= 0 || row.UUID != strings.Repeat("1", 32) || len(row.Contexts) != 1 || row.Contexts[0].UID != 9 || !equalEnvironment(row.Environment, []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC"}) {
					t.Fatal("fake collector lost identity", row, err)
				}
			} else if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatal("failed collector published identity", err)
			}
		})
	}
	if e = writeJSON(filepath.Join(out, "proof.json"), map[string]any{"source": sources, "inputs": inputs, "compiler": selected.Compiler, "dependencies": deps, "objects": objects, "ELF": metadata, "cases": 5, "description": fmt.Sprint("CPU fake collector only; no CUDA headers/DSOs/device")}); e != nil {
		t.Fatal(e)
	}
}
