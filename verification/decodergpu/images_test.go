package decodergpu

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func imageFixture(t *testing.T) (ImagePolicy, []ObservedModule, []ObservedFunction, []ObservedLaunch) {
	t.Helper()
	payload := testFile(t, t.TempDir(), "original.cubin", "host-only fake original image")
	image := DeviceImage{"host-fake", payload, map[string]uint32{}}
	functions, launches := []ObservedFunction{}, []ObservedLaunch{}
	for i, name := range decoderSymbols {
		stage := uint64(i + 1)
		image.Symbols[name] = uint32(i + 10)
		functions = append(functions, ObservedFunction{stage, 9, 7, uint64(i + 10), stage, name})
		for _, site := range []uint64{0, 1} {
			launches = append(launches, ObservedLaunch{1, 9, stage, 307, site, stage, 0, name})
		}
	}
	return ImagePolicy{Schema: "host-fake", Images: []DeviceImage{image}}, []ObservedModule{{0, 1, 7, 0, 29, 29, payload}}, functions, launches
}

func TestSelectedDecoderImageRequiresExactModuleFunctionAndLaunch(t *testing.T) {
	p, modules, functions, launches := imageFixture(t)
	if b, err := bindSelectedImages(modules, functions, launches, p); err != nil || len(b) != 4 {
		t.Fatalf("complete host-only binding: %v %v", b, err)
	}
	for _, name := range []string{"image", "context", "module", "index", "index-overflow", "launch-error", "launch-name", "duplicate-function", "empty-functions", "unpaired-launch"} {
		t.Run(name, func(t *testing.T) {
			m, f, l := slices.Clone(modules), slices.Clone(functions), slices.Clone(launches)
			switch name {
			case "image":
				m[0].Payload.SHA256 = strings.Repeat("0", 64)
			case "context":
				f[0].Context++
			case "module":
				f[0].Module++
			case "index":
				f[0].Index++
			case "index-overflow":
				f[0].Index += 1 << 32
			case "launch-error":
				l[1].Result = 1
			case "launch-name":
				l[0].Name = "other_kernel"
			case "duplicate-function":
				f = append(f, f[0])
			case "empty-functions":
				f = nil
			case "unpaired-launch":
				l = l[:7]
			}
			if _, err := bindSelectedImages(m, f, l, p); err == nil {
				t.Fatal("unqualified selected-code binding accepted")
			}
		})
	}
}

func TestImageRawCustodyAndNarrowModuleOnlyFailure(t *testing.T) {
	p, modules, functions, launches := imageFixture(t)
	dir := t.TempDir()
	bytes, err := os.ReadFile(modules[0].Payload.Path)
	if err != nil {
		t.Fatal(err)
	}
	modules[0].Observed, modules[0].Retained = uint64(len(bytes)), uint64(len(bytes))
	if err = os.WriteFile(filepath.Join(dir, "module-0.cubin"), bytes, 0600); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string) {
		t.Helper()
		if e := os.WriteFile(filepath.Join(dir, name), []byte(body), 0600); e != nil {
			t.Fatal(e)
		}
	}
	write("modules.tsv", fmt.Sprintf("0\t1\t7\t0\t%d\t%d\t%s\tmodule-0.cubin\n", len(bytes), len(bytes), modules[0].Payload.SHA256))
	var fs, ls strings.Builder
	for _, f := range functions {
		fmt.Fprintf(&fs, "%d\t%d\t%d\t%d\t%d\t%s\n", f.ID, f.Context, f.Module, f.Index, f.DeliveredStage, f.Name)
	}
	for _, l := range launches {
		fmt.Fprintf(&ls, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n", l.Context, l.UID, l.Correlation, l.Callback, l.Site, l.Stage, l.Result, l.Name)
	}
	write("functions.tsv", fs.String())
	write("launches.tsv", ls.String())
	write("status.txt", "callback_error=0\nmodules=1\nfunctions=4\nlaunches=8\n")
	if r, e := ReadImageReport(dir, p); e != nil || len(r.Bindings) != 4 || len(r.Files) != 5 || r.ModuleOnly {
		t.Fatalf("retained complete binding: %+v %v", r, e)
	}
	write("functions.tsv", "")
	write("status.txt", "callback_error=0\nmodules=1\nfunctions=0\nlaunches=8\n")
	if r, e := ReadImageReport(dir, p); e == nil || !r.ModuleOnly || len(r.Modules) != 1 || r.Status != "failed" {
		t.Fatalf("module-only collection qualified: %+v %v", r, e)
	}
	write("module-0.cubin", "changed image")
	if _, e := ReadImageReport(dir, p); e == nil {
		t.Fatal("changed retained image accepted")
	}
}

func TestProducerCopiesSurviveLiveDriftAndRejectCopyDrift(t *testing.T) {
	dir := t.TempDir()
	original := testFile(t, dir, "oracle.cpp", "original native producer")
	bindings, copies, err := snapshotPins(map[string]string{original.Path: original.SHA256}, filepath.Join(dir, "native-inputs"))
	if err != nil {
		t.Fatal(err)
	}
	named := filepath.Join(dir, "native-inputs/oracle.cpp")
	if err = os.WriteFile(named, []byte("original native producer"), 0400); err != nil {
		t.Fatal(err)
	}
	copies[named] = original.SHA256
	p := ClosedBuildPlan{Directory: dir, SourceBindings: bindings, SnapshotPins: copies}
	if err = os.WriteFile(original.Path, []byte("new observer consumer source"), 0600); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]string{original.Path: original.SHA256}
	if err = checkProducerInputs(inputs, p, true); err == nil {
		t.Fatal("live source drift accepted as current")
	}
	if err = checkProducerInputs(inputs, p, false); err != nil {
		t.Fatal("immutable producer rejected later source drift", err)
	}
	copy := bindings[original.Path]
	if err = os.Chmod(copy, 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(copy, []byte("forged producer copy"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = checkProducerInputs(inputs, p, false); err == nil {
		t.Fatal("changed immutable producer accepted")
	}
}
