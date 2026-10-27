package hacvcsaux

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const NativeInputsSHA256 = "5532c32a6b6ff13238e563a0ce4ec873a7018300f70abf5848195210c77dcf4d"

type NativeEvidence struct {
	Schema, Status, Selection, MatrixSHA256, InputsSHA256, CompilerBaselineSHA256 string
	Oracle                                                                        closedbuild.Pin
	Build                                                                         closedbuild.Pin
	Inputs                                                                        closedbuild.Pin
	Files                                                                         map[string]string
}

type OracleBuild struct {
	Schema, Status, CompilerBaselineSHA256 string
	Output                                 closedbuild.Pin
	Inputs, Objects                        map[string]string
	OriginalObjects, SourcePins            map[string]closedbuild.Pin
}

type Comparison struct {
	Schema, Selection, MatrixSHA256, OutputWordRejection string
	Evidence, Oracle                                     closedbuild.Pin
	Cases, Runs, ComparedWords                           int
	WorkspaceInitialized, WorkspaceUntouched             map[string]int
}

type captureMetadata struct {
	Schema            string `json:"schema"`
	Case              string `json:"case"`
	MatrixSHA256      string `json:"matrix_sha256"`
	Workers           int    `json:"workers"`
	WorkspaceFill     int32  `json:"workspace_fill"`
	DestinationFill   int32  `json:"destination_fill"`
	BatchSize         int    `json:"batch_size"`
	FullTLSTM         int    `json:"full_t_lstm"`
	Status            int    `json:"status"`
	PoolCreateStatus  int    `json:"thread_pool_create_status"`
	PoolDestroyStatus int    `json:"thread_pool_destroy_status"`
	CopyCount         int    `json:"copy_count"`
	CopyKind          int    `json:"copy_kind"`
	Stream            uint64 `json:"stream"`
	InputUnchanged    bool   `json:"input_unchanged"`
	WorkspaceGuardsOK bool   `json:"workspace_guards_ok"`
	SourceOffsets     []int  `json:"source_workspace_offsets"`
	Copies            []struct {
		Name          string `json:"name"`
		Words         int    `json:"words"`
		Capacity      int    `json:"capacity"`
		CapturedWords int    `json:"captured_words"`
		GuardsOK      bool   `json:"guards_ok"`
	} `json:"copies"`
}

type captureReader struct {
	root  string
	files map[string]string
	used  map[string]bool
}

func (r *captureReader) read(name string, size int) ([]byte, error) {
	if filepath.IsAbs(name) || filepath.Clean(name) != name || name == ".." || strings.HasPrefix(name, "../") || r.files[name] == "" {
		return nil, fmt.Errorf("capture file is outside independent evidence: %s", name)
	}
	path := filepath.Join(r.root, name)
	actual, err := filepath.EvalSymlinks(path)
	if err != nil || actual != path {
		return nil, fmt.Errorf("capture file alias differs: %s", name)
	}
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || size >= 0 && info.Size() != int64(size) || size < 0 && info.Size() > 2<<20 {
		return nil, fmt.Errorf("capture file length or type differs: %s", name)
	}
	sha, err := runner.SHA(path)
	if err != nil || sha != r.files[name] {
		return nil, fmt.Errorf("capture file pin differs: %s", name)
	}
	r.used[name] = true
	return os.ReadFile(path)
}

func (r *captureReader) words(name string, count int) ([]int32, error) {
	body, err := r.read(name, 4*count)
	if err != nil {
		return nil, err
	}
	out := make([]int32, count)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(body[4*i:]))
	}
	return out, nil
}

func (r *captureReader) json(name string, result any) error {
	body, err := r.read(name, -1)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, result)
}

func compareWords(name string, goWords, nativeWords []int32) error {
	if len(goWords) != len(nativeWords) {
		return fmt.Errorf("%s length Go=%d native=%d", name, len(goWords), len(nativeWords))
	}
	for i := range goWords {
		if goWords[i] != nativeWords[i] {
			return fmt.Errorf("%s[%d] Go=%d native=%d", name, i, goWords[i], nativeWords[i])
		}
	}
	return nil
}

func checkGuards(name string, guarded, payload []int32) error {
	if len(guarded) != len(payload)+64 {
		return fmt.Errorf("%s guard length differs", name)
	}
	for i := 0; i < 32; i++ {
		if guarded[i] != 0x6b497d31 || guarded[len(guarded)-1-i] != 0x6b497d31 {
			return fmt.Errorf("%s canary changed at %d", name, i)
		}
	}
	return compareWords(name+" guarded payload", payload, guarded[32:len(guarded)-32])
}

func compareMaps(body []byte, allowed map[string]string, loaderCache closedbuild.Pin) error {
	seen := map[string]bool{}
	for _, line := range strings.Split(string(body), "\n") {
		fields := strings.Fields(line)
		if len(fields) <= 5 || strings.HasPrefix(fields[5], "[") {
			continue
		}
		if len(fields) != 6 || !filepath.IsAbs(fields[5]) {
			return fmt.Errorf("unexpected native mapping: %s", line)
		}
		path, err := filepath.EvalSymlinks(fields[5])
		if err != nil {
			return fmt.Errorf("unapproved native mapping: %s", fields[5])
		}
		expected, required := allowed[path]
		if path == loaderCache.Path {
			if fields[5] != loaderCache.Path || fields[1] != "r--p" && fields[1] != "r--s" || loaderCache.SHA256 == "" {
				return fmt.Errorf("loader cache mapping must be the pinned read-only non-executable file: %s", line)
			}
			expected = loaderCache.SHA256
		} else if !required {
			return fmt.Errorf("unapproved native mapping: %s", fields[5])
		}
		sha, err := runner.SHA(path)
		if err != nil || sha != expected {
			return fmt.Errorf("mapped native file changed: %s", path)
		}
		if required {
			seen[path] = true
		}
	}
	for path := range allowed {
		if !seen[path] {
			return fmt.Errorf("closed native file not observed in maps: %s", path)
		}
	}
	return nil
}

func CompareNative(evidencePath, evidenceSHA string) (Comparison, error) {
	var result Comparison
	pin, err := closedbuild.PinFile(evidencePath)
	if err != nil || evidenceSHA == "" || pin.SHA256 != evidenceSHA {
		return result, fmt.Errorf("independent native evidence pin differs")
	}
	body, err := os.ReadFile(pin.Path)
	if err != nil {
		return result, err
	}
	var evidence NativeEvidence
	if err := json.Unmarshal(body, &evidence); err != nil {
		return result, err
	}
	wantObjects := map[string]string{"utils_lstm.cpp.o": "7598a9c75f9b4d3f3936e3871a31155a8e4b63c417017b7f5b2132db96051243", "utils_lstm.cu.o": "ce998c4411202c7c1c1d6240bad5677dfb8c1d69118355bf9affef6170045117", "thread_pool.cpp.o": "89cdf4ea0968f38e557787203b8a8b180dcac74c6df077bf89bf18d837b13f2d"}
	if evidence.Schema != "hac-vcs-aux-native-evidence/v1" || evidence.Status != "captured_public_host_cpu" || evidence.MatrixSHA256 != MatrixSHA256 || evidence.InputsSHA256 != NativeInputsSHA256 || evidence.CompilerBaselineSHA256 != CompilerBaselineSHA256 || evidence.Selection != "first" && evidence.Selection != "all" {
		return result, fmt.Errorf("native producer identity or frozen inputs differ")
	}
	inputPin, err := closedbuild.PinFile(evidence.Inputs.Path)
	if err != nil || inputPin != evidence.Inputs || inputPin.SHA256 != NativeInputsSHA256 {
		return result, fmt.Errorf("native runtime input pin differs")
	}
	buildPin, err := closedbuild.PinFile(evidence.Build.Path)
	if err != nil || buildPin != evidence.Build {
		return result, fmt.Errorf("independent native build pin differs")
	}
	buildBody, err := os.ReadFile(buildPin.Path)
	if err != nil {
		return result, err
	}
	var build OracleBuild
	if err := json.Unmarshal(buildBody, &build); err != nil {
		return result, err
	}
	if build.Schema != "hac-vcs-aux-build/v1" || build.Status != "built_outputs_unexecuted" || build.CompilerBaselineSHA256 != CompilerBaselineSHA256 || build.Output != evidence.Oracle || len(build.OriginalObjects) != len(wantObjects) || len(build.Inputs) == 0 || len(build.Objects) != 2 {
		return result, fmt.Errorf("native build producer contract differs")
	}
	for name, expectedSHA := range wantObjects {
		original := build.OriginalObjects[name]
		actual, err := closedbuild.PinFile(original.Path)
		if err != nil || actual != original || original.SHA256 != expectedSHA || build.Inputs[actual.Path] != expectedSHA {
			return result, fmt.Errorf("original native object differs: %s", name)
		}
	}
	for path, expectedSHA := range build.Inputs {
		actual, err := closedbuild.PinFile(path)
		if err != nil || actual.SHA256 != expectedSHA {
			return result, fmt.Errorf("declared native build input changed: %s", path)
		}
	}
	for path, expectedSHA := range build.Objects {
		actual, err := closedbuild.PinFile(path)
		if err != nil || actual.SHA256 != expectedSHA {
			return result, fmt.Errorf("retained native compilation object changed: %s", path)
		}
	}
	for _, name := range []string{"oracle.cpp", "transport.cpp", "capture.h"} {
		path, err := filepath.Abs(filepath.Join("verification/hacvcsaux/native", name))
		if err != nil {
			return result, err
		}
		recorded := build.SourcePins[name]
		original, err := closedbuild.PinFile(recorded.Path)
		if err != nil || original != recorded {
			return result, fmt.Errorf("retained native producer source changed: %s", name)
		}
		sha, err := runner.SHA(path)
		if err != nil || recorded.SHA256 != sha || build.Inputs[recorded.Path] != sha {
			return result, fmt.Errorf("native producer source pin differs: %s", path)
		}
	}
	oracle, err := InspectOracle(evidence.Oracle.Path, evidence.Oracle.SHA256)
	if err != nil {
		return result, err
	}
	compiler, err := CheckCompiler()
	if err != nil {
		return result, err
	}
	loaderCache := compiler.Compiler.LoaderCache
	if loaderCache.Path != "/etc/ld.so.cache" || loaderCache.SHA256 == "" {
		return result, fmt.Errorf("independent compiler loader-cache identity differs")
	}
	reader := captureReader{root: filepath.Dir(pin.Path), files: evidence.Files, used: map[string]bool{}}
	allowed := map[string]string{}
	for _, input := range oracle.Closure.Inputs {
		allowed[input.Path] = input.SHA256
	}
	for _, name := range []string{"maps-before", "maps-after"} {
		body, err := reader.read(name, -1)
		if err != nil {
			return result, err
		}
		if err := compareMaps(body, allowed, loaderCache); err != nil {
			return result, err
		}
	}
	matrix, err := ReadMatrix()
	if err != nil {
		return result, err
	}
	cases := matrix.Cases
	if evidence.Selection == "first" {
		cases = cases[:1]
	}
	var transport struct {
		Schema                                                                                             string
		FatBinary, Functions, Variables, End, Unregister, Copies, ExpectedCopies, ErrorStrings, Unexpected int
		Complete                                                                                           bool
	}
	transportBody, err := reader.read("transport-final.json", -1)
	if err != nil {
		return result, err
	}
	var rawTransport map[string]json.RawMessage
	if err := json.Unmarshal(transportBody, &rawTransport); err != nil {
		return result, err
	}
	for key, target := range map[string]any{"schema": &transport.Schema, "fat_binary": &transport.FatBinary, "functions": &transport.Functions, "variables": &transport.Variables, "end": &transport.End, "unregister": &transport.Unregister, "copies": &transport.Copies, "expected_copies": &transport.ExpectedCopies, "error_strings": &transport.ErrorStrings, "unexpected": &transport.Unexpected, "complete": &transport.Complete} {
		if err := json.Unmarshal(rawTransport[key], target); err != nil {
			return result, fmt.Errorf("constructor transport evidence %s: %w", key, err)
		}
	}
	if transport.Schema != "hac-vcs-aux-transport/v1" || transport.FatBinary != 1 || transport.Functions != 6 || transport.Variables != 24 || transport.End != 1 || transport.Unregister != 1 || transport.Copies != 16*len(cases) || transport.ExpectedCopies != transport.Copies || transport.ErrorStrings != 0 || transport.Unexpected != 0 || !transport.Complete {
		return result, fmt.Errorf("constructor, transport, or unexpected-device-call control differs")
	}
	result = Comparison{Schema: "hac-vcs-aux-comparison/v1", Selection: evidence.Selection, MatrixSHA256: MatrixSHA256, Evidence: pin, Oracle: evidence.Oracle, Cases: len(cases), WorkspaceInitialized: map[string]int{}, WorkspaceUntouched: map[string]int{}}
	for _, fixture := range cases {
		if err := compareFixture(&reader, matrix, fixture, &result); err != nil {
			return result, fmt.Errorf("%s: %w", fixture.ID, err)
		}
	}
	if len(reader.used) != len(evidence.Files) {
		return result, fmt.Errorf("native evidence includes undeclared or unread capture files")
	}
	for name, expectedSHA := range evidence.Files {
		sha, err := runner.SHA(filepath.Join(reader.root, name))
		if err != nil || sha != expectedSHA {
			return result, fmt.Errorf("native capture changed during comparison: %s", name)
		}
	}
	for _, inputs := range []map[string]string{build.Inputs, build.Objects, {pin.Path: pin.SHA256, inputPin.Path: inputPin.SHA256, buildPin.Path: buildPin.SHA256}} {
		for path, expectedSHA := range inputs {
			sha, err := runner.SHA(path)
			if err != nil || sha != expectedSHA {
				return result, fmt.Errorf("native evidence input changed during comparison: %s", path)
			}
		}
	}
	return result, nil
}

func compareFixture(reader *captureReader, matrix Matrix, fixture Fixture, result *Comparison) error {
	aux, err := fixture.Aux(matrix.InputStride)
	if err != nil {
		return err
	}
	packing, err := aux.PackLSTM()
	if err != nil {
		return err
	}
	workspaceCount, err := aux.Workspace()
	if err != nil {
		return err
	}
	withSentinels := len(packing.InLayout) + 2*len(aux.SizesOut())
	full := len(packing.ForwardEncoding)
	back := len(packing.BackwardEncoding)
	initialized := 2*withSentinels + 2*full + back
	expectedOffsets := []int{2*withSentinels + full + back, withSentinels, withSentinels + full, 2*withSentinels + full}
	names := []string{"OutLayout", "ForwardEncoding", "InLayout", "BackwardEncoding"}
	expected := [][]int32{packing.OutLayout, packing.ForwardEncoding, packing.InLayout, packing.BackwardEncoding}
	capacities := []int{packing.OutLayoutCapacity, full, len(packing.InLayout), back}
	input := make([]int32, len(aux.SizesOut()))
	for i, size := range aux.SizesOut() {
		input[i] = int32(size)
	}
	for _, workers := range matrix.ThreadPoolWorkers {
		var firstWorkspace []int32
		for fill := range matrix.WorkspaceFills {
			run := fmt.Sprintf("%s-pool%d-fill%d", fixture.ID, workers, fill)
			var metadata captureMetadata
			if err := reader.json(run+"/capture.json", &metadata); err != nil {
				return err
			}
			if metadata.Schema != "hac-vcs-aux-capture/v1" || metadata.Case != fixture.ID || metadata.MatrixSHA256 != MatrixSHA256 || metadata.Workers != workers || metadata.WorkspaceFill != matrix.WorkspaceFills[fill] || metadata.DestinationFill != matrix.DestinationFills[fill] || metadata.BatchSize != fixture.BatchSize || metadata.FullTLSTM != aux.TLSTM() || metadata.Status != 0 || metadata.PoolCreateStatus != 0 || metadata.PoolDestroyStatus != 0 || metadata.CopyCount != 4 || metadata.CopyKind != 1 || metadata.Stream != 0x456780 || !metadata.InputUnchanged || !metadata.WorkspaceGuardsOK || !reflect.DeepEqual(metadata.SourceOffsets, expectedOffsets) || len(metadata.Copies) != 4 {
				return fmt.Errorf("%s public-call metadata differs", run)
			}
			workspace, err := reader.words(run+"/workspace-after.i32", workspaceCount)
			if err != nil {
				return err
			}
			guarded, err := reader.words(run+"/workspace-guarded.i32", workspaceCount+64)
			if err != nil {
				return err
			}
			if err := checkGuards(run+" workspace", guarded, workspace); err != nil {
				return err
			}
			if fill == 0 {
				firstWorkspace = workspace
			} else {
				for i := range workspace {
					if i < initialized && firstWorkspace[i] != workspace[i] || i >= initialized && (firstWorkspace[i] != matrix.WorkspaceFills[0] || workspace[i] != matrix.WorkspaceFills[1]) {
						return fmt.Errorf("%s workspace initialization mask differs at %d", run, i)
					}
				}
				result.WorkspaceInitialized[fmt.Sprintf("%s-pool%d", fixture.ID, workers)] = initialized
				result.WorkspaceUntouched[fmt.Sprintf("%s-pool%d", fixture.ID, workers)] = workspaceCount - initialized
			}
			for _, name := range []string{"input-before.i32", "input-after.i32"} {
				words, err := reader.words(run+"/"+name, len(input))
				if err != nil {
					return err
				}
				if err := compareWords(run+" "+name, input, words); err != nil {
					return err
				}
			}
			for i, name := range names {
				copy := metadata.Copies[i]
				if copy.Name != name || copy.Words != len(expected[i]) || copy.Capacity != capacities[i] || copy.CapturedWords != len(expected[i]) || !copy.GuardsOK {
					return fmt.Errorf("%s upload %d shape/order differs", run, i)
				}
				words, err := reader.words(run+"/"+name+".i32", len(expected[i]))
				if err != nil {
					return err
				}
				if err := compareWords(run+" "+name, expected[i], words); err != nil {
					return err
				}
				if err := compareWords(run+" "+name+" source", words, workspace[expectedOffsets[i]:expectedOffsets[i]+len(words)]); err != nil {
					return err
				}
				destination, err := reader.words(run+"/"+name+"-destination.i32", capacities[i])
				if err != nil {
					return err
				}
				if err := compareWords(run+" "+name+" transported prefix", words, destination[:len(words)]); err != nil {
					return err
				}
				for _, word := range destination[len(words):] {
					if word != matrix.DestinationFills[fill] {
						return fmt.Errorf("%s %s untouched tail changed", run, name)
					}
				}
				guarded, err := reader.words(run+"/"+name+"-guarded.i32", capacities[i]+64)
				if err != nil {
					return err
				}
				if err := checkGuards(run+" "+name, guarded, destination); err != nil {
					return err
				}
				if result.OutputWordRejection == "" && name == "InLayout" {
					mutated := append([]int32(nil), expected[i]...)
					mutated[0] ^= 1
					drift := compareWords(name, mutated, words)
					if drift == nil || !strings.HasPrefix(drift.Error(), "InLayout[0]") {
						return fmt.Errorf("initialized output-word rejection control failed")
					}
					result.OutputWordRejection = drift.Error()
				}
				result.ComparedWords += len(words)
			}
			result.Runs++
		}
	}
	return nil
}
