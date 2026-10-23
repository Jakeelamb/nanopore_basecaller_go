// Package endsoracle builds source-pinned end detection and edlib controls.
package endsoracle

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
const EdlibCommit = "42aa8fa7051fb2aa1c4f144c5eda22a0e753f026"
const DefaultEdlibRoot = "/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/source-input/dorado/3rdparty/edlib"

var SourcePins = map[string]string{
	"dorado/demux/AdapterDetector.cpp":                   "2aafffb7ad668377923394772a03181adc96f343eb797d696d092c158dd14b78",
	"dorado/demux/include/demux/AdapterDetector.h":       "82aa02335578c4cfa6348998e61570cab82f7a721f49df89a37b16a6c19df52c",
	"dorado/demux/adapter_primer_kits.cpp":               "ac0a8390968e258c006f827a6f5983d99df7307ef736cc3a86c8ea7c558ffc80",
	"dorado/demux/adapter_primer_kits.h":                 "83e947a1ce86e4208d56f2f43174f213d2a6147d25728d21d427cb6d2998f6e5",
	"dorado/demux/adapter_info.cpp":                      "30ec413dd093117323790b6e854328aaeacdd488752c48c669e5d92b15208935",
	"dorado/demux/include/demux/adapter_info.h":          "f6f48e51885fb36237857b900bc572573e30c3572ebd5e3ad0e8aa9762aa0c16",
	"dorado/demux/Trimmer.cpp":                           "253852c8f7f96d9706164b5d28371c3a0e6702c55f87cd000e2dce6ed0ff9f52",
	"dorado/models/kits.cpp":                             "ca3a496efdc34444a161e11ba11f750587d3fee940e23d00f483c53cf53884b5",
	"dorado/models/include/models/kits.h":                "7fe1c1578805df68ea6c4d916202bce5fffc0c4cf655ff0e4172855c1b630df2",
	"dorado/utils/include/utils/string_utils.h":          "a46e8f2cb56248590bcf50b7dcbb0d5c1810700c25040979ce1985d51f8c22bb",
	"dorado/utils/include/utils/sequence_utils.h":        "ada13fc46424ba0d068075012b90dc267dce058d8c2a5d6d591b00820f634c6c",
	"dorado/utils/include/utils/types.h":                 "1c15b080a2c10b760d5cc5d4f5e79960ad28e42751e555fc7f16314e17896219",
	"dorado/utils/sequence_utils.cpp":                    "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2",
	"dorado/hts_utils/include/hts_utils/hts_types.h":     "653f2f71de8bc57847349154bd45a6f66120cb56fcaf847a8ccc9b694a1ae65b",
	"dorado/read_pipeline/nodes/AdapterDetectorNode.cpp": "df4a068a50857e67bb6165bfcf514213517fb70931a2e85fc8104cdd40523597",
}

var EdlibPins = map[string]string{
	"edlib/src/edlib.cpp":   "a9c2b4a44820453d4991b129b373d04d82a59fc0aaabfad3d8da838c380ab0bd",
	"edlib/include/edlib.h": "62d0529a87aa6e701c767fb9ce3786c759068f454c129641b2be38da6237ccb5",
}

func hash(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

func verify(root string, pins map[string]string) error {
	for name, want := range pins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return err
		}
		if got := hash(data); got != want {
			return fmt.Errorf("source hash mismatch %s: got %s want %s", name, got, want)
		}
	}
	return nil
}

func VerifySources(root, edlib string) error {
	if err := verify(root, SourcePins); err != nil {
		return err
	}
	return verify(edlib, EdlibPins)
}

func sourceFunction(data []byte, signature string) ([]byte, error) {
	start := bytes.Index(data, []byte(signature))
	if start < 0 {
		return nil, fmt.Errorf("missing source function %s", signature)
	}
	open := bytes.IndexByte(data[start:], '{')
	if open < 0 {
		return nil, fmt.Errorf("missing source body %s", signature)
	}
	depth := 0
	for i := start + open; i < len(data); i++ {
		if data[i] == '{' {
			depth++
		}
		if data[i] == '}' {
			depth--
			if depth == 0 {
				return bytes.Clone(data[start : i+1]), nil
			}
		}
	}
	return nil, fmt.Errorf("unclosed source function %s", signature)
}

func region(data []byte, first, after string) ([]byte, error) {
	start := bytes.Index(data, []byte(first))
	if start < 0 {
		return nil, fmt.Errorf("missing region start %q", first)
	}
	end := bytes.Index(data[start:], []byte(after))
	if end < 0 {
		return nil, fmt.Errorf("missing region end %q", after)
	}
	return bytes.Clone(data[start : start+end]), nil
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}

// Build requires a new directory and compiles original edlib source, never a prebuilt archive.
func Build(root, edlib, output string) (string, error) {
	if err := os.Mkdir(output, 0700); err != nil {
		return "", err
	}
	if err := VerifySources(root, edlib); err != nil {
		_ = writeJSON(filepath.Join(output, "rejection.json"), map[string]any{"time": time.Now().UTC(), "error": err.Error()})
		return "", err
	}
	pins := map[string]string{}
	put := func(name string, data []byte) error {
		path := filepath.Join(output, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return err
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			return err
		}
		pins[name] = hash(data)
		return nil
	}
	for name := range SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return "", err
		}
		if err := put(filepath.Join("source", name), data); err != nil {
			return "", err
		}
	}
	for name := range EdlibPins {
		data, err := os.ReadFile(filepath.Join(edlib, name))
		if err != nil {
			return "", err
		}
		if err := put(filepath.Join("edlib-source", name), data); err != nil {
			return "", err
		}
	}
	root, edlib = filepath.Join(output, "source"), filepath.Join(output, "edlib-source")
	if err := VerifySources(root, edlib); err != nil {
		return "", err
	}
	for _, part := range []struct{ signature, name string }{
		{"EdlibAlignConfig init_edlib_config_for_adapters(", "config"},
		{"dorado::SingleEndResult copy_results(", "copy"},
		{"dorado::SingleEndResult align(", "align"},
		{"dorado::SingleEndResult get_best_result(", "best"},
		{"std::vector<AdapterDetector::Query>& AdapterDetector::get_primer_sequences(", "primer-queries"},
		{"AdapterScoreResult AdapterDetector::detect(", "detect"},
		{"PrimerClassification AdapterDetector::classify_primers(", "classify"},
	} {
		data, err := os.ReadFile(filepath.Join(root, "dorado/demux/AdapterDetector.cpp"))
		if err != nil {
			return "", err
		}
		slice, err := sourceFunction(data, part.signature)
		if err != nil {
			return "", err
		}
		if err := put(filepath.Join("functions", part.name+".cpp"), slice); err != nil {
			return "", err
		}
	}
	for _, part := range []struct{ path, signature, name string }{
		{"dorado/demux/adapter_primer_kits.cpp", "AdapterPrimerManager::AdapterPrimerManager()", "manager-constructor.slice.cpp"},
		{"dorado/demux/adapter_primer_kits.cpp", "std::vector<Candidate> AdapterPrimerManager::get_candidates(", "candidates.slice.cpp"},
		{"dorado/demux/Trimmer.cpp", "std::pair<int, int> Trimmer::determine_trim_interval(AdapterScoreResult&", "trim.slice.cpp"},
		{"dorado/utils/sequence_utils.cpp", "std::string\nreverse_complement_impl(", "reverse-impl.slice.cpp"},
		{"dorado/utils/sequence_utils.cpp", "std::string reverse_complement(", "reverse.slice.cpp"},
	} {
		data, err := os.ReadFile(filepath.Join(root, part.path))
		if err != nil {
			return "", err
		}
		slice, err := sourceFunction(data, part.signature)
		if err != nil {
			return "", err
		}
		if err := put(part.name, slice); err != nil {
			return "", err
		}
	}
	for _, part := range []struct{ path, first, after, name string }{
		{"dorado/demux/adapter_primer_kits.cpp", "enum class AdapterCode", "struct CustomItem", "tables.slice.cpp"},
		{"dorado/utils/include/utils/types.h", "struct SingleEndResult", "using BarcodeFilterSet", "types.slice.h"},
		{"dorado/hts_utils/include/hts_utils/hts_types.h", "enum class StrandOrientation", "struct BarcodeScoreResult", "orientation.slice.h"},
		{"dorado/read_pipeline/nodes/AdapterDetectorNode.cpp", "        std::pair<int, int> trim_interval = adapter_trim_interval;", "\n    }\n}", "intersection.slice.cpp"},
	} {
		data, err := os.ReadFile(filepath.Join(root, part.path))
		if err != nil {
			return "", err
		}
		if part.name == "intersection.slice.cpp" {
			start := bytes.Index(data, []byte("void AdapterDetectorNode::process_read(SimplexRead& read)"))
			if start < 0 {
				return "", fmt.Errorf("missing simplex detector method")
			}
			data = data[start:]
		}
		slice, err := region(data, part.first, part.after)
		if err != nil {
			return "", err
		}
		if err := put(part.name, slice); err != nil {
			return "", err
		}
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate oracle")
	}
	for _, name := range []string{"oracle.cpp", "shim/utils/types.h", "shim/hts_utils/hts_types.h", "shim/utils/log_utils.h", "shim/spdlog/spdlog.h", "shim/htslib/sam.h"} {
		data, err := os.ReadFile(filepath.Join(filepath.Dir(here), name))
		if err != nil {
			return "", err
		}
		if err := put(name, data); err != nil {
			return "", err
		}
	}
	executable := filepath.Join(output, "ends-oracle")
	args := []string{"-std=c++20", "-O2", "-ffp-contract=off", "-include", "memory", "-include", "unordered_map", "-include", "cmath", "-I" + filepath.Join(output, "shim"), "-I" + output, "-I" + filepath.Join(output, "source/dorado/demux/include"), "-I" + filepath.Join(output, "source/dorado/demux"), "-I" + filepath.Join(output, "source/dorado/utils/include"), "-I" + filepath.Join(output, "source/dorado/models/include"), "-I" + filepath.Join(output, "edlib-source/edlib/include"), filepath.Join(output, "oracle.cpp"), filepath.Join(output, "source/dorado/models/kits.cpp"), filepath.Join(output, "source/dorado/demux/adapter_info.cpp"), filepath.Join(output, "edlib-source/edlib/src/edlib.cpp"), "-o", executable}
	command := exec.Command("c++", args...)
	started := time.Now().UTC()
	log, buildErr := command.CombinedOutput()
	finished := time.Now().UTC()
	if err := put("build.log", log); err != nil {
		return "", err
	}
	exit := -1
	if command.ProcessState != nil {
		exit = command.ProcessState.ExitCode()
	}
	compiler, err := exec.Command("c++", "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	compilerPath, err := exec.LookPath("c++")
	if err != nil {
		return "", err
	}
	compilerPath, err = filepath.EvalSymlinks(compilerPath)
	if err != nil {
		return "", err
	}
	compilerBytes, err := os.ReadFile(compilerPath)
	if err != nil {
		return "", err
	}
	compilerTools := map[string]string{}
	for _, name := range []string{"cc1plus", "as", "ld"} {
		data, err := exec.Command("c++", "-print-prog-name="+name).Output()
		if err != nil {
			return "", err
		}
		path := strings.TrimSpace(string(data))
		if !filepath.IsAbs(path) {
			path, err = exec.LookPath(path)
			if err != nil {
				return "", err
			}
		}
		path, err = filepath.EvalSymlinks(path)
		if err != nil {
			return "", err
		}
		data, err = os.ReadFile(path)
		if err != nil {
			return "", err
		}
		compilerTools[path] = hash(data)
	}
	receipt := map[string]any{"source_commit": SourceCommit, "edlib_commit": EdlibCommit, "source_pins": SourcePins, "edlib_pins": EdlibPins, "inputs": pins, "args": args, "compiler": string(compiler), "compiler_path": compilerPath, "compiler_sha256": hash(compilerBytes), "started": started, "finished": finished, "exit_code": exit, "qualification": "unchanged full detector, kit metadata and primer aux; unchanged table/manager/trim/reverse/intersection slices; source-built edlib; container and logger shims"}
	receipt["compiler_tools"] = compilerTools
	if buildErr == nil {
		deps, err := exec.Command("ldd", executable).CombinedOutput()
		if err != nil {
			return "", err
		}
		if err := put("ldd.txt", deps); err != nil {
			return "", err
		}
		libraries := map[string]string{}
		for _, field := range strings.Fields(string(deps)) {
			if strings.HasPrefix(field, "/") {
				b, err := os.ReadFile(field)
				if err != nil {
					return "", err
				}
				libraries[field] = hash(b)
			}
		}
		receipt["runtime_libraries"] = libraries
		binary, err := os.ReadFile(executable)
		if err != nil {
			return "", err
		}
		receipt["executable_sha256"] = hash(binary)
	}
	if err := writeJSON(filepath.Join(output, "build-receipt.json"), receipt); err != nil {
		return "", err
	}
	if buildErr != nil {
		return "", fmt.Errorf("compile native end detector: %w\n%s", buildErr, log)
	}
	return executable, nil
}
