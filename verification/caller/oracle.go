// Package calleroracle builds source-pinned FASTQ metadata and mean-quality controls.
package calleroracle

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

var SourcePins = map[string]string{
	"dorado/utils/sequence_utils.cpp":                  "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2",
	"dorado/read_pipeline/base/messages.cpp":           "5db12169f5eb376bfa70425a5f17fa31eab6be5d027d681caec81dd105ba35e0",
	"dorado/data_loader/DataLoader.cpp":                "6e2d7858c2c39cbb08955e127c2d1979805667fdcfc6187b9dd910a8de14db88",
	"dorado/utils/time_utils.cpp":                      "5b4e928ece39f04d4c22dd8310c1a7f8c6878a7663e86bb46fec6fae9ed4f75d",
	"dorado/utils/include/utils/time_utils.h":          "0746e20fd10bc072acf5ffad67944bdd5470cd842886e7be2832030f4285a2c8",
	"dorado/hts_utils/hts_file.cpp":                    "493186872cc3ffb0716f2c45b8466c8d9231b2a3e59adb7122621b6c48666c98",
	"dorado/hts_utils/include/hts_utils/hts_types.h":   "653f2f71de8bc57847349154bd45a6f66120cb56fcaf847a8ccc9b694a1ae65b",
	"dorado/hts_writer/HtsFileWriter.cpp":              "d29a4389353d00e476afc9d2bc110e9b21b36d95f6d66ee7ac85232d1a2cf48e",
	"dorado/hts_writer/StreamHtsFileWriter.cpp":        "165d59528d96bd49d9114d109e493a062652b78db09b2f2c4d7f444f647247c1",
	"dorado/hts_writer/StructuredHtsFileWriter.cpp":    "1d7355329d3a865457aadbc4a927e077fda5b9ddf7fdf9dfca8cbaefb1199ba4",
	"dorado/read_pipeline/nodes/WriterNode.cpp":        "39e0f302b3c39ee088a4bc4d133a94d746e6f4f39f886d5072492cb61bd3c026",
	"dorado/read_pipeline/nodes/HtsWriterNode.cpp":     "853f332f13397826aba000e18ee5b4bc530934e508183cdf2a39d8be76520c0e",
	"dorado/read_pipeline/nodes/ReadToBamTypeNode.cpp": "5a4e9e7313342602cd44ef8086356e6577e67cbe8a1f83271c6aa71ea5a001b8",
}

func VerifySources(root string) error {
	for name, want := range SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return err
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
			return fmt.Errorf("Dorado source hash mismatch: %s: got %s want %s", name, got, want)
		}
	}
	return nil
}

func function(data []byte, signature string) ([]byte, error) {
	start := bytes.Index(data, []byte(signature))
	if start < 0 {
		return nil, fmt.Errorf("missing source function %s", signature)
	}
	open := bytes.IndexByte(data[start:], '{') + start
	depth := 0
	for i := open; i < len(data); i++ {
		if data[i] == '{' {
			depth++
		}
		if data[i] == '}' {
			depth--
			if depth == 0 {
				return append(bytes.Clone(data[start:i+1]), '\n'), nil
			}
		}
	}
	return nil, fmt.Errorf("unclosed source function %s", signature)
}

// Build extracts whole source functions verbatim and compiles original time_utils.cpp.
// ReadCommon and writer containers are test-only shims. System HTSlib serializes
// tags with the original source's selected tag list and insertion order.
func Build(root, output string) (string, error) {
	if err := VerifySources(root); err != nil {
		return "", err
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return "", err
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate caller oracle")
	}
	pins := map[string]string{}
	for _, part := range []struct{ path, signature, file string }{
		{"dorado/utils/sequence_utils.cpp", "float mean_qscore_from_qstring(", "mean.slice.cpp"},
		{"dorado/read_pipeline/base/messages.cpp", "std::string ReadCommon::generate_read_group(", "group.slice.cpp"},
		{"dorado/read_pipeline/base/messages.cpp", "float ReadCommon::calculate_mean_qscore(", "quality.slice.cpp"},
		{"dorado/read_pipeline/base/messages.cpp", "void ReadCommon::generate_read_tags(", "tags.slice.cpp"},
		{"dorado/hts_writer/HtsFileWriter.cpp", "void HtsFileWriter::prepare_item(", "writer.slice.cpp"},
		{"dorado/hts_utils/include/hts_utils/hts_types.h", "inline std::string to_string(TrimFlags", "trim.slice.cpp"},
	} {
		data, err := os.ReadFile(filepath.Join(root, part.path))
		if err != nil {
			return "", err
		}
		slice, err := function(data, part.signature)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(output, part.file), slice, 0600); err != nil {
			return "", err
		}
		pins[part.file] = fmt.Sprintf("%x", sha256.Sum256(slice))
	}
	data, err := os.ReadFile(filepath.Join(root, "dorado/hts_utils/hts_file.cpp"))
	if err != nil {
		return "", err
	}
	start := bytes.Index(data, []byte("constexpr std::array fastq_aux_tags"))
	if start < 0 {
		return "", fmt.Errorf("missing FASTQ tag list")
	}
	end := bytes.IndexByte(data[start:], ';') + start + 1
	if end <= start {
		return "", fmt.Errorf("unterminated FASTQ tag list")
	}
	tags := data[start:end]
	if err := os.WriteFile(filepath.Join(output, "selected.slice.cpp"), tags, 0600); err != nil {
		return "", err
	}
	pins["selected.slice.cpp"] = fmt.Sprintf("%x", sha256.Sum256(tags))
	data, err = os.ReadFile(filepath.Join(root, "dorado/data_loader/DataLoader.cpp"))
	if err != nil {
		return "", err
	}
	start = bytes.Index(data, []byte("auto start_time_ms = ((read_data.start_sample * 1000) / run_sample_rate);"))
	if start < 0 {
		return "", fmt.Errorf("missing POD5 sample clock")
	}
	end = bytes.Index(data[start:], []byte("    new_read->run_acquisition_start_time_ms")) + start
	if end < start {
		return "", fmt.Errorf("missing POD5 sample clock endpoint")
	}
	clock := data[start:end]
	if err := os.WriteFile(filepath.Join(output, "clock.slice.cpp"), clock, 0600); err != nil {
		return "", err
	}
	pins["clock.slice.cpp"] = fmt.Sprintf("%x", sha256.Sum256(clock))
	driver, err := os.ReadFile(filepath.Join(filepath.Dir(here), "oracle.cpp"))
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "oracle.cpp"), driver, 0600); err != nil {
		return "", err
	}
	pins["oracle.cpp"] = fmt.Sprintf("%x", sha256.Sum256(driver))
	executable := filepath.Join(output, "caller-oracle")
	args := []string{"-std=c++20", "-O2", "-ffp-contract=off", "-I" + output, "-I" + filepath.Join(root, "dorado/utils/include"), filepath.Join(output, "oracle.cpp"), filepath.Join(root, "dorado/utils/time_utils.cpp"), "-lhts", "-o", executable}
	command := exec.Command("c++", args...)
	started := time.Now().UTC()
	log, buildErr := command.CombinedOutput()
	if err := os.WriteFile(filepath.Join(output, "build.log"), log, 0600); err != nil {
		return "", err
	}
	exitCode := -1
	if command.ProcessState != nil {
		exitCode = command.ProcessState.ExitCode()
	}
	compiler, compilerErr := exec.Command("c++", "--version").CombinedOutput()
	if compilerErr != nil {
		return "", compilerErr
	}
	htslib, htsErr := exec.Command("pkg-config", "--modversion", "htslib").CombinedOutput()
	if htsErr != nil {
		return "", htsErr
	}
	receipt := map[string]any{"source_commit": SourceCommit, "source_pins": SourcePins, "slices_and_driver": pins, "args": args, "compiler": string(compiler), "system_htslib": strings.TrimSpace(string(htslib)), "started": started, "finished": time.Now().UTC(), "exit_code": exitCode, "qualification": "extracted metadata/quality/writer functions and untouched time functions; shimmed containers; system HTSlib; no model, POD5 or inference"}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "build-receipt.json"), append(encoded, '\n'), 0600); err != nil {
		return "", err
	}
	if buildErr != nil {
		return "", fmt.Errorf("compile caller metadata oracle: %w\n%s", buildErr, log)
	}
	return executable, nil
}
