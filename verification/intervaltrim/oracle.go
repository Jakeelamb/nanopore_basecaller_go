// Package intervaltrimoracle compiles pinned Dorado interval-trim and filter source.
package intervaltrimoracle

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
	"dorado/demux/Trimmer.cpp":                                                   "253852c8f7f96d9706164b5d28371c3a0e6702c55f87cd000e2dce6ed0ff9f52",
	"dorado/read_pipeline/base/messages.cpp":                                     "5db12169f5eb376bfa70425a5f17fa31eab6be5d027d681caec81dd105ba35e0",
	"dorado/read_pipeline/nodes/ReadFilterNode.cpp":                              "cd4668dfc020880e2fe8d1c4f919d19eae637cad3bc144a3e44a6c569a2fad93",
	"dorado/torch_utils/include/torch_utils/trim.h":                              "a8781942df2f2b067c87144d59012b072dbe29c9c230cf05ac17d9b3e147ed47",
	"dorado/utils/sequence_utils.cpp":                                            "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2",
	"dorado/torch_utils/trim.cpp":                                                "7aaacedeb85b9d79cf04d181f9c272a214eeae339af9c21735ce84eba5205d1b",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h": "a8ff4bced80c61f07070d6f5ed17ad3c2367c4f7d2a4564f54b5a7e47aec01cb",
}

func hash(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

func VerifySources(root string) error {
	for name, want := range SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return err
		}
		if got := hash(data); got != want {
			return fmt.Errorf("Dorado source hash mismatch for %s: %s != %s", name, got, want)
		}
	}
	return nil
}

func sourceFunction(data []byte, signature string) ([]byte, error) {
	start := bytes.Index(data, []byte(signature))
	if start < 0 {
		return nil, fmt.Errorf("missing function %s", signature)
	}
	open := bytes.IndexByte(data[start:], '{')
	if open < 0 {
		return nil, fmt.Errorf("missing function body %s", signature)
	}
	open += start
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
	return nil, fmt.Errorf("unclosed function %s", signature)
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}

// Build requires a new output directory. Every source file, extracted function,
// shim, driver, compiler and linked runtime is retained or hashed there.
func Build(root, output string) (string, error) {
	if err := os.Mkdir(output, 0700); err != nil {
		return "", err
	}
	if err := VerifySources(root); err != nil {
		_ = writeJSON(filepath.Join(output, "rejection.json"), map[string]any{"time": time.Now().UTC(), "error": err.Error()})
		return "", err
	}
	pins := map[string]string{}
	for name := range SourcePins {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return "", err
		}
		dst := filepath.Join(output, "source", name)
		if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
			return "", err
		}
		if err := os.WriteFile(dst, data, 0600); err != nil {
			return "", err
		}
	}
	for _, p := range []struct{ path, signature, name string }{
		{"dorado/demux/Trimmer.cpp", "void Trimmer::trim_sequence(SimplexRead&", "trim.slice.cpp"},
		{"dorado/demux/Trimmer.cpp", "void trim_torch_tensor(", "tensor.slice.cpp"},
		{"dorado/torch_utils/trim.cpp", "std::tuple<int, std::vector<uint8_t>> trim_move_table(", "move.slice.cpp"},
		{"dorado/torch_utils/trim.cpp", "std::string trim_sequence(", "string.slice.cpp"},
		{"dorado/torch_utils/include/torch_utils/trim.h", "template <typename Container>\nContainer trim_vector(", "vector.slice.cpp"},
		{"dorado/utils/sequence_utils.cpp", "float mean_qscore_from_qstring(", "mean.slice.cpp"},
		{"dorado/read_pipeline/base/messages.cpp", "float ReadCommon::calculate_mean_qscore(", "quality.slice.cpp"},
	} {
		data, err := os.ReadFile(filepath.Join(root, p.path))
		if err != nil {
			return "", err
		}
		slice, err := sourceFunction(data, p.signature)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(output, p.name), slice, 0600); err != nil {
			return "", err
		}
		pins[p.name] = hash(slice)
	}
	filterSource, err := os.ReadFile(filepath.Join(root, "dorado/read_pipeline/nodes/ReadFilterNode.cpp"))
	if err != nil {
		return "", err
	}
	start := bytes.Index(filterSource, []byte("(m_min_qscore > 0 &&"))
	if start < 0 {
		return "", fmt.Errorf("missing filter expression")
	}
	start-- // Include the source if-condition's outer parenthesis.
	stop := bytes.Index(filterSource[start:], []byte(" {"))
	if stop < 0 {
		return "", fmt.Errorf("missing filter expression end")
	}
	expression := filterSource[start : start+stop]
	if err := os.WriteFile(filepath.Join(output, "filter.slice.cpp"), expression, 0600); err != nil {
		return "", err
	}
	pins["filter.slice.cpp"] = hash(expression)
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate oracle driver")
	}
	driver, err := os.ReadFile(filepath.Join(filepath.Dir(here), "oracle.cpp"))
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "oracle.cpp"), driver, 0600); err != nil {
		return "", err
	}
	pins["oracle.cpp"] = hash(driver)
	compiler, err := exec.LookPath("c++")
	if err != nil {
		return "", err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return "", err
	}
	compilerData, err := os.ReadFile(compiler)
	if err != nil {
		return "", err
	}
	version, err := exec.Command(compiler, "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	executable := filepath.Join(output, "interval-oracle")
	args := []string{"-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-I" + output, filepath.Join(output, "oracle.cpp"), "-o", executable}
	if err := commandReceipt(output, "build", exec.Command(compiler, args...), nil); err != nil {
		return "", err
	}
	exeData, err := os.ReadFile(executable)
	if err != nil {
		return "", err
	}
	ldd, err := exec.Command("ldd", executable).CombinedOutput()
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(output, "ldd.txt"), ldd, 0600); err != nil {
		return "", err
	}
	dependencies := map[string]string{}
	for _, line := range strings.Split(string(ldd), "\n") {
		for _, field := range strings.Fields(line) {
			if strings.HasPrefix(field, "/") {
				data, err := os.ReadFile(field)
				if err != nil {
					return "", err
				}
				dependencies[field] = hash(data)
			}
		}
	}
	receipt := map[string]any{"source_commit": SourceCommit, "source_pins": SourcePins, "slices_and_driver": pins, "compiler": compiler, "compiler_sha256": hash(compilerData), "compiler_version": string(version), "executable_sha256": hash(exeData), "runtime_pins": dependencies, "qualification": "seven unchanged complete source functions plus exact filter expression; accessed ReadCommon container shim; uint16 one-dimensional positive unit slice shim; RNA helper and mods error stubs; float32 threshold extension of native size_t CLI; no inference or GPU"}
	if err := writeJSON(filepath.Join(output, "manifest.json"), receipt); err != nil {
		return "", err
	}
	return executable, nil
}

func commandReceipt(dir, name string, cmd *exec.Cmd, input []byte) error {
	cmd.Stdin = bytes.NewReader(input)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	started := time.Now().UTC()
	runErr := cmd.Run()
	finished := time.Now().UTC()
	code := -1
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	for name, data := range map[string][]byte{name + ".stdout": stdout.Bytes(), name + ".stderr": stderr.Bytes()} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			return err
		}
	}
	receipt := map[string]any{"command": cmd.Args, "started": started, "finished": finished, "exit_code": code, "stdin_sha256": hash(input), "stdout_sha256": hash(stdout.Bytes()), "stderr_sha256": hash(stderr.Bytes())}
	if err := writeJSON(filepath.Join(dir, name+".receipt.json"), receipt); err != nil {
		return err
	}
	if runErr != nil {
		return fmt.Errorf("%s: %w: %s", name, runErr, stderr.String())
	}
	return nil
}
