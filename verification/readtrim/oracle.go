// Package readtrimoracle compiles four unchanged pinned Dorado functions.
package readtrimoracle

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

var SourcePins = map[string]string{
	"dorado/read_pipeline/base/read_utils.cpp":                                   "41b2130f6c484504af84c0578645d6a8380c89883910219d839c022ca5a37615",
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
		{"dorado/read_pipeline/base/read_utils.cpp", "int64_t find_mux_change_trim_seq_index(", "index.slice.cpp"},
		{"dorado/read_pipeline/base/read_utils.cpp", "void mux_change_trim_read(", "mux.slice.cpp"},
		{"dorado/utils/sequence_utils.cpp", "int64_t sequence_to_move_table_index(", "move.slice.cpp"},
		{"dorado/torch_utils/trim.cpp", "std::string trim_sequence(", "string.slice.cpp"},
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
	executable := filepath.Join(output, "mux-oracle")
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
	receipt := map[string]any{"source_commit": SourceCommit, "source_pins": SourcePins, "slices_and_driver": pins, "compiler": compiler, "compiler_sha256": hash(compilerData), "compiler_version": string(version), "executable_sha256": hash(exeData), "runtime_pins": dependencies, "qualification": "four unchanged source functions; exact accessed ReadCommon fields; uint16 slice-only vector shim with runtime self-check; no ATen arithmetic, inference or GPU"}
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

// EncodeInput defines the little-endian, length-prefixed oracle protocol.
func EncodeInput(reads []readtrim.MuxRead) []byte {
	var data bytes.Buffer
	for _, r := range reads {
		encodeRead(&data, r)
	}
	return data.Bytes()
}
func encodeRead(data *bytes.Buffer, r readtrim.MuxRead) {
	if r.IsMuxChange {
		data.WriteByte(1)
	} else {
		data.WriteByte(0)
	}
	for _, v := range []any{r.Stride, r.NumMinknowEvents, r.PreTrimSequenceLength, uint32(len(r.Sequence))} {
		_ = binary.Write(data, binary.LittleEndian, v)
	}
	data.WriteString(r.Sequence)
	data.WriteString(r.Qstring)
	_ = binary.Write(data, binary.LittleEndian, uint32(len(r.Moves)))
	data.Write(r.Moves)
	_ = binary.Write(data, binary.LittleEndian, uint32(len(r.Signal)))
	_ = binary.Write(data, binary.LittleEndian, r.Signal)
}

// Compare preserves exact binary inputs and both outputs before comparison.
func Compare(executable, dir string, reads []readtrim.MuxRead) error {
	input := EncodeInput(reads)
	if err := os.WriteFile(filepath.Join(dir, "inputs.bin"), input, 0600); err != nil {
		return err
	}
	var expected bytes.Buffer
	for i, r := range reads {
		result, err := readtrim.MuxChangeTrim(r)
		if err != nil {
			return fmt.Errorf("case %d: %w", i, err)
		}
		encodeRead(&expected, result.Read)
		_ = binary.Write(&expected, binary.LittleEndian, result.NumSamples)
	}
	if err := os.WriteFile(filepath.Join(dir, "go-output.bin"), expected.Bytes(), 0600); err != nil {
		return err
	}
	if err := commandReceipt(dir, "native", exec.Command(executable), input); err != nil {
		return err
	}
	actual, err := os.ReadFile(filepath.Join(dir, "native.stdout"))
	if err != nil {
		return err
	}
	if !bytes.Equal(actual, expected.Bytes()) {
		return fmt.Errorf("native MUX output mismatch: got %s want %s", hash(actual), hash(expected.Bytes()))
	}
	return writeJSON(filepath.Join(dir, "comparison.json"), map[string]any{"cases": len(reads), "exact_bytes": true, "input_sha256": hash(input), "output_sha256": hash(actual)})
}
