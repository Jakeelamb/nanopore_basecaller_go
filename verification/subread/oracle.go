// Package subreadoracle builds source-pinned, CPU-only child materialization evidence.
package subreadoracle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

var SourcePins = map[string]string{
	"dorado/splitter/DuplexReadSplitter.cpp":                                      "13e8a88e4aecda1dcb44eb58e2d8aa30dec76b7202e10f680dd1f6419710200e",
	"dorado/splitter/splitter_utils.cpp":                                          "33a055cf5c8ef3db49500c992e9c832682e4c175a231ae2b854d5705de8a9a48",
	"dorado/read_pipeline/base/read_utils.cpp":                                    "41b2130f6c484504af84c0578645d6a8380c89883910219d839c022ca5a37615",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h":  "a8ff4bced80c61f07070d6f5ed17ad3c2367c4f7d2a4564f54b5a7e47aec01cb",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/SimplexRead.h": "a6e225079228a55ff0077477bf7d7d0b2c74e9b8d2441c659dee522da745b663",
	"dorado/utils/sequence_utils.cpp":                                             "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2",
	"dorado/utils/uuid_utils.cpp":                                                 "36f849d2873ae819a8e3eed3e47f286fb5f60f1550d4e226e0afaacbdf7c677b",
	"dorado/utils/crypto_utils.cpp":                                               "a361a2e2461c6dc23e81706e2e514233699030e5f56e143a8afda4eeba51519f",
	"dorado/utils/time_utils.cpp":                                                 "5b4e928ece39f04d4c22dd8310c1a7f8c6878a7663e86bb46fec6fae9ed4f75d",
	"dorado/utils/include/utils/uuid_utils.h":                                     "f55a962ebfee6432dd9d5fa1070bcc9541cb17e835fddb88a5a0ee3126c69b7b",
	"dorado/utils/include/utils/crypto_utils.h":                                   "d7a75c4950beef5b9ad6427c2cfdefbf564091baeae65f0899b9bd495f5c92bc",
	"dorado/utils/include/utils/time_utils.h":                                     "0746e20fd10bc072acf5ffad67944bdd5470cd842886e7be2832030f4285a2c8",
}

func Hash(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }
func SaveJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}
func VerifySources(root string) error {
	for path, want := range SourcePins {
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			return err
		}
		if Hash(data) != want {
			return fmt.Errorf("source hash mismatch for %s", path)
		}
	}
	return nil
}

// Command retains clocks, exits and raw streams before returning a failure.
func Command(dir, name string, cmd *exec.Cmd, input []byte) ([]byte, error) {
	cmd.Stdin = bytes.NewReader(input)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	start := time.Now().UTC()
	runErr := cmd.Run()
	end := time.Now().UTC()
	exit := -1
	if cmd.ProcessState != nil {
		exit = cmd.ProcessState.ExitCode()
	}
	for path, data := range map[string][]byte{name + ".stdin": input, name + ".stdout": stdout.Bytes(), name + ".stderr": stderr.Bytes()} {
		if err := os.WriteFile(filepath.Join(dir, path), data, 0600); err != nil {
			return nil, err
		}
	}
	receipt := map[string]any{"command": cmd.Args, "started": start, "finished": end, "exit_code": exit, "stdin_sha256": Hash(input), "stdout_sha256": Hash(stdout.Bytes()), "stderr_sha256": Hash(stderr.Bytes())}
	if cmd.Env != nil {
		env := map[string]string{}
		for _, entry := range cmd.Env {
			key, value, _ := strings.Cut(entry, "=")
			switch key {
			case "CGO_ENABLED", "DORADO_SOURCE", "SUBREAD_MUTATIONS", "SUBREAD_ARTIFACT_ROOT", "GOOS", "GOARCH":
				env[key] = value
			}
		}
		receipt["verification_environment"] = env
	}
	if runErr != nil {
		receipt["error"] = runErr.Error()
	}
	if err := SaveJSON(filepath.Join(dir, name+".receipt.json"), receipt); err != nil {
		return nil, err
	}
	if runErr != nil {
		return stdout.Bytes(), fmt.Errorf("%s: %w: %s", name, runErr, stderr.Bytes())
	}
	return stdout.Bytes(), nil
}

func sourceBody(data []byte, signature string) ([]byte, error) {
	start := bytes.Index(data, []byte(signature))
	if start < 0 {
		return nil, fmt.Errorf("source signature missing: %s", signature)
	}
	open := bytes.IndexByte(data[start:], '{')
	if open < 0 {
		return nil, fmt.Errorf("source body missing: %s", signature)
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
	return nil, fmt.Errorf("source body unclosed: %s", signature)
}

// Build uses unchanged function bodies and full original crypto/UUID/time units.
// FlatSignal replaces only immutable one-dimensional uint16 storage and slicing.
func Build(source, output string) (string, error) {
	if err := os.Mkdir(output, 0700); err != nil {
		return "", err
	}
	if err := VerifySources(source); err != nil {
		_ = SaveJSON(filepath.Join(output, "rejection.json"), map[string]any{"utc": time.Now().UTC(), "error": err.Error()})
		return "", err
	}
	for path := range SourcePins {
		data, err := os.ReadFile(filepath.Join(source, path))
		if err != nil {
			return "", err
		}
		dst := filepath.Join(output, "source", path)
		if err = os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
			return "", err
		}
		if err = os.WriteFile(dst, data, 0600); err != nil {
			return "", err
		}
	}
	pins := map[string]string{}
	for _, slice := range []struct{ path, signature, name string }{
		{"dorado/utils/sequence_utils.cpp", "std::vector<uint64_t> moves_to_map(", "moves_to_map"},
		{"dorado/read_pipeline/base/read_utils.cpp", "SimplexReadPtr shallow_copy_read(", "shallow_copy_read"},
		{"dorado/splitter/splitter_utils.cpp", "void assign_subread_parent_id(", "assign_subread_parent_id"},
		{"dorado/splitter/splitter_utils.cpp", "SimplexReadPtr subread(", "subread"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "std::vector<SimplexReadPtr> DuplexReadSplitter::subreads(", "subreads"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "template <typename SplitFinder>\nvoid DuplexReadSplitter::apply_split_finder(", "apply_split_finder"},
		{"dorado/splitter/DuplexReadSplitter.cpp", "std::vector<SimplexReadPtr> DuplexReadSplitter::split(", "split"},
	} {
		data, err := os.ReadFile(filepath.Join(source, slice.path))
		if err != nil {
			return "", err
		}
		body, err := sourceBody(data, slice.signature)
		if err != nil {
			return "", err
		}
		name := slice.name + ".slice.cpp"
		if err = os.WriteFile(filepath.Join(output, name), body, 0600); err != nil {
			return "", err
		}
		pins[name] = Hash(body)
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot locate driver")
	}
	driver, err := os.ReadFile(filepath.Join(filepath.Dir(here), "oracle.cpp"))
	if err != nil {
		return "", err
	}
	if err = os.WriteFile(filepath.Join(output, "oracle.cpp"), driver, 0600); err != nil {
		return "", err
	}
	pins["oracle.cpp"] = Hash(driver)
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
	version, err := Command(output, "compiler-version", exec.Command(compiler, "--version"), nil)
	if err != nil {
		return "", err
	}
	executable := filepath.Join(output, "subread-oracle")
	flags := []string{"-std=c++20", "-O2", "-Wall", "-Wextra", "-ffp-contract=off", "-I" + output, "-I" + filepath.Join(output, "source/dorado/utils/include")}
	units := []string{filepath.Join(output, "oracle.cpp"), filepath.Join(output, "source/dorado/utils/uuid_utils.cpp"), filepath.Join(output, "source/dorado/utils/crypto_utils.cpp"), filepath.Join(output, "source/dorado/utils/time_utils.cpp")}
	args := append(append(append([]string{}, flags...), units...), "-lcrypto", "-o", executable)
	manifest := map[string]any{"source_commit": SourceCommit, "source_pins": SourcePins, "slices_driver_headers": pins, "compiler": compiler, "compiler_sha256": Hash(compilerData), "compiler_version": string(version), "qualification": "unchanged native child functions; full original UUID/OpenSSL/time units; flat uint16 signal storage shim; supplied-spacer detection seam; no pore detection, caller, inference or GPU"}
	if err = SaveJSON(filepath.Join(output, "manifest.json"), manifest); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	if _, err = Command(output, "build", exec.CommandContext(ctx, compiler, args...), nil); err != nil {
		return "", err
	}
	exe, err := os.ReadFile(executable)
	if err != nil {
		return "", err
	}
	manifest["executable_sha256"] = Hash(exe)
	deps, err := Command(output, "ldd", exec.Command("ldd", executable), nil)
	if err != nil {
		return "", err
	}
	libs := map[string]string{}
	for _, field := range strings.Fields(string(deps)) {
		if strings.HasPrefix(field, "/") {
			data, e := os.ReadFile(field)
			if e != nil {
				return "", e
			}
			libs[field] = Hash(data)
		}
	}
	manifest["runtime_pins"] = libs
	for i, unit := range units {
		args := append(append([]string{}, flags...), "-M", unit)
		deps, e := Command(output, fmt.Sprintf("headers-%d", i), exec.Command(compiler, args...), nil)
		if e != nil {
			return "", e
		}
		joined := strings.ReplaceAll(string(deps), "\\\n", " ")
		colon := strings.IndexByte(joined, ':')
		if colon < 0 {
			return "", fmt.Errorf("invalid header dependencies")
		}
		for _, path := range strings.Fields(joined[colon+1:]) {
			data, e := os.ReadFile(path)
			if e != nil {
				return "", e
			}
			pins[path] = Hash(data)
		}
	}
	if err = VerifySources(source); err != nil {
		return "", err
	}
	if err = SaveJSON(filepath.Join(output, "manifest.json"), manifest); err != nil {
		return "", err
	}
	return executable, nil
}

type Case struct {
	Name    string
	Read    subread.Read
	Spacers []subread.Interval
}
type UUIDCase struct {
	Parent string
	Index  uint64
}

// The field-order protocol preserves every float bit and every metadata value.
func encode(w io.Writer, value reflect.Value) error {
	switch value.Kind() {
	case reflect.Struct:
		for i := 0; i < value.NumField(); i++ {
			if err := encode(w, value.Field(i)); err != nil {
				return err
			}
		}
	case reflect.String:
		if value.Len() > 1<<24 {
			return fmt.Errorf("string exceeds protocol bound")
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(value.Len())); err != nil {
			return err
		}
		_, err := io.WriteString(w, value.String())
		return err
	case reflect.Slice:
		if value.Len() > 1<<24 {
			return fmt.Errorf("slice exceeds protocol bound")
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(value.Len())); err != nil {
			return err
		}
		for i := 0; i < value.Len(); i++ {
			if err := encode(w, value.Index(i)); err != nil {
				return err
			}
		}
	default:
		return binary.Write(w, binary.LittleEndian, value.Interface())
	}
	return nil
}
func decode(r *bytes.Reader, value reflect.Value) error {
	switch value.Kind() {
	case reflect.Struct:
		for i := 0; i < value.NumField(); i++ {
			if err := decode(r, value.Field(i)); err != nil {
				return err
			}
		}
	case reflect.String, reflect.Slice:
		var n uint32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return err
		}
		if n > 1<<24 {
			return fmt.Errorf("protocol count exceeds bound")
		}
		if value.Kind() == reflect.String {
			if uint64(n) > uint64(r.Len()) {
				return io.ErrUnexpectedEOF
			}
			data := make([]byte, n)
			_, err := io.ReadFull(r, data)
			if err != nil {
				return err
			}
			value.SetString(string(data))
			return nil
		}
		if uint64(n) > uint64(r.Len()) {
			return io.ErrUnexpectedEOF
		}
		value.Set(reflect.MakeSlice(value.Type(), int(n), int(n)))
		for i := 0; i < int(n); i++ {
			if err := decode(r, value.Index(i)); err != nil {
				return err
			}
		}
	default:
		return binary.Read(r, binary.LittleEndian, value.Addr().Interface())
	}
	return nil
}
func EncodeRead(read subread.Read) []byte {
	var b bytes.Buffer
	if err := encode(&b, reflect.ValueOf(read)); err != nil {
		panic(err)
	}
	return b.Bytes()
}
func EncodeInput(cases []Case, ids []UUIDCase) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString("SUBREAD1")
	if len(cases) > 10000 || len(ids) > 10000 {
		return nil, fmt.Errorf("too many protocol cases")
	}
	if err := binary.Write(&b, binary.LittleEndian, uint32(len(cases))); err != nil {
		return nil, err
	}
	for _, c := range cases {
		if err := encode(&b, reflect.ValueOf(c.Read)); err != nil {
			return nil, err
		}
		if err := encode(&b, reflect.ValueOf(c.Spacers)); err != nil {
			return nil, err
		}
	}
	if err := binary.Write(&b, binary.LittleEndian, uint32(len(ids))); err != nil {
		return nil, err
	}
	for _, id := range ids {
		if err := encode(&b, reflect.ValueOf(id)); err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}
func DecodeOutput(data []byte, cases, ids int) ([][]subread.Read, []string, error) {
	r := bytes.NewReader(data)
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, nil, err
	}
	if int(n) != cases {
		return nil, nil, fmt.Errorf("native case count differs")
	}
	out := make([][]subread.Read, cases)
	for i := range out {
		if err := decode(r, reflect.ValueOf(&out[i]).Elem()); err != nil {
			return nil, nil, err
		}
	}
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, nil, err
	}
	if int(n) != ids {
		return nil, nil, fmt.Errorf("native UUID count differs")
	}
	uuids := make([]string, ids)
	for i := range uuids {
		if err := decode(r, reflect.ValueOf(&uuids[i]).Elem()); err != nil {
			return nil, nil, err
		}
	}
	if r.Len() != 0 {
		return nil, nil, fmt.Errorf("trailing native output")
	}
	return out, uuids, nil
}
