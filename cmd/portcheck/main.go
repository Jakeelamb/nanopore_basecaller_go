// portcheck pins native source and records reproducible CPU parity checks.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const sourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

// CPU core qualification stays independent of optional native tensor/model tiers.
var cpuTestCommand = []string{"go", "test", "-count=1", "-v",
	"./cmd/portcheck", "./cmd/readcompare", "./internal/chunk", "./internal/signal",
	"./internal/decode", "./internal/stitch", "./internal/recordcompare",
	"./verification/backend/...", "./verification/scaling", "./verification/stitch", "./verification/decode"}

var sourceFiles = []string{
	"LICENCE.txt",
	"dorado/read_pipeline/base/chunk.cpp",
	"dorado/read_pipeline/base/include/read_pipeline/base/chunk.h",
	"dorado/read_pipeline/base/stitch.cpp",
	"dorado/read_pipeline/base/include/read_pipeline/base/stitch.h",
	"dorado/read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h",
	"dorado/read_pipeline/nodes/BasecallerNode.cpp",
	"dorado/read_pipeline/nodes/ScalerNode.cpp",
	"dorado/torch_utils/tensor_utils.cpp",
	"dorado/utils/include/utils/math_utils.h",
	"dorado/utils/include/utils/string_utils.h",
	"tests/ChunkTest.cpp",
	"tests/StitchTest.cpp",
	"dorado/basecall/decode/beam_search.cpp",
}

type sourcePin struct {
	Commit     string            `json:"commit"`
	Files      map[string]string `json:"files"`
	Submodules map[string]string `json:"submodules"`
}

type receipt struct {
	Schema       int               `json:"schema"`
	Key          string            `json:"key"`
	Source       sourcePin         `json:"source"`
	Inputs       map[string]string `json:"inputs"`
	Environment  map[string]string `json:"environment"`
	Command      []string          `json:"command"`
	Started      string            `json:"started"`
	Finished     string            `json:"finished"`
	Exit         int               `json:"exit"`
	StdoutSHA256 string            `json:"stdout_sha256"`
	StderrSHA256 string            `json:"stderr_sha256"`
	Artifacts    map[string]string `json:"artifacts"`
	Failure      string            `json:"failure,omitempty"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: portcheck source|verify --source PATH [--artifacts PATH] [--refresh]")
	}
	f := flag.NewFlagSet(args[0], flag.ContinueOnError)
	defaultSource := os.Getenv("DORADO_SOURCE")
	if defaultSource == "" {
		defaultSource = "external/dorado-source"
	}
	source := f.String("source", defaultSource, "pinned Dorado source checkout")
	artifacts := f.String("artifacts", "artifacts/parity", "ignored receipt directory")
	refresh := f.Bool("refresh", false, "make a new attempt even when valid evidence exists")
	if err := f.Parse(args[1:]); err != nil {
		return err
	}
	if f.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}
	absolute, err := filepath.Abs(*source)
	if err != nil {
		return err
	}
	pin, err := inspectSource(absolute)
	if err != nil {
		return err
	}
	if args[0] == "source" {
		return writeJSON(os.Stdout, pin)
	}
	if args[0] != "verify" {
		return fmt.Errorf("unknown command %q", args[0])
	}
	var expected sourcePin
	data, err := os.ReadFile("provenance/dorado-source.json")
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &expected); err != nil {
		return err
	}
	if !sameJSON(pin, expected) {
		return errors.New("source pin mismatch; old parity evidence does not qualify this source")
	}
	inputs, err := hashInputs(".")
	if err != nil {
		return err
	}
	environment, err := toolEnvironment()
	if err != nil {
		return err
	}
	// The absolute source path affects compiler/test behavior and belongs in the key.
	environment["DORADO_SOURCE"] = absolute
	identity := struct {
		Source      sourcePin
		Inputs      map[string]string
		Environment map[string]string
	}{pin, inputs, environment}
	encoded, err := json.Marshal(identity)
	if err != nil {
		return err
	}
	key := digest(encoded)
	if !*refresh {
		if prior, ok := cachedAttempt(*artifacts, key); ok {
			fmt.Printf("REUSED %s\nCPU package and native algorithm evidence only.\n", prior)
			return nil
		}
	}
	if err := os.MkdirAll(*artifacts, 0755); err != nil {
		return err
	}
	attempt, err := os.MkdirTemp(*artifacts, key[:16]+"-")
	if err != nil {
		return err
	}
	out, err := os.Create(filepath.Join(attempt, "stdout"))
	if err != nil {
		return err
	}
	defer out.Close()
	errout, err := os.Create(filepath.Join(attempt, "stderr"))
	if err != nil {
		return err
	}
	defer errout.Close()
	r := receipt{Schema: 1, Key: key, Source: pin, Inputs: inputs, Environment: environment,
		Command: append([]string(nil), cpuTestCommand...), Started: time.Now().UTC().Format(time.RFC3339Nano)}
	cmd := exec.Command(r.Command[0], r.Command[1:]...)
	attemptAbsolute, err := filepath.Abs(attempt)
	if err != nil {
		return err
	}
	cmd.Env = append(os.Environ(), "DORADO_SOURCE="+absolute,
		"SCALING_ARTIFACT_DIR="+filepath.Join(attemptAbsolute, "scaling"),
		"STITCH_ARTIFACT_DIR="+filepath.Join(attemptAbsolute, "stitch"),
		"CHUNK_ARTIFACT_DIR="+filepath.Join(attemptAbsolute, "chunk"),
		"DECODE_ARTIFACT_DIR="+filepath.Join(attemptAbsolute, "decode"),
		"GOWORK=off")
	cmd.Stdout, cmd.Stderr = out, errout
	runErr := cmd.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if runErr != nil {
		r.Exit = 1
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			r.Exit = exitErr.ExitCode()
		}
	}
	afterInputs, inputErr := hashInputs(".")
	afterSource, sourceErr := inspectSource(absolute)
	if inputErr != nil || sourceErr != nil || !sameJSON(inputs, afterInputs) || !sameJSON(pin, afterSource) {
		r.Exit = 1
		runErr = errors.New("verification inputs changed while tests were running")
	}
	if err := out.Sync(); err != nil {
		return err
	}
	if err := errout.Sync(); err != nil {
		return err
	}
	r.StdoutSHA256, err = hashFile(filepath.Join(attempt, "stdout"))
	if err != nil {
		return err
	}
	r.StderrSHA256, err = hashFile(filepath.Join(attempt, "stderr"))
	if err != nil {
		return err
	}
	r.Artifacts, err = hashAttemptArtifacts(attempt)
	if err != nil {
		return err
	}
	logs, err := os.ReadFile(filepath.Join(attempt, "stdout"))
	if err != nil {
		return err
	}
	if runErr == nil {
		runErr = qualifyLog(logs)
	}
	if runErr != nil {
		r.Exit = 1
		r.Failure = runErr.Error()
	}
	if err := saveJSON(filepath.Join(attempt, "receipt.json"), r); err != nil {
		return err
	}
	if runErr != nil {
		return fmt.Errorf("FAIL %s: %w", attempt, runErr)
	}
	fmt.Printf("PASS %s\nCPU package and native algorithm evidence only.\n", attempt)
	return nil
}

func inspectSource(source string) (sourcePin, error) {
	p := sourcePin{Files: map[string]string{}}
	cmd := exec.Command("git", "-C", source, "rev-parse", "HEAD")
	b, err := cmd.Output()
	if err != nil {
		return p, fmt.Errorf("read source HEAD: %w", err)
	}
	p.Commit = strings.TrimSpace(string(b))
	if p.Commit != sourceCommit {
		return p, fmt.Errorf("expected source %s, found %s", sourceCommit, p.Commit)
	}
	status, err := exec.Command("git", "-C", source, "status", "--porcelain", "--untracked-files=all").Output()
	if err != nil {
		return p, fmt.Errorf("read source status: %w", err)
	}
	if len(bytes.TrimSpace(status)) != 0 {
		return p, errors.New("Dorado source checkout is modified; use a clean pinned source oracle")
	}
	for _, name := range sourceFiles {
		hash, err := hashFile(filepath.Join(source, name))
		if err != nil {
			return p, err
		}
		p.Files[name] = hash
	}
	p.Submodules = map[string]string{}
	for name, want := range map[string]string{"dorado/3rdparty/NVTX": "e170594ac7cf1dac584da473d4ca9301087090c1"} {
		path := filepath.Join(source, name)
		got, err := exec.Command("git", "-C", path, "rev-parse", "HEAD").Output()
		if err != nil || strings.TrimSpace(string(got)) != want {
			return p, fmt.Errorf("required submodule %s is missing or has wrong HEAD", name)
		}
		dirty, err := exec.Command("git", "-C", path, "status", "--porcelain", "--untracked-files=all").Output()
		if err != nil || len(bytes.TrimSpace(dirty)) != 0 {
			return p, fmt.Errorf("required submodule %s is not clean", name)
		}
		p.Submodules[name] = want
	}
	return p, nil
}

func hashInputs(root string) (map[string]string, error) {
	inputs := map[string]string{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == filepath.Join(root, ".humanlayer", "tasks") && entry.IsDir() {
			return filepath.SkipDir
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", "artifacts", "bin", "external":
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("verification input symlink is unsupported: %s", path)
		}
		switch filepath.Ext(path) {
		case ".go", ".mod", ".sum", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".s", ".S", ".sh", ".json", ".py", ".toml":
			hash, err := hashFile(path)
			if err != nil {
				return err
			}
			inputs[filepath.ToSlash(path)] = hash
		}
		return nil
	})
	return inputs, err
}

func toolEnvironment() (map[string]string, error) {
	result := map[string]string{"GOOS": runtime.GOOS, "GOARCH": runtime.GOARCH}
	for _, name := range []string{"GOFLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_AUDIT"} {
		if os.Getenv(name) != "" {
			return nil, fmt.Errorf("unset %s for controlled native parity; external overlays and compiler injection are not pinned", name)
		}
	}
	for name, args := range map[string][]string{
		"go-version":       {"go", "version"},
		"go-env":           {"go", "env", "GOTOOLCHAIN", "CGO_ENABLED", "GOFLAGS", "GOEXPERIMENT", "GOAMD64", "GOOS", "GOARCH", "GOWORK", "GOMOD", "CC", "CXX", "CGO_CFLAGS", "CGO_CXXFLAGS", "CGO_LDFLAGS"},
		"cxx-version":      {"c++", "--version"},
		"gxx-version":      {"g++", "--version"},
		"samtools-version": {"samtools", "--version"},
	} {
		command := exec.Command(args[0], args[1:]...)
		command.Env = append(os.Environ(), "GOWORK=off")
		b, err := command.CombinedOutput()
		if err != nil {
			return nil, fmt.Errorf("%s: %w: %s", name, err, b)
		}
		result[name] = string(b)
	}
	flags, err := exec.Command("go", "env", "GOFLAGS").Output()
	if err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(flags)) != 0 {
		return nil, errors.New("effective GOFLAGS must be empty for controlled parity")
	}
	cgo, err := exec.Command("go", "env", "CGO_ENABLED").Output()
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(string(cgo)) != "1" {
		return nil, errors.New("controlled native decoder parity requires CGO_ENABLED=1")
	}
	compilers, err := exec.Command("go", "env", "CC", "CXX").Output()
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(string(compilers)) != "gcc\ng++" {
		return nil, errors.New("controlled parity requires effective Go CC=gcc and CXX=g++")
	}
	for _, name := range []string{"PATH", "CC", "CXX", "LD_LIBRARY_PATH", "GODEBUG", "GOMAXPROCS", "GOFLAGS", "CGO_CFLAGS", "CGO_LDFLAGS", "GLIBC_TUNABLES"} {
		result[name] = os.Getenv(name)
	}
	for _, name := range []string{"go", "c++", "g++", "gcc", "samtools"} {
		path, err := exec.LookPath(name)
		if err != nil {
			return nil, err
		}
		hash, err := hashFile(path)
		if err != nil {
			return nil, err
		}
		result[name+"-executable-sha256"] = hash
	}
	// Numerical libraries can change float32 rounding without a source change.
	for _, name := range []string{"libm.so.6", "libc.so.6", "libstdc++.so.6", "libgcc_s.so.1"} {
		b, err := exec.Command("g++", "-print-file-name="+name).Output()
		if err != nil {
			return nil, err
		}
		path := strings.TrimSpace(string(b))
		if !filepath.IsAbs(path) {
			return nil, fmt.Errorf("cannot identify native runtime %s", name)
		}
		hash, err := hashFile(path)
		if err != nil {
			return nil, err
		}
		result[name+"-sha256"] = hash
	}
	return result, nil
}

func cachedAttempt(root, key string) (string, bool) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", false
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		var r receipt
		b, err := os.ReadFile(filepath.Join(dir, "receipt.json"))
		if err != nil || json.Unmarshal(b, &r) != nil || r.Schema != 1 || r.Key != key || r.Exit != 0 || r.Finished == "" {
			continue
		}
		identity := struct {
			Source      sourcePin
			Inputs      map[string]string
			Environment map[string]string
		}{r.Source, r.Inputs, r.Environment}
		encoded, err := json.Marshal(identity)
		if err != nil || digest(encoded) != key || !sameJSON(r.Command, cpuTestCommand) {
			continue
		}
		outHash, outErr := hashFile(filepath.Join(dir, "stdout"))
		errHash, errErr := hashFile(filepath.Join(dir, "stderr"))
		artifacts, artifactsErr := hashAttemptArtifacts(dir)
		logs, logErr := os.ReadFile(filepath.Join(dir, "stdout"))
		if outErr == nil && errErr == nil && logErr == nil && qualifyLog(logs) == nil && artifactsErr == nil && outHash == r.StdoutSHA256 && errHash == r.StderrSHA256 && sameJSON(artifacts, r.Artifacts) {
			return dir, true
		}
	}
	return "", false
}

func qualifyLog(logs []byte) error {
	if bytes.Contains(logs, []byte("--- SKIP:")) {
		return errors.New("skipped tests cannot qualify native parity")
	}
	for _, name := range []string{"TestNativeParity", "TestPinnedNativeScaling", "TestPinnedNativeStitch", "TestPinnedNativeDecode"} {
		if !bytes.Contains(logs, []byte("--- PASS: "+name+" (")) {
			return fmt.Errorf("required native gate %s did not pass", name)
		}
	}
	return nil
}

func hashAttemptArtifacts(root string) (map[string]string, error) {
	result := map[string]string{}
	err := filepath.WalkDir(root, func(path string, e fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if e.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if rel == "receipt.json" || rel == "stdout" || rel == "stderr" {
			return nil
		}
		if e.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("unexpected receipt artifact symlink %s", path)
		}
		hash, err := hashFile(path)
		if err != nil {
			return err
		}
		result[filepath.ToSlash(rel)] = hash
		return nil
	})
	return result, err
}

func sameJSON(a, b any) bool {
	x, xErr := json.Marshal(a)
	y, yErr := json.Marshal(b)
	return xErr == nil && yErr == nil && bytes.Equal(x, y)
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeJSON(w io.Writer, value any) error {
	e := json.NewEncoder(w)
	e.SetIndent("", "  ")
	return e.Encode(value)
}

func saveJSON(path string, value any) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if err := writeJSON(f, value); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
