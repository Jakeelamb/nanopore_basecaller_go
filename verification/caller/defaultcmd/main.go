// Command defaultcmd qualifies cached default postprocessing without inference.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type runner struct {
	dir   string
	steps []step
}
type step struct {
	Name                       string
	Args                       []string
	Started, Finished          time.Time
	Exit                       int
	StdoutSHA256, StderrSHA256 string
}

func sha(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func (r *runner) command(name string, args ...string) ([]byte, []byte, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	s := step{Name: name, Args: args, Started: time.Now().UTC(), Exit: -1}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	var out, errout bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errout
	runErr := cmd.Run()
	s.Finished = time.Now().UTC()
	if cmd.ProcessState != nil {
		s.Exit = cmd.ProcessState.ExitCode()
	}
	s.StdoutSHA256, s.StderrSHA256 = sha(out.Bytes()), sha(errout.Bytes())
	r.steps = append(r.steps, s)
	if err := os.WriteFile(filepath.Join(r.dir, name+".stdout"), out.Bytes(), 0600); err != nil {
		return nil, nil, -1, err
	}
	if err := os.WriteFile(filepath.Join(r.dir, name+".stderr"), errout.Bytes(), 0600); err != nil {
		return nil, nil, -1, err
	}
	if err := writeJSON(filepath.Join(r.dir, name+".receipt.json"), s); err != nil {
		return nil, nil, -1, err
	}
	return out.Bytes(), errout.Bytes(), s.Exit, runErr
}
func (r *runner) pass(name string, args ...string) ([]byte, error) {
	out, errout, code, err := r.command(name, args...)
	if err != nil || code != 0 {
		return nil, fmt.Errorf("%s exited %d: %w\n%s", name, code, err, errout)
	}
	return out, nil
}
func writeJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
func read(path string) ([]byte, error) { return os.ReadFile(path) }
func copyDir(from, to string) error {
	if err := os.Mkdir(to, 0700); err != nil {
		return err
	}
	entries, err := os.ReadDir(from)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return errors.New("unexpected nested postcapture")
		}
		b, err := read(filepath.Join(from, entry.Name()))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(to, entry.Name()), b, 0600); err != nil {
			return err
		}
	}
	return nil
}
func pins(paths []string) (map[string]string, error) {
	result := map[string]string{}
	for _, root := range paths {
		if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			b, e := os.ReadFile(path)
			if e != nil {
				return e
			}
			result[path] = sha(b)
			return nil
		}); err != nil {
			return nil, err
		}
	}
	return result, nil
}
func samePins(a, b map[string]string) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}
func object(b []byte) (map[string]any, error) {
	var m map[string]any
	e := json.Unmarshal(b, &m)
	return m, e
}
func endpoint(p map[string]any, key string) (map[string]any, error) {
	v, ok := p[key].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("missing %s", key)
	}
	return v, nil
}
func replacePayload(dir string, m map[string]any, key string, data []byte) error {
	p, err := endpoint(m, key)
	if err != nil {
		return err
	}
	name, ok := p["File"].(string)
	if !ok {
		return errors.New("missing payload file")
	}
	if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
		return err
	}
	p["Bytes"], p["SHA256"] = len(data), sha(data)
	return nil
}
func qualify(r *runner, hac, sup, stock string) (map[string]any, error) {
	roots := []string{"internal/caller", "internal/postcall", "internal/align", "internal/ends", "internal/splitter", "internal/subread", "internal/readtrim", "cmd/basecaller", "verification/caller", "docs/caller.md"}
	sourceBefore, err := pins(roots)
	if err != nil {
		return nil, err
	}
	inputRoots := []string{filepath.Dir(hac), filepath.Dir(sup), stock}
	inputsBefore, err := pins(inputRoots)
	if err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(r.dir, "sources-before.json"), sourceBefore); err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(r.dir, "inputs-before.json"), inputsBefore); err != nil {
		return nil, err
	}
	binary := filepath.Join(r.dir, "basecaller")
	if _, err := r.pass("build", "go", "build", "-o", binary, "./cmd/basecaller"); err != nil {
		return nil, err
	}
	if _, err := exec.LookPath("strace"); err != nil {
		return nil, errors.New("strace is required for default replay qualification")
	}
	publication, err := publicationFaults(r, binary, hac)
	if err != nil {
		return nil, err
	}
	type modelCase struct {
		name, core, disabled, defaultControl, defaultHash string
		bases                                             int
	}
	models := []modelCase{
		{"hac", hac, filepath.Join(stock, "stock-hac-dev1-001/stdout"), filepath.Join(stock, "stock-hac-default-dev1-001/stdout"), "8e07d67e28ac3eb6c6f9311590d749ed7f5faaa06614196b93272878f57a529a", 1615},
		{"sup", sup, filepath.Join(stock, "stock-sup-dev1-001/stdout"), filepath.Join(stock, "stock-sup-default-dev1-001/stdout"), "4b9a63131ded41457816f13b420a7f74f0faf5ad32e7c057115d3d733f17952d", 1619},
	}
	var comparisons []map[string]any
	for _, model := range models {
		defaultBytes, err := read(model.defaultControl)
		if err != nil {
			return nil, err
		}
		if sha(defaultBytes) != model.defaultHash {
			return nil, fmt.Errorf("%s stock default hash changed", model.name)
		}
		disabledBytes, err := read(model.disabled)
		if err != nil {
			return nil, err
		}
		baseline, err := r.pass(model.name+"-core", binary, "--replay", model.core)
		if err != nil {
			return nil, err
		}
		if !bytes.Equal(baseline, disabledBytes) {
			return nil, errors.New("core regression differs")
		}
		for _, trim := range []bool{true, false} {
			for _, split := range []bool{true, false} {
				name := fmt.Sprintf("%s-trim-%t-split-%t", model.name, trim, split)
				capture := filepath.Join(r.dir, name+"-capture")
				args := []string{binary, "--replay", model.core, "--postprocess", "default", "--capture", capture}
				if !trim {
					args = append(args, "--no-trim")
				}
				if !split {
					args = append(args, "--disable-read-splitting")
				}
				actual, err := r.pass(name, args...)
				if err != nil {
					return nil, err
				}
				expected := disabledBytes
				if trim {
					expected = defaultBytes
				}
				if !bytes.Equal(actual, expected) {
					return nil, fmt.Errorf("%s differs from retained stock control", name)
				}
				replay, err := r.pass(name+"-replay", binary, "--replay", filepath.Join(capture, "processed-manifest.json"))
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(replay, actual) {
					return nil, fmt.Errorf("%s final replay differs", name)
				}
				comparisons = append(comparisons, map[string]any{"model": model.name, "trim": trim, "split": split, "raw_stock_exact": true, "processed_replay_exact": true, "fastq_sha256": sha(actual), "records": 1, "bases": len(bytes.Split(actual, []byte("\n"))[1])})
			}
		}
	}
	seed := filepath.Join(r.dir, "hac-trim-true-split-true-capture")
	negatives := []string{"missing", "malformed", "changed-output", "wrong-hash", "policy-swapped", "core-swapped", "core-relative", "payload-traversal", "symlink", "manifest-trailing", "unknown-field", "output-count"}
	for _, kind := range negatives {
		dir := filepath.Join(r.dir, "reject-"+kind)
		if err := copyDir(seed, dir); err != nil {
			return nil, err
		}
		path := filepath.Join(dir, "processed-manifest.json")
		data, err := read(path)
		if err != nil {
			return nil, err
		}
		m, err := object(data)
		if err != nil {
			return nil, err
		}
		switch kind {
		case "missing":
			if err := os.Remove(filepath.Join(dir, "processed-output.json")); err != nil {
				return nil, err
			}
		case "malformed":
			if err := replacePayload(dir, m, "Output", []byte("{")); err != nil {
				return nil, err
			}
		case "changed-output":
			b, err := read(filepath.Join(dir, "processed-output.json"))
			if err != nil {
				return nil, err
			}
			o, err := object(b)
			if err != nil {
				return nil, err
			}
			o["Reads"] = []any{}
			b, err = json.Marshal(o)
			if err != nil {
				return nil, err
			}
			if err := replacePayload(dir, m, "Output", b); err != nil {
				return nil, err
			}
		case "wrong-hash":
			p, err := endpoint(m, "FASTQ")
			if err != nil {
				return nil, err
			}
			p["SHA256"] = strings.Repeat("0", 64)
		case "policy-swapped":
			b, err := read(filepath.Join(dir, "processed-policy.json"))
			if err != nil {
				return nil, err
			}
			p, err := object(b)
			if err != nil {
				return nil, err
			}
			p["Trim"] = false
			b, err = json.Marshal(p)
			if err != nil {
				return nil, err
			}
			if err := replacePayload(dir, m, "Policy", b); err != nil {
				return nil, err
			}
		case "core-swapped":
			p, err := endpoint(m, "Core")
			if err != nil {
				return nil, err
			}
			b, err := read(sup)
			if err != nil {
				return nil, err
			}
			p["Path"], p["Bytes"], p["SHA256"] = sup, len(b), sha(b)
		case "core-relative":
			p, err := endpoint(m, "Core")
			if err != nil {
				return nil, err
			}
			p["Path"] = "../manifest.json"
		case "payload-traversal":
			p, err := endpoint(m, "Output")
			if err != nil {
				return nil, err
			}
			p["File"] = "../processed-output.json"
		case "symlink":
			target := filepath.Join(dir, "processed.fastq")
			if err := os.Rename(target, target+".saved"); err != nil {
				return nil, err
			}
			if err := os.Symlink(target+".saved", target); err != nil {
				return nil, err
			}
		case "manifest-trailing":
			data = append(data, []byte("{}")...)
		case "unknown-field":
			m["Unknown"] = true
		case "output-count":
			m["OutputReads"] = 2
		}
		if kind != "manifest-trailing" {
			data, err = json.Marshal(m)
			if err != nil {
				return nil, err
			}
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			return nil, err
		}
		out, _, code, runErr := r.command("reject-"+kind, binary, "--replay", path)
		if code == 0 || runErr == nil || len(out) != 0 {
			return nil, fmt.Errorf("%s did not reject with empty stdout", kind)
		}
	}
	tracePath := filepath.Join(r.dir, "default.strace")
	traced, err := r.pass("trace", "strace", "-f", "-e", "trace=execve,openat", "-o", tracePath, binary, "--replay", filepath.Join(seed, "processed-manifest.json"))
	if err != nil {
		return nil, err
	}
	expected, err := read(models[0].defaultControl)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(traced, expected) {
		return nil, errors.New("traced FASTQ differs")
	}
	trace, err := read(tracePath)
	if err != nil {
		return nil, err
	}
	var execs []string
	for _, line := range strings.Split(string(trace), "\n") {
		if strings.Contains(line, "execve(") {
			execs = append(execs, line)
		}
	}
	if len(execs) != 1 || !strings.Contains(execs[0], binary) {
		return nil, fmt.Errorf("replay launched unexpected processes: %v", execs)
	}
	for _, blocked := range []string{"libtorch", "libpod5", "libnanopore_tensor", "/dorado\"", "/models/"} {
		if bytes.Contains(trace, []byte(blocked)) {
			return nil, fmt.Errorf("replay initialized unsupported dependency %s", blocked)
		}
	}
	sourceAfter, err := pins(roots)
	if err != nil {
		return nil, err
	}
	inputAfter, err := pins(inputRoots)
	if err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(r.dir, "sources-after.json"), sourceAfter); err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(r.dir, "inputs-after.json"), inputAfter); err != nil {
		return nil, err
	}
	if !samePins(sourceBefore, sourceAfter) || !samePins(inputsBefore, inputAfter) {
		return nil, errors.New("source or retained input changed during qualification")
	}
	sort.Strings(negatives)
	return map[string]any{"schema": 1, "qualified": "cached plain DNA CPU default composition; no inference", "comparisons": comparisons, "manifest_publication_faults": publication, "negative_controls": negatives, "negative_stdout_bytes": 0, "core_and_stock_immutable": true, "sources_unchanged": true, "trace_execve_count": len(execs), "no_model_pod5_or_dorado_initialization": true, "steps": r.steps}, nil
}
func main() {
	root, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	hac := flag.String("hac", filepath.Join(root, "artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json"), "cached HAC core")
	sup := flag.String("sup", filepath.Join(root, "artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json"), "cached SUP core")
	stock := flag.String("stock", filepath.Join(root, "artifacts/native-cpu-control"), "retained stock controls root")
	output := flag.String("output", "", "new evidence directory")
	flag.Parse()
	if *output == "" {
		*output, err = os.MkdirTemp(filepath.Join(root, "artifacts/caller"), "default-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	} else {
		err = os.Mkdir(*output, 0700)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	absolute, err := filepath.Abs(*output)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(absolute)
	r := &runner{dir: absolute}
	result, runErr := qualify(r, *hac, *sup, *stock)
	if runErr != nil {
		result = map[string]any{"error": runErr.Error(), "steps": r.steps}
		if err := writeJSON(filepath.Join(absolute, "failure.json"), result); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		fmt.Fprintln(os.Stderr, runErr)
		os.Exit(1)
	}
	if err := writeJSON(filepath.Join(absolute, "summary.json"), result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stdout, "PASS cached four-mode postprocessing for HAC and SUP; manifest write and sync failures; 12 rejected CLI captures; trace and immutable inputs verified")
}
