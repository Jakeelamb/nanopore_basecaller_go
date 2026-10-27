// Package closedbuild closes the maintained tensor compiler recipe without
// loading its outputs. Runtime admission is deliberately a separate operation.
package closedbuild

import (
	"context"
	"debug/elf"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Pin = nativeproof.Pin

func PinFile(path string) (Pin, error) {
	p, e := filepath.Abs(path)
	if e != nil {
		return Pin{}, e
	}
	p, e = filepath.EvalSymlinks(p)
	if e != nil {
		return Pin{}, e
	}
	s, e := os.Stat(p)
	if e != nil {
		return Pin{}, e
	}
	if !s.Mode().IsRegular() {
		return Pin{}, fmt.Errorf("regular file required: %s", p)
	}
	h, e := runner.SHA(p)
	return Pin{Path: p, SHA256: h}, e
}
func verify(p Pin) error {
	q, e := PinFile(p.Path)
	if e != nil {
		return e
	}
	if p.SHA256 == "" || p.SHA256 != q.SHA256 {
		return fmt.Errorf("input pin changed: %s", p.Path)
	}
	return nil
}
func read(p Pin, v any) error {
	if e := verify(p); e != nil {
		return e
	}
	b, e := os.ReadFile(p.Path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func Write(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	_, e = f.Write(append(b, '\n'))
	return errors.Join(e, f.Close())
}
func add(m map[string]string, p Pin) error {
	if e := verify(p); e != nil {
		return e
	}
	q, e := PinFile(p.Path)
	if e != nil {
		return e
	}
	if old, ok := m[q.Path]; ok && old != q.SHA256 {
		return fmt.Errorf("conflicting input %s", q.Path)
	}
	m[q.Path] = q.SHA256
	return nil
}
func checkPins(m map[string]string) error {
	for _, p := range slices.Sorted(maps.Keys(m)) {
		if e := verify(Pin{Path: p, SHA256: m[p]}); e != nil {
			return e
		}
	}
	return nil
}
func equal(a, b any) bool { return reflect.DeepEqual(a, b) }

type Tool struct {
	Name       string
	Executable Pin
}
type Domain struct {
	Name string
	Root Pin
	ELF  nativeproof.ELFClosure
}
type Compiler struct {
	Tools       []Tool
	Plugins     []Pin
	LoaderCache Pin
	SearchDirs  []string
	Domains     []Domain
	Inputs      map[string]string
}
type CompilerContract struct {
	Schema   string
	Compiler Compiler
	Recipe   struct{ Compiler Pin }
}

func (c Compiler) Check() error {
	if len(c.Tools) != 10 || len(c.Plugins) < 1 || len(c.Domains) != len(c.Tools)+len(c.Plugins) || c.LoaderCache.Path != "/etc/ld.so.cache" || !slices.Equal(c.SearchDirs, []string{"/usr/lib", "/lib"}) {
		return fmt.Errorf("incomplete selected compiler contract")
	}
	roots := map[string]Pin{}
	for _, t := range c.Tools {
		if _, ok := roots[t.Name]; ok {
			return fmt.Errorf("duplicate compiler tool")
		}
		roots[t.Name] = t.Executable
	}
	for _, name := range []string{"driver", "cc1plus", "collect2", "as", "ld", "nm", "lto-wrapper", "lto1", "cc1", "gcc-driver"} {
		if roots[name].Path == "" {
			return fmt.Errorf("selected tool missing: %s", name)
		}
	}
	plugins := []Pin{c.Plugins[0]}
	paths, e := filepath.Glob("/usr/lib/bfd-plugins/*.so*")
	if e != nil {
		return e
	}
	for _, path := range paths {
		p, e := PinFile(path)
		if e != nil {
			return e
		}
		if !slices.Contains(plugins, p) {
			plugins = append(plugins, p)
		}
	}
	sortPins := func(p []Pin) { slices.SortFunc(p, func(a, b Pin) int { return strings.Compare(a.Path, b.Path) }) }
	sortPins(plugins)
	want := slices.Clone(c.Plugins)
	sortPins(want)
	if !slices.Equal(plugins, want) {
		return fmt.Errorf("BFD plugin family changed")
	}
	for _, p := range c.Plugins {
		roots["plugin:"+p.Path] = p
	}
	inputs := map[string]string{}
	for _, d := range c.Domains {
		if roots[d.Name] != d.Root {
			return fmt.Errorf("compiler root changed: %s", d.Name)
		}
		delete(roots, d.Name)
		a, e := nativeproof.CloseELF([]Pin{d.Root}, c.SearchDirs)
		if e != nil {
			return e
		}
		if !equal(a, d.ELF) {
			return fmt.Errorf("compiler ELF domain changed: %s", d.Name)
		}
		for _, n := range a.Nodes {
			if n.Class != elf.ELFCLASS64 || n.Data != elf.ELFDATA2LSB || n.Machine != elf.EM_X86_64 {
				return fmt.Errorf("compiler ELF outside x86_64")
			}
		}
		for _, p := range a.Inputs {
			inputs[p.Path] = p.SHA256
		}
	}
	if len(roots) != 0 {
		return fmt.Errorf("compiler domains omitted")
	}
	if e := add(inputs, c.LoaderCache); e != nil {
		return e
	}
	if !equal(inputs, c.Inputs) {
		return fmt.Errorf("compiler closure union differs")
	}
	return checkPins(inputs)
}

// Limits reads effective cgroup files rather than trusting launch arguments.
func Limits() (map[string]string, error) {
	b, e := os.ReadFile("/proc/self/cgroup")
	if e != nil {
		return nil, e
	}
	rel := ""
	for _, s := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(s, "0::/") {
			rel = strings.TrimPrefix(s, "0::/")
		}
	}
	if rel == "" || strings.Contains(rel, "..") {
		return nil, fmt.Errorf("bounded cgroup v2 required")
	}
	v := map[string]string{"cgroup": rel, "child_timeout_seconds": "180"}
	for _, n := range []string{"cpu.max", "memory.max", "memory.swap.max", "pids.max"} {
		b, e = os.ReadFile(filepath.Join("/sys/fs/cgroup", rel, n))
		if e != nil {
			return nil, e
		}
		v[n] = strings.TrimSpace(string(b))
	}
	p := strings.Fields(v["cpu.max"])
	if len(p) != 2 {
		return nil, fmt.Errorf("two CPU limit required")
	}
	q, e := strconv.ParseUint(p[0], 10, 64)
	if e != nil {
		return nil, e
	}
	period, e := strconv.ParseUint(p[1], 10, 64)
	if e != nil || period == 0 || q == 0 || q/period > 2 || q/period == 2 && q%period != 0 {
		return nil, fmt.Errorf("CPU exceeds2")
	}
	memory, e := strconv.ParseUint(v["memory.max"], 10, 64)
	if e != nil || memory == 0 || memory > 3<<30 || v["memory.swap.max"] != "0" {
		return nil, fmt.Errorf("3GiB/swap0 required")
	}
	tasks, e := strconv.ParseUint(v["pids.max"], 10, 64)
	if e != nil || tasks == 0 || tasks > 64 {
		return nil, fmt.Errorf("at most64 tasks required")
	}
	return v, nil
}
func command(out, label, cwd string, args, env []string) (r runner.CommandReceipt, err error) {
	if _, err = Limits(); err != nil {
		return r, err
	}
	r = runner.CommandReceipt{Args: slices.Clone(args), Environment: append([]string{"CLOSED_BUILD_CWD=" + cwd}, env...), Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
	a, e := os.OpenFile(r.Stdout, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return r, e
	}
	b, e := os.OpenFile(r.Stderr, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		a.Close()
		return r, e
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, args[0], args[1:]...)
	c.Dir = cwd
	c.Env = slices.Clone(env)
	c.Stdout = a
	c.Stderr = b
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	c.Cancel = func() error { return syscall.Kill(-c.Process.Pid, syscall.SIGKILL) }
	c.WaitDelay = 2 * time.Second
	err = c.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if c.ProcessState != nil {
		r.Exit = c.ProcessState.ExitCode()
	}
	err = errors.Join(err, ctx.Err(), a.Close(), b.Close())
	_, e = Limits()
	err = errors.Join(err, e)
	r.StdoutSHA, e = runner.SHA(r.Stdout)
	err = errors.Join(err, e)
	r.StderrSHA, e = runner.SHA(r.Stderr)
	return r, errors.Join(err, e)
}
