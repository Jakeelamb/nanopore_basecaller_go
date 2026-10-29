package graph

import (
	"context"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

type identityWire struct {
	Schema                      string
	Executable                  nativeproof.Pin
	Seeds                       []nativeproof.Pin
	LazyProviders               []nativeproof.Pin
	SearchDirs                  []string
	Closure                     nativeproof.ELFClosure
	Args, Environment           []string
	ExpectedPCI, ExpectedDriver string
	TimeoutSeconds              int
}

var pciRE = regexp.MustCompile(`^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$`)
var driverRE = regexp.MustCompile(`^[0-9]+(\.[0-9]+)+$`)

func identityArgs(p identityWire, ordinal int) []string {
	return []string{p.Executable.Path, "-i", strconv.Itoa(ordinal), "--query-gpu=index,uuid,pci.bus_id,driver_version", "--format=csv,noheader,nounits"}
}
func readIdentity(pin nativeproof.Pin) (p identityWire, err error) {
	if err = readJSON(pin, &p); err != nil {
		return
	}
	if p.Schema != "nanopore-sup-tiny-device-plan/v1" || filepath.Base(p.Executable.Path) != "nvidia-smi" || !pciRE.MatchString(p.ExpectedPCI) || !driverRE.MatchString(p.ExpectedDriver) || p.TimeoutSeconds < 1 || p.TimeoutSeconds > 60 || len(p.Seeds) == 0 || len(p.SearchDirs) == 0 {
		return p, fmt.Errorf("independent identity tool/PCI/driver/environment/bound required")
	}
	found := false
	seen := map[string]bool{}
	for _, s := range p.Seeds {
		if seen[s.Path] {
			return p, fmt.Errorf("duplicate identity seed")
		}
		seen[s.Path] = true
		if _, e := sizedPin(s); e != nil {
			return p, e
		}
		found = found || s == p.Executable
	}
	if !found {
		return p, fmt.Errorf("identity executable not in selected closure")
	}
	selected := map[string]nativeproof.Pin{p.Executable.Path: p.Executable}
	for _, v := range p.LazyProviders {
		if _, exists := selected[v.Path]; exists {
			return p, fmt.Errorf("duplicate/root lazy provider")
		}
		selected[v.Path] = v
	}
	if len(selected) != len(p.Seeds) {
		return p, fmt.Errorf("identity seeds must be exactly root and explicit lazy providers")
	}
	for _, v := range p.Seeds {
		if selected[v.Path] != v {
			return p, fmt.Errorf("identity seed lacks explicit selection")
		}
	}
	for _, dir := range p.SearchDirs {
		if err = canonical(dir); err != nil {
			return
		}
	}
	for _, v := range p.Closure.Inputs {
		if _, e := sizedPin(v); e != nil {
			return p, e
		}
	}
	closure, e := nativeproof.CloseELF(p.Seeds, p.SearchDirs)
	if e != nil || !sameIdentityClosure(closure, p.Closure) {
		return p, fmt.Errorf("selected identity ELF closure differs: %v", e)
	}
	return
}

// Original identity JSON has already rejected nulls. InspectELF represents
// absent dynamic string tables with nil; compare those ordered lists by value.
func sameIdentityClosure(a, b nativeproof.ELFClosure) bool {
	if !slices.Equal(a.Inputs, b.Inputs) || len(a.Nodes) != len(b.Nodes) || len(a.Bindings) != len(b.Bindings) {
		return false
	}
	for i, x := range a.Nodes {
		y := b.Nodes[i]
		if x.Pin != y.Pin || x.Class != y.Class || x.Data != y.Data || x.Machine != y.Machine || x.Type != y.Type || x.Interpreter != y.Interpreter || x.SONAME != y.SONAME || !slices.Equal(x.Needed, y.Needed) || !slices.Equal(x.RPath, y.RPath) || !slices.Equal(x.RunPath, y.RunPath) {
			return false
		}
	}
	for i, x := range a.Bindings {
		y := b.Bindings[i]
		if x.From != y.From || x.Kind != y.Kind || x.Name != y.Name || x.Target != y.Target || !slices.Equal(x.SearchDirs, y.SearchDirs) {
			return false
		}
	}
	return true
}

// Source witness format is compact32hex only; a GPU- prefix or hyphenated
// source UUID must not be guessed into an identity match.
func sourceUUID(s string) ([]byte, error) {
	if len(s) != 32 || s != strings.ToLower(s) {
		return nil, fmt.Errorf("SUP source UUID must be compact32hex")
	}
	b, e := hex.DecodeString(s)
	if e != nil || len(b) != 16 {
		return nil, fmt.Errorf("invalid SUP compact UUID")
	}
	return b, nil
}
func queriedUUID(s string) ([]byte, error) {
	if len(s) != 40 || !strings.HasPrefix(s, "GPU-") {
		return nil, fmt.Errorf("metadata UUID must be GPU-8-4-4-4-12")
	}
	value := s[4:]
	for _, i := range []int{8, 13, 18, 23} {
		if value[i] != '-' {
			return nil, fmt.Errorf("metadata UUID separator differs")
		}
	}
	compact := strings.ReplaceAll(value, "-", "")
	if len(compact) != 32 {
		return nil, fmt.Errorf("metadata UUID length differs")
	}
	b, e := hex.DecodeString(compact)
	if e != nil || len(b) != 16 {
		return nil, fmt.Errorf("metadata UUID hex differs")
	}
	return b, nil
}
func checkIdentityBody(body []byte, p identityWire, ordinal int, source string) error {
	want, e := sourceUUID(source)
	if e != nil {
		return e
	}
	text := strings.TrimSpace(string(body))
	if text == "" || strings.ContainsAny(text, "\r\n") {
		return fmt.Errorf("exactly one device metadata line required")
	}
	fields := strings.Split(text, ",")
	if len(fields) != 4 {
		return fmt.Errorf("metadata field count differs")
	}
	for i := range fields {
		fields[i] = strings.TrimSpace(fields[i])
	}
	if fields[0] != strconv.Itoa(ordinal) || fields[2] != p.ExpectedPCI || fields[3] != p.ExpectedDriver {
		return fmt.Errorf("ordinal/PCI/driver identity differs")
	}
	got, e := queriedUUID(fields[1])
	if e != nil || !slices.Equal(want, got) {
		return fmt.Errorf("source UUID metadata differs: %v", e)
	}
	return nil
}
func identityCommand(w admissionWire, name string, p identityWire) supgpu.GuardCommand {
	return supgpu.GuardCommand{Directory: w.Command.Directory, Output: w.Command.Output + "-identity-" + name, Args: slices.Clone(p.Args)}
}

// The identity options do not refer back to their guard plan, avoiding hash cycles.
func identityPolicy(p identityWire, plan supgpu.GuardPlan, audit supgpu.Pin) error {
	root, err := sizedPin(p.Executable)
	if err != nil {
		return err
	}
	policy := plan.Policy
	if policy.Reference != root || policy.Observer != root || policy.Audit != audit || !slices.Equal(policy.Directories, p.SearchDirs) || !slices.Equal(p.Environment, supgpu.SelectedModelEnvironment(policy)) {
		return fmt.Errorf("metadata root/auditor/directories/clean environment differs")
	}
	// Only the executable's dependency graph is mandatory at startup. Explicit
	// lazy seeds and their dependencies remain optional until actually opened.
	startup := map[string]bool{root.Path: true}
	for changed := true; changed; {
		changed = false
		for _, b := range p.Closure.Bindings {
			if startup[b.From] && !startup[b.Target.Path] {
				startup[b.Target.Path] = true
				changed = true
			}
		}
	}
	allowed := map[string]supgpu.RuntimeCode{}
	for _, c := range policy.Allowed {
		allowed[c.Object.Path] = c
	}
	count := 0
	for _, m := range p.Closure.Nodes {
		if m.Pin == p.Executable {
			continue
		}
		v, e := sizedPin(m.Pin)
		if e != nil {
			return e
		}
		c, ok := allowed[v.Path]
		if !ok || c.Object != v || c.SONAME != m.SONAME || !slices.Equal(c.Needed, m.Needed) || c.Startup != startup[v.Path] {
			return fmt.Errorf("metadata provider/startup ELF selection differs: %s", v.Path)
		}
		count++
	}
	if count != len(policy.Allowed) {
		return fmt.Errorf("metadata policy contains unselected provider")
	}
	return nil
}

func executeIdentity(ctx context.Context, w admissionWire, name string, p identityWire, ordinal int, source string) (r supgpu.GuardedRun, err error) {
	for _, v := range p.Closure.Inputs {
		if _, e := sizedPin(v); e != nil {
			return r, e
		}
	}
	closure, e := nativeproof.CloseELF(p.Seeds, p.SearchDirs)
	if e != nil || !sameIdentityClosure(closure, p.Closure) {
		return r, fmt.Errorf("live metadata tool closure changed: %v", e)
	}
	commandCtx, cancel := context.WithTimeout(ctx, time.Duration(p.TimeoutSeconds)*time.Second)
	defer cancel()
	r, err = supgpu.ExecuteGuarded(commandCtx, w.MetadataGuardPlan, w.MetadataContract, identityCommand(w, name, p))
	if err == nil {
		err = validateIdentity(r, w, name, p, ordinal, source)
	}
	return
}
func validateIdentity(run supgpu.GuardedRun, w admissionWire, name string, p identityWire, ordinal int, source string) error {
	if _, e := supgpu.ReplayGuarded(w.MetadataGuardPlan, w.MetadataContract, identityCommand(w, name, p), run); e != nil {
		return e
	}
	r := run.Command
	a, e := time.Parse(time.RFC3339Nano, r.Started)
	b, f := time.Parse(time.RFC3339Nano, r.Finished)
	if e != nil || f != nil || b.Before(a) || b.Sub(a) > time.Duration(p.TimeoutSeconds+1)*time.Second || r.Exit != 0 || !slices.Equal(r.Args, p.Args) || !slices.Equal(r.Args, identityArgs(p, ordinal)) {
		return fmt.Errorf("metadata exact argv/environment/exit/clocks differ")
	}
	pin, e := sizedPin(nativeproof.Pin{Path: r.Stdout, SHA256: r.StdoutSHA})
	if e != nil {
		return e
	}
	body, e := readSized(pin.Path, pin, 64<<10)
	if e != nil {
		return e
	}
	if _, e = sizedPin(nativeproof.Pin{Path: r.Stderr, SHA256: r.StderrSHA}); e != nil {
		return e
	}
	return checkIdentityBody(body, p, ordinal, source)
}
