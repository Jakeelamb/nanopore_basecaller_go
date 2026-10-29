package supgpu

import (
	"debug/elf"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// PhaseContract is a caller-owned, independently pinned exact phase sequence.
// It is not inferred from a target or a saved transcript.
type PhaseContract struct {
	Schema string   `json:"schema"`
	Role   string   `json:"role"`
	Action string   `json:"action"`
	Names  []string `json:"names"`
}

// GuardPlan contains the source-specific selected runtime and its consumed
// custody inputs. The caller authenticates this plan in its admission manifest.
// No SUP model, stock executable or compiler is required by this interface.
type GuardPlan struct {
	Schema   string        `json:"schema"`
	Policy   RuntimePolicy `json:"policy"`
	Contract Pin           `json:"phase_contract"`
	Inputs   []Pin         `json:"inputs"`
}

// GuardCommand is the expected invocation, supplied separately at execution
// and replay. A receipt cannot select its own argv or output directory.
type GuardCommand struct {
	Directory string   `json:"directory"`
	Output    string   `json:"output"`
	Args      []string `json:"args"`
}

// GuardedRun binds an actual command and the unchanged core transcript format
// to the externally pinned plan and phase contract.
type GuardedRun struct {
	Schema     string                `json:"schema"`
	Plan       Pin                   `json:"plan"`
	Contract   Pin                   `json:"phase_contract"`
	Command    runner.CommandReceipt `json:"command"`
	Transcript Pin                   `json:"transcript"`
}

var phaseNameRE = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}$`)

func validatePhaseContract(c PhaseContract) error {
	if c.Schema != "nanopore-audited-phase-contract/v1" || (c.Role != "reference" && c.Role != "observer") || !phaseNameRE.MatchString(c.Action) || len(c.Names) < 1 || len(c.Names) > 32 || c.Names[0] != "pre-model" {
		return fmt.Errorf("invalid independently expected phase contract")
	}
	seen := map[string]bool{}
	for _, name := range c.Names {
		if !phaseNameRE.MatchString(name) || seen[name] {
			return fmt.Errorf("duplicate/invalid expected phase name")
		}
		seen[name] = true
	}
	return nil
}

func readGuardPlan(planPin, contractPin Pin) (p GuardPlan, c PhaseContract, err error) {
	for _, v := range []Pin{planPin, contractPin} {
		if err = verifyPin("", v); err != nil {
			return
		}
	}
	if err = readPinnedJSON(planPin.Path, planPin.SHA256, &p); err != nil {
		return
	}
	if p.Schema != "nanopore-audited-guard-plan/v1" || p.Contract != contractPin || len(p.Inputs) == 0 {
		err = fmt.Errorf("guard plan differs from independent phase/custody contract")
		return
	}
	if err = readPinnedJSON(contractPin.Path, contractPin.SHA256, &c); err != nil {
		return
	}
	if err = validatePhaseContract(c); err != nil {
		return
	}
	if err = validateGuardPolicy(p.Policy); err != nil {
		return
	}
	seen := map[string]bool{}
	for _, v := range p.Inputs {
		if !filepath.IsAbs(v.Path) || seen[v.Path] {
			err = fmt.Errorf("guard custody input path duplicate/nonabsolute")
			return
		}
		seen[v.Path] = true
		if err = verifyPin("", v); err != nil {
			return
		}
	}
	return
}

func validateGuardPolicy(p RuntimePolicy) error {
	if p.Schema != "nanopore-supgpu-selected-runtime/v1" || p.Lazy != "audit-every-loader-object-before-initializers/v1" || len(p.Directories) == 0 {
		return fmt.Errorf("selected runtime/initializer-veto policy differs")
	}
	pins := []Pin{p.Reference, p.Observer, p.Audit}
	allowed := map[string]Pin{}
	paths := map[string]bool{}
	for _, c := range p.Allowed {
		if c.Tier == "" || (c.SONAME != "" && allowed[c.SONAME].Path != "") || paths[c.Object.Path] {
			return fmt.Errorf("runtime allowed provider is missing/ambiguous")
		}
		if c.SONAME != "" {
			allowed[c.SONAME] = c.Object
		}
		paths[c.Object.Path] = true
		pins = append(pins, c.Object)
	}
	for _, c := range p.Alternates {
		if c.SONAME == "" || c.Tier == "" || paths[c.Object.Path] || c.Object == p.Reference || c.Object == p.Observer || c.Object == p.Audit {
			return fmt.Errorf("rejected runtime provider conflicts with selected code")
		}
		paths[c.Object.Path] = true
		pins = append(pins, c.Object)
	}
	dirs := map[string]bool{}
	for _, dir := range p.Directories {
		if !filepath.IsAbs(dir) || dirs[dir] {
			return fmt.Errorf("runtime loader directory duplicate/nonabsolute")
		}
		dirs[dir] = true
	}
	for _, v := range pins {
		if !filepath.IsAbs(v.Path) {
			return fmt.Errorf("runtime code pin must be absolute")
		}
		if err := verifyPin("", v); err != nil {
			return err
		}
	}
	for _, c := range p.Allowed {
		if c.SONAME != "" {
			continue
		}
		if c.Startup || c.Object.Path == p.Reference.Path || c.Object.Path == p.Observer.Path || c.Object.Path == p.Audit.Path {
			return fmt.Errorf("unnamed runtime image cannot be startup/root/audit code")
		}
		m, err := nativeproof.InspectELF(c.Object.Path)
		if err != nil {
			return fmt.Errorf("unnamed runtime image inspection: %w", err)
		}
		if m.Pin.Path != c.Object.Path || m.Pin.SHA256 != c.Object.SHA256 || m.Type != elf.ET_DYN || m.SONAME != "" || m.Interpreter != "" || !slices.Equal(m.Needed, c.Needed) {
			return fmt.Errorf("unnamed runtime image differs from canonical ELF metadata")
		}
		// Recheck the sized pin after inspection; nil and [] Needed compare equal,
		// but dependency order is retained and every name needs a named selection.
		if err := verifyPin("", c.Object); err != nil {
			return err
		}
		for _, name := range c.Needed {
			if name == "" || allowed[name].Path == "" {
				return fmt.Errorf("unnamed runtime dependency has no selected named provider: %q", name)
			}
		}
	}
	return nil
}

func validateGuardCommand(command GuardCommand, p GuardPlan, c PhaseContract) error {
	root := p.Policy.Reference
	if c.Role == "observer" {
		root = p.Policy.Observer
	}
	if !filepath.IsAbs(command.Directory) || !filepath.IsAbs(command.Output) || len(command.Args) == 0 || command.Args[0] != root.Path {
		return fmt.Errorf("guard command differs from selected role/absolute directories")
	}
	return nil
}

// ExecuteGuarded launches one selected target through the existing audited
// loader/phase core. The caller must obtain its exact native-phase admission
// before calling; this API does not invent a second boolean admission policy.
// Output must be a fresh directory. Failure receipts and maps are retained.
func ExecuteGuarded(planPin, contractPin Pin, command GuardCommand) (GuardedRun, error) {
	return executeGuarded(planPin, contractPin, command, readCodeMaps)
}

func executeGuarded(planPin, contractPin Pin, command GuardCommand, readMaps mapReader) (run GuardedRun, err error) {
	run = GuardedRun{Schema: "nanopore-audited-guard-run/v1", Plan: planPin, Contract: contractPin}
	p, c, err := readGuardPlan(planPin, contractPin)
	if err != nil {
		return run, err
	}
	if err = validateGuardCommand(command, p, c); err != nil {
		return run, err
	}
	if err = os.Mkdir(command.Output, 0755); err != nil {
		return run, err
	}
	defer func() {
		if e := writeJSON(filepath.Join(command.Output, c.Role+"-guarded.json"), run); err == nil {
			err = e
		}
	}()
	run.Command, err = executeGuardCore(command.Directory, command.Output, c.Role, c.Action, command.Args, p.Policy, planPin, c.Names, readMaps)
	var e error
	run.Transcript, e = pin(filepath.Join(command.Output, c.Role+"-guard.json"))
	if err == nil {
		err = e
	}
	return run, err
}

// ReplayGuarded validates saved bytes only. Expected plan, phase contract and
// command must come from the caller's authenticated admission, never from run.
// A rehashed/truncated receipt cannot replace the independent phase sequence.
func ReplayGuarded(planPin, contractPin Pin, command GuardCommand, run GuardedRun) ([]Pin, error) {
	p, c, err := readGuardPlan(planPin, contractPin)
	if err != nil {
		return nil, err
	}
	if err = validateGuardCommand(command, p, c); err != nil {
		return nil, err
	}
	if run.Schema != "nanopore-audited-guard-run/v1" || run.Plan != planPin || run.Contract != contractPin || !reflect.DeepEqual(run.Command.Args, command.Args) || run.Command.Stdout != filepath.Join(command.Output, c.Role+".stdout") || run.Command.Stderr != filepath.Join(command.Output, c.Role+".stderr") || run.Transcript.Path != filepath.Join(command.Output, c.Role+"-guard.json") {
		return nil, fmt.Errorf("guarded run differs from independently expected invocation")
	}
	if err = verifyCommand(run.Command); err != nil {
		return nil, err
	}
	if err = verifyPin("", run.Transcript); err != nil {
		return nil, err
	}
	var g GuardReceipt
	if err = readPinnedJSON(run.Transcript.Path, run.Transcript.SHA256, &g); err != nil {
		return nil, err
	}
	expectedEnv := append([]string{"SUPGPU_COMMAND_CWD=" + command.Directory, "SUPGPU_GATE_FDS=3,4,5,6"}, modelEnvironment(p.Policy)...)
	expectedEnv = append(expectedEnv, fmt.Sprintf("SUPGPU_TARGET_PID=%d", g.PID))
	if !reflect.DeepEqual(run.Command.Environment, expectedEnv) {
		return nil, fmt.Errorf("guarded actual command environment differs")
	}
	return replayGuardCore(g, p.Policy, planPin, c.Role, c.Action, c.Names)
}
