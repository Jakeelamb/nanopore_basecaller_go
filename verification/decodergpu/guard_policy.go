package decodergpu

import (
	"debug/elf"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const decoderBuildSHA = "5a4bfa0161d337f62e897cb88d083e01b06600fc2ab338000c04e56954abfcd1"
const decoderBinarySHA = "897f6e25716177bbb6db7a619e256c406935c127c0a45cf92b56da5daea601f5"
const auditorSHA = "e6be01d8c8c67348efec0ef3e14674cbd0654b6f3f6b7498a766b01ad3e1ab4c"
const qualifiedRuntimeSHA = "f1fae0cc36df9c503f5d8554d053276744368214499a947b2f94d493754d25ec"
const roleWatcherSHA = "271cb2c23de4dee042515c41e66ff8ad07593df5852ccfdba046b1838fef80c0"

func decoderPhases() supgpu.PhaseContract {
	return supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "reference", Action: "decoder", Names: []string{"pre-model", "pre-forward", "post-back-guide", "post-beam-search", "post-posterior", "post-staged"}}
}

func guardPin(f File) (supgpu.Pin, error) {
	if err := f.verify(); err != nil {
		return supgpu.Pin{}, err
	}
	info, err := os.Stat(f.Path)
	if err != nil {
		return supgpu.Pin{}, err
	}
	return supgpu.Pin{Path: f.Path, SHA256: f.SHA256, Bytes: info.Size()}, nil
}

// The proof is a separately accepted auditor producer, not a new D01 build.
type auditorProof struct {
	Schema     string                  `json:"schema"`
	Audit      supgpu.Pin              `json:"auditor"`
	Runtime    supgpu.Pin              `json:"accepted_runtime_metadata"`
	Source     supgpu.Pin              `json:"auditor_source"`
	Object     supgpu.Pin              `json:"auditor_object"`
	Depfile    supgpu.Pin              `json:"auditor_depfile"`
	PreDepfile supgpu.Pin              `json:"auditor_precompile_depfile"`
	LinkMap    supgpu.Pin              `json:"auditor_link_map"`
	Commands   []runner.CommandReceipt `json:"auditor_commands"`
}

func readAuditor(p File) (a auditorProof, q supgpu.RuntimePlan, inputs map[string]string, err error) {
	inputs = map[string]string{}
	if err = readJSON(p, &a); err != nil {
		return
	}
	if a.Schema != "nanopore-supgpu-auditor-producer-proof/v1" || a.Audit.SHA256 != auditorSHA || a.Source.SHA256 != "25836caa2b5f923b805e054d79672fd46f131ec968f58a2e8718fab88a44cdd4" || a.Runtime.SHA256 != qualifiedRuntimeSHA || len(a.Commands) != 2 {
		err = fmt.Errorf("accepted auditor/qualification identity differs")
		return
	}
	for _, pin := range []supgpu.Pin{a.Audit, a.Runtime, a.Source, a.Object, a.Depfile, a.PreDepfile, a.LinkMap} {
		f := File{pin.Path, pin.SHA256}
		var actual supgpu.Pin
		actual, err = guardPin(f)
		if err != nil {
			return
		}
		if actual != pin {
			err = fmt.Errorf("auditor proof file identity differs")
			return
		}
		inputs[pin.Path] = pin.SHA256
	}
	for _, c := range a.Commands {
		if err = commandEvidence(c); err != nil {
			return
		}
		inputs[c.Stdout] = c.StdoutSHA
		inputs[c.Stderr] = c.StderrSHA
	}
	if err = readJSON(File{a.Runtime.Path, a.Runtime.SHA256}, &q); err != nil {
		return
	}
	if q.Schema != "nanopore-supgpu-runtime-plan/v3" || q.Policy.Audit != a.Audit {
		err = fmt.Errorf("auditor and qualified runtime binding differs")
		return
	}
	f, e := elf.Open(a.Audit.Path)
	if e != nil {
		err = e
		return
	}
	defer f.Close()
	needed, e := f.ImportedLibraries()
	if e != nil {
		err = e
		return
	}
	imports, e := f.ImportedSymbols()
	if e != nil {
		err = e
		return
	}
	if len(needed) != 0 || len(imports) != 0 {
		err = fmt.Errorf("accepted initializer auditor has dependencies/imports")
		return
	}
	inputs[p.Path] = p.SHA256
	return
}

func selectedDecoderPolicy(closure nativeproof.ELFClosure, binary, audit supgpu.Pin, qualification supgpu.RuntimePolicy, directories []string) (p supgpu.RuntimePolicy, err error) {
	p = supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: binary, Observer: binary, Audit: audit, Allowed: []supgpu.RuntimeCode{}, Alternates: []supgpu.RuntimeCode{}, Directories: []string{}}
	for _, dir := range directories {
		actual, e := filepath.EvalSymlinks(dir)
		if e != nil {
			return p, e
		}
		if !slices.Contains(p.Directories, actual) {
			p.Directories = append(p.Directories, actual)
		}
	}
	byName := map[string]int{}
	rootNeeded := []string{}
	rootFound := false
	cupti := false
	for _, node := range closure.Nodes {
		if node.Pin.Path == binary.Path {
			if node.Pin.SHA256 != binary.SHA256 {
				return p, fmt.Errorf("decoder binary pin differs")
			}
			rootNeeded = append(rootNeeded, node.Needed...)
			rootFound = true
			continue
		}
		if node.Pin.Path == audit.Path {
			if node.Pin.SHA256 != audit.SHA256 {
				return p, fmt.Errorf("auditor binary pin differs")
			}
			continue
		}
		if node.SONAME == "" || strings.HasPrefix(node.SONAME, "libtorch") || strings.HasPrefix(node.SONAME, "libc10") || strings.HasPrefix(node.SONAME, "libcudnn") {
			return p, fmt.Errorf("object outside standalone decoder policy: %s", node.Pin.Path)
		}
		if _, exists := byName[node.SONAME]; exists {
			return p, fmt.Errorf("ambiguous decoder provider %s", node.SONAME)
		}
		object, e := guardPin(File{node.Pin.Path, node.Pin.SHA256})
		if e != nil {
			return p, e
		}
		tier := "d01-unique-prospective-provider"
		for _, selected := range qualification.Allowed {
			if selected.SONAME == node.SONAME {
				if selected.Object != object {
					return p, fmt.Errorf("qualified provider tier differs for %s", node.SONAME)
				}
				tier = selected.Tier
			}
		}
		for _, rejected := range qualification.Alternates {
			if rejected.SONAME == node.SONAME {
				if rejected.Object == object {
					return p, fmt.Errorf("known wrong decoder provider tier")
				}
				rejected.Needed = append([]string{}, rejected.Needed...)
				p.Alternates = append(p.Alternates, rejected)
			}
		}
		byName[node.SONAME] = len(p.Allowed)
		p.Allowed = append(p.Allowed, supgpu.RuntimeCode{Object: object, SONAME: node.SONAME, Needed: append([]string{}, node.Needed...), Tier: tier})
		cupti = cupti || strings.HasPrefix(node.SONAME, "libcupti.so")
	}
	if !rootFound || !cupti {
		return p, fmt.Errorf("observed decoder/CUPTI closure missing")
	}
	seen := map[string]bool{}
	for len(rootNeeded) > 0 {
		name := rootNeeded[0]
		rootNeeded = rootNeeded[1:]
		if seen[name] {
			continue
		}
		seen[name] = true
		i, ok := byName[name]
		if !ok {
			return p, fmt.Errorf("startup decoder provider absent for %s", name)
		}
		p.Allowed[i].Startup = true
		rootNeeded = append(rootNeeded, p.Allowed[i].Needed...)
	}
	return p, nil
}
