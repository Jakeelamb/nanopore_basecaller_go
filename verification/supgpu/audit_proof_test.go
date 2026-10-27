package supgpu

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type auditProducerProof struct {
	Schema             string                  `json:"schema"`
	Controller         Pin                     `json:"controller"`
	AcceptedRuntime    Pin                     `json:"accepted_runtime_metadata"`
	AcceptedBuild      Pin                     `json:"accepted_build_metadata"`
	CompilerPlan       Pin                     `json:"compiler_plan_metadata"`
	ProducerController Pin                     `json:"auditor_producer_controller_metadata"`
	ProducerBinary     Pin                     `json:"auditor_producer_binary"`
	Audit              Pin                     `json:"auditor"`
	Source             Pin                     `json:"auditor_source"`
	Object             Pin                     `json:"auditor_object"`
	Depfile            Pin                     `json:"auditor_depfile"`
	PreDepfile         Pin                     `json:"auditor_precompile_depfile"`
	LinkMap            Pin                     `json:"auditor_link_map"`
	Commands           []runner.CommandReceipt `json:"auditor_commands"`
	Compiler           Pin                     `json:"compiler"`
	CompilerRuntime    CompilerRuntime         `json:"compiler_runtime"`
	QualifiedBuildDir  string                  `json:"qualified_build_directory"`
	CPU                []RuntimeCode           `json:"selected_cpu_providers"`
	Cache              Pin                     `json:"loader_cache"`
	SearchDirs         []string                `json:"cpu_search_directories"`
}

var auditCPUName = map[string]bool{"libc.so.6": true, "libstdc++.so.6": true, "libm.so.6": true, "libgcc_s.so.1": true, "ld-linux-x86-64.so.2": true, "libdl.so.2": true, "libpthread.so.0": true}

func auditCPUProviders(r RuntimePlan) ([]RuntimeCode, []string) {
	cpu := []RuntimeCode{}
	dirs := []string{}
	for _, c := range r.Policy.Allowed {
		if auditCPUName[c.SONAME] {
			cpu = append(cpu, c)
			dir := filepath.Dir(c.Object.Path)
			if !containsString(dirs, dir) {
				dirs = append(dirs, dir)
			}
		}
	}
	sort.Strings(dirs)
	return cpu, dirs
}

// This preparation authenticates the entire accepted producer once. Subsequent
// CPU controls consume the auditor and this proof, not NN/Torch/CUDA images.
func prepareAuditorProof(controller, runtimePin Pin, out string) (p auditProducerProof, err error) {
	p = auditProducerProof{Schema: "nanopore-supgpu-auditor-producer-proof/v1", Controller: controller, AcceptedRuntime: runtimePin}
	if _, err = readAuditController(controller, true); err != nil {
		return
	}
	var r RuntimePlan
	if err = readPinnedJSON(runtimePin.Path, runtimePin.SHA256, &r); err != nil {
		return
	}
	var b Build
	if err = readPinnedJSON(r.Build.Path, r.Build.SHA256, &b); err != nil {
		return
	}
	if err = verifyBuild(b, b.PreparedSHA256); err != nil {
		return
	}
	if _, err = readRuntimePlan(runtimePin, b, r.Build, false); err != nil {
		return
	}
	var compiler BuildPlan
	if err = readPinnedJSON(b.Plan.Path, b.Plan.SHA256, &compiler); err != nil {
		return
	}
	p.AcceptedBuild, p.CompilerPlan, p.ProducerController = r.Build, b.Plan, compiler.Controller
	oldController, e := readController(compiler.Controller, false)
	if e != nil {
		err = e
		return
	}
	p.ProducerBinary = oldController.Binary
	p.Audit, p.Object, p.Depfile, p.PreDepfile, p.LinkMap = b.Audit, b.Objects[5], b.Depfiles[5], b.PreDepfiles[5], b.LinkMaps[2]
	p.Source, err = pin(compiler.Commands[5].Argv[len(compiler.Commands[5].Argv)-3])
	if err != nil {
		return
	}
	p.Commands = []runner.CommandReceipt{b.Commands[5], b.Commands[8]}
	p.Compiler, p.CompilerRuntime, p.QualifiedBuildDir = compiler.Compiler, compiler.CompilerRuntime, compiler.Commands[0].Directory
	p.CPU, p.SearchDirs = auditCPUProviders(r)
	p.Cache = r.LoaderCache
	if _, e = os.Stat(out); !os.IsNotExist(e) {
		err = fmt.Errorf("auditor proof output must be fresh")
		return
	}
	err = writeJSON(out, p)
	return
}
func readAuditorProof(v Pin) (p auditProducerProof, err error) {
	if err = readPinnedJSON(v.Path, v.SHA256, &p); err != nil {
		return
	}
	if p.Schema != "nanopore-supgpu-auditor-producer-proof/v1" || len(p.Commands) != 2 {
		err = fmt.Errorf("invalid auditor producer proof")
		return
	}
	for _, v := range []Pin{p.Controller, p.AcceptedRuntime, p.AcceptedBuild, p.CompilerPlan, p.ProducerController, p.ProducerBinary, p.Audit, p.Source, p.Object, p.Depfile, p.PreDepfile, p.LinkMap, p.Cache, p.Compiler} {
		if err = verifyPin("", v); err != nil {
			return
		}
	}
	for _, c := range p.Commands {
		if err = verifyCommand(c); err != nil {
			return
		}
	}
	if err = verifyAuditELF(p.Audit); err != nil {
		return
	}
	var r RuntimePlan
	if err = readPinnedJSON(p.AcceptedRuntime.Path, p.AcceptedRuntime.SHA256, &r); err != nil {
		return
	}
	cpu, dirs := auditCPUProviders(r)
	if r.Build != p.AcceptedBuild || r.Controller != p.ProducerController || r.LoaderCache != p.Cache || !reflect.DeepEqual(cpu, p.CPU) || !reflect.DeepEqual(dirs, p.SearchDirs) {
		err = fmt.Errorf("auditor proof selected CPU metadata differs")
		return
	}
	for _, c := range p.CPU {
		if err = verifyPin("", c.Object); err != nil {
			return
		}
	}
	var old BuildPlan
	if err = readPinnedJSON(p.CompilerPlan.Path, p.CompilerPlan.SHA256, &old); err != nil {
		return
	}
	if old.Compiler != p.Compiler || !reflect.DeepEqual(old.CompilerRuntime, p.CompilerRuntime) || old.Commands[0].Directory != p.QualifiedBuildDir {
		err = fmt.Errorf("auditor proof compiler identity differs")
		return
	}
	return
}
