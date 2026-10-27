package hacvcsaux

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type OracleELFEvidence struct {
	Schema      string
	Oracle      closedbuild.Pin
	Closure     nativeproof.ELFClosure
	RuntimeHeld bool
}

func InspectOracle(path, expectedSHA string) (OracleELFEvidence, error) {
	var result OracleELFEvidence
	if expectedSHA == "" {
		return result, fmt.Errorf("independent oracle output pin required")
	}
	pin, err := closedbuild.PinFile(path)
	if err != nil {
		return result, err
	}
	if pin.SHA256 != expectedSHA {
		return result, fmt.Errorf("oracle output pin differs")
	}
	compiler, err := CheckCompiler()
	if err != nil {
		return result, err
	}
	body, err := os.ReadFile(compiler.Baseline.Path)
	if err != nil {
		return result, err
	}
	var baseline struct{ Runtime []nativeproof.ELFMetadata }
	if err := json.Unmarshal(body, &baseline); err != nil {
		return result, err
	}
	allowed := map[string]closedbuild.Pin{}
	for _, node := range baseline.Runtime {
		switch node.SONAME {
		case "libstdc++.so.6", "libgcc_s.so.1", "libc.so.6", "libm.so.6", "libmvec.so.1", "ld-linux-x86-64.so.2":
			if prior, exists := allowed[node.SONAME]; exists && prior != node.Pin {
				return result, fmt.Errorf("independent runtime baseline has ambiguous SONAME %s", node.SONAME)
			}
			allowed[node.SONAME] = node.Pin
		}
	}
	closure, err := nativeproof.CloseELF([]closedbuild.Pin{pin}, []string{"/usr/lib"})
	if err != nil {
		return result, err
	}
	for _, node := range closure.Nodes {
		if node.Class != elf.ELFCLASS64 || node.Data != elf.ELFDATA2LSB || node.Machine != elf.EM_X86_64 {
			return result, fmt.Errorf("oracle runtime is outside the pinned x86-64 ABI")
		}
		if node.Pin != pin && allowed[node.SONAME] != node.Pin {
			return result, fmt.Errorf("oracle runtime has an unapproved DSO: %s", node.Pin.Path)
		}
	}
	f, err := elf.Open(pin.Path)
	if err != nil {
		return result, err
	}
	defer f.Close()
	symbols, err := f.Symbols()
	if err != nil {
		return result, err
	}
	required := map[string]bool{"host_lstm_preprocess": false, "koi_create_thread_pool": false, "koi_destroy_thread_pool": false,
		"_ZN3koi4lstm21create_auxiliary_dataEPKiiPjiP13KoiThreadPoolbbiii": false,
		"cudaMemcpyAsync": false, "__cudaRegisterFatBinary": false, "__cudaRegisterFunction": false,
		"__cudaRegisterVar": false, "__cudaRegisterFatBinaryEnd": false, "__cudaUnregisterFatBinary": false,
		"cudaGetDevice": false, "cudaDeviceGetAttribute": false, "cudaGetLastError": false,
		"__cudaPushCallConfiguration": false, "__cudaPopCallConfiguration": false, "__cudaGetKernel": false, "__cudaLaunchKernel": false}
	for _, symbol := range symbols {
		if _, needed := required[symbol.Name]; needed && symbol.Section != elf.SHN_UNDEF {
			required[symbol.Name] = true
		}
	}
	for name, present := range required {
		if !present {
			return result, fmt.Errorf("original oracle or verification transport symbol absent: %s", name)
		}
	}
	return OracleELFEvidence{Schema: "hac-vcs-aux-oracle-elf/v1", Oracle: pin, Closure: closure, RuntimeHeld: true}, nil
}

const CompilerBaselinePath = "/home/jake/Projects/nanopore_basecaller_go/artifacts/tensor/backend-closed-plan-005/plan.json"
const CompilerBaselineSHA256 = "b2d33b287e9135d85fc3ac6191de33c7aa9e70eda5a029f3a02b89a460b76d7a"

type CompilerEvidence struct {
	Schema            string
	Baseline          closedbuild.Pin
	Compiler          closedbuild.Compiler
	EffectiveCgroup   map[string]string
	ExecutionAdmitted bool
}

// CheckCompiler reuses the maintained closed compiler's complete tool, plugin,
// interpreter, DSO, loader-cache, and symlink checks. It starts no native child.
func CheckCompiler() (CompilerEvidence, error) {
	var result CompilerEvidence
	pin, err := closedbuild.PinFile(CompilerBaselinePath)
	if err != nil {
		return result, err
	}
	if pin.SHA256 != CompilerBaselineSHA256 {
		return result, fmt.Errorf("independent closed compiler baseline changed")
	}
	body, err := os.ReadFile(pin.Path)
	if err != nil {
		return result, err
	}
	var baseline struct {
		Compiler closedbuild.Compiler
	}
	if err := json.Unmarshal(body, &baseline); err != nil {
		return result, err
	}
	if err := baseline.Compiler.Check(); err != nil {
		return result, err
	}
	limits, err := closedbuild.Limits()
	if err != nil {
		return result, err
	}
	return CompilerEvidence{Schema: "hac-vcs-aux-compiler-evidence/v1", Baseline: pin, Compiler: baseline.Compiler, EffectiveCgroup: limits}, nil
}
