package decodergpu

import (
	"bufio"
	"debug/elf"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

var decoderSymbols = [4]string{"back_guide_step", "beam_search_step", "compute_posts_step", "decode_step"}

type DeviceImage struct {
	Architecture string
	Image        File
	Symbols      map[string]uint32
}
type ImagePolicy struct {
	Schema string
	Member File
	Images []DeviceImage
}

func WriteImagePolicy(out string, policy ImagePolicy) error {
	dir, err := runner.NewAttempt(out)
	if err != nil {
		return err
	}
	return writeJSON(filepath.Join(dir, "images.json"), policy)
}

func CheckImages(policyFile File, observation string) (ImageReport, error) {
	policyFile, err := normalizeAnchor(policyFile)
	if err != nil {
		return ImageReport{}, err
	}
	var policy ImagePolicy
	if err = readJSON(policyFile, &policy); err != nil {
		return ImageReport{}, err
	}
	if len(policy.Images) != 2 {
		return ImageReport{}, fmt.Errorf("fixed prospective image policy differs")
	}
	expected, err := PlanDeviceImages(filepath.Dir(policy.Member.Path), policy.Images[1].Image.Path)
	if err != nil {
		return ImageReport{}, err
	}
	if !reflect.DeepEqual(expected, policy) {
		return ImageReport{}, fmt.Errorf("prospective member/image/symbol policy changed")
	}
	return ReadImageReport(observation, policy)
}

func PlanDeviceImages(static, sm120a string) (p ImagePolicy, err error) {
	p.Schema = "nanopore-decoder-device-images/v1"
	p.Member, err = artifactPin(filepath.Join(static, "decoder_lib.cu.o"))
	if err != nil || p.Member.SHA256 != "930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f" {
		return p, fmt.Errorf("selected original decoder member differs: %v", err)
	}
	for _, spec := range []struct{ name, path, sha string }{
		{"sm_120", filepath.Join(static, "decoder_lib.sm_120.cubin"), "292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938"},
		{"sm_120a", sm120a, "8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40"},
	} {
		image, e := artifactPin(spec.path)
		if e != nil || image.SHA256 != spec.sha {
			return p, fmt.Errorf("pinned %s decoder image differs: %v", spec.name, e)
		}
		f, e := elf.Open(image.Path)
		if e != nil {
			return p, e
		}
		symbols, e := f.Symbols()
		valid := f.Class == elf.ELFCLASS64 && f.Data == elf.ELFDATA2LSB && f.Machine == elf.EM_CUDA
		f.Close()
		if e != nil || !valid {
			return p, fmt.Errorf("decoder image ELF metadata differs: %v", e)
		}
		d := DeviceImage{spec.name, image, map[string]uint32{}}
		for i, symbol := range symbols {
			for _, name := range decoderSymbols {
				if symbol.Name == name && elf.ST_TYPE(symbol.Info) == elf.STT_FUNC && symbol.Section != elf.SHN_UNDEF {
					if d.Symbols[name] != 0 {
						return p, fmt.Errorf("ambiguous decoder symbol %s", name)
					}
					d.Symbols[name] = uint32(i + 1) // debug/elf omits ELF's null symbol.
				}
			}
		}
		if len(d.Symbols) != 4 {
			return p, fmt.Errorf("decoder cubin lacks original four kernel symbols")
		}
		p.Images = append(p.Images, d)
	}
	return p, nil
}

type ObservedModule struct {
	Slot, Context, Module, DeliveredStage, Observed, Retained uint64
	Payload                                                   File
}
type ObservedFunction struct {
	ID, Context, Module, Index, DeliveredStage uint64
	Name                                       string
}
type ObservedLaunch struct {
	Context, UID, Correlation, Callback, Site, Stage uint64
	Result                                           int64
	Name                                             string
}
type ImageBinding struct {
	Stage                             uint64
	Context, ContextID, Module, Index uint64
	Name, Architecture                string
	Image                             File
}
type ImageReport struct {
	Schema, Status, Error string
	Directory             string
	Modules               []ObservedModule
	Functions             []ObservedFunction
	Launches              []ObservedLaunch
	Bindings              []ImageBinding
	ModuleOnly            bool
	Files                 map[string]string
}

func imageRows(path string, fields int, read func([]string) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 4096), 8192)
	count := 0
	for s.Scan() {
		count++
		if count > 512 {
			return fmt.Errorf("observer row capacity exceeded")
		}
		row := strings.Split(s.Text(), "\t")
		if len(row) != fields {
			return fmt.Errorf("observer row field count differs")
		}
		if err = read(row); err != nil {
			return err
		}
	}
	return s.Err()
}
func imageNumbers(row []string) ([]uint64, error) {
	values := make([]uint64, len(row))
	for i, word := range row {
		n, err := strconv.ParseUint(word, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid observer integer: %w", err)
		}
		values[i] = n
	}
	return values, nil
}

// ReadImageReport retains narrower module-only evidence on a missing function
// collection, but returns an error. Such a report cannot qualify numerical use.
func ReadImageReport(dir string, policy ImagePolicy) (r ImageReport, err error) {
	r = ImageReport{Schema: "nanopore-decoder-selected-code/v1", Status: "failed", Directory: dir, Files: map[string]string{}}
	defer func() {
		r.ModuleOnly = len(r.Modules) > 0 && len(r.Bindings) != 4
		if err != nil {
			r.Error = err.Error()
		}
	}()
	for _, name := range []string{"modules.tsv", "functions.tsv", "launches.tsv", "status.txt"} {
		if err = filePin(r.Files, filepath.Join(dir, name)); err != nil {
			return r, err
		}
	}
	status, err := os.ReadFile(filepath.Join(dir, "status.txt"))
	if err != nil {
		return r, err
	}
	err = imageRows(filepath.Join(dir, "modules.tsv"), 8, func(row []string) error {
		n, e := imageNumbers(row[:6])
		if e != nil {
			return e
		}
		if row[7] != fmt.Sprintf("module-%d.cubin", n[0]) || n[0] != uint64(len(r.Modules)) || n[0] >= 16 || n[1] == 0 || n[4] == 0 || n[4] != n[5] || n[5] > 1<<20 {
			return fmt.Errorf("incomplete or unbounded observed module")
		}
		payload, e := artifactPin(filepath.Join(dir, row[7]))
		if e != nil {
			return e
		}
		info, e := os.Stat(payload.Path)
		if e != nil || uint64(info.Size()) != n[5] || payload.SHA256 != row[6] {
			return fmt.Errorf("module raw payload size/digest differs")
		}
		r.Files[payload.Path] = payload.SHA256
		r.Modules = append(r.Modules, ObservedModule{n[0], n[1], n[2], n[3], n[4], n[5], payload})
		return nil
	})
	if err != nil {
		return r, err
	}
	err = imageRows(filepath.Join(dir, "functions.tsv"), 6, func(row []string) error {
		n, e := imageNumbers(row[:5])
		if e == nil {
			r.Functions = append(r.Functions, ObservedFunction{n[0], n[1], n[2], n[3], n[4], row[5]})
		}
		return e
	})
	if err != nil {
		return r, err
	}
	err = imageRows(filepath.Join(dir, "launches.tsv"), 8, func(row []string) error {
		n, e := imageNumbers(row[:6])
		if e != nil {
			return e
		}
		rc, e := strconv.ParseInt(row[6], 10, 32)
		if e == nil {
			r.Launches = append(r.Launches, ObservedLaunch{n[0], n[1], n[2], n[3], n[4], n[5], rc, row[7]})
		}
		return e
	})
	if err != nil {
		return r, err
	}
	want := fmt.Sprintf("callback_error=0\nmodules=%d\nfunctions=%d\nlaunches=%d\n", len(r.Modules), len(r.Functions), len(r.Launches))
	if string(status) != want {
		return r, fmt.Errorf("CUPTI callback failure or record count differs")
	}
	r.Bindings, err = bindSelectedImages(r.Modules, r.Functions, r.Launches, policy)
	if err == nil {
		r.Status = "selected_decoder_code_qualified"
	}
	return r, err
}

func bindSelectedImages(modules []ObservedModule, functions []ObservedFunction, launches []ObservedLaunch, policy ImagePolicy) ([]ImageBinding, error) {
	if len(modules) == 0 || len(functions) == 0 || len(launches) != 8 {
		return nil, fmt.Errorf("exact four original kernel launches/function records unavailable")
	}
	known := map[uint64]DeviceImage{}
	for i, module := range modules {
		for j := 0; j < i; j++ {
			if modules[j].Context == module.Context && modules[j].Module == module.Module {
				return nil, fmt.Errorf("ambiguous observed module identity")
			}
		}
		for _, image := range policy.Images {
			if image.Image.SHA256 == module.Payload.SHA256 {
				known[uint64(i)] = image
			}
		}
		if _, ok := known[uint64(i)]; !ok {
			return nil, fmt.Errorf("observed module differs from original selected member images")
		}
	}
	result := []ImageBinding{}
	for stage, symbol := range decoderSymbols {
		var enter, exit *ObservedLaunch
		for i := range launches {
			l := &launches[i]
			if l.Stage == uint64(stage+1) {
				if l.Name != symbol || l.Context == 0 || l.UID == 0 || (l.Callback != 307 && l.Callback != 442 && l.Callback != 652 && l.Callback != 653) {
					return nil, fmt.Errorf("unexpected original stage launch/context/callback")
				}
				switch l.Site {
				case 0:
					if enter != nil {
						return nil, fmt.Errorf("duplicate kernel entry")
					}
					enter = l
				case 1:
					if exit != nil {
						return nil, fmt.Errorf("duplicate kernel exit")
					}
					exit = l
				default:
					return nil, fmt.Errorf("unknown launch callback site")
				}
			}
		}
		if enter == nil || exit == nil || enter.Correlation != exit.Correlation || enter.Context != exit.Context || enter.UID != exit.UID || enter.Callback != exit.Callback || exit.Result != 0 {
			return nil, fmt.Errorf("missing or failed paired original kernel launch")
		}
		var function *ObservedFunction
		for i := range functions {
			f := &functions[i]
			if f.Name == symbol && f.Context == enter.UID {
				if function != nil {
					return nil, fmt.Errorf("ambiguous function/context binding")
				}
				function = f
			}
		}
		if function == nil {
			return nil, fmt.Errorf("launch lacks timestamp-free FUNCTION record")
		}
		matched := false
		for i, module := range modules {
			image := known[uint64(i)]
			if module.Context == enter.Context && module.Module == function.Module && uint64(image.Symbols[symbol]) == function.Index && function.Index != 0 {
				result = append(result, ImageBinding{uint64(stage + 1), enter.Context, enter.UID, module.Module, function.Index, symbol, image.Architecture, module.Payload})
				matched = true
			}
		}
		if !matched {
			return nil, fmt.Errorf("function symbol index/module/context lacks exact selected image binding")
		}
	}
	return result, nil
}
