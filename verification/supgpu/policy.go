package supgpu

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

type RuntimeCode struct {
	Object  Pin      `json:"object"`
	SONAME  string   `json:"soname"`
	Needed  []string `json:"needed"`
	Tier    string   `json:"tier"`
	Startup bool     `json:"startup"`
}

type RuntimePolicy struct {
	Schema      string        `json:"schema"`
	Lazy        string        `json:"lazy_policy"`
	Reference   Pin           `json:"reference"`
	Observer    Pin           `json:"observer"`
	Audit       Pin           `json:"audit"`
	Allowed     []RuntimeCode `json:"allowed"`
	Alternates  []RuntimeCode `json:"rejected_alternates"`
	Directories []string      `json:"loader_directories"`
}

func makeRuntimePolicy(r RuntimePlan, b Build, p BuildPlan) (RuntimePolicy, error) {
	var prepared Prepared
	if err := readPinnedJSON(p.Prepared.Path, p.Prepared.SHA256, &prepared); err != nil {
		return RuntimePolicy{}, err
	}
	if err := verifyPin("", prepared.Profile); err != nil {
		return RuntimePolicy{}, err
	}
	raw, err := os.ReadFile(prepared.Profile.Path)
	if err != nil {
		return RuntimePolicy{}, err
	}
	var profile struct {
		Runtime map[string]string `json:"runtime_sha256"`
	}
	if err = json.Unmarshal(raw, &profile); err != nil {
		return RuntimePolicy{}, err
	}
	observed := map[string]string{}
	for path, sha := range profile.Runtime {
		actual, err := filepath.EvalSymlinks(path)
		if err != nil {
			return RuntimePolicy{}, err
		}
		observed[actual] = sha
	}
	qualified := ""
	for _, binding := range p.LinkBindings {
		if filepath.Base(binding.Input.Path) == "libtorch_cuda.so" {
			qualified = filepath.Dir(binding.Input.Path)
		}
	}
	if qualified == "" {
		return RuntimePolicy{}, fmt.Errorf("qualified Torch directory is absent")
	}
	code := []RuntimeCode{}
	for _, node := range r.Nodes {
		if node.Object == b.Reference || node.Object == b.Observer {
			continue
		}
		f, err := elf.Open(node.Object.Path)
		if err != nil {
			return RuntimePolicy{}, err
		}
		names, err := f.DynString(elf.DT_SONAME)
		f.Close()
		if err != nil || len(names) != 1 || names[0] == "" {
			return RuntimePolicy{}, fmt.Errorf("runtime object has no unique SONAME: %s", node.Object.Path)
		}
		code = append(code, RuntimeCode{node.Object, names[0], node.Needed, "", false})
	}
	policy, err := selectRuntimePolicy(code, observed, qualified, b.Reference, b.Observer, b.Audit)
	if err != nil {
		return policy, err
	}
	byName := map[string]int{}
	for i, c := range policy.Allowed {
		byName[c.SONAME] = i
	}
	queue := []string{}
	for _, n := range r.Nodes {
		if n.Object == b.Reference || n.Object == b.Observer {
			queue = append(queue, n.Needed...)
		}
	}
	seen := map[string]bool{}
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		if seen[name] {
			continue
		}
		seen[name] = true
		i, ok := byName[name]
		if !ok {
			return policy, fmt.Errorf("selected runtime provider missing for %s", name)
		}
		policy.Allowed[i].Startup = true
		queue = append(queue, policy.Allowed[i].Needed...)
	}
	return policy, nil
}

func selectRuntimePolicy(code []RuntimeCode, observed map[string]string, qualified string, reference, observer, audit Pin) (RuntimePolicy, error) {
	p := RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: reference, Observer: observer, Audit: audit, Allowed: []RuntimeCode{}, Alternates: []RuntimeCode{}, Directories: []string{qualified}}
	groups := map[string][]RuntimeCode{}
	names := []string{}
	for _, c := range code {
		if _, ok := groups[c.SONAME]; !ok {
			names = append(names, c.SONAME)
		}
		groups[c.SONAME] = append(groups[c.SONAME], c)
	}
	sort.Strings(names)
	for _, name := range names {
		members := groups[name]
		chosen := -1
		tier := "unique-frozen"
		for i, c := range members {
			match := observed[c.Object.Path] == c.Object.SHA256
			if strings.HasPrefix(name, "libcudnn") {
				match = filepath.Dir(c.Object.Path) == qualified
				tier = "p01-qualified-cudnn"
			} else if match {
				tier = "p01-observed"
			}
			if match {
				if chosen != -1 {
					return p, fmt.Errorf("ambiguous selected runtime tier for %s", name)
				}
				chosen = i
			}
		}
		if chosen == -1 && len(members) == 1 && !strings.HasPrefix(name, "libcudnn") {
			chosen = 0
		}
		if chosen < 0 {
			if strings.HasPrefix(name, "libcudnn") {
				for _, c := range members {
					c.Tier = "rejected-alternate"
					p.Alternates = append(p.Alternates, c)
				}
				continue
			}
			return p, fmt.Errorf("no uniquely qualified runtime provider for %s", name)
		}
		for i, c := range members {
			if i == chosen {
				c.Tier = tier
				if c.Object == audit {
					c.Tier = "verification-loader-audit"
				}
				p.Allowed = append(p.Allowed, c)
				if strings.Contains(c.Object.Path, "/cuda/") && !containsString(p.Directories, filepath.Dir(c.Object.Path)) {
					p.Directories = append(p.Directories, filepath.Dir(c.Object.Path))
				}
			} else {
				c.Tier = "rejected-alternate"
				p.Alternates = append(p.Alternates, c)
			}
		}
	}
	if !containsString(p.Directories, "/usr/lib") {
		p.Directories = append(p.Directories, "/usr/lib")
	}
	return p, nil
}

func containsString(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}

func modelEnvironment(p RuntimePolicy) []string {
	return append(nativeEnvironment(), "LD_AUDIT="+p.Audit.Path, "LD_LIBRARY_PATH="+strings.Join(p.Directories, ":"), "LD_BIND_NOW=1")
}

func requireSelectedCode(actual Pin, p RuntimePolicy, role string) error {
	root := p.Reference
	if role == "observer" {
		root = p.Observer
	}
	if actual == root || actual == p.Audit {
		return nil
	}
	for _, c := range p.Allowed {
		if actual == c.Object {
			return nil
		}
	}
	for _, c := range p.Alternates {
		if actual == c.Object {
			return fmt.Errorf("known wrong runtime tier rejected: %s", actual.Path)
		}
	}
	return fmt.Errorf("unknown mapped code rejected: %s", actual.Path)
}

func requireSelectedMappings(actual []Pin, p RuntimePolicy, role string, startup bool) error {
	seen := map[string]Pin{}
	for _, a := range actual {
		if err := requireSelectedCode(a, p, role); err != nil {
			return err
		}
		seen[a.Path] = a
	}
	if startup {
		root := p.Reference
		if role == "observer" {
			root = p.Observer
		}
		for _, required := range append([]Pin{root, p.Audit}, startupPins(p)...) {
			if seen[required.Path] != required {
				return fmt.Errorf("required startup image absent: %s", required.Path)
			}
		}
	}
	return nil
}

func requireCaptureRuntime(actual []Pin, p RuntimePolicy) error {
	for _, a := range actual {
		if a == p.Observer {
			continue
		}
		if err := requireSelectedCode(a, p, "reference"); err != nil {
			return err
		}
	}
	return nil
}

func startupPins(p RuntimePolicy) []Pin {
	out := []Pin{}
	for _, c := range p.Allowed {
		if c.Startup {
			out = append(out, c.Object)
		}
	}
	return out
}

func verifyRuntimePolicy(r RuntimePlan, b Build, p BuildPlan) error {
	expected, err := makeRuntimePolicy(r, b, p)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expected, r.Policy) {
		return fmt.Errorf("selected runtime policy changed")
	}
	return nil
}

func verifyAuditELF(p Pin) error {
	if err := verifyPin("", p); err != nil {
		return err
	}
	f, err := elf.Open(p.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	needed, err := f.ImportedLibraries()
	if err != nil {
		return err
	}
	if len(needed) != 0 {
		return fmt.Errorf("loader auditor must have no DT_NEEDED")
	}
	imports, err := f.ImportedSymbols()
	if err != nil {
		return err
	}
	if len(imports) != 0 {
		return fmt.Errorf("loader auditor has unresolved external symbols")
	}
	syms, err := f.DynamicSymbols()
	if err != nil {
		return err
	}
	for _, name := range []string{"la_version", "la_objopen", "la_objclose"} {
		found := false
		for _, s := range syms {
			found = found || (s.Name == name && s.Section != elf.SHN_UNDEF)
		}
		if !found {
			return fmt.Errorf("loader audit export absent: %s", name)
		}
	}
	return nil
}
