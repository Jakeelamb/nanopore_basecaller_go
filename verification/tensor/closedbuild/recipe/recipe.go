// Package recipe is the maintained tensor builder's declarative compiler recipe.
// Constructing a recipe performs no file access or command execution.
package recipe

import (
	"fmt"
	"path/filepath"
)

type Options struct {
	Compiler, Torch, Archive, Dorado, CUDA, Koi, Weight, Family string
}
type Recipe struct {
	Schema                    string
	Options                   Options
	Common, Link, FixtureLink []string
}
type Unit struct {
	Name, Source, Output string
	Flags, Link          []string
	Shared               bool
}

func New(o Options) (Recipe, error) {
	if o.Family != "none" && o.Family != "13.0" {
		return Recipe{}, fmt.Errorf("unsupported CUDA family %q", o.Family)
	}
	lib := filepath.Join(o.Torch, "lib")
	r := Recipe{Schema: "nanopore-tensor-compiler-recipe/v1", Options: o}
	r.Common = []string{"-std=c++17", "-O1", "-g0", "-fPIC", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-Iinternal/tensor", "-I" + filepath.Join(o.Torch, "include"), "-I" + filepath.Join(o.Torch, "include/torch/csrc/api/include")}
	r.FixtureLink = []string{"-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10"}
	r.Link = append([]string{}, r.FixtureLink...)
	if o.Family == "13.0" {
		r.Common = append(r.Common, "-DNT_ENABLE_CUDA=1", "-I"+filepath.Join(o.CUDA, "include"), "-I"+filepath.Join(o.Koi, "include"))
		r.Link = append(r.Link, "-L"+filepath.Join(o.CUDA, "lib64"), "-Wl,-rpath,"+filepath.Join(o.CUDA, "lib64"), "-ltorch_cuda", "-lc10_cuda", "-lcudart", "-lcublas", "-lcuda", filepath.Join(o.Koi, "lib/libkoi.a"))
	}
	return r, nil
}
func (r Recipe) Unit(name, out string) (Unit, error) {
	u := Unit{Name: name, Flags: append([]string{}, r.Common...), Link: append([]string{}, r.Link...)}
	switch name {
	case "backend":
		u.Source = "verification/tensor/backend.cpp"
		u.Output = filepath.Join(out, "libnanopore_tensor.so")
		u.Shared = true
	case "fixtures":
		u.Source = "verification/tensor/fixtures.cpp"
		u.Output = filepath.Join(out, "create-fixtures")
		u.Link = append([]string{}, r.FixtureLink...)
	case "failure-test":
		u.Source = "verification/tensor/cuda_failure_test.cpp"
		u.Output = filepath.Join(out, "cuda-failure-test")
		u.Flags = []string{"-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror"}
		u.Link = nil
	case "cuda-oracle":
		if r.Options.Family != "13.0" {
			return Unit{}, fmt.Errorf("CUDA oracle requires CUDA family13.0")
		}
		u.Source = "verification/tensor/cuda_oracle.cpp"
		u.Output = filepath.Join(out, "cuda-source-oracle")
		u.Flags = append(u.Flags, "-I"+out)
	default:
		return Unit{}, fmt.Errorf("unsupported compiler profile %q", name)
	}
	return u, nil
}

// Combined retains the maintained default builder's single-driver invocation.
func (r Recipe) Combined(name, out, depfile string) ([]string, error) {
	u, e := r.Unit(name, out)
	if e != nil {
		return nil, e
	}
	a := append([]string{r.Options.Compiler}, u.Flags...)
	if depfile != "" {
		a = append(a, "-MMD", "-MF", depfile)
	}
	if u.Shared {
		a = append(a, "-shared")
	}
	a = append(a, u.Source, "-o", u.Output)
	return append(a, u.Link...), nil
}
