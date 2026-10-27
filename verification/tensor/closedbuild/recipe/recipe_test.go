package recipe

import (
	"reflect"
	"testing"
)

func TestMaintainedRecipeArgv(t *testing.T) {
	o := Options{Compiler: "/qualified/g++-15", Torch: "/torch", CUDA: "/cuda", Koi: "/koi", Family: "13.0"}
	r, e := New(o)
	if e != nil {
		t.Fatal(e)
	}
	common := []string{"-std=c++17", "-O1", "-g0", "-fPIC", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-Iinternal/tensor", "-I/torch/include", "-I/torch/include/torch/csrc/api/include", "-DNT_ENABLE_CUDA=1", "-I/cuda/include", "-I/koi/include"}
	link := []string{"-L/torch/lib", "-Wl,-rpath,/torch/lib", "-ltorch_cpu", "-lc10", "-L/cuda/lib64", "-Wl,-rpath,/cuda/lib64", "-ltorch_cuda", "-lc10_cuda", "-lcudart", "-lcublas", "-lcuda", "/koi/lib/libkoi.a"}
	want := append([]string{o.Compiler}, common...)
	want = append(want, "-MMD", "-MF", "/out/backend.d", "-shared", "verification/tensor/backend.cpp", "-o", "/out/libnanopore_tensor.so")
	want = append(want, link...)
	got, e := r.Combined("backend", "/out", "/out/backend.d")
	if e != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("maintained backend argv changed: %v %v", got, e)
	}
	f, e := r.Combined("failure-test", "/out", "")
	if e != nil || !reflect.DeepEqual(f, []string{o.Compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "verification/tensor/cuda_failure_test.cpp", "-o", "/out/cuda-failure-test"}) {
		t.Fatalf("failure probe changed: %v %v", f, e)
	}
	for _, name := range []string{"fixtures", "cuda-oracle"} {
		u, e := r.Unit(name, "/out")
		if e != nil || u.Shared {
			t.Fatal(name, e)
		}
	}
	if _, e = r.Unit("all", "/out"); e == nil {
		t.Fatal("implicit all profiles accepted")
	}
}
