// Package backend inventories pinned prerequisites. It does not qualify a
// callable backend, model inference or GPU execution.
package backend

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Artifact struct {
	Name, Path, SHA256, URL string
	Bytes                   int64
}
type Check struct {
	Name, Path, Status, ExpectedSHA256, ObservedSHA256, Detail string
	Bytes                                                      int64
}
type Report struct {
	Status           string
	BackendQualified bool
	CUDAFamily       string
	Checks           []Check
	MissingOrInvalid int
}

func Archives(family string) ([]Artifact, error) {
	common := Artifact{"pod5", "lib_pod5-0.3.39-linux-x64.tar.gz", "7fcf1d9f1da6f1bd13167782e900462fbadb0db57c666911bdba9bbdfd4e7e06", "https://github.com/nanoporetech/pod5-file-format/releases/download/0.3.39/lib_pod5-0.3.39-linux-x64.tar.gz", 44034293}
	var torchHash, koiHash string
	var torchBytes, koiBytes int64
	switch family {
	case "13.0":
		torchHash = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"
		koiHash = "0600a138761b17467bd3a1f8933c6a6a02e17415a9da377f22bc14357c3101ca"
		torchBytes = 1281640079
		koiBytes = 58019481
	case "12.8":
		torchHash = "ed09249f48bc3e20130307e86a729e8bc8a97aed794c149b60efcae172654a9d"
		koiHash = "bfb3bea6f1bb7f7e6e602d4899db14f422a46519ce69e2d1b645050eba307aad"
		torchBytes = 2016718479
		koiBytes = 69125663
	default:
		return nil, fmt.Errorf("unsupported pinned CUDA family %q", family)
	}
	torch := "torch-2.9.0.0-ont-CUDA-" + family + "-linux-x64-cxx11-abi.zip"
	koi := "libkoi-0.7.5-Linux-x86_64-cuda-" + family + ".tar.gz"
	cdn := "https://cdn.oxfordnanoportal.com/software/analysis/"
	return []Artifact{{"torch", torch, torchHash, cdn + torch, torchBytes}, {"koi", koi, koiHash, cdn + koi, koiBytes}, common}, nil
}

func Inspect(sourceRoot, dependencyRoot, cudaRoot, family string) Report {
	report := Report{Status: "missing_or_invalid_prerequisites", CUDAFamily: family}
	add := func(check Check) {
		report.Checks = append(report.Checks, check)
		if check.Status != "verified" && check.Status != "present_unqualified" {
			report.MissingOrInvalid++
		}
	}
	for _, item := range []Artifact{
		{Name: "Torch CMake source", Path: "cmake/Torch.cmake", SHA256: "9ec0d638208067cf8c08c4b3bc61b43000240bc21e60752895a838562f9369fb"},
		{Name: "Koi CMake source", Path: "cmake/Koi.cmake", SHA256: "57e32a620b983d7ab62740b2541a54c9412c98f8602b4c4ea0cf46012ac51877"},
		{Name: "POD5 CMake source", Path: "cmake/Pod5.cmake", SHA256: "9be161fa49c299cbd4fbf9dd0275f123016d9bbc702390390741ee647b154799"},
	} {
		add(CheckFile(filepath.Join(sourceRoot, item.Path), item))
	}
	archives, err := Archives(family)
	if err != nil {
		add(Check{Name: "CUDA family", Status: "invalid", Detail: err.Error()})
		return report
	}
	for _, item := range archives {
		add(CheckFile(filepath.Join(dependencyRoot, "archives", item.Path), item))
	}
	for _, path := range []string{"torch/include/ATen/ATen.h", "torch/include/torch/csrc/api/include/torch/torch.h", "torch/share/cmake/Torch/TorchConfig.cmake", "torch/lib/libtorch.so", "koi/include/koi.h", "koi/lib/libkoi.a", "pod5/include/pod5_format/c_api.h", "pod5/lib64/libpod5_format.a", "pod5/lib64/libarrow.a", "pod5/lib64/libjemalloc_pic.a", "pod5/lib64/libzstd.a"} {
		add(CheckFile(filepath.Join(dependencyRoot, path), Artifact{Name: path}))
	}
	for _, path := range []string{"include/cuda_runtime_api.h", "lib64/libcudart.so", "bin/nvcc"} {
		add(CheckFile(filepath.Join(cudaRoot, path), Artifact{Name: "CUDA " + path}))
	}
	command := exec.Command(filepath.Join(cudaRoot, "bin/nvcc"), "--version")
	out, err := command.CombinedOutput()
	check := Check{Name: "CUDA compiler version", Path: command.Path, Status: "invalid", Detail: strings.TrimSpace(string(out))}
	if err != nil {
		check.Detail = err.Error()
	} else if family == "13.0" && strings.Contains(string(out), "release 13.") || family == "12.8" && strings.Contains(string(out), "release 12.8") {
		check.Status = "verified"
	} else {
		check.Status = "present_unqualified"
		check.Detail += "; compiler differs from dependency family; a C++-only bridge may not need nvcc, compilation and linkage remain unqualified"
	}
	add(check)
	if report.MissingOrInvalid == 0 {
		report.Status = "prerequisites_present_extraction_and_backend_unqualified"
	}
	return report
}

// CheckFile verifies a declared pin. Without an expected pin it records only
// presence and the observed checksum, never claiming extraction integrity.
func CheckFile(path string, item Artifact) Check {
	result := Check{Name: item.Name, Path: path, ExpectedSHA256: item.SHA256, Status: "missing"}
	file, err := os.Open(path)
	if err != nil {
		result.Detail = err.Error()
		return result
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		result.Status = "invalid"
		result.Detail = "not a regular file"
		return result
	}
	result.Bytes = info.Size()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		result.Status = "invalid"
		result.Detail = err.Error()
		return result
	}
	result.ObservedSHA256 = hex.EncodeToString(hash.Sum(nil))
	if item.SHA256 == "" {
		result.Status = "present_unqualified"
		return result
	}
	if result.ObservedSHA256 != item.SHA256 || item.Bytes > 0 && result.Bytes != item.Bytes {
		result.Status = "invalid"
		result.Detail = "checksum or size mismatch"
		return result
	}
	result.Status = "verified"
	return result
}
