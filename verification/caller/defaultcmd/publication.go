package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func publicationFaults(r *runner, binary, core string) ([]map[string]any, error) {
	var results []map[string]any
	for _, fault := range []struct{ syscall, when string }{{"write", "4"}, {"fsync", "1"}} {
		name := "manifest-fault-" + fault.syscall
		capture := filepath.Join(r.dir, name+"-capture")
		tracePath := filepath.Join(r.dir, name+".strace")
		out, _, code, runErr := r.command(name, "env", "GOMAXPROCS=1", "GODEBUG=asyncpreemptoff=1", "strace", "-f", "-yy", "-e", "trace=openat,write,fsync,close,linkat,unlinkat", "-e", "inject="+fault.syscall+":error=ENOSPC:when="+fault.when, "-o", tracePath, binary, "--replay", core, "--postprocess", "default", "--capture", capture)
		if code != 1 || runErr == nil || len(out) != 0 {
			return nil, fmt.Errorf("%s must fail with exit 1 and empty stdout, got %d", name, code)
		}
		trace, err := read(tracePath)
		if err != nil {
			return nil, err
		}
		var injected string
		for _, line := range strings.Split(string(trace), "\n") {
			if strings.Contains(line, "(INJECTED)") && strings.Contains(line, "ENOSPC") && strings.Contains(line, fault.syscall+"(") {
				if injected != "" {
					return nil, fmt.Errorf("%s injected more than one failure", name)
				}
				injected = line
			}
		}
		if injected == "" {
			return nil, fmt.Errorf("%s did not inject its syscall failure", name)
		}
		if _, err := os.Lstat(filepath.Join(capture, "processed-manifest.json")); !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%s left a final manifest after failed %s: %v", name, fault.syscall, err)
		}
		if !strings.Contains(injected, capture+"/.processed-manifest-") {
			return nil, fmt.Errorf("%s fault missed the temporary manifest: %s", name, injected)
		}
		entries, err := os.ReadDir(capture)
		if err != nil {
			return nil, err
		}
		if len(entries) != 3 {
			return nil, fmt.Errorf("%s left %d files instead of three prerequisites", name, len(entries))
		}
		for _, entry := range entries {
			switch entry.Name() {
			case "processed-policy.json", "processed-output.json", "processed.fastq":
			default:
				return nil, fmt.Errorf("%s left unexpected file %s", name, entry.Name())
			}
		}
		files, err := pins([]string{capture})
		if err != nil {
			return nil, err
		}
		results = append(results, map[string]any{"syscall": fault.syscall, "exit": code, "stdout_bytes": len(out), "final_manifest_absent": true, "temporary_manifest_removed": true, "injected": injected, "trace_sha256": sha(trace), "retained_payloads": files})
	}
	return results, nil
}
