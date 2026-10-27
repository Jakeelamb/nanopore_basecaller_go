package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacvcsaux"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	unit := flag.String("unit", "", "inputs, compiler-check, artifact-check, or compare")
	out := flag.String("out", "", "fresh native input file")
	oracle := flag.String("oracle", "", "unexecuted native oracle")
	oracleSHA := flag.String("oracle-sha", "", "independent native output pin")
	evidence := flag.String("evidence", "", "genuine native capture manifest")
	evidenceSHA := flag.String("evidence-sha", "", "independent native capture pin")
	flag.Parse()
	if flag.NArg() != 0 || *unit != "inputs" && *unit != "compiler-check" && *unit != "artifact-check" && *unit != "compare" {
		fmt.Fprintln(os.Stderr, "use -unit inputs, compiler-check, artifact-check, or compare")
		os.Exit(2)
	}
	if *unit == "compare" {
		result, err := hacvcsaux.CompareNative(*evidence, *evidenceSHA)
		if err == nil {
			err = json.NewEncoder(os.Stdout).Encode(result)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *unit == "artifact-check" {
		result, err := hacvcsaux.InspectOracle(*oracle, *oracleSHA)
		if err == nil {
			err = json.NewEncoder(os.Stdout).Encode(result)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *unit == "compiler-check" {
		result, err := hacvcsaux.CheckCompiler()
		if err == nil {
			err = json.NewEncoder(os.Stdout).Encode(result)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *out == "" {
		fmt.Fprintln(os.Stderr, "inputs requires -out NEW_FILE")
		os.Exit(2)
	}
	if err := hacvcsaux.PrepareInputs(*out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sha, err := runner.SHA(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("matrix_sha256=%s inputs_sha256=%s\n", hacvcsaux.MatrixSHA256, sha)
}
