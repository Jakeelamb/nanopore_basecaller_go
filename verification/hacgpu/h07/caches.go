package h07

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type CacheTensor struct {
	Role, DType, File, SHA256 string
	Shape                     []int64
	Bytes                     int64
}

// ReadCacheDescriptors validates the native three-cache publication. The
// producer receipt and frozen build/runtime closure must be authenticated by
// the caller before treating these bytes as native evidence.
func ReadCacheDescriptors(dir string) ([]CacheTensor, error) {
	file, err := os.Open(filepath.Join(dir, "caches.tsv"))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.Comma = '\t'
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) != 4 || len(rows[0]) != 1 || rows[0][0] != "nanopore-hac-output-caches/v1" {
		return nil, fmt.Errorf("incomplete or unknown native projection cache publication")
	}
	wants := []CacheTensor{{Role: "down", DType: "Char", Shape: []int64{128, 1024}, Bytes: 131072}, {Role: "down-scale", DType: "Half", Shape: []int64{128}, Bytes: 256}, {Role: "up", DType: "Half", Shape: []int64{1024, 128}, Bytes: 262144}}
	for i, want := range wants {
		row := rows[i+1]
		if len(row) != 5 {
			return nil, fmt.Errorf("invalid cache row")
		}
		shape := []string{}
		for _, n := range want.Shape {
			shape = append(shape, strconv.FormatInt(n, 10))
		}
		if row[0] != want.Role || row[1] != want.DType || row[2] != strings.Join(shape, ",") || row[3] != strconv.FormatInt(want.Bytes, 10) || row[4] != want.Role+".bin" {
			return nil, fmt.Errorf("native cache descriptor mismatch for %s", want.Role)
		}
		body, err := os.ReadFile(filepath.Join(dir, row[4]))
		if err != nil {
			return nil, err
		}
		if int64(len(body)) != want.Bytes {
			return nil, fmt.Errorf("native cache byte count differs for %s", want.Role)
		}
		wants[i].File = row[4]
		wants[i].SHA256 = sum(body)
	}
	return wants, nil
}
