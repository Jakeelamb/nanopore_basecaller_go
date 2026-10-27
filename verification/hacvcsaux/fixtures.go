package hacvcsaux

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
)

//go:embed fixtures.json
var fixtureJSON []byte

type Fixture struct {
	ID               string `json:"id"`
	BatchSize        int    `json:"batch_size"`
	FullChunkSizeOut int    `json:"full_chunk_size_out"`
	SizesOut         []int  `json:"sizes_out"`
	Repeat           int    `json:"repeat"`
}

type Matrix struct {
	Schema            string    `json:"schema"`
	DoradoHead        string    `json:"dorado_head"`
	SubbatchSize      int       `json:"subbatch_size"`
	InputStride       int       `json:"input_stride"`
	WorkspaceFills    []int32   `json:"workspace_fills"`
	DestinationFills  []int32   `json:"destination_fills"`
	ThreadPoolWorkers []int     `json:"thread_pool_workers"`
	Cases             []Fixture `json:"cases"`
	GoRejections      []string  `json:"go_rejections"`
	Comparison        string    `json:"comparison"`
	ExcludedData      string    `json:"excluded_data"`
}

const MatrixSHA256 = "d4602d3fd2ac2e45515a949087b82fe50fe5a0a92e3dcffc66b82979d0c9e906"

func ReadMatrix() (Matrix, error) {
	h := sha256.Sum256(fixtureJSON)
	if hex.EncodeToString(h[:]) != MatrixSHA256 {
		return Matrix{}, fmt.Errorf("frozen fixture matrix changed")
	}
	var m Matrix
	if err := json.Unmarshal(fixtureJSON, &m); err != nil {
		return m, err
	}
	if m.Schema != "nanopore-hac-vcs-aux-fixtures/v1" || m.DoradoHead != "8b8fc5d36a9c0baab262a743cb175b5878e38ca6" || m.SubbatchSize != 32 || m.InputStride != 6 {
		return m, fmt.Errorf("unsupported fixture profile")
	}
	return m, nil
}

func (f Fixture) Aux(stride int) (*gpubatch.Aux, error) {
	if stride != 6 || f.Repeat < 1 || f.FullChunkSizeOut < 1 || f.FullChunkSizeOut > gpubatch.MaxCoordinate/stride || len(f.SizesOut) == 0 || f.Repeat > gpubatch.MaxCoordinate/len(f.SizesOut) {
		return nil, fmt.Errorf("invalid fixture %q geometry", f.ID)
	}
	sizes := make([]int, 0, f.Repeat*len(f.SizesOut))
	for i := 0; i < f.Repeat; i++ {
		for _, size := range f.SizesOut {
			if size < 1 || size > f.FullChunkSizeOut {
				return nil, fmt.Errorf("invalid fixture %q chunk size", f.ID)
			}
			sizes = append(sizes, stride*size)
		}
	}
	cfg := gpubatch.Config{Lane: gpubatch.HAC, ChunkSize: stride * f.FullChunkSizeOut, BatchSize: f.BatchSize, Stride: stride, InnerStride: stride, Granularity: stride, VariableChunks: true}
	return gpubatch.NewAux(cfg, sizes)
}
