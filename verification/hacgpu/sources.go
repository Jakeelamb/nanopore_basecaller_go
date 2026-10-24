// Package hacgpu holds source-pinned host oracles and the native stage-capture
// harness for the HAC GPU caller port.
//
// H01 is verification-only. It never implements caller arithmetic: it compiles the
// pinned stock Dorado static libraries, drives the untouched CRF graph through the
// same call sequence `CRFModelImpl::run_koi` uses, and records the exact live bytes
// of every stage output plus the complete raw parent backing storage. Every
// downstream HAC card consumes those captures instead of running the graph again.
package hacgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// StageSources pins the exact lines the oracle's call sequence mirrors. The
// sequence in native/oracle.cpp is a literal transcription of these ranges.
var StageSources = []Source{
	{
		Path:   "dorado/basecall/model/CRFModel.cpp",
		SHA256: "37f1f9712c3e61957ff17099d4cffb606c6cc784d2973c040bcc77129b5f9964",
		Lines:  "87-145 run_koi: working-memory reservation, backing allocation, input copy, convs/rnns/linear1/linear2, scores",
	},
	{
		Path:   "dorado/basecall/crf_utils.cpp",
		SHA256: "0758b6f4d57693566ab15cb0332255ff75ea01ac4f529c04329cb1aac13caf0f",
		Lines:  "182-197 load_crf_model; 22-80 load_lstm_model_weights tensor names",
	},
	{
		Path:   "dorado/config/BasecallModelConfig.cpp",
		SHA256: "cfa62bbe112e942e70a74761e6968bdff880e6d30a92fcb89ae612ab8067bf4d",
		Lines:  "510-515 load_model_config",
	},
	{
		Path:   "dorado/nn/ConvStack.cpp",
		SHA256: "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2",
		Lines:  "191-211 ConvLayer::reserve_working_memory; 214-373 run_koi",
	},
	{
		Path:   "dorado/nn/FLSTMStack.cpp",
		SHA256: "97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160",
		Lines:  "125-295 reserve_working_memory and run_koi",
	},
	{
		Path:   "dorado/nn/FactorisedLinearLayer.cpp",
		SHA256: "2f41d8688ad3aab27c20fa257320f66e8f32eb9c29348163b7b297833a636288",
		Lines:  "52-106 reserve_working_memory and run_koi",
	},
	{
		Path:   "dorado/nn/WorkingMemory.cpp",
		SHA256: "4b512195a11e418ce35f8a60cf050dc499cec1b3c5504fa6ce8e21d6986e2b0d",
		Lines:  "30-33 tensor_bytes; 35-59 next; 83-102 next_TC; 108-114 allocate_backing_tensor",
	},
	{
		Path:   "dorado/nn/include/nn/WorkingMemory.h",
		SHA256: "271f1c1e4451832eacfb8553a1f72902fa8795040c1e36aa282f08d214f1aa6e",
		Lines:  "55-82 public state: ALIGNMENT, reservation_bytes, backing_tensor, current, layout, N/T/C",
	},
}

// VerifyStageSources re-hashes every pinned stage source under the read-only
// Dorado checkout and rejects any drift before a capture is attempted.
func VerifyStageSources(doradoSource string) error {
	if doradoSource == "" {
		return fmt.Errorf("pinned Dorado source path is required")
	}
	for _, s := range StageSources {
		path := filepath.Join(doradoSource, s.Path)
		got, err := fileSHA(path)
		if err != nil {
			return err
		}
		if got != s.SHA256 {
			return fmt.Errorf("pinned source changed: %s", s.Path)
		}
	}
	return nil
}

// PinStageSources records the verified pins in a manifest map.
func PinStageSources(doradoSource string, into map[string]string) error {
	if err := VerifyStageSources(doradoSource); err != nil {
		return err
	}
	for _, s := range StageSources {
		path := filepath.Join(doradoSource, s.Path)
		got, err := fileSHA(path)
		if err != nil {
			return err
		}
		into[path] = got
	}
	return nil
}

func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
