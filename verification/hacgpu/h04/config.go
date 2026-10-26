package h04

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h02"
)

// ModelFLSTM is the recurrent geometry the pinned model config declares.
//
// `BasecallModelConfig.cpp:242-281` derives it from the encoder sublayers:
// `lstm_size` is the last convolution's output size, `lstm_layers` is the number
// of `flstm_softout` blocks and `lstm_inner_dim` is their shared `inner_dim`. The
// per-block `reverse` key is never read for FLSTM models: `CRFModel.cpp:38` passes
// `first_reverse = true` as a literal and `FLSTMStackImpl` derives each layer's
// direction as `first_reverse ^ (layer & 1)` (`FLSTMStack.cpp:163`). The consumer
// records that declared rule rather than trusting the unread config key.
type ModelFLSTM struct {
	Layers   int
	Size     int
	InnerDim int
	Features int
}

// ModelFLSTMConfig reads the pinned config and returns the declared geometry. A
// missing key, a mixed LSTM/FLSTM stack or two different inner dimensions is a
// rejection: the port must not guess the shape it is packing for.
func ModelFLSTMConfig(modelDir string) (ModelFLSTM, error) {
	var out ModelFLSTM
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.toml"))
	if err != nil {
		return out, err
	}
	section := ""
	block := map[string]string{}
	blocks := []map[string]string{}
	flush := func() {
		if section == "encoder.sublayers" && len(block) > 0 {
			blocks = append(blocks, block)
			block = map[string]string{}
		}
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			flush()
			section = strings.Trim(line, "[]")
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if section == "input" && key == "features" {
			parsed, perr := strconv.Atoi(value)
			if perr != nil {
				return out, fmt.Errorf("input features %q: %w", value, perr)
			}
			out.Features = parsed
		}
		if section == "encoder.sublayers" {
			block[key] = value
		}
	}
	flush()

	flstm, plainLSTM := 0, 0
	for i, b := range blocks {
		switch strings.Trim(b["type"], `"`) {
		case "flstm_softout":
			flstm++
			inner, perr := strconv.Atoi(b["inner_dim"])
			if perr != nil {
				return out, fmt.Errorf("flstm sublayer %d has no integer inner_dim", i)
			}
			if out.InnerDim == 0 {
				out.InnerDim = inner
			} else if out.InnerDim != inner {
				return out, fmt.Errorf("flstm sublayer %d declares inner_dim %d, earlier blocks declare %d",
					i, inner, out.InnerDim)
			}
		case "lstm":
			plainLSTM++
		}
	}
	if flstm == 0 {
		return out, fmt.Errorf("model config declares no flstm sublayers")
	}
	if plainLSTM > 0 {
		return out, fmt.Errorf("model config mixes %d lstm and %d flstm sublayers", plainLSTM, flstm)
	}
	out.Layers = flstm

	convs, err := h02.ModelConvs(modelDir)
	if err != nil {
		return out, err
	}
	if len(convs) == 0 {
		return out, fmt.Errorf("model config declares no convolutions")
	}
	out.Size = int(convs[len(convs)-1].COut)
	if out.Features <= 0 {
		return out, fmt.Errorf("model config declares input features %d", out.Features)
	}
	return out, nil
}
