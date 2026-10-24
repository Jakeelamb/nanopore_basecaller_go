package h02

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
)

// ModelConvs reads the `[[encoder.sublayers]]` convolution blocks of the pinned
// model config in source order. The values are declared, not inferred: a missing
// key, an unknown activation or a declared padding that differs from
// `winlen / 2` (the value ConvStack.cpp:223 actually passes) is a hard rejection.
func ModelConvs(modelDir string) ([]ConvSpec, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.toml"))
	if err != nil {
		return nil, err
	}
	var out []ConvSpec
	section := ""
	block := map[string]string{}
	flush := func() error {
		if section != "encoder.sublayers" || block["type"] != `"convolution"` {
			return nil
		}
		spec := ConvSpec{Index: len(out), Activation: strings.Trim(block["activation"], `"`)}
		for key, dst := range map[string]*int64{
			"insize": &spec.CIn, "size": &spec.COut, "winlen": &spec.WinLen,
			"stride": &spec.Stride, "padding": &spec.Padding,
		} {
			value, ok := block[key]
			if !ok {
				return fmt.Errorf("convolution sublayer %d has no %q", spec.Index, key)
			}
			parsed, perr := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			if perr != nil {
				return fmt.Errorf("convolution sublayer %d %s is not an integer: %w", spec.Index, key, perr)
			}
			*dst = parsed
		}
		if spec.CIn < 1 || spec.COut < 1 || spec.WinLen < 1 || spec.Stride < 1 {
			return fmt.Errorf("convolution sublayer %d geometry is not positive", spec.Index)
		}
		if _, aerr := convActivation(spec.Activation); aerr != nil {
			return fmt.Errorf("convolution sublayer %d: %w", spec.Index, aerr)
		}
		if spec.WinLen%2 != 1 {
			return fmt.Errorf("convolution sublayer %d window %d is not odd", spec.Index, spec.WinLen)
		}
		// ConvStack.cpp:223 passes `params.winlen / 2`, not `params.padding`. A
		// config whose declared padding differs would not be the call the port
		// models, so it is refused instead of silently substituted.
		if spec.Padding != spec.WinLen/2 {
			return fmt.Errorf("convolution sublayer %d declares padding %d but the source passes winlen/2 = %d",
				spec.Index, spec.Padding, spec.WinLen/2)
		}
		out = append(out, spec)
		return nil
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			if err = flush(); err != nil {
				return nil, err
			}
			section = strings.Trim(line, "[]")
			block = map[string]string{}
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		block[strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	if err = flush(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("model config %s declares no convolution sublayer", modelDir)
	}
	for i := 1; i < len(out); i++ {
		if out[i].CIn != out[i-1].COut {
			return nil, fmt.Errorf("convolution sublayer %d C_in %d does not follow sublayer %d C_out %d",
				i, out[i].CIn, i-1, out[i-1].COut)
		}
	}
	return out, nil
}

// convParams turns the declared sublayers into caller parameters. Only the last
// layer's layout is decided by the source, through
// `get_koi_lstm_input_layout(lp.size, lp.inner_dim, lp.activation)`
// (ConvStack.cpp:138-139); every earlier layer keeps the NTC default
// (ConvStack.h:32).
func convParams(convs []ConvSpec) ([]model.SmallConvParams, error) {
	out := make([]model.SmallConvParams, len(convs))
	for i, c := range convs {
		activation, err := convActivation(c.Activation)
		if err != nil {
			return nil, fmt.Errorf("convolution sublayer %d: %w", i, err)
		}
		layout := model.GPUStorageNTC
		if i == len(convs)-1 {
			layout, err = lastConvLayout(c)
			if err != nil {
				return nil, err
			}
		}
		out[i] = model.SmallConvParams{
			CIn: c.CIn, COut: c.COut, WinLen: c.WinLen, Stride: c.Stride,
			Activation: activation, OutputLayout: layout,
		}
	}
	return out, nil
}

func convActivation(name string) (model.SmallConvActivation, error) {
	switch name {
	case "swish":
		return model.SmallConvSwish, nil
	case "tanh":
		return model.SmallConvTanh, nil
	}
	return 0, fmt.Errorf("unrecognised convolution activation %q", name)
}

// lastConvLayout is the narrow slice of `get_koi_lstm_input_layout` this pinned
// graph needs, and it refuses everything it does not model rather than guessing.
//
// For a plain convolution `inner_dim == 0`, so the `flstm` branch is skipped and
// the layout is `NTC` for a quantised size (96 or 128), `CUTLASS_TNC_I8` for a
// TANH layer of size 129..1024 that is a multiple of 128, and `CUBLAS_TN2C`
// otherwise (ConvStack.cpp:66-102).
func lastConvLayout(c ConvSpec) (model.GPUStorageLayout, error) {
	switch {
	case c.COut == 96 || c.COut == 128:
		return model.GPUStorageNTC, nil
	case c.COut > 128 && c.COut <= 1024 && c.COut%128 == 0:
		switch c.Activation {
		case "tanh":
			return model.GPUStorageCUTLASSI8, nil
		case "swish":
			return model.GPUStorageCUTLASSF16, nil
		}
	}
	return 0, fmt.Errorf("last convolution C_out %d activation %q is outside the modelled Koi layout rule",
		c.COut, c.Activation)
}
