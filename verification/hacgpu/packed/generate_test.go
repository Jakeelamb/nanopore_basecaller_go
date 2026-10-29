package packed

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const pinnedSourceRoot = "/home/jake/Projects/nanopore_basecaller_go/external/dorado-source"

// These tests deliberately require the actual pinned source. They do not
// silently skip or replace it with a synthetic producer fixture.
func TestActualSourceAnchorsAndReversal(t *testing.T) {
	for _, spec := range sourceSpecs() {
		t.Run(filepath.Base(spec.Path), func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join(pinnedSourceRoot, spec.Path))
			if err != nil {
				t.Fatal(err)
			}
			if digest(b) != spec.SHA {
				t.Fatal("actual source SHA differs")
			}
			observed, err := instrument(b, spec)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := reverse(observed, spec.Patches)
			if err != nil || !bytes.Equal(restored, b) {
				t.Fatalf("exact reversal: %v", err)
			}
			if bytes.Equal(observed, b) {
				t.Fatal("no observation calls")
			}
			corrupt := append([]byte(nil), b...)
			corrupt[len(corrupt)/2] ^= 1
			if _, err := instrument(corrupt, spec); err == nil {
				t.Fatal("corrupted actual source accepted")
			}
			for _, patch := range spec.Patches {
				// Exercise uniqueness after a separately recomputed test SHA so
				// the anchor control is independent of the whole-file gate.
				bad := strings.Replace(string(b), patch.Anchor, patch.Anchor+patch.Anchor, 1)
				control := spec
				control.SHA = digest([]byte(bad))
				if _, err := instrument([]byte(bad), control); err == nil {
					t.Fatalf("duplicate anchor accepted: %s", patch.Name)
				}
			}
			broken := bytes.Replace(observed, []byte("packed_hac_observer::"), []byte("broken_observer::"), 1)
			if _, err := reverse(broken, spec.Patches); err == nil {
				t.Fatal("corrupted insertion reversed")
			}
		})
	}
}

func TestGenerateActualCopiesAndManifestRejection(t *testing.T) {
	out := filepath.Join(t.TempDir(), "generated")
	g, err := Generate(pinnedSourceRoot, out)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateGeneration(g); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	if err := strictJSON(b, new(Generated)); err != nil {
		t.Fatal(err)
	}
	header, err := json.Marshal(g.Header)
	if err != nil {
		t.Fatal(err)
	}
	controls := map[string][2]string{
		"root-case":          {`"schema":`, `"SCHEMA":`},
		"schema-alias":       {`"schema":"PackedHACGeneration/v1"`, `"schema":"wrong","SCHEMA":"PackedHACGeneration/v1"`},
		"nested-case":        {`"path":`, `"PATH":`},
		"nested-alias":       {`"before":""`, `"before":"wrong","BEFORE":""`},
		"missing-insertion":  {`"before":"",`, ``},
		"null-insertion":     {`"before":""`, `"before":null`},
		"missing-reversible": {`,"reversible":true`, ``},
		"null-header":        {`"header":` + string(header), `"header":null`},
	}
	for name, change := range controls {
		t.Run(name, func(t *testing.T) {
			bad := strings.Replace(string(b), change[0], change[1], 1)
			if bad == string(b) {
				t.Fatal("control did not change generation")
			}
			if err := strictJSON([]byte(bad), new(Generated)); err == nil {
				t.Fatal("invalid generation declaration accepted")
			}
		})
	}
	if _, err := Generate(pinnedSourceRoot, out); err == nil {
		t.Fatal("existing output accepted")
	}
	g.Units[0].Insertions[0].After += "// unauthorized\n"
	if err := ValidateGeneration(g); err == nil {
		t.Fatal("altered insertion manifest accepted")
	}
}

func TestGraphOperandHooks(t *testing.T) {
	for _, spec := range sourceSpecs() {
		if strings.HasSuffix(spec.Path, "FLSTMStack.cpp") {
			for _, p := range spec.Patches {
				if p.Name == "lstm-call-before" && (!strings.Contains(p.Before, "device_bwd_encoding") || !strings.Contains(p.Before, "wm.T + (aux && reverse)")) {
					t.Fatal("encoded source call binding missing")
				}
			}
		}
		if strings.HasSuffix(spec.Path, "FactorisedLinearLayer.cpp") {
			if !strings.Contains(spec.Patches[1].Anchor, "aux ? 3 : 2") {
				t.Fatal("packed row3 binding missing")
			}
		}
	}
}

func TestActualFastConvolutionAccumulatorOperand(t *testing.T) {
	for _, spec := range sourceSpecs() {
		if !strings.HasSuffix(spec.Path, "ConvStack.cpp") {
			continue
		}
		original, err := os.ReadFile(filepath.Join(pinnedSourceRoot, spec.Path))
		if err != nil {
			t.Fatal(err)
		}
		observed, err := instrument(original, spec)
		if err != nil {
			t.Fatal(err)
		}
		const selected = "packed_hac_observer::AccumulatorMode{use_f32_accum ? KOI_F32 : KOI_F16}"
		found := false
		for _, p := range spec.Patches {
			if p.Name == "fast-status" {
				found = true
				if !strings.Contains(p.After, selected) || !strings.Contains(string(observed), p.Anchor+p.After) {
					t.Fatal("actual selected fast accumulator operand missing")
				}
			} else if strings.Contains(p.Before+p.After, "packed_hac_observer::conv(") && !strings.Contains(p.Before+p.After, ", std::nullopt);") {
				t.Fatal("nonfast phase invents an accumulator mode")
			}
		}
		if !found || strings.Count(string(observed), selected) != 1 {
			t.Fatal("fast accumulator insertion missing or duplicated")
		}
	}
	for _, declaration := range []string{
		"struct AccumulatorMode {\n    int koi_type;\n};",
		"std::optional<AccumulatorMode> accumulator_mode;",
		"std::optional<AccumulatorMode> accumulator_mode);",
	} {
		if !bytes.Contains(observerHeader, []byte(declaration)) {
			t.Fatalf("typed accumulator/event contract missing: %s", declaration)
		}
	}
}

func TestActualFallbackStatusAvailability(t *testing.T) {
	for _, spec := range sourceSpecs() {
		if !strings.HasSuffix(spec.Path, "ConvStack.cpp") {
			continue
		}
		original, err := os.ReadFile(filepath.Join(pinnedSourceRoot, spec.Path))
		if err != nil {
			t.Fatal(err)
		}
		observed, err := instrument(original, spec)
		if err != nil {
			t.Fatal(err)
		}
		// Check actual-source scope and call order as text, not native behavior.
		// Skipping fast convolution makes its status unavailable at fallback
		// entry; attempting it retains res, including a failed native result.
		const want = `        packed_hac_observer::conv(conv_layer_num, "scatter-allocated", wm, aux, in, out_ntc, w_device, w_t_device, b_device, false, 0, std::nullopt);
        const bool try_fast_conv = utils::get_dev_opt<bool>("koi_conv", true);
        int res = -1;
        if (try_fast_conv) {
            utils::ScopedProfileRange spr3("host_convolution", 4);
            const bool use_f32_accum = utils::get_dev_opt<bool>("koi_conv_f32", false);
            const int N = aux ? 1 : wm.N;
            res = host_convolution(stream, N, T_in, C_in, C_out, params.winlen, params.stride,
                                   padding, int(in.stride(0)), int(out_ntc.stride(0)),
                                   int(out_ntc.stride(1)),
                                   in.slice(1, padding, torch::indexing::None).data_ptr(),
                                   out_ntc.data_ptr(), out_layout, w_device.data_ptr(),
                                   b_device.data_ptr(), get_koi_activation(params.activation),
                                   KOI_F16, out_type, use_f32_accum ? KOI_F32 : KOI_F16);
        packed_hac_observer::conv(conv_layer_num, "fast-result", wm, aux, in, out_ntc, w_device, w_t_device, b_device, true, res, packed_hac_observer::AccumulatorMode{use_f32_accum ? KOI_F32 : KOI_F16});
        }
        if (res != KOI_SUCCESS) {
            utils::ScopedProfileRange spr3("host_linear", 4);
        packed_hac_observer::conv(conv_layer_num, "fallback-before", wm, aux, in, out_ntc, w_device, w_t_device, b_device, try_fast_conv, res, std::nullopt);
            res = host_linear(stream, KOI_F16, get_koi_activation(params.activation), out_type,
                              aux ? 1 : wm.N, T_out, C_in * params.winlen, C_out, int(in.stride(0)),
                              params.stride * C_in, int(out_ntc.stride(0)), int(out_ntc.stride(1)),
                              in.data_ptr(), w_device.data_ptr(), out_ntc.data_ptr(), out_layout,
                              nullptr, b_device.data_ptr());
        packed_hac_observer::conv(conv_layer_num, "fallback-result", wm, aux, in, out_ntc, w_device, w_t_device, b_device, true, res, std::nullopt);
        }
        if (res != KOI_SUCCESS) {`
		if strings.Count(string(observed), want) != 1 {
			t.Fatal("fallback status availability, source scope or actual call order differs")
		}
		if strings.Count(string(observed), `utils::get_dev_opt<bool>("koi_conv", true)`) != 1 {
			t.Fatal("fast convolution selection reevaluated")
		}
		return
	}
	t.Fatal("actual convolution source specification missing")
}
