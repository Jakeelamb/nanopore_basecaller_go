package packed

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed native/observer.h
var observerHeader []byte

type Insertion struct {
	Name    string `json:"name"`
	Section string `json:"section"`
	Anchor  string `json:"anchor"`
	Before  string `json:"before"`
	After   string `json:"after"`
}

type GeneratedUnit struct {
	Source     Pin         `json:"source"`
	Original   Pin         `json:"original_copy"`
	Observed   Pin         `json:"observed_copy"`
	Insertions []Insertion `json:"insertions"`
}

type Generated struct {
	Schema     string          `json:"schema"`
	Status     string          `json:"status"`
	Header     Pin             `json:"header"`
	Units      []GeneratedUnit `json:"units"`
	Reversible bool            `json:"reversible"`
}

type sourceSpec struct {
	Path, SHA string
	Patches   []Insertion
}

func before(name, section, anchor, text string) Insertion {
	return Insertion{name, section, anchor, text, ""}
}
func after(name, section, anchor, text string) Insertion {
	return Insertion{name, section, anchor, "", text}
}
func include(anchor string) Insertion {
	return after("observer-header", "", anchor, "#include \"observer.h\"\n")
}

func sourceSpecs() []sourceSpec {
	const graph = "at::Tensor CRFModelImpl::run_koi("
	const conv = "void ConvStackImpl::ConvLayer::run_koi("
	const lstm = "void FLSTMStackImpl::forward_koi("
	const proj = "void FactorisedLinearLayerImpl::forward_koi("
	const aux = "void AuxiliaryData::create_auxiliary_data("
	convCall := func(phase, out, has string) string {
		value := "0"
		accumulator := "std::nullopt"
		if has != "false" {
			value = "res"
		}
		if phase == "fast-result" {
			accumulator = "packed_hac_observer::AccumulatorMode{use_f32_accum ? KOI_F32 : KOI_F16}"
		}
		return "        packed_hac_observer::conv(conv_layer_num, \"" + phase + "\", wm, aux, in, " + out + ", w_device, w_t_device, b_device, " + has + ", " + value + ", " + accumulator + ");\n"
	}
	lstmCall := func(phase string) string {
		return "        packed_hac_observer::lstm(layer, \"" + phase + "\", wm, aux, inout, dn_bfr, dn_scale_bfr, workspace_bfr, aux ? (reverse ? aux->device_bwd_encoding : aux->device_fwd_encoding) : at::Tensor{}, wm.T + (aux && reverse), reverse ? -1 : 1, parity, {{\"dn_ih\", device_dn_weights_ih_[layer]}, {\"dn_ih_scale\", device_dn_weights_scale_ih_[layer]}, {\"dn_hh\", device_dn_weights_hh_[layer]}, {\"dn_hh_scale\", device_dn_weights_scale_hh_[layer]}, {\"up\", device_up_weights_[layer]}, {\"up_scale\", device_up_weights_scale_[layer]}, {\"bias\", device_up_bias_[layer]}});\n"
	}
	projCall := func(phase string) string {
		return "    packed_hac_observer::projection(\"" + phase + "\", wm, aux, in_bfr, dn_bfr, out_bfr, aux ? aux->device_out_layout : at::Tensor{}, parity, SCALE, {{\"dn\", device_dn_weight_}, {\"dn_scale\", device_dn_weight_scale_}, {\"up\", device_up_weight_}});\n"
	}
	return []sourceSpec{
		{"dorado/basecall/model/CRFModel.cpp", "37f1f9712c3e61957ff17099d4cffb606c6cc784d2973c040bcc77129b5f9964", []Insertion{
			include("#include \"CRFModel.h\"\n"),
			before("reserved", graph, "    wm.allocate_backing_tensor(in.device());", "    packed_hac_observer::graph(\"reserved\", wm, aux);\n"),
			after("allocated", graph, "    wm.allocate_backing_tensor(in.device());\n", "    packed_hac_observer::graph(\"allocated\", wm, aux);\n"),
			before("input", graph, "    convs->run_koi(wm, aux);", "    packed_hac_observer::graph(\"input-before-convs\", wm, aux);\n"),
			after("convs", graph, "    convs->run_koi(wm, aux);\n", "    packed_hac_observer::graph(\"after-convs\", wm, aux);\n"),
			after("rnns", graph, "    rnns->run_koi(wm, aux);\n", "    packed_hac_observer::graph(\"after-rnns\", wm, aux);\n"),
			after("projection", graph, "    linear1->run_koi(wm, aux);\n", "    packed_hac_observer::graph(\"after-linear1\", wm, aux);\n"),
		}},
		{"dorado/nn/ConvStack.cpp", "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2", []Insertion{
			include("#include \"nn/ConvStack.h\"\n"),
			after("small-allocated", conv, "        out = out.narrow(1, output_T_padding, T_out);\n", convCall("small-before-convolution", "out", "false")),
			before("small-repair-before", conv, "        if (aux) {\n            if (host_convolution_postprocess", convCall("small-after-convolution-before-repair", "out", "false")),
			before("small-repair-after", conv, "        }\n    } else if (cutlass_conv) {\n", convCall("small-after-repair", "out", "false")),
			before("padding-before", conv, "        in.slice(1, 0, padding) = 0;", "        packed_hac_observer::tensor(\"conv-before-padding-zero\", wm, aux, {{\"input\", in}});\n"),
			after("padding-after", conv, "        in.slice(1, -padding, torch::indexing::None) = 0;\n", "        packed_hac_observer::tensor(\"conv-after-padding-zero\", wm, aux, {{\"input\", in}});\n"),
			before("scatter-allocated", conv, "        const bool try_fast_conv =", convCall("scatter-allocated", "out_ntc", "false")),
			after("fast-status", conv, "                                   KOI_F16, out_type, use_f32_accum ? KOI_F32 : KOI_F16);\n", convCall("fast-result", "out_ntc", "true")),
			before("fallback-before", conv, "            res = host_linear(stream, KOI_F16, get_koi_activation(params.activation), out_type,", convCall("fallback-before", "out_ntc", "try_fast_conv")),
			after("fallback-status", conv, "                              nullptr, b_device.data_ptr());\n", convCall("fallback-result", "out_ntc", "true")),
			before("scatter-repair-before", conv, "            res = host_convolution_postprocess(\n", convCall("scatter-before-repair", "out_ntc", "true")),
			after("scatter-repair-status", conv, "                    out_ntc.data_ptr(), out_layout, w_t_device.data_ptr(), b_device.data_ptr());\n", convCall("scatter-after-repair", "out_ntc", "true")),
		}},
		{"dorado/nn/FLSTMStack.cpp", "97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160", []Insertion{
			include("#include <tuple>\n"),
			before("sentinels-before", lstm, "    if (aux == nullptr) {", "    packed_hac_observer::tensor(\"lstm-before-sentinels\", wm, aux, {{\"inout\", inout}});\n"),
			before("sentinels-after", lstm, "    auto stream = at::cuda::getCurrentCUDAStream().stream();", "    packed_hac_observer::tensor(\"lstm-after-sentinels\", wm, aux, {{\"inout\", inout}});\n"),
			before("scratch-before", lstm, "    temp_bfr.zero_();", "    packed_hac_observer::tensor(\"lstm-before-scratch-zero\", wm, aux, {{\"scratch\", temp_bfr}});\n"),
			after("scratch-after", lstm, "    temp_bfr.zero_();\n", "    packed_hac_observer::tensor(\"lstm-after-scratch-zero\", wm, aux, {{\"scratch\", temp_bfr}});\n"),
			before("lstm-call-before", lstm, "        host_factorised_lstm(\n", lstmCall("before-call")),
			after("lstm-call-after", lstm, "                device_up_weights_scale_[layer].data_ptr(), device_up_bias_[layer].data_ptr());\n", lstmCall("after-call")),
			after("lstm-orientation", lstm, "        wm.is_input_to_rev_lstm = !reverse;\n", lstmCall("after-orientation")),
		}},
		{"dorado/nn/FactorisedLinearLayer.cpp", "2f41d8688ad3aab27c20fa257320f66e8f32eb9c29348163b7b297833a636288", []Insertion{
			include("#include \"nn/FactorisedLinearLayer.h\"\n"),
			after("row3", proj, "    auto in_bfr = wm.current.narrow(0, aux ? 3 : 2, wm.T);\n", "    packed_hac_observer::tensor(\"projection-row3-before-mutation\", wm, aux, {{\"input\", in_bfr}});\n"),
			before("down-before", proj, "    dn_bfr.zero_();", "    packed_hac_observer::tensor(\"projection-before-down-zero\", wm, aux, {{\"down\", dn_bfr}});\n"),
			after("down-after", proj, "    dn_bfr.zero_();\n", "    packed_hac_observer::tensor(\"projection-after-down-zero\", wm, aux, {{\"down\", dn_bfr}});\n"),
			before("projection-call-before", proj, "    host_factorised_linear(stream, wm.N, wm.T, parity, in_bfr.data_ptr(),", projCall("before-call")),
			after("projection-call-after", proj, "                           SCALE, out_layout, out_bfr.data_ptr());\n", projCall("after-call")),
			after("compact-scores", proj, "    wm.current = out_bfr.view({wm.N, wm.T, wm.C});\n", projCall("compact-scores")),
		}},
		{"dorado/nn/AuxiliaryData.cpp", "06ed728c6cc9e9ce09d80e492f1cecbba92c527ba91fd7a1d9b31c1250c4c140", []Insertion{
			include("#include \"nn/AuxiliaryData.h\"\n"),
			after("constructed", "AuxiliaryData::AuxiliaryData(", "        cs /= stride_out_;\n    }\n", "    packed_hac_observer::aux(\"constructed\", *this, workspace_, chunk_sizes, chunk_intervals_, chunk_table_, false, 0);\n"),
			before("upload-before", aux, "        const int status = host_lstm_preprocess(", "        packed_hac_observer::aux(\"before-preprocess\", *this, workspace_, {}, chunk_intervals_, chunk_table_, false, 0);\n"),
			after("upload-after", aux, "                device_bwd_encoding.data_ptr<std::int32_t>());\n", "        packed_hac_observer::aux(\"after-preprocess\", *this, workspace_, {}, chunk_intervals_, chunk_table_, true, status);\n"),
			after("intervals-table-uploaded", aux, "                at::from_blob(std::data(chunk_table_), {total_num_varlen_chunks_, 2}, cpu_options)\n                        .to(gpu_options);\n", "        packed_hac_observer::aux(\"after-all-uploads\", *this, workspace_, {}, chunk_intervals_, chunk_table_, true, status);\n"),
		}},
	}
}

// section bounds observations to a method, avoiding similar fixed-path text.
func section(body, marker string) (int, int, error) {
	if marker == "" {
		return 0, len(body), nil
	}
	if strings.Count(body, marker) != 1 {
		return 0, 0, fmt.Errorf("unique method marker required: %s", marker)
	}
	start := strings.Index(body, marker)
	// All selected pinned methods end at a column-zero closing brace.
	end := strings.Index(body[start:], "\n}\n")
	if end < 0 {
		return 0, 0, fmt.Errorf("method terminator missing")
	}
	return start, start + end + 3, nil
}

func instrument(original []byte, spec sourceSpec) ([]byte, error) {
	if digest(original) != spec.SHA {
		return nil, fmt.Errorf("whole-file source SHA differs: %s", spec.Path)
	}
	body := string(original)
	for _, p := range spec.Patches {
		start, end, err := section(body, p.Section)
		if err != nil {
			return nil, err
		}
		part := body[start:end]
		if strings.Count(part, p.Anchor) != 1 {
			return nil, fmt.Errorf("unique anchor required: %s/%s", spec.Path, p.Name)
		}
		part = strings.Replace(part, p.Anchor, p.Before+p.Anchor+p.After, 1)
		body = body[:start] + part + body[end:]
	}
	restored, err := reverse([]byte(body), spec.Patches)
	if err != nil || !bytes.Equal(restored, original) {
		return nil, fmt.Errorf("exact reversal failed: %s: %v", spec.Path, err)
	}
	return []byte(body), nil
}

func reverse(observed []byte, patches []Insertion) ([]byte, error) {
	body := string(observed)
	for i := len(patches) - 1; i >= 0; i-- {
		p := patches[i]
		start, end, err := section(body, p.Section)
		if err != nil {
			return nil, err
		}
		part := body[start:end]
		text := p.Before + p.Anchor + p.After
		if strings.Count(part, text) != 1 {
			return nil, fmt.Errorf("unique reverse insertion required: %s", p.Name)
		}
		body = body[:start] + strings.Replace(part, text, p.Anchor, 1) + body[end:]
	}
	return []byte(body), nil
}

func generationPins(g Generated) []Pin {
	p := []Pin{g.Header}
	for _, u := range g.Units {
		p = append(p, u.Source, u.Original, u.Observed)
	}
	return p
}

// ValidateGeneration recomputes expected insertions from this controller, not
// from a caller-controlled insertion manifest.
func ValidateGeneration(g Generated) error {
	cache, _, err := collectInputs(generationPins(g))
	if err != nil {
		return err
	}
	return validateGeneration(g, cache)
}

func validateGeneration(g Generated, cache map[string][]byte) error {
	specs := sourceSpecs()
	if g.Schema != GenerationSchema || g.Status != "source-only-uncompiled" || !g.Reversible || len(g.Units) != len(specs) {
		return fmt.Errorf("generation contract differs")
	}
	if !bytes.Equal(cache[g.Header.Path], observerHeader) {
		return fmt.Errorf("observer ABI differs")
	}
	for i, spec := range specs {
		u := g.Units[i]
		if u.Source.SHA256 != spec.SHA || !strings.HasSuffix(u.Source.Path, "/"+spec.Path) || len(u.Insertions) != len(spec.Patches) {
			return fmt.Errorf("source order/identity differs")
		}
		for j, p := range spec.Patches {
			if u.Insertions[j] != p {
				return fmt.Errorf("insertion manifest differs")
			}
		}
		original := cache[u.Source.Path]
		if !bytes.Equal(original, cache[u.Original.Path]) {
			return fmt.Errorf("original copy differs")
		}
		want, e := instrument(original, spec)
		if e != nil {
			return e
		}
		if !bytes.Equal(want, cache[u.Observed.Path]) {
			return fmt.Errorf("observed copy differs")
		}
	}
	return nil
}

// Generate copies the five actual source units; it never invokes a compiler.
func Generate(sourceRoot, out string) (g Generated, err error) {
	if err = canonical(sourceRoot); err != nil {
		return g, err
	}
	specs := sourceSpecs()
	pins := []Pin{}
	originals, observed := [][]byte{}, [][]byte{}
	for _, spec := range specs {
		path := filepath.Join(sourceRoot, filepath.FromSlash(spec.Path))
		if err = canonical(path); err != nil {
			return g, err
		}
		info, e := os.Stat(path)
		if e != nil {
			return g, e
		}
		pin := Pin{path, spec.SHA, info.Size()}
		b, e := readPin(pin)
		if e != nil {
			return g, e
		}
		obs, e := instrument(b, spec)
		if e != nil {
			return g, e
		}
		pins = append(pins, pin)
		originals, observed = append(originals, b), append(observed, obs)
	}
	// Keep outputs outside the entire source tree, not merely individual files.
	if beneath(sourceRoot, out) || beneath(out, sourceRoot) {
		return g, fmt.Errorf("source tree/output overlap")
	}
	if err = freshOutput(out, pins); err != nil {
		return g, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return g, err
	}
	g.Schema, g.Status, g.Reversible = GenerationSchema, "source-only-uncompiled", true
	if g.Header, err = writeNew(filepath.Join(out, "observer.h"), observerHeader); err != nil {
		return g, err
	}
	for i, spec := range specs {
		base := filepath.Base(spec.Path)
		u := GeneratedUnit{Source: pins[i], Insertions: spec.Patches}
		if u.Original, err = writeNew(filepath.Join(out, "original-"+base), originals[i]); err != nil {
			return g, err
		}
		if u.Observed, err = writeNew(filepath.Join(out, "observed-"+base), observed[i]); err != nil {
			return g, err
		}
		g.Units = append(g.Units, u)
	}
	if err = ValidateGeneration(g); err != nil {
		return g, err
	}
	_, err = writeJSON(filepath.Join(out, "generated.json"), g)
	return g, err
}
