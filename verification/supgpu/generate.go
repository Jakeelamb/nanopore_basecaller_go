package supgpu

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed native/observer.h
var observerHeader string

//go:embed native/graph_oracle.cpp
var graphOracle string

//go:embed native/audit.c
var loaderAudit string

type Insertion struct {
	Source       string `json:"source"`
	BlockSHA256  string `json:"block_sha256"`
	AnchorSHA256 string `json:"anchor_sha256"`
	Count        int    `json:"count"`
}
type Generated struct {
	Schema         string      `json:"schema"`
	PreparedSHA256 string      `json:"prepared_sha256"`
	Files          []Pin       `json:"files"`
	Insertions     []Insertion `json:"insertions"`
}

func shaText(s string) string { v := sha256.Sum256([]byte(s)); return hex.EncodeToString(v[:]) }

// Generate adds observer-only statements to verified immutable copies. It
// verifies whole files before checking unique anchors in bounded functions.
// The original stock checkout and qualified build tree are never written.
func Generate(preparedPath, preparedSHA, out string) (Generated, error) {
	m := Generated{Schema: "nanopore-supgpu-observer-source/v1", PreparedSHA256: preparedSHA, Files: []Pin{}, Insertions: []Insertion{}}
	p, err := ReadPrepared(preparedPath, preparedSHA)
	if err != nil {
		return m, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return m, err
	}
	write := func(name, body string) error {
		path := filepath.Join(out, name)
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			return err
		}
		v, err := pin(path)
		if err == nil {
			m.Files = append(m.Files, v)
		}
		return err
	}
	if err = write("supgpu_observer.h", observerHeader); err != nil {
		return m, err
	}
	if err = write("graph_oracle.cpp", graphOracle); err != nil {
		return m, err
	}
	if err = write("loader_audit.c", loaderAudit); err != nil {
		return m, err
	}
	for index, expected := range sourcePins[:3] {
		raw, e := os.ReadFile(p.Sources[index].Path)
		if e != nil {
			return m, e
		}
		body := string(raw)
		if shaText(body) != expected.sha {
			return m, fmt.Errorf("source changed before generation")
		}
		changes := [][2]string{}
		replace := func(begin, end, anchor, extra string) error {
			a := strings.Index(body, begin)
			if a < 0 {
				return fmt.Errorf("missing bounded function %s", begin)
			}
			z := len(body)
			if end != "" {
				q := strings.Index(body[a+len(begin):], end)
				if q < 0 {
					return fmt.Errorf("missing function end %s", end)
				}
				z = a + len(begin) + q
			}
			block := body[a:z]
			if strings.Count(block, anchor) != 1 {
				return fmt.Errorf("anchor occurrence count is not one in %s", begin)
			}
			original := string(raw)
			oa := strings.Index(original, begin)
			oz := len(original)
			if end != "" {
				oz = oa + len(begin) + strings.Index(original[oa+len(begin):], end)
			}
			m.Insertions = append(m.Insertions, Insertion{expected.path, shaText(original[oa:oz]), shaText(anchor), 1})
			body = body[:a] + strings.Replace(block, anchor, extra, 1) + body[z:]
			changes = append(changes, [2]string{anchor, extra})
			return nil
		}
		switch index {
		case 0:
			begin, end := "void TxEncoderImpl::koi_forward(", "void TxEncoderImpl::koi_volta_forward("
			cache := "        this->remove_bits();"
			if err = replace(begin, end, cache, "        supgpu::cache(this, false);\n"+cache+"\n        supgpu::cache(this, true);"); err != nil {
				return m, err
			}
			if err = replace(begin, end, "    int res = KOI_SUCCESS;", `    supgpu::encoder(this, "enter", {{"residual", x_f16, {'N','T','C','t','c'}}, {"int8", x, {'N','T','C','t','c'}, 'C'}, {"scale", scaled_tensor.scale}, {"counter", ctr}});
    int res = KOI_SUCCESS;`); err != nil {
				return m, err
			}
			points := []struct{ anchor, stage, args string }{
				{"                             ctr[0].data_ptr<int>());", "qkv", `{{"qkv", qkv, {'N','T',0,'H','D','t','d'}}, {"counter",ctr}}`},
				{"            res = koi_masked_attention(stream, win_upper, win_lower, &out_qkv, &out_attn);\n        }", "attention", `{{"attention",t_out_attn,{'N','T','H','D','t','d'}}}`},
				{"                         ctr[1].data_ptr<int>());", "projection", `{{"projection",t_out_proj,{'N','T','C','t','c'}},{"counter",ctr}}`},
				{"                                   use_f8 ? &out_rms_f8 : nullptr, use_f8 ? nullptr : in_i8_ptr);", "norm1", `{{"residual",x_f16,{'N','T','C','t','c'}},{"rms_fp8",t_rms_out,{'N','T','C','t','c'}}}`},
				{"                            use_f32_accum, use_uswiglu);", "swiglu", `{{"swiglu",t_fc1_out,{'M','N','m','n'}},{"counter",ctr}}`},
				{"                         ctr[3].data_ptr<int>());", "fc2", `{{"fc2",t_fc2_out,{'N','T','C','t','c'}},{"counter",ctr}}`},
				{"                                   nullptr, in_i8_ptr);", "norm2", `{{"residual",x_f16,{'N','T','C','t','c'}},{"int8",x,{'N','T','C','t','c'},'C'},{"scale",scaled_tensor.scale}}`},
			}
			for _, v := range points {
				descriptors := map[string]string{
					"qkv":        `{{"input",&in},{"weights",&weights_qkv},{"sincos",&sincos},{"output",&out_qkv}}`,
					"attention":  `{{"input",&out_qkv},{"output",&out_attn}}`,
					"projection": `{{"input",&out_attn_mk},{"weight",&proj_w},{"bias",&proj_b},{"output",&out_proj_mn}}`,
					"norm1":      `{{"input",&out_proj_ntc},{"residual",&in_f16},{"weight",&res_weights},{"output",&in_f16},{"fp8",&out_rms_f8}}`,
					"swiglu":     `{{"input",&in_mk},{"weight",&fc1_wts},{"output",&fc1_out}}`,
					"fc2":        `{{"input",&fc1_out_mk},{"weight",&fc2_wts},{"output",&fc2_out_mn}}`,
					"norm2":      `{{"input",&fc2_out_ntc},{"residual",&in_f16},{"weight",&res2_weights},{"output",&in_f16},{"int8",&in}}`,
				}[v.stage]
				if err = replace(begin, end, v.anchor, v.anchor+"\n        supgpu::check(res);\n        supgpu::encoder(this, \""+v.stage+"\", "+v.args+", "+descriptors+");"); err != nil {
					return m, err
				}
			}
			stackBegin, stackEnd := "at::Tensor TxEncoderStackImpl::forward(", "#if !DORADO_ORIN && (CUDA_VERSION / 1000) < 13"
			anchor := "        for (auto &layer : layer_vec) {\n            layer->koi_forward"
			if err = replace(stackBegin, stackEnd, anchor, `        supgpu::emit("encoder.input", -1, {{"half",tiled_f16,{'N','T','C','t','c'}},{"int8",scaled_tensor.t,{'N','T','C','t','c'}},{"scale",scaled_tensor.scale}});
`+anchor); err != nil {
				return m, err
			}
			anchor = "        return untiled_f16;\n    }"
			if err = replace(stackBegin, stackEnd, anchor, "        supgpu::emit(\"encoder.output\", -1, {{\"output\",untiled_f16}});\n"+anchor); err != nil {
				return m, err
			}
		case 1:
			begin, end := "at::Tensor ConvStackImpl::forward(at::Tensor x)", "ConvStackImpl::ConvLayer::ConvLayer("
			if err = replace(begin, end, "    for (auto &layer : layers) {", "    int supgpu_conv_index = 0;\n    for (auto &layer : layers) {"); err != nil {
				return m, err
			}
			anchor := "        }\n    }\n    // Output is [N, T_out, C_out]"
			if err = replace(begin, end, anchor, "        }\n        supgpu::emit(\"conv\" + std::to_string(supgpu_conv_index++), -1, {{\"output\",x}});\n    }\n    // Output is [N, T_out, C_out]"); err != nil {
				return m, err
			}
		case 2:
			begin, end := "at::Tensor TxModelImpl::forward(", "}  // namespace dorado::basecall::model"
			a := strings.Index(string(raw), begin)
			z := strings.Index(string(raw)[a:], end) + a
			block := string(raw)[a:z]
			first, last := strings.Index(block, "{"), strings.LastIndex(block, "\n}")
			if first < 0 || last <= first {
				return m, fmt.Errorf("cannot extract reference module sequence")
			}
			sequence := block[first+1 : last]
			previous := -1
			for _, v := range []struct{ call, name, value string }{{"h = convs->forward(input);", "conv4", "h.transpose(1, 2)"}, {"h = tx_encoder(h, aux);", "encoder.output", "h"}, {"h = tx_decoder(h);", "upsample", "h"}, {"h = crf(h);", "scores", "h"}} {
				pos := strings.Index(sequence, v.call)
				if strings.Count(sequence, v.call) != 1 || pos <= previous {
					return m, fmt.Errorf("reference stock call order changed")
				}
				previous = pos
				sequence = strings.Replace(sequence, v.call, v.call+"\n        supgpu::reference_boundary(\""+v.name+"\", "+v.value+");", 1)
			}
			wrapper := "static at::Tensor reference_sequence(dorado::basecall::model::TxModelImpl &model, const at::Tensor &input) {\n    using namespace dorado;\n    auto convs=model.convs;auto tx_encoder=model.tx_encoder;auto tx_decoder=model.tx_decoder;auto crf=model.crf;\n    dorado::nn::AuxiliaryData *aux=nullptr;\n" + sequence + "\n}\n"
			if err = write("reference_sequence.inc", wrapper); err != nil {
				return m, err
			}
			if err = replace(begin, end, "    at::Tensor h;", "    supgpu::emit(\"input\", -1, {{\"input\",input}});\n    at::Tensor h;"); err != nil {
				return m, err
			}
			if err = replace(begin, end, "        h = tx_decoder(h);", "        h = tx_decoder(h);\n        supgpu::emit(\"upsample\", -1, {{\"output\",h}});"); err != nil {
				return m, err
			}
			if err = replace(begin, end, "        h = crf(h);", "        h = crf(h);\n        supgpu::emit(\"crf.weight\", -1, {{\"weight\",crf->linear->weight}});\n        supgpu::emit(\"scores\", -1, {{\"output\",h}});"); err != nil {
				return m, err
			}
		}
		restored := body
		for i := len(changes) - 1; i >= 0; i-- {
			v := changes[i]
			if strings.Count(restored, v[1]) != 1 {
				return m, fmt.Errorf("observer transformation cannot be reversed exactly")
			}
			restored = strings.Replace(restored, v[1], v[0], 1)
		}
		if restored != string(raw) {
			return m, fmt.Errorf("observer changed source outside declared insertions")
		}
		if err = write(filepath.Base(expected.path), "#include \"supgpu_observer.h\"\n"+body); err != nil {
			return m, err
		}
	}
	if err = writeJSON(filepath.Join(out, "generated.json"), m); err != nil {
		return m, err
	}
	return m, nil
}
