package h06

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const FLSTMSourceSHA = "97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160"

type Insertion struct{ Name, Anchor, Text string }
type Generated struct {
	Schema, Source, SourceSHA, Observed, ObservedSHA, Header, HeaderSHA, HeaderSource string
	Insertions                                                                        []Insertion
	Reversible                                                                        bool
}

func digest(body []byte) string { h := sha256.Sum256(body); return hex.EncodeToString(h[:]) }

func observedSource(original []byte) ([]byte, []Insertion, error) {
	const method = "void FLSTMStackImpl::forward_cublas(WorkingMemory &wm) {"
	if bytes.Count(original, []byte(method)) != 1 {
		return nil, nil, fmt.Errorf("unique source half method required")
	}
	start := bytes.Index(original, []byte(method))
	prefix, body := string(original[:start]), string(original[start:])
	patches := []Insertion{
		{"cache", "        }\n\n        for (int t = 0; t < wm.T; ++t) {", "        }\n\n        h06_observer::caches(layer, device_dn_weights_ih_[layer], device_dn_weights_hh_[layer], device_up_weights_[layer], device_up_bias_[layer]);\n\n        for (int t = 0; t < wm.T; ++t) {"},
		{"before", "            // gate calculation\n", "            h06_observer::before_step(layer, t, t_i, t_h, t_o, up_bfr, state_bfr);\n\n            // gate calculation\n"},
		{"after", "                               up_bfr.data_ptr(), state_bfr.data_ptr(), inout[t_o].data_ptr());", "                               up_bfr.data_ptr(), state_bfr.data_ptr(), inout[t_o].data_ptr());\n            h06_observer::after_step(layer, t, t_i, t_h, t_o, dn_bfr, up_bfr, state_bfr, inout[t_o]);"},
		{"layer", "        wm.is_input_to_rev_lstm = !reverse;", "        h06_observer::layer_output(layer, inout);\n        wm.is_input_to_rev_lstm = !reverse;"},
	}
	for _, p := range patches {
		if strings.Count(body, p.Anchor) != 1 {
			return nil, nil, fmt.Errorf("source anchor %s is absent or ambiguous", p.Name)
		}
		body = strings.Replace(body, p.Anchor, p.Text, 1)
	}
	header := Insertion{"include", "#include <tuple>\n", "#include <tuple>\n#include \"h06_observer.h\"\n"}
	if strings.Count(prefix, header.Anchor) != 1 {
		return nil, nil, fmt.Errorf("unique header anchor required")
	}
	prefix = strings.Replace(prefix, header.Anchor, header.Text, 1)
	patches = append([]Insertion{header}, patches...)
	observed := []byte(prefix + body)
	restored := string(observed)
	for i := len(patches) - 1; i >= 0; i-- {
		p := patches[i]
		if strings.Count(restored, p.Text) != 1 {
			return nil, nil, fmt.Errorf("insertion %s cannot be reversed", p.Name)
		}
		restored = strings.Replace(restored, p.Text, p.Anchor, 1)
	}
	if !bytes.Equal([]byte(restored), original) {
		return nil, nil, fmt.Errorf("observer source reversal differs")
	}
	return observed, patches, nil
}

// GenerateObserver copies the pinned source, adding observation calls without
// replacing its arithmetic or control flow. It never executes a native tool.
func GenerateObserver(source, out string) (r Generated, err error) {
	if source, err = filepath.Abs(source); err != nil {
		return r, err
	}
	if out, err = filepath.Abs(out); err != nil {
		return r, err
	}
	original, err := os.ReadFile(source)
	if err != nil {
		return r, err
	}
	if digest(original) != FLSTMSourceSHA {
		return r, fmt.Errorf("pinned FLSTM source differs")
	}
	observed, patches, err := observedSource(original)
	if err != nil {
		return r, err
	}
	headerPath := "verification/hacgpu/h06/native/h06_observer.h"
	header, err := os.ReadFile(headerPath)
	if err != nil {
		return r, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return r, err
	}
	r = Generated{Schema: "nanopore-hac-half-observer/v1", Source: source, SourceSHA: digest(original), Observed: filepath.Join(out, "FLSTMStack-observed.cpp"), ObservedSHA: digest(observed), Header: filepath.Join(out, "h06_observer.h"), HeaderSHA: digest(header), HeaderSource: headerPath, Insertions: patches, Reversible: true}
	for name, data := range map[string][]byte{r.Observed: observed, r.Header: header} {
		if err = os.WriteFile(name, data, 0644); err != nil {
			return r, err
		}
	}
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return r, err
	}
	err = os.WriteFile(filepath.Join(out, "generated.json"), append(raw, '\n'), 0644)
	return r, err
}
