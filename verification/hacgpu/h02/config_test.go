package h02

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
)

// pinnedModelDir locates the pinned HAC model. NANOPORE_MODEL_DIR wins; the
// fallback is the layout the other model tests use, derived from DORADO_SOURCE.
// A missing model skips, because the model is data, not source.
func pinnedModelDir(t *testing.T) string {
	t.Helper()
	const name = "dna_r10.4.1_e8.2_400bps_hac@v6.0.0"
	candidates := []string{}
	if dir := os.Getenv("NANOPORE_MODEL_DIR"); dir != "" {
		candidates = append(candidates, dir)
	}
	if source := os.Getenv("DORADO_SOURCE"); source != "" {
		if resolved, err := filepath.EvalSymlinks(source); err == nil {
			source = resolved
		}
		candidates = append(candidates, filepath.Join(filepath.Dir(filepath.Dir(source)), "models", name))
	}
	for _, dir := range candidates {
		if _, err := os.Stat(filepath.Join(dir, "config.toml")); err == nil {
			return dir
		}
	}
	t.Skip("pinned HAC model is not present; set NANOPORE_MODEL_DIR")
	return ""
}

// TestModelConvsReadsThePinnedStack pins the declared sublayers of the pinned
// config. These values are the config's, not a copy of the plan's expectations.
func TestModelConvsReadsThePinnedStack(t *testing.T) {
	convs, err := ModelConvs(pinnedModelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	want := []ConvSpec{
		{Index: 0, CIn: 1, COut: 16, WinLen: 5, Stride: 1, Padding: 2, Activation: "swish"},
		{Index: 1, CIn: 16, COut: 16, WinLen: 5, Stride: 1, Padding: 2, Activation: "swish"},
		{Index: 2, CIn: 16, COut: 1024, WinLen: 19, Stride: 6, Padding: 9, Activation: "tanh"},
	}
	if len(convs) != len(want) {
		t.Fatalf("%d convolutions, want %d", len(convs), len(want))
	}
	for i := range want {
		if convs[i] != want[i] {
			t.Fatalf("convolution %d = %+v, want %+v", i, convs[i], want[i])
		}
	}
}

// TestConvParamsLayouts pins the layout rule: earlier layers keep the NTC
// default and only the last layer is decided by the pinned Koi rule.
func TestConvParamsLayouts(t *testing.T) {
	convs, err := ModelConvs(pinnedModelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	params, err := convParams(convs)
	if err != nil {
		t.Fatal(err)
	}
	if len(params) != 3 || params[0].OutputLayout != model.GPUStorageNTC ||
		params[1].OutputLayout != model.GPUStorageNTC || params[2].OutputLayout != model.GPUStorageCUTLASSI8 {
		t.Fatalf("layouts %v %v %v", params[0].OutputLayout, params[1].OutputLayout, params[2].OutputLayout)
	}
	plan, err := model.PlanHACSmallConvs(64, 192, 1, 7782400, params)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.Layers[1].Params.OutputTPadding; got != 9 {
		t.Fatalf("conv1 output padding %d, want the last window/2 = 9", got)
	}
}

func TestLastConvLayoutRule(t *testing.T) {
	cases := []struct {
		spec ConvSpec
		want model.GPUStorageLayout
		ok   bool
	}{
		{ConvSpec{COut: 96, Activation: "swish"}, model.GPUStorageNTC, true},
		{ConvSpec{COut: 128, Activation: "swish"}, model.GPUStorageNTC, true},
		{ConvSpec{COut: 1024, Activation: "tanh"}, model.GPUStorageCUTLASSI8, true},
		{ConvSpec{COut: 1024, Activation: "swish"}, model.GPUStorageCUTLASSF16, true},
		{ConvSpec{COut: 256, Activation: "tanh"}, model.GPUStorageCUTLASSI8, true},
		{ConvSpec{COut: 64, Activation: "swish"}, 0, false},
		{ConvSpec{COut: 1152, Activation: "tanh"}, 0, false},
		{ConvSpec{COut: 200, Activation: "tanh"}, 0, false},
	}
	for _, tt := range cases {
		got, err := lastConvLayout(tt.spec)
		if tt.ok != (err == nil) {
			t.Fatalf("%+v: err = %v, want ok %v", tt.spec, err, tt.ok)
		}
		if tt.ok && got != tt.want {
			t.Fatalf("%+v: layout %v, want %v", tt.spec, got, tt.want)
		}
	}
}

// TestModelConvsRejections pins the config guards: a declared padding the source
// would not pass, an unknown activation, a missing key and a broken channel
// chain are all hard rejections.
func TestModelConvsRejections(t *testing.T) {
	pinned, err := os.ReadFile(filepath.Join(pinnedModelDir(t), "config.toml"))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct{ name, body, want string }{
		{"padding", strings.Replace(string(pinned), "padding = 2", "padding = 3", 1), "padding 3"},
		{"activation", strings.Replace(string(pinned), `activation = "swish"`, `activation = "relu"`, 1), "unrecognised convolution activation"},
		{"missing key", strings.Replace(string(pinned), "stride = 6", "", 1), "has no \"stride\""},
		{"channel chain", strings.Replace(string(pinned), "insize = 16", "insize = 15", 1), "does not follow"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte(tt.body), 0o600); err != nil {
				t.Fatal(err)
			}
			_, err := ModelConvs(dir)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
		})
	}
	if _, err := convActivation("softmax"); err == nil {
		t.Fatal("unknown activation accepted")
	}
}
