package h06

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const patchFixture = `#include <tuple>
namespace dorado::nn {
void FLSTMStackImpl::forward_cublas(WorkingMemory &wm) {
    for (int layer = 0; layer < 7; ++layer) {
        if (new_cache) {
            populate_cache();
        }

        for (int t = 0; t < wm.T; ++t) {
            const int t_i = reverse ? (wm.T - t) : (2 + t);
            const int t_o = t_i + (reverse ? 1 : -1);
            const int t_h = t_i + (reverse ? 2 : -2);
            utils::matmul_f16(input, down, gates);
            // gate calculation
            host_lstm_step_f16(stream, wm.N, C_, C_, true, device_up_bias_[layer].data_ptr(),
                               up_bfr.data_ptr(), state_bfr.data_ptr(), inout[t_o].data_ptr());
        }
        wm.is_input_to_rev_lstm = !reverse;
    }
}
}
`

func stripObservations(body string) string {
	var lines []string
	for _, line := range strings.Split(body, "\n") {
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "h06_observer::") || trim == `#include "h06_observer.h"` {
			continue
		}
		lines = append(lines, line)
	}
	return strings.Join(strings.Fields(strings.Join(lines, "\n")), " ")
}

func TestHalfObserverAddsOnlyReversibleObservationCalls(t *testing.T) {
	got, patches, err := observedSource([]byte(patchFixture))
	if err != nil {
		t.Fatal(err)
	}
	if len(patches) != 5 || stripObservations(string(got)) != stripObservations(patchFixture) {
		t.Fatal("observer changed original arithmetic/control tokens")
	}
	for _, call := range []string{"caches", "before_step", "after_step", "layer_output"} {
		if strings.Count(string(got), "h06_observer::"+call+"(") != 1 {
			t.Fatal("missing or repeated observation", call)
		}
	}
}

func TestHalfObserverRejectsMissingOrAmbiguousSourceAnchors(t *testing.T) {
	for _, body := range []string{"", strings.Replace(patchFixture, "// gate calculation", "// changed", 1), patchFixture + patchFixture, strings.Replace(patchFixture, "            // gate calculation\n", "            // gate calculation\n            // gate calculation\n", 1)} {
		if _, _, err := observedSource([]byte(body)); err == nil {
			t.Fatal("changed/ambiguous source admitted")
		}
	}
}

func TestHalfObserverRejectsUnpinnedSourceBeforeOutput(t *testing.T) {
	source := filepath.Join(t.TempDir(), "FLSTMStack.cpp")
	if err := os.WriteFile(source, []byte(patchFixture), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "not-created")
	if _, err := GenerateObserver(source, out); err == nil || !strings.Contains(err.Error(), "pinned FLSTM source differs") {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatal("rejected source created output", err)
	}
}
