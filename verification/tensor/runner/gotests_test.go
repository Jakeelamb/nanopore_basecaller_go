package runner

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNativeGoEvidenceRequiresRealTests(t *testing.T) {
	const pkg = "example/native"
	for _, tt := range []struct {
		name, raw string
		pass      bool
	}{
		{"no tests is not native success", `{"Action":"output","Package":"example/native","Output":"testing: warning: no tests to run"}
{"Action":"pass","Package":"example/native"}
`, false},
		{"real executed tests", `{"Action":"pass","Package":"example/native","Test":"TestNativeCUDA/one"}
{"Action":"pass","Package":"example/native","Test":"TestNativeCUDA"}
{"Action":"pass","Package":"example/native"}
`, true},
		{"missing subtest", `{"Action":"pass","Package":"example/native","Test":"TestNativeCUDA"}`, false},
		{"skip", `{"Action":"skip","Package":"example/native","Test":"TestNativeCUDA/one"}
{"Action":"pass","Package":"example/native","Test":"TestNativeCUDA"}`, false},
		{"failure", `{"Action":"fail","Package":"example/native","Test":"TestNativeCUDA"}`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "go.json")
			if err := os.WriteFile(path, []byte(tt.raw), 0600); err != nil {
				t.Fatal(err)
			}
			err := RequireGoTests(path, pkg, []string{"TestNativeCUDA", "TestNativeCUDA/one"})
			if (err == nil) != tt.pass {
				t.Fatal(err)
			}
		})
	}
}
