package supgpu

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runtimeTestFile(t *testing.T, dir, name string, body []byte) Pin {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRuntimeAnchorsSeparateOldProducerAndCurrentConsumer(t *testing.T) {
	c, save, source, _ := syntheticController(t)
	dir := t.TempDir()
	binary, err := os.ReadFile(c.Binary.Path)
	if err != nil {
		t.Fatal(err)
	}
	old := c
	old.Binary = runtimeTestFile(t, dir, "old-compiler-controller", binary)
	old.Commands = append(old.Commands[:0:0], c.Commands...)
	old.Commands[2].Args = []string{old.Commands[0].Args[0], "build", "-buildvcs=false", "-o", old.Binary.Path, "./verification/supgpu/cmd"}
	producer := save(old)
	// The old live source is superseded. Its original immutable snapshot remains
	// the producer evidence, while the new consumer records the current bytes.
	if err = os.WriteFile(source, []byte("new current consumer source"), 0600); err != nil {
		t.Fatal(err)
	}
	c.Snapshots = append(c.Snapshots[:0:0], c.Snapshots...)
	var before strings.Builder
	for i, s := range c.Snapshots {
		p, err := pin(s.Original)
		if err != nil {
			t.Fatal(err)
		}
		if s.Original == source {
			body, err := os.ReadFile(source)
			if err != nil {
				t.Fatal(err)
			}
			c.Snapshots[i].Copy = runtimeTestFile(t, dir, "new-source-copy", body)
		}
		fmt.Fprintf(&before, "%s  %s\n", p.SHA256, p.Path)
	}
	c.Before = runtimeTestFile(t, dir, "new-before.sha256", []byte(before.String()))
	consumer := save(c)
	prepared := runtimeTestFile(t, dir, "prepared", []byte("original prepared input"))
	build := runtimeTestFile(t, dir, "build", []byte("accepted compiler build"))
	b := Build{PreparedSHA256: prepared.SHA256}
	p := BuildPlan{Prepared: prepared, Controller: producer}
	r := RuntimePlan{Schema: runtimePlanSchema, Family: "fixed-sup-cuda13-dlopen/v1", Build: build, Prepared: prepared, Controller: producer, Consumer: consumer}
	if err = verifyRuntimeAnchors(r, b, p, build, true); err != nil {
		t.Fatalf("old producer/current consumer rejected: %v", err)
	}
	if _, err = readController(producer, true); err == nil {
		t.Fatal("old producer was incorrectly considered current")
	}
	for _, field := range []string{"build", "prepared", "producer"} {
		bad := r
		switch field {
		case "build":
			bad.Build = prepared
		case "prepared":
			bad.Prepared = build
		case "producer":
			bad.Controller = consumer
		}
		if err = verifyRuntimeAnchors(bad, b, p, build, true); err == nil {
			t.Fatalf("wrong %s anchor accepted", field)
		}
	}
	alternate := c
	alternate.Binary = runtimeTestFile(t, dir, "alternate-consumer", binary)
	alternate.Commands = append(alternate.Commands[:0:0], c.Commands...)
	alternate.Commands[2].Args = []string{c.Commands[0].Args[0], "build", "-buildvcs=false", "-o", alternate.Binary.Path, "./verification/supgpu/cmd"}
	r.Consumer = save(alternate)
	if err = verifyRuntimeAnchors(r, b, p, build, true); err == nil || !strings.Contains(err.Error(), "executing controller binary differs") {
		t.Fatalf("wrong current consumer binary accepted: %v", err)
	}
	if err = verifyRuntimeAnchors(r, b, p, build, false); err != nil {
		t.Fatalf("historical import required importer=producer: %v", err)
	}
	r.Consumer = consumer
	if err = os.WriteFile(source, []byte("consumer source drift"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = verifyRuntimeAnchors(r, b, p, build, true); err == nil || !strings.Contains(err.Error(), "source/tool drift") {
		t.Fatalf("current consumer source drift accepted: %v", err)
	}
	if err = verifyRuntimeAnchors(r, b, p, build, false); err != nil {
		t.Fatalf("historical consumer snapshots could not be imported: %v", err)
	}
	if err = os.WriteFile(old.Binary.Path, []byte("changed old producer"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = verifyRuntimeAnchors(r, b, p, build, false); err == nil {
		t.Fatal("changed old producer binary accepted by historical import")
	}
}

func TestProfileRuntimeSeedSeparatesOnlyCanonicalLoaderCache(t *testing.T) {
	dir := t.TempDir()
	cache := runtimeTestFile(t, dir, "ld.so.cache", []byte("glibc metadata, not ELF"))
	alias := filepath.Join(dir, "cache-alias.so")
	if err := os.Symlink(cache.Path, alias); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{cache.Path, alias} {
		p, code, err := profileRuntimeSeed(path, cache.SHA256, cache)
		if err != nil || code || p != cache {
			t.Fatalf("declared loader metadata was not separated: %+v %t %v", p, code, err)
		}
	}
	if _, _, err := profileRuntimeSeed(cache.Path, strings.Repeat("0", 64), cache); err == nil {
		t.Fatal("loader cache omitted its retained-profile SHA check")
	}
	code := metadataELF(t, filepath.Join(dir, "libcode.so"), "", "")
	if p, elf, err := profileRuntimeSeed(code.Path, code.SHA256, cache); err != nil || !elf || p != code {
		t.Fatalf("valid inert ELF rejected: %+v %t %v", p, elf, err)
	}
	other := runtimeTestFile(t, dir, "unrecognized.so", []byte("not ELF code"))
	if _, _, err := profileRuntimeSeed(other.Path, other.SHA256, cache); err == nil || !strings.Contains(err.Error(), "code input is not ELF") {
		t.Fatalf("unrecognized non-ELF code was skipped: %v", err)
	}
	if err := os.WriteFile(cache.Path, []byte("changed loader cache"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := profileRuntimeSeed(cache.Path, cache.SHA256, cache); err == nil {
		t.Fatal("changed actual cache bytes accepted")
	}
	current, err := pin(cache.Path)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = profileRuntimeSeed(cache.Path, current.SHA256, cache); err == nil {
		t.Fatal("mid-plan loader cache change accepted with a replacement profile hash")
	}
}
