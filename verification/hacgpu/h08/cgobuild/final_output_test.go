package cgobuild

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func finalLinkFixture(t *testing.T) (Plan, Receipt, map[string]operation) {
	t.Helper()
	root := t.TempDir()
	p := Plan{Role: H08Command, Options: Options{Root: root, GoRoot: filepath.Join(root, "go"), WorkRoot: filepath.Join(root, "work")}, Environment: map[string]string{"GOOS": "linux", "GOARCH": "amd64", "GOROOT": filepath.Join(root, "go"), "CGO_LDFLAGS": ""}}
	put := func(path string, b []byte) Pin {
		t.Helper()
		if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(path, b, 0600); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(path)
		if e != nil {
			t.Fatal(e)
		}
		return q
	}
	pre, final := smallFIPSELF(t)
	s := Session{Schema: "nanopore-h08-cgo-session/v1", Phase: "build", Directory: filepath.Join(root, "session"), Driver: put(filepath.Join(root, "driver"), []byte("driver"))}
	r := Receipt{Session: saveQueryFixture(t, filepath.Join(s.Directory, "session.json"), s), Output: put(outputPath(p), final)}
	tool := Pin{Path: filepath.Join(p.Options.GoRoot, "pkg/tool/linux_amd64/link"), SHA256: go1273LinuxAMD64LinkSHA}
	p.GoTools = map[string]string{tool.Path: tool.SHA256}
	target := filepath.Join(p.Options.WorkRoot, "exe/a.out")
	args := []string{"-o", target, "-importcfg", filepath.Join(p.Options.WorkRoot, "importcfg.link"), "-buildmode=exe", "-buildid=generated", "-linkmode=external", "-buildid=", "-extld=" + `"` + s.Driver.Path + `"`, filepath.Join(p.Options.WorkRoot, "_pkg_.a")}
	l := &Lease{Schema: "nanopore-h08-go-tool-lease/v1", Kind: "link", Package: selectionTarget(p), Directory: root, Tool: tool, Args: args}
	gd, nd := filepath.Join(s.Directory, "commands/go"), filepath.Join(s.Directory, "commands/native")
	lease := saveQueryFixture(t, filepath.Join(gd, "lease.json"), l)
	inputs := map[string]string{tool.Path: tool.SHA256}
	snap := map[string]Pin{tool.Path: {Path: filepath.Join(root, "unread-tool-metadata"), SHA256: tool.SHA256}}
	command := func(dir string, args, env []string) runner.CommandReceipt {
		out, err := put(filepath.Join(dir, "stdout"), nil), put(filepath.Join(dir, "stderr"), nil)
		c := runner.CommandReceipt{Args: args, Environment: env, Exit: 0, Started: "2026-10-26T01:00:00Z", Finished: "2026-10-26T01:00:10Z", Stdout: out.Path, StdoutSHA: out.SHA256, Stderr: err.Path, StderrSHA: err.SHA256}
		saveQueryFixture(t, filepath.Join(dir, "command.json"), c)
		return c
	}
	env, e := toolEnvironment(p, s, r.Session, "link", false, l, sourceToolOverrides(p, "link", false, l))
	if e != nil {
		t.Fatal(e)
	}
	env = append(env, "H08_CGO_LEASE="+lease.Path, "H08_CGO_LEASE_SHA="+lease.SHA256)
	goLink := operation{Schema: "nanopore-h08-cgo-operation/v1", Role: "go-tool-link", Status: "accepted", Directory: root, Lease: l, OriginalArgs: append([]string{tool.Path}, args...), Inputs: maps.Clone(inputs), Snapshots: maps.Clone(snap), Expected: []int{0}, Output: Pin{Path: target, SHA256: r.Output.SHA256}, OutputSnapshot: put(filepath.Join(root, "snap/final"), final), Outputs: map[string]string{target: target}, ToolOverrides: sourceToolOverrides(p, "link", false, l)}
	goLink.Commands = []runner.CommandReceipt{command(gd, goLink.OriginalArgs, env)}
	goLink.Before = saveQueryFixture(t, filepath.Join(gd, "before.json"), inputs)
	nArgs := []string{"-m64", "-o", target, filepath.Join(p.Options.WorkRoot, "go.o")}
	native := operation{Schema: goLink.Schema, Role: "go-link", Status: "accepted", Directory: root, Lease: l, OriginalArgs: nArgs, Inputs: maps.Clone(inputs), Snapshots: maps.Clone(snap), Expected: []int{0}, OutputSnapshot: put(filepath.Join(root, "snap/native"), pre)}
	native.Output = Pin{Path: target, SHA256: native.OutputSnapshot.SHA256}
	native.Request = &ccRequest{Role: "go-link", Link: true, Args: nArgs, Output: target, OutputTarget: target}
	argv := append([]string{p.CC.Path}, nArgs...)
	argv = append(argv, "-MD", "-MF", filepath.Join(nd, "actual.d"), "-Wl,-t", "-Wl,-Map="+filepath.Join(nd, "link.map"))
	native.Commands = []runner.CommandReceipt{command(filepath.Join(nd, "actual"), argv, append(environment(p), "TERM=dumb"))}
	native.Before = saveQueryFixture(t, filepath.Join(nd, "actual/before.json"), inputs)
	return p, r, map[string]operation{gd: goLink, nd: native}
}

func TestFinalOutputLinkCustody(t *testing.T) {
	for _, test := range []struct{ name, want string }{
		{"exact", ""}, {"missing native", "pair required"}, {"duplicate native", "duplicate native"}, {"duplicate Go", "duplicate go-tool"},
		{"wrong tool", "linker/lease differs"}, {"wrong lease", "linker/lease differs"}, {"wrong argv", "lease argv differs"},
		{"wrong command", "command/environment differs"}, {"wrong environment", "command/environment differs"}, {"wrong native command", "command/environment differs"},
		{"wrong exit", "command argv/clocks/exit"}, {"wrong output", "output target differs"}, {"wrong final", "final output bytes differ"},
		{"wrong snapshot", "successful link command/snapshot"}, {"wrong before", "before manifest differs"}, {"missing snapshot", "successful link command/snapshot"},
		{"changed preimage", "final checksum mismatch"}, {"changed final", "other final bytes changed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, r, ops := finalLinkFixture(t)
			var gd, nd string
			for dir, op := range ops {
				if op.Role == "go-tool-link" {
					gd = dir
				} else {
					nd = dir
				}
			}
			g, n := ops[gd], ops[nd]
			switch test.name {
			case "missing native":
				delete(ops, nd)
			case "duplicate native":
				ops[nd+"2"] = n
			case "duplicate Go":
				ops[gd+"2"] = g
			case "wrong tool":
				p.GoTools[g.Lease.Tool.Path] = "other"
			case "wrong lease":
				l := *n.Lease
				l.Package = "other"
				n.Lease = &l
			case "wrong argv":
				g.OriginalArgs = slices.Clone(g.OriginalArgs)
				g.OriginalArgs[1] = "-o=other"
			case "wrong command":
				g.Commands[0].Args = []string{"other-linker"}
			case "wrong environment":
				g.Commands[0].Environment = append(g.Commands[0].Environment, "GOARCH=arm64")
			case "wrong native command":
				n.Commands[0].Args = []string{"other-cc"}
			case "wrong exit":
				g.Commands[0].Exit = 1
			case "wrong output":
				n.Output.Path += "other"
			case "wrong final":
				r.Output.SHA256 = digest([]byte("other"))
			case "wrong snapshot":
				n.OutputSnapshot = g.OutputSnapshot
			case "wrong before":
				g.Before.SHA256 = digest(nil)
			case "missing snapshot":
				delete(g.Snapshots, g.Lease.Tool.Path)
			case "changed preimage", "changed final":
				op := &n
				offset := 0x300
				if test.name == "changed final" {
					op = &g
					offset = 0x350
				}
				b, e := os.ReadFile(op.OutputSnapshot.Path)
				if e != nil {
					t.Fatal(e)
				}
				b[offset] ^= 1
				if e = os.WriteFile(op.OutputSnapshot.Path, b, 0600); e != nil {
					t.Fatal(e)
				}
				q, e := closedPin(op.OutputSnapshot.Path)
				if e != nil {
					t.Fatal(e)
				}
				op.OutputSnapshot = q
				op.Output.SHA256 = q.SHA256
				if op == &g {
					r.Output.SHA256 = q.SHA256
					if e = os.WriteFile(r.Output.Path, b, 0600); e != nil {
						t.Fatal(e)
					}
				}
			}
			for _, op := range []operation{g, n} {
				for _, c := range op.Commands {
					saveQueryFixture(t, filepath.Join(filepath.Dir(c.Stdout), "command.json"), c)
				}
			}
			if test.name != "missing native" {
				ops[nd] = n
			}
			ops[gd] = g
			e := finalOutputBindings(p, r, ops)
			if test.want == "" {
				if e != nil {
					t.Fatal(e)
				}
			} else if e == nil || !strings.Contains(e.Error(), test.want) {
				t.Fatalf("want %q, got %v", test.want, e)
			}
		})
	}
}

func TestFIPSLinkArgvProfile(t *testing.T) {
	for _, test := range []struct {
		name  string
		args  []string
		valid bool
	}{
		{"empty effective buildid", nil, true},
		{"separate empty buildid", []string{"-buildid", ""}, true},
		{"nonempty buildid", []string{"-buildid=other"}, false},
		{"B separate", []string{"-B", "gobuildid"}, false}, {"B inline", []string{"-B=gobuildid"}, false}, {"B attached", []string{"-Bgobuildid"}, false}, {"B double dash", []string{"--B=gobuildid"}, false},
		{"fipso separate", []string{"-fipso", "other"}, false}, {"fipso inline", []string{"-fipso=other"}, false}, {"fipso double dash", []string{"--fipso=other"}, false},
		{"buildmode prefix", []string{"-buildmode=exetra"}, false}, {"linkmode prefix", []string{"-linkmode=externalother"}, false},
		{"buildid double dash", []string{"--buildid="}, false}, {"unknown", []string{"-unknown"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, r, ops := finalLinkFixture(t)
			for dir, g := range ops {
				if g.Role != "go-tool-link" {
					continue
				}
				args := slices.Clone(g.Lease.Args)
				args = append(args[:len(args)-1], append(test.args, g.Lease.Args[len(g.Lease.Args)-1])...)
				l := *g.Lease
				l.Args = args
				g.Lease = &l
				g.OriginalArgs = append([]string{l.Tool.Path}, args...)
				ops[dir] = g
				for nd, n := range ops {
					if n.Role == "go-link" {
						n.Lease = &l
						ops[nd] = n
					}
				}
				var s Session
				if e := read(r.Session, &s); e != nil {
					t.Fatal(e)
				}
				_, e := fipsLinkOutput(args, s.Driver)
				if test.valid && e != nil {
					t.Fatal(e)
				}
				if !test.valid && e == nil {
					t.Fatal("invalid argv accepted")
				}
				if !test.valid && finalOutputBindings(p, r, ops) == nil {
					t.Fatal("common body accepted invalid profile")
				}
				break
			}
		})
	}
}

func TestFIPSLinkSelectedExternalDriver(t *testing.T) {
	driver := Pin{Path: "/selected/driver"}
	base := []string{"-o", "/private/a.out", "-importcfg", "/private/importcfg.link", "-buildmode=exe", "-linkmode=external", "-buildid=", "-extld=", "/private/main.a"}
	for _, test := range []struct {
		name, value string
		valid       bool
	}{
		{"bare", driver.Path, true}, {"generated quoted", `"` + driver.Path + `"`, true},
		{"foreign", "/foreign/driver", false}, {"quoted foreign", `"/foreign/driver"`, false},
		{"extra argument", `"` + driver.Path + `" -foreign`, false},
		{"extra command", `"` + driver.Path + `";/foreign/driver`, false},
		{"malformed", `"` + driver.Path, false}, {"mismatched", `"` + driver.Path + `'`, false},
		{"unsupported escape", `\"` + driver.Path + `\"`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			args := slices.Clone(base)
			args[7] += test.value
			out, e := fipsLinkOutput(args, driver)
			if test.valid {
				if e != nil || out != "/private/a.out" {
					t.Fatalf("selected driver rejected: %s %v", out, e)
				}
			} else if e == nil || !strings.Contains(e.Error(), "unsupported FIPS linker profile: -extld=") {
				t.Fatalf("external driver missed profile gate: %v", e)
			}
		})
	}
	for _, flag := range []string{"-extld", "--extld=" + driver.Path, "-extld=" + driver.Path} {
		args := append(slices.Clone(base[:len(base)-1]), flag, base[len(base)-1])
		args[7] += driver.Path
		if _, e := fipsLinkOutput(args, driver); e == nil {
			t.Fatalf("separate/double-dash/duplicate driver flag accepted: %v", args)
		}
	}
}
