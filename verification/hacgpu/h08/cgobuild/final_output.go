package cgobuild

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"
)

const go1273LinuxAMD64LinkSHA = "97df419b7cf8f4bb8e76c3623b8a1d371c6818533c4be04f0a3563ff6169843d"

func fipsLinkOutput(args []string, driver Pin) (string, error) {
	output, buildid := "", "missing"
	counts := map[string]int{}
	positional := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "-") {
			if positional || i != len(args)-1 || !strings.HasSuffix(a, ".a") {
				return "", fmt.Errorf("unsupported FIPS linker positional argument")
			}
			positional = true
			continue
		}
		key, value, inline := strings.Cut(a, "=")
		switch key {
		case "-o", "-importcfg", "-buildid", "-X":
			if !inline {
				if i+1 == len(args) {
					return "", fmt.Errorf("missing FIPS linker flag value")
				}
				i++
				value = args[i]
			}
			counts[key]++
			if key == "-o" {
				output = value
			}
			if key == "-buildid" {
				buildid = value
			}
			if key != "-buildid" && key != "-X" && (counts[key] != 1 || value == "") {
				return "", fmt.Errorf("duplicate/empty FIPS linker flag")
			}
		case "-buildmode", "-linkmode", "-compressdwarf", "-extld":
			counts[key]++
			valid := selectedExternalLinker(value, driver)
			if key != "-extld" {
				valid = value == map[string]string{"-buildmode": "exe", "-linkmode": "external", "-compressdwarf": "false"}[key]
			}
			if !inline || !valid || counts[key] != 1 {
				return "", fmt.Errorf("unsupported FIPS linker profile: %s", a)
			}
		default:
			return "", fmt.Errorf("unsupported FIPS linker flag: %s", a)
		}
	}
	if !positional || output == "" || buildid != "" || counts["-buildmode"] != 1 || counts["-linkmode"] != 1 || counts["-extld"] != 1 || counts["-importcfg"] != 1 {
		return "", fmt.Errorf("FIPS linker requires executable external link and effective empty buildid")
	}
	return output, nil
}

func finalOutputBindings(p Plan, r Receipt, operations map[string]operation) error {
	_, e := finalOutputTransition(p, r, operations)
	return e
}

func finalOutputTransition(p Plan, r Receipt, operations map[string]operation) (fipsByteTransition, error) {
	reject := func(s string) (fipsByteTransition, error) {
		return fipsByteTransition{}, fmt.Errorf("final link binding: %s", s)
	}
	if r.Output.Path != outputPath(p) {
		return reject("final output differs from selected Go -o")
	}
	var native, goLink operation
	nd, gd := "", ""
	for dir, op := range operations {
		switch op.Role {
		case "go-link":
			if nd != "" {
				return reject("duplicate native go-link")
			}
			nd, native = dir, op
		case "go-tool-link":
			if gd != "" {
				return reject("duplicate go-tool-link")
			}
			gd, goLink = dir, op
		}
	}
	if nd == "" || gd == "" {
		return reject("exact native/Go link pair required")
	}
	var s Session
	if e := read(r.Session, &s); e != nil {
		return fipsByteTransition{}, e
	}
	if s.Schema != "nanopore-h08-cgo-session/v1" || s.Phase != "build" || s.Plan != r.Plan || s.Controller != r.Controller || !filepath.IsAbs(s.Directory) {
		return reject("selected build session differs")
	}
	for _, dir := range []string{nd, gd} {
		if !under(filepath.Join(s.Directory, "commands"), dir) {
			return reject("link outside selected session")
		}
	}
	l := goLink.Lease
	tool := Pin{Path: filepath.Join(p.Options.GoRoot, "pkg/tool/linux_amd64/link"), SHA256: go1273LinuxAMD64LinkSHA}
	if p.Environment["GOOS"] != "linux" || p.Environment["GOARCH"] != "amd64" || p.GoTools[tool.Path] != tool.SHA256 || l == nil || l.Tool != tool || l.Schema != "nanopore-h08-go-tool-lease/v1" || l.Kind != "link" || l.Package != selectionTarget(p) || l.Directory != goLink.Directory || native.Directory != goLink.Directory || !reflect.DeepEqual(native.Lease, l) {
		return reject("selected Go1.27.3 linker/lease differs")
	}
	if len(goLink.OriginalArgs) < 2 || goLink.OriginalArgs[0] != tool.Path || !slices.Equal(goLink.OriginalArgs[1:], l.Args) {
		return reject("Go linker lease argv differs")
	}
	output, e := fipsLinkOutput(l.Args, s.Driver)
	if e != nil {
		return fipsByteTransition{}, e
	}
	target := absArg(goLink.Directory, output)
	if !under(p.Options.WorkRoot, target) || goLink.Outputs[target] != goLink.Output.Path || native.Output.Path != goLink.Output.Path || native.Request == nil || !native.Request.Link || native.Request.Role != "go-link" || native.Request.OutputTarget != native.Output.Path || absArg(native.Directory, native.Request.Output) != target || !slices.Equal(native.OriginalArgs, native.Request.Args) {
		return reject("native/Go output target differs")
	}
	if goLink.Output.SHA256 != r.Output.SHA256 {
		return reject("final output bytes differ from successful Go link")
	}
	for _, op := range []operation{native, goLink} {
		if op.Schema != "nanopore-h08-cgo-operation/v1" || op.Status != "accepted" || op.Error != "" || len(op.Commands) != 1 || !slices.Equal(op.Expected, []int{0}) || op.Output.SHA256 == "" || op.OutputSnapshot.SHA256 != op.Output.SHA256 || op.Inputs[tool.Path] != tool.SHA256 || op.Snapshots[tool.Path].SHA256 != tool.SHA256 {
			return reject("successful link command/snapshot omitted")
		}
		c := op.Commands[0]
		if e := checkCommand(c.Args, []int{0}, c.Started, c.Finished, c.Stdout, c.StdoutSHA, c.Stderr, c.StderrSHA, c.Exit); e != nil {
			return fipsByteTransition{}, e
		}
		bp, e := closedPin(filepath.Join(filepath.Dir(c.Stdout), "before.json"))
		if e != nil || bp != op.Before {
			return reject("link before manifest differs")
		}
	}
	lp, e := closedPin(filepath.Join(gd, "lease.json"))
	if e != nil {
		return fipsByteTransition{}, e
	}
	var actual Lease
	if e = read(lp, &actual); e != nil || !reflect.DeepEqual(actual, *l) {
		return reject("Go link lease artifact differs")
	}
	env, e := toolEnvironment(p, s, r.Session, "link", false, l, goLink.ToolOverrides)
	if e != nil {
		return fipsByteTransition{}, e
	}
	env = append(env, "H08_CGO_LEASE="+lp.Path, "H08_CGO_LEASE_SHA="+lp.SHA256)
	if filepath.Dir(goLink.Commands[0].Stdout) != gd || filepath.Dir(native.Commands[0].Stdout) != filepath.Join(nd, "actual") {
		return reject("link command directory differs")
	}
	if e = beforeCommand(goLink.Commands[0], goLink.OriginalArgs, env, goLink.Inputs, goLink); e != nil {
		return fipsByteTransition{}, e
	}
	argv := append([]string{p.CC.Path}, native.Request.Args...)
	argv = append(argv, "-MD", "-MF", filepath.Join(nd, "actual.d"), "-Wl,-t", "-Wl,-Map="+filepath.Join(nd, "link.map"))
	if e = beforeCommand(native.Commands[0], argv, append(environment(p), "TERM=dumb"), native.Inputs, native); e != nil {
		return fipsByteTransition{}, e
	}
	gs, _ := time.Parse(time.RFC3339Nano, goLink.Commands[0].Started)
	gf, _ := time.Parse(time.RFC3339Nano, goLink.Commands[0].Finished)
	ns, _ := time.Parse(time.RFC3339Nano, native.Commands[0].Started)
	nf, _ := time.Parse(time.RFC3339Nano, native.Commands[0].Finished)
	if ns.Before(gs) || nf.After(gf) {
		return reject("native link outside Go linker clock")
	}
	for _, q := range []Pin{native.OutputSnapshot, goLink.OutputSnapshot, r.Output} {
		if e = verifyPin(q); e != nil {
			return fipsByteTransition{}, e
		}
	}
	pre, e := os.ReadFile(native.OutputSnapshot.Path)
	if e != nil {
		return fipsByteTransition{}, e
	}
	final, e := os.ReadFile(goLink.OutputSnapshot.Path)
	if e != nil {
		return fipsByteTransition{}, e
	}
	return verifyGo1273FIPSByteTransition(pre, final)
}
