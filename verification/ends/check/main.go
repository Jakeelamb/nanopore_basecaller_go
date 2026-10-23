package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/align"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/ends"
	endsoracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/ends"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func sha(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }
func saveJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}
func token(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

type detectorCase struct {
	Name, Sequence, Kit string
	Options             ends.Options
}
type alignCase struct {
	Query, Target string
	Equality      align.Equality
}

func run() error {
	root := flag.String("source", "external/dorado-source", "pinned Dorado checkout")
	edlib := flag.String("edlib", endsoracle.DefaultEdlibRoot, "pinned edlib checkout")
	out := flag.String("output", "", "new retained attempt directory")
	buildOnly := flag.Bool("build-only", false, "build oracle without running comparisons")
	flag.Parse()
	if *out == "" {
		if err := os.MkdirAll("artifacts/ends", 0700); err != nil {
			return err
		}
		p, err := os.MkdirTemp("artifacts/ends", "attempt-"+time.Now().UTC().Format("20060102T150405Z")+"-")
		if err != nil {
			return err
		}
		*out = p
	} else if err := os.Mkdir(*out, 0700); err != nil {
		return err
	}
	absolute, err := filepath.Abs(*out)
	if err != nil {
		return err
	}
	*out = absolute
	fmt.Println(*out)
	if err := implementationPins(*out); err != nil {
		return err
	}
	executable, err := endsoracle.Build(*root, *edlib, filepath.Join(*out, "native"))
	if err != nil {
		return err
	}
	if *buildOnly {
		return nil
	}
	if err := custodyControls(*root, *edlib, *out); err != nil {
		return err
	}
	var cases []alignCase
	for _, alphabet := range []string{"AC", "AN"} {
		queries := words(alphabet, 1, 5)
		targets := words(alphabet, 0, 6)
		for _, q := range queries {
			for _, t := range targets {
				for _, eq := range []align.Equality{align.EqualityExact, align.EqualityNWildcard} {
					cases = append(cases, alignCase{q, t, eq})
				}
			}
		}
	}
	rng := rand.New(rand.NewSource(85391))
	for i := 0; i < 1200; i++ {
		q, t := randomDNA(rng, 1+rng.Intn(100)), randomDNA(rng, rng.Intn(151))
		cases = append(cases, alignCase{q, t, align.EqualityExact}, alignCase{q, t, align.EqualityNWildcard})
	}
	for _, qlen := range []int{1, 24, 27, 63, 64, 65, 99, 100, 127, 128, 129} {
		for _, tlen := range []int{0, 1, 24, 63, 64, 65, 75, 150} {
			for _, equality := range []align.Equality{align.EqualityExact, align.EqualityNWildcard} {
				cases = append(cases, alignCase{strings.Repeat("A", qlen), strings.Repeat("C", tlen), equality}, alignCase{strings.Repeat("A", qlen), strings.Repeat("A", tlen), equality})
			}
		}
	}
	var input bytes.Buffer
	for _, c := range cases {
		fmt.Fprintf(&input, "ALIGN %s %s %d\n", token(c.Query), token(c.Target), c.Equality)
	}
	lines, err := native(executable, *out, "alignment", input.Bytes())
	if err != nil {
		return err
	}
	if len(lines) != len(cases) {
		return fmt.Errorf("native alignment line count %d != %d", len(lines), len(cases))
	}
	latestMismatches, reverseMismatches := 0, 0
	var firstLatest any
	for i, c := range cases {
		var want align.Location
		if err := json.Unmarshal(lines[i], &want); err != nil {
			return err
		}
		got, err := align.HW(c.Query, c.Target, c.Equality)
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("alignment mismatch case %d %q %q eq %d: Go %+v native %+v", i, c.Query, c.Target, c.Equality, got, want)
		}
		latest := alternative(c.Query, c.Target, c.Equality, true)
		if latest != want {
			latestMismatches++
			if firstLatest == nil {
				firstLatest = map[string]any{"query": c.Query, "target": c.Target, "equality": c.Equality, "latest_start": latest, "native": want}
			}
		}
		if reverseShape(c.Query, c.Target, c.Equality) != want {
			reverseMismatches++
		}
	}
	if latestMismatches == 0 || reverseMismatches != 0 {
		return fmt.Errorf("tie probe did not discriminate: latest mismatches=%d reverse mismatches=%d", latestMismatches, reverseMismatches)
	}
	if err := saveJSON(filepath.Join(*out, "alignment-summary.json"), map[string]any{"cases": len(cases), "production_mismatches": 0, "reverse_shw_mismatches": reverseMismatches, "latest_start_mismatches": latestMismatches, "first_latest_start_counterexample": firstLatest}); err != nil {
		return err
	}
	fmt.Printf("alignment: %d native comparisons passed; latest-start rejected by %d cases; reverse-SHW agrees\n", len(cases), latestMismatches)
	lines, err = native(executable, *out, "candidates", []byte("CANDIDATES\n"))
	if err != nil {
		return err
	}
	wantCandidates := [][3]string{{"LSK110", adapterFront, adapterRear}, {"cDNA_FWD", primerFront, rc(primerRear)}, {"cDNA_REV", primerRear, rc(primerFront)}}
	var candidates [][3]string
	if err := json.Unmarshal(lines[0], &candidates); err != nil {
		return err
	}
	if !reflect.DeepEqual(candidates, wantCandidates) {
		return fmt.Errorf("native default candidate mismatch: %v", candidates)
	}
	detectors := detectorCases(rng)
	capturePins := map[string]string{}
	fixtureData, err := os.ReadFile("verification/ends/testdata/captures.json")
	if err != nil {
		return err
	}
	var fixtures []struct{ Path, ManifestSHA256, OutputSHA256 string }
	if err := json.Unmarshal(fixtureData, &fixtures); err != nil {
		return err
	}
	if len(fixtures) != 2 {
		return fmt.Errorf("expected retained HAC and SUP capture pins")
	}
	for _, fixture := range fixtures {
		path := fixture.Path
		manifestData, err := os.ReadFile(filepath.Join(path, "manifest.json"))
		if err != nil {
			return err
		}
		if sha(manifestData) != fixture.ManifestSHA256 {
			return fmt.Errorf("capture manifest hash mismatch %s", path)
		}
		capturePins[filepath.Join(path, "manifest.json")] = sha(manifestData)
		var manifest struct{ Output struct{ File, SHA256 string } }
		if err := json.Unmarshal(manifestData, &manifest); err != nil {
			return err
		}
		if manifest.Output.File != "output.json" || manifest.Output.SHA256 != fixture.OutputSHA256 {
			return fmt.Errorf("capture manifest lacks output pin %s", path)
		}
		data, err := os.ReadFile(filepath.Join(path, "output.json"))
		if err != nil {
			return err
		}
		if sha(data) != manifest.Output.SHA256 {
			return fmt.Errorf("capture hash mismatch %s", path)
		}
		capturePins[filepath.Join(path, "output.json")] = sha(data)
		var output struct{ Sequence string }
		if err := json.Unmarshal(data, &output); err != nil {
			return err
		}
		if output.Sequence == "" {
			return fmt.Errorf("empty retained capture")
		}
		detectors = append(detectors, detectorCase{path, output.Sequence, "sqk-lsk114", ends.DefaultOptions()})
	}
	input.Reset()
	for _, c := range detectors {
		kit := strings.ToUpper(c.Kit)
		if kit == "" {
			kit = "ANY"
		}
		fmt.Fprintf(&input, "DETECT %s %s %d %d\n", kit, token(c.Sequence), bit(c.Options.Adapters), bit(c.Options.Primers))
	}
	lines, err = native(executable, *out, "detector", input.Bytes())
	if err != nil {
		return err
	}
	if len(lines) != len(detectors) {
		return fmt.Errorf("native detector line count mismatch")
	}
	conflicts, nonemptyConflicts, forward, reverse := 0, 0, 0, 0
	for i, c := range detectors {
		var want ends.Result
		if err := json.Unmarshal(lines[i], &want); err != nil {
			return err
		}
		got, err := ends.Detect(c.Sequence, c.Kit, c.Options)
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("detector mismatch %s: Go %+v native %+v", c.Name, got, want)
		}
		if got.Conflict {
			conflicts++
			if c.Sequence != "" {
				nonemptyConflicts++
			}
		}
		if got.PrimerClassification.Orientation == ends.Forward {
			forward++
		}
		if got.PrimerClassification.Orientation == ends.Reverse {
			reverse++
		}
	}
	if nonemptyConflicts == 0 || forward == 0 || reverse == 0 {
		return fmt.Errorf("missing detector coverage conflicts=%d forward=%d reverse=%d", conflicts, forward, reverse)
	}
	if err := saveJSON(filepath.Join(*out, "detector-summary.json"), map[string]any{"cases": len(detectors), "mismatches": 0, "conflicts": conflicts, "nonempty_conflicts": nonemptyConflicts, "forward": forward, "reverse": reverse, "retained_capture_pins": capturePins}); err != nil {
		return err
	}
	fmt.Printf("detector: %d exact comparisons passed, including retained HAC and SUP\n", len(detectors))
	command := exec.Command("go", "test", "./internal/ends", "-run", "Native", "-count=1", "-v")
	helperEvidence := filepath.Join(*out, "native-helper-evidence")
	if err := os.Mkdir(helperEvidence, 0700); err != nil {
		return err
	}
	command.Env = append(os.Environ(), "ENDS_NATIVE_ORACLE="+executable, "ENDS_NATIVE_EVIDENCE="+helperEvidence)
	helperOutput, err := record(*out, "native-helper-tests", command, nil)
	if err != nil {
		return err
	}
	for _, name := range []string{"--- PASS: TestNativeBestResult", "--- PASS: TestNativeTrimInterval"} {
		if !bytes.Contains(helperOutput, []byte(name)) {
			return fmt.Errorf("native helper check did not execute %s", name)
		}
	}
	fmt.Println("native candidate selection and threshold controls passed")
	return saveJSON(filepath.Join(*out, "summary.json"), map[string]any{"alignment_cases": len(cases), "detector_cases": len(detectors), "status": "pass", "scope": "standalone default SQK-LSK114 end detection; no caller or inference qualification"})
}

func implementationPins(output string) error {
	pins := map[string]string{}
	for _, root := range []string{"internal/align", "internal/ends", "verification/ends"} {
		if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			pins[path] = sha(data)
			return nil
		}); err != nil {
			return err
		}
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return err
	}
	goPath, err = filepath.EvalSymlinks(goPath)
	if err != nil {
		return err
	}
	goBytes, err := os.ReadFile(goPath)
	if err != nil {
		return err
	}
	head, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return err
	}
	return saveJSON(filepath.Join(output, "implementation-inputs.json"), map[string]any{"source_sha256": pins, "go_runtime": runtime.Version(), "go_executable": goPath, "go_executable_sha256": sha(goBytes), "checkout_head": strings.TrimSpace(string(head))})
}

func custodyControls(root, edlib, output string) error {
	base := filepath.Join(output, "custody-controls")
	if err := os.Mkdir(base, 0700); err != nil {
		return err
	}
	doradoCopy, edlibCopy := filepath.Join(base, "dorado"), filepath.Join(base, "edlib")
	for _, set := range []struct {
		from, to string
		pins     map[string]string
	}{{root, doradoCopy, endsoracle.SourcePins}, {edlib, edlibCopy, endsoracle.EdlibPins}} {
		for path := range set.pins {
			data, err := os.ReadFile(filepath.Join(set.from, path))
			if err != nil {
				return err
			}
			dest := filepath.Join(set.to, path)
			if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
				return err
			}
			if err := os.WriteFile(dest, data, 0600); err != nil {
				return err
			}
		}
	}
	var receipts []map[string]any
	for _, set := range []struct {
		copyRoot string
		pins     map[string]string
	}{{doradoCopy, endsoracle.SourcePins}, {edlibCopy, endsoracle.EdlibPins}} {
		for path := range set.pins {
			file := filepath.Join(set.copyRoot, path)
			original, err := os.ReadFile(file)
			if err != nil {
				return err
			}
			mutated := append(bytes.Clone(original), '\n')
			if err := os.WriteFile(file, mutated, 0600); err != nil {
				return err
			}
			attempt := filepath.Join(base, fmt.Sprintf("rejection-%02d", len(receipts)))
			started := time.Now().UTC()
			_, buildErr := endsoracle.Build(doradoCopy, edlibCopy, attempt)
			if buildErr == nil || !strings.Contains(buildErr.Error(), "source hash mismatch") {
				return fmt.Errorf("source mutation was not rejected: %s: %v", path, buildErr)
			}
			if err := os.WriteFile(filepath.Join(attempt, "mutated-input"), mutated, 0600); err != nil {
				return err
			}
			receipts = append(receipts, map[string]any{"path": path, "original_sha256": sha(original), "mutated_sha256": sha(mutated), "started": started, "finished": time.Now().UTC(), "rejected": true, "error": buildErr.Error()})
			if err := os.WriteFile(file, original, 0600); err != nil {
				return err
			}
		}
	}
	if err := saveJSON(filepath.Join(base, "summary.json"), receipts); err != nil {
		return err
	}
	fmt.Printf("source custody: %d mutations rejected before compilation\n", len(receipts))
	return nil
}
func bit(b bool) int {
	if b {
		return 1
	}
	return 0
}
func native(exe, out, name string, input []byte) ([][]byte, error) {
	data, err := record(out, name, exec.Command(exe), input)
	if err != nil {
		return nil, err
	}
	var lines [][]byte
	s := bufio.NewScanner(bytes.NewReader(data))
	s.Buffer(make([]byte, 4096), 1<<20)
	for s.Scan() {
		lines = append(lines, bytes.Clone(s.Bytes()))
	}
	return lines, s.Err()
}
func record(out, name string, command *exec.Cmd, input []byte) ([]byte, error) {
	if err := os.WriteFile(filepath.Join(out, name+".input"), input, 0600); err != nil {
		return nil, err
	}
	command.Stdin = bytes.NewReader(input)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	started := time.Now().UTC()
	err := command.Run()
	finished := time.Now().UTC()
	exit := -1
	if command.ProcessState != nil {
		exit = command.ProcessState.ExitCode()
	}
	for suffix, data := range map[string][]byte{"stdout": stdout.Bytes(), "stderr": stderr.Bytes()} {
		if e := os.WriteFile(filepath.Join(out, name+"."+suffix), data, 0600); e != nil {
			return nil, e
		}
	}
	if e := saveJSON(filepath.Join(out, name+".receipt.json"), map[string]any{"args": command.Args, "env_additions": filterOracleEnv(command.Env), "started": started, "finished": finished, "exit_code": exit, "input_sha256": sha(input), "stdout_sha256": sha(stdout.Bytes()), "stderr_sha256": sha(stderr.Bytes())}); e != nil {
		return nil, e
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w\n%s\n%s", name, err, stdout.Bytes(), stderr.Bytes())
	}
	return stdout.Bytes(), nil
}
func filterOracleEnv(env []string) []string {
	var out []string
	for _, v := range env {
		if strings.HasPrefix(v, "ENDS_NATIVE_ORACLE=") || strings.HasPrefix(v, "ENDS_NATIVE_EVIDENCE=") {
			out = append(out, v)
		}
	}
	return out
}
func words(alphabet string, minLen, maxLen int) []string {
	out := []string{}
	var visit func(string)
	visit = func(s string) {
		if len(s) >= minLen {
			out = append(out, s)
		}
		if len(s) == maxLen {
			return
		}
		for i := range alphabet {
			visit(s + alphabet[i:i+1])
		}
	}
	visit("")
	return out
}
func randomDNA(rng *rand.Rand, n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = "ACGTN"[rng.Intn(5)]
	}
	return string(b)
}
func equal(a, b byte, e align.Equality) bool {
	return a == b || e == align.EqualityNWildcard && (a == 'N' && strings.ContainsRune("ACGT", rune(b)) || b == 'N' && strings.ContainsRune("ACGT", rune(a)))
}

// The rejected alternative carries the latest start through distance ties.
func alternative(q, t string, e align.Equality, latest bool) align.Location {
	if t == "" {
		return align.Location{Distance: len(q), Start: -1, End: -1}
	}
	type cell struct{ d, s int }
	p, c := make([]cell, len(q)+1), make([]cell, len(q)+1)
	for i := range p {
		p[i] = cell{i, 0}
	}
	best := align.Location{Distance: len(q), Start: 0, End: -1, Valid: true}
	if len(q)%64 == 0 {
		best.Distance++
	}
	for j := range t {
		c[0] = cell{0, j + 1}
		for i := 1; i <= len(q); i++ {
			cost := 1
			if equal(q[i-1], t[j], e) {
				cost = 0
			}
			v := cell{p[i-1].d + cost, p[i-1].s}
			for _, a := range []cell{{c[i-1].d + 1, c[i-1].s}, {p[i].d + 1, p[i].s}} {
				if a.d < v.d || a.d == v.d && (latest && a.s > v.s || !latest && a.s < v.s) {
					v = a
				}
			}
			c[i] = v
		}
		if c[len(q)].d < best.Distance {
			best = align.Location{Distance: c[len(q)].d, Start: c[len(q)].s, End: j, Valid: true}
		}
		p, c = c, p
	}
	return best
}

// The second shape computes the first end, then reproduces reversed SHW last-end selection.
func reverseShape(q, t string, e align.Equality) align.Location {
	first := alternative(q, t, e, false)
	if !first.Valid || first.End < 0 {
		return first
	}
	end := first.End
	rquery := reverse(q)
	rtarget := reverse(t[:end+1])
	p, c := make([]int, len(q)+1), make([]int, len(q)+1)
	for i := range p {
		p[i] = i
	}
	last := -1
	if p[len(q)] == first.Distance {
		last = -1
	}
	for j := range rtarget {
		c[0] = j + 1
		for i := 1; i <= len(q); i++ {
			cost := 1
			if equal(rquery[i-1], rtarget[j], e) {
				cost = 0
			}
			c[i] = min(p[i-1]+cost, min(c[i-1]+1, p[i]+1))
		}
		if c[len(q)] == first.Distance {
			last = j
		}
		p, c = c, p
	}
	first.Start = end - last
	return first
}
func reverse(s string) string {
	b := []byte(s)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return string(b)
}

const adapterFront = "CCTGTACTTCGTTCAGTTACGTATTGC"
const adapterRear = "AGCAATACGTAACTGAAC"
const primerFront = "TTTCTGTTGGTGCTGATATTGCTGGG"
const primerRear = "ACTTGCCTGTCGCTCTATCTTCTTT"

func rc(s string) string {
	b := []byte(reverse(s))
	for i := range b {
		b[i] = "TGCA"[strings.IndexByte("ACGT", b[i])]
	}
	return string(b)
}
func detectorCases(rng *rand.Rand) []detectorCase {
	var out []detectorCase
	add := func(name, s string) {
		out = append(out, detectorCase{name, s, ends.SupportedKit, ends.DefaultOptions()})
	}
	body := strings.Repeat("ACGT", 60)
	add("disjoint_adapter_primer_intervals", strings.Repeat("G", 10)+rc(primerRear)+strings.Repeat("G", 5)+adapterFront+strings.Repeat("G", 40))
	for _, c := range []struct{ name, s string }{{"empty", ""}, {"no_hit", strings.Repeat("G", 240)}, {"N_equality", strings.Repeat("N", 240)}, {"adapter_front", adapterFront + body}, {"adapter_rear", body + adapterRear}, {"adapter_both", adapterFront + body + adapterRear}, {"primer_forward", primerFront + body + rc(primerRear)}, {"primer_reverse", primerRear + body + rc(primerFront)}, {"primer_front_only", primerFront + body}, {"primer_rear_only", body + rc(primerRear)}, {"primer_conflicting_names", primerFront + body + rc(primerFront)}, {"GEN10X_excluded", "CTACACGACGCTCTTCCGATCT" + body + rc("GTACTCTGCGTTGATACCACTGCTT")}, {"adapter_and_primer", adapterFront + primerFront + body + rc(primerRear) + adapterRear}, {"overlap", adapterFront + adapterRear}, {"cross_interval", rc(primerRear) + strings.Repeat("G", 8) + adapterFront}} {
		add(c.name, c.s)
	}
	for _, n := range []int{1, 4, 5, 17, 18, 23, 24, 25, 26, 27, 74, 75, 76, 149, 150, 151, 300} {
		add(fmt.Sprintf("clipped_%d", n), randomDNA(rng, n))
	}
	for _, n := range []int{49, 50, 51, 74, 75, 76, 124, 125, 126, 149, 150, 151} {
		add(fmt.Sprintf("front_window_boundary_%d", n), strings.Repeat("G", n)+adapterFront+body)
		add(fmt.Sprintf("rear_window_boundary_%d", n), body+adapterRear+strings.Repeat("G", n))
	}
	for i := 0; i <= 8; i++ {
		b := []byte(primerFront)
		for j := 0; j < i; j++ {
			b[j] = "ACGT"[(strings.IndexByte("ACGT", b[j])+1)%4]
		}
		add(fmt.Sprintf("threshold_mutations_%d", i), string(b)+body)
	}
	for i := 0; i < 300; i++ {
		s := randomDNA(rng, rng.Intn(350))
		if i%3 == 0 {
			s = primerFront + s + rc(primerRear)
		}
		if i%5 == 0 {
			s = adapterFront + s + adapterRear
		}
		add(fmt.Sprintf("seeded_%d", i), s)
	}
	out = append(out, detectorCase{"missing_kit", adapterFront + body + adapterRear, "", ends.DefaultOptions()}, detectorCase{"explicit_ANY", primerFront + body, "ANY", ends.DefaultOptions()})
	for _, options := range []ends.Options{{}, {Adapters: true}, {Primers: true}} {
		out = append(out, detectorCase{"flags", adapterFront + primerFront + body + rc(primerRear) + adapterRear, ends.SupportedKit, options})
		out = append(out, detectorCase{"empty_flags", "", ends.SupportedKit, options})
	}
	return out
}
