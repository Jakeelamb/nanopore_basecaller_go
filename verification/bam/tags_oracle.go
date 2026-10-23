// Native tag oracle for the unaligned BAM writer.
//
// The oracle compiles the pinned Dorado tag emitters against real htslib and
// runs them on synthetic ReadCommon values. It is a narrow build: the marked
// slices of messages.cpp, sequence_utils.cpp, hts_types.h and ReadCommon.h are
// copied byte for byte, and the class body plus get_raw_data_samples,
// generate_modbase_tags and trace_log are shims. The shims are listed in
// docs/bam-writer.md.
//
// This is a native tag-emission differential, not GPU evidence, not
// end-to-end basecalling and not biological evidence.
package bamverify

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
)

const TagsOracleSchema = "bam-tags-oracle-receipt/1"

// Pinned Dorado inputs. A whole-file pin is stronger than a slice pin: any
// edit to the file invalidates the oracle before a slice is taken.
const (
	tagsMessagesPath = "dorado/read_pipeline/base/messages.cpp"
	tagsMessagesSHA  = "5db12169f5eb376bfa70425a5f17fa31eab6be5d027d681caec81dd105ba35e0"
	tagsSequencePath = "dorado/utils/sequence_utils.cpp"
	tagsSequenceSHA  = "d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2"
	tagsHtsTypesPath = "dorado/hts_utils/include/hts_utils/hts_types.h"
	tagsHtsTypesSHA  = "653f2f71de8bc57847349154bd45a6f66120cb56fcaf847a8ccc9b694a1ae65b"
	tagsCommonPath   = "dorado/read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h"
	tagsCommonSHA    = "a8ff4bced80c61f07070d6f5ed17ad3c2367c4f7d2a4564f54b5a7e47aec01cb"
)

//go:embed native/tags_oracle.cpp
var tagsOracleTemplate string

// SourceSlice records one extracted region so a reviewer can reproduce it.
type SourceSlice struct {
	Path        string `json:"path"`
	Marker      string `json:"marker"`
	Start       int    `json:"start_byte"`
	End         int    `json:"end_byte"`
	SHA256      string `json:"slice_sha256"`
	WholeSHA256 string `json:"whole_file_sha256"`
}

// TagsOracleCase is one synthetic read and the Go verdict it should produce.
type TagsOracleCase struct {
	Name string
	// Ported cases must match the native aux bytes exactly.
	Ported bool
	// Reject is non-empty when bam.Tags must refuse the request; the value is
	// the required substring of the error.
	Reject string
	// Gap names the tags the pinned source emits and Go does not. A gap case
	// compares tag names only and never claims byte parity.
	Gap []string
	In  TagsOracleInput
}

// TagsOracleInput is the synthetic ReadCommon the oracle builds.
type TagsOracleInput struct {
	ReadID       string
	Seq          string
	Qstring      string
	NumTrimmed   uint64
	RawSamples   uint64
	SampleRate   uint64
	Mux          uint32
	Channel      int32
	StartTime    string
	ReadNumber   int32
	Filename     string
	ModelStride  int32
	PoreType     string
	EndReason    string
	MinknowEvent uint64
	Moves        []byte
	EmitMoves    bool
	Shift        float32
	Scale        float32
	Scaling      string
	RunID        string
	ModelName    string
	Barcode      string
	Barcoding    *TagsOracleBarcode
	Orientation  int8
	UMI          string
	ParentID     string
	SplitPoint   uint32
	RNA          *TagsOracleRNA
	IsDuplex     bool
	IsRNAModel   bool
	MeanQStart   uint32
	DuplexParent bool
}

type TagsOracleBarcode struct {
	Variant                string
	BarcodeScore           float32
	TopStart, TopEnd       int32
	TopScore               float32
	BottomStart, BottomEnd int32
	BottomScore            float32
}

type TagsOracleRNA struct {
	Length, Anchor int32
	B0Start, B0End int32
	B1Start, B1End int32
}

type TagsOracleCaseResult struct {
	Name          string   `json:"name"`
	Ported        bool     `json:"ported"`
	NativeBytes   int      `json:"native_aux_bytes"`
	NativeSHA256  string   `json:"native_aux_sha256"`
	GoSHA256      string   `json:"go_aux_sha256,omitempty"`
	NativeTags    []string `json:"native_tags"`
	GoTags        []string `json:"go_tags,omitempty"`
	ByteIdentical bool     `json:"byte_identical"`
	Rejected      bool     `json:"rejected,omitempty"`
	RejectDetail  string   `json:"reject_detail,omitempty"`
	Gap           []string `json:"gap,omitempty"`
	Passed        bool     `json:"passed"`
	Detail        string   `json:"detail,omitempty"`
}

// TagsOracleReceipt is the reproducible evidence for one oracle run.
type TagsOracleReceipt struct {
	Schema     string                 `json:"schema"`
	Started    string                 `json:"started"`
	Finished   string                 `json:"finished"`
	Compiler   string                 `json:"compiler"`
	Htslib     string                 `json:"htslib"`
	Slices     []SourceSlice          `json:"slices"`
	FixtureSHA string                 `json:"fixture_sha256"`
	SplicedSHA string                 `json:"spliced_source_sha256"`
	BinarySHA  string                 `json:"oracle_binary_sha256"`
	OutputSHA  string                 `json:"oracle_output_sha256"`
	Cases      []TagsOracleCaseResult `json:"cases"`
	Controls   []TagsOracleControl    `json:"controls"`
	Passed     bool                   `json:"passed"`
}

// TagsOracleControl proves the comparison is sensitive to a changed input.
type TagsOracleControl struct {
	Name     string `json:"name"`
	Expected string `json:"expected"`
	Observed string `json:"observed"`
	Passed   bool   `json:"passed"`
}

// TagsOracleCases returns the pinned case matrix. The ported cases cover every
// conditional simplex tag the writer claims; the gap cases name what the pinned
// source emits and Go does not.
func TagsOracleCases() []TagsOracleCase {
	base := TagsOracleInput{
		ReadID: "6b1f9d0e-2c3a-4f5b-8d7e-1a2b3c4d5e6f", Seq: "ACGTACGTAC",
		Qstring: "IIIIIIIIII", NumTrimmed: 120, RawSamples: 4800, SampleRate: 4000,
		Mux: 1, Channel: 2372, StartTime: "2024-06-19T21:36:39.995000+00:00",
		ReadNumber: 4484, Filename: "PAW70337_66b2eea5_de8117b1_33.pod5",
		ModelStride: 6, PoreType: "not_set", EndReason: "signal_positive",
		MinknowEvent: 3653, Shift: 93.99999, Scale: 24, Scaling: "pa",
		RunID: "66b2eea5-f887-4690-bd76-97fd9c4e416b", ModelName: "dna_r10.4.1_e8.2_400bps_hac@v6.0.0",
		MeanQStart: 60,
	}
	with := func(f func(*TagsOracleInput)) TagsOracleInput {
		out := base
		f(&out)
		return out
	}
	return []TagsOracleCase{
		{Name: "simplex-minimal", Ported: true, In: base},
		{Name: "simplex-primer-umi", Ported: true, In: with(func(i *TagsOracleInput) {
			i.Orientation = 1
			i.UMI = "ACGTACGT"
		})},
		{Name: "simplex-split", Ported: true, In: with(func(i *TagsOracleInput) {
			i.ParentID = "6b1f9d0e-2c3a-4f5b-8d7e-000000000000"
			i.SplitPoint = 2048
		})},
		{Name: "simplex-moves", Ported: true, In: with(func(i *TagsOracleInput) {
			i.Moves = []byte{1, 0, 1, 1, 0, 1}
			i.EmitMoves = true
		})},
		{Name: "simplex-full", Ported: true, In: with(func(i *TagsOracleInput) {
			i.Orientation = 1
			i.UMI = "ACGTACGT"
			i.ParentID = "6b1f9d0e-2c3a-4f5b-8d7e-000000000000"
			i.SplitPoint = 2048
			i.Moves = []byte{1, 0, 1, 1, 0, 1}
			i.EmitMoves = true
		})},
		{Name: "simplex-duplex-parent", Ported: true, In: with(func(i *TagsOracleInput) {
			i.DuplexParent = true
		})},
		{Name: "duplex-read", Ported: true, In: with(func(i *TagsOracleInput) {
			i.IsDuplex = true
			i.ParentID = "6b1f9d0e-2c3a-4f5b-8d7e-000000000000"
		})},
		{Name: "duplex-minimal", Ported: true, In: with(func(i *TagsOracleInput) {
			i.IsDuplex = true
			i.RunID, i.ModelName = "", ""
			i.PoreType = ""
		})},
		{Name: "barcode", Ported: true, In: with(func(i *TagsOracleInput) {
			i.Barcode = "barcode01"
			i.Barcoding = &TagsOracleBarcode{
				Variant: "barcode01", BarcodeScore: 12.5,
				TopStart: 10, TopEnd: 42, TopScore: 9.25,
				BottomStart: 480, BottomEnd: 512, BottomScore: 8.75,
			}
		})},
		{Name: "barcode-unclassified", Ported: true, In: with(func(i *TagsOracleInput) {
			i.Barcode = "unclassified"
		})},
		{Name: "barcode-unclassified-score", Reject: "no barcode tags", In: with(func(i *TagsOracleInput) {
			i.Barcode = "unclassified"
			i.Barcoding = &TagsOracleBarcode{
				Variant: "barcode01", BarcodeScore: 12.5,
				TopStart: 10, TopEnd: 42, TopScore: 9.25,
				BottomStart: 480, BottomEnd: 512, BottomScore: 8.75,
			}
		})},
		{Name: "rna-polytail-window", Ported: true, In: with(func(i *TagsOracleInput) {
			// A trailing poly-A run makes find_rna_polya return 10, so the mean
			// Q-score must use qstring[:10] rather than the whole string. The
			// quality values differ sharply so a wrong window cannot match.
			i.IsRNAModel = true
			i.Seq = "CGTCAGTACG" + strings.Repeat("A", 20)
			i.Qstring = strings.Repeat("5", 10) + strings.Repeat("!", 20)
			i.RNA = &TagsOracleRNA{Length: -1, Anchor: 300, B0Start: 290, B0End: 340, B1Start: -1, B1End: -1}
		})},
		{Name: "rna-polytail", Ported: true, In: with(func(i *TagsOracleInput) {
			i.IsRNAModel = true
			i.Seq = strings.Repeat("A", 20) + "CGTCAGTACG"
			i.Qstring = strings.Repeat("I", len(i.Seq))
			i.RNA = &TagsOracleRNA{Length: 18, Anchor: 300, B0Start: 290, B0End: 340, B1Start: -1, B1End: -1}
		})},
	}
}

// BuildTagsOracle splices the pinned slices into the template and compiles it
// against htslib. It never writes to the Dorado checkout.
func BuildTagsOracle(srcRoot, outDir, cxx string) (string, []SourceSlice, string, error) {
	sources := []struct{ path, sha string }{
		{tagsMessagesPath, tagsMessagesSHA},
		{tagsSequencePath, tagsSequenceSHA},
		{tagsHtsTypesPath, tagsHtsTypesSHA},
		{tagsCommonPath, tagsCommonSHA},
	}
	text := make(map[string]string, len(sources))
	for _, s := range sources {
		data, err := os.ReadFile(filepath.Join(srcRoot, s.path))
		if err != nil {
			return "", nil, "", err
		}
		if got := HashBytes(data); got != s.sha {
			return "", nil, "", fmt.Errorf("pinned source %s has sha %s, want %s", s.path, got, s.sha)
		}
		text[s.path] = string(data)
	}

	type splice struct{ marker, path, start string }
	splices := []splice{
		{"//__STRAND__", tagsHtsTypesPath, "enum class StrandOrientation"},
		{"//__TO_CHAR__", tagsHtsTypesPath, "inline char to_char("},
		{"//__BARCODE__", tagsHtsTypesPath, "struct BarcodeScoreResult {"},
		{"//__PRIMER__", tagsHtsTypesPath, "struct PrimerClassification {"},
		{"//__ATTRIBUTES__", tagsCommonPath, "struct Attributes {"},
		{"//__READ_GROUP__", tagsMessagesPath, "std::string ReadCommon::generate_read_group() const {"},
		{"//__READ_TAGS__", tagsMessagesPath, "void ReadCommon::generate_read_tags("},
		{"//__DUPLEX_TAGS__", tagsMessagesPath, "void ReadCommon::generate_duplex_read_tags("},
		{"//__MEAN_QSCORE__", tagsMessagesPath, "float ReadCommon::calculate_mean_qscore() const {"},
		{"//__EXTRACT__", tagsMessagesPath, "std::vector<BamPtr> ReadCommon::extract_sam_lines("},
		{"//__FIND_POLYA__", tagsSequencePath, "size_t find_rna_polya(const std::string& seq) {"},
		{"//__MEAN_QSTRING__", tagsSequencePath, "float mean_qscore_from_qstring(std::string_view qstring) {"},
	}
	out := tagsOracleTemplate
	var slices []SourceSlice
	for _, sp := range splices {
		block, start, end, err := extractBlock(text[sp.path], sp.start)
		if err != nil {
			return "", nil, "", fmt.Errorf("%s: %w", sp.path, err)
		}
		if !strings.Contains(out, sp.marker) {
			return "", nil, "", fmt.Errorf("template marker %s is missing", sp.marker)
		}
		out = strings.Replace(out, sp.marker, block, 1)
		slices = append(slices, SourceSlice{
			Path: sp.path, Marker: sp.marker, Start: start, End: end,
			SHA256: HashBytes([]byte(block)), WholeSHA256: HashBytes([]byte(text[sp.path])),
		})
	}
	if strings.Contains(out, "//__") {
		return "", nil, "", errors.New("template still has an unreplaced marker")
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", nil, "", err
	}
	cpp := filepath.Join(outDir, "tags_oracle.spliced.cpp")
	if err := os.WriteFile(cpp, []byte(out), 0o644); err != nil {
		return "", nil, "", err
	}
	flags, err := htslibFlags()
	if err != nil {
		return "", nil, "", err
	}
	bin := filepath.Join(outDir, "tags_oracle")
	args := []string{"-std=c++17", "-O2", "-o", bin, cpp}
	args = append(args, flags...)
	cmd := exec.Command(cxx, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", nil, "", fmt.Errorf("%s: %w: %.4096s", cxx, err, stderr.String())
	}
	return bin, slices, cpp, nil
}

func htslibFlags() ([]string, error) {
	out, err := exec.Command("pkg-config", "--cflags", "--libs", "htslib").Output()
	if err != nil {
		return nil, fmt.Errorf("pkg-config htslib: %w", err)
	}
	fields := strings.Fields(string(out))
	if len(fields) == 0 {
		return nil, errors.New("pkg-config returned no htslib flags")
	}
	return fields, nil
}

// extractBlock returns the text from start through its matching closing brace.
// It skips string, character, line-comment and block-comment regions so braces
// inside a trace_log format string do not end the block.
func extractBlock(src, start string) (string, int, int, error) {
	i := strings.Index(src, start)
	if i < 0 {
		return "", 0, 0, fmt.Errorf("source marker %q not found", start)
	}
	open := -1
	for k := i; k < len(src); k++ {
		if src[k] == '{' {
			open = k
			break
		}
		if src[k] == ';' {
			return "", 0, 0, fmt.Errorf("marker %q has no body", start)
		}
	}
	if open < 0 {
		return "", 0, 0, fmt.Errorf("marker %q has no opening brace", start)
	}
	const (
		normal = iota
		lineComment
		blockComment
		strLit
		charLit
	)
	state := normal
	depth := 0
	for k := open; k < len(src); k++ {
		c := src[k]
		switch state {
		case normal:
			switch c {
			case '/':
				if k+1 < len(src) && src[k+1] == '/' {
					state = lineComment
					k++
				} else if k+1 < len(src) && src[k+1] == '*' {
					state = blockComment
					k++
				}
			case '"':
				state = strLit
			case '\'':
				state = charLit
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					end := k + 1
					// A type definition ends with "};"; a function does not.
					for j := end; j < len(src); j++ {
						if src[j] == ' ' || src[j] == '\t' || src[j] == '\n' || src[j] == '\r' {
							continue
						}
						if src[j] == ';' {
							end = j + 1
						}
						break
					}
					return src[i:end], i, end, nil
				}
			}
		case lineComment:
			if c == '\n' {
				state = normal
			}
		case blockComment:
			if c == '*' && k+1 < len(src) && src[k+1] == '/' {
				state = normal
				k++
			}
		case strLit:
			if c == '\\' {
				k++
			} else if c == '"' {
				state = normal
			}
		case charLit:
			if c == '\\' {
				k++
			} else if c == '\'' {
				state = normal
			}
		}
	}
	return "", 0, 0, fmt.Errorf("marker %q has an unterminated body", start)
}

// WriteTagsOracleFixture serializes the case matrix in the oracle's binary
// protocol.
func WriteTagsOracleFixture(path string, cases []TagsOracleCase) error {
	var b bytes.Buffer
	b.WriteString("BTAGF001")
	putU32(&b, uint32(len(cases)))
	for _, c := range cases {
		in := c.In
		putStr(&b, in.ReadID)
		putStr(&b, in.Seq)
		putStr(&b, in.Qstring)
		putU64(&b, in.NumTrimmed)
		putU64(&b, in.RawSamples)
		putU64(&b, in.SampleRate)
		putU32(&b, in.Mux)
		putU32(&b, uint32(in.Channel))
		putStr(&b, in.StartTime)
		putU32(&b, uint32(in.ReadNumber))
		putStr(&b, in.Filename)
		putU32(&b, uint32(in.ModelStride))
		putStr(&b, in.PoreType)
		putStr(&b, in.EndReason)
		putU64(&b, in.MinknowEvent)
		if in.EmitMoves {
			b.WriteByte(1)
			putU32(&b, uint32(len(in.Moves)))
			b.Write(in.Moves)
		} else {
			b.WriteByte(0)
		}
		putU32(&b, math.Float32bits(in.Shift))
		putU32(&b, math.Float32bits(in.Scale))
		putStr(&b, in.Scaling)
		putStr(&b, in.RunID)
		putStr(&b, in.ModelName)
		putStr(&b, in.Barcode)
		if in.Barcoding != nil {
			b.WriteByte(1)
			putStr(&b, in.Barcoding.Variant)
			putU32(&b, math.Float32bits(in.Barcoding.BarcodeScore))
			putU32(&b, uint32(in.Barcoding.TopStart))
			putU32(&b, uint32(in.Barcoding.TopEnd))
			putU32(&b, math.Float32bits(in.Barcoding.TopScore))
			putU32(&b, uint32(in.Barcoding.BottomStart))
			putU32(&b, uint32(in.Barcoding.BottomEnd))
			putU32(&b, math.Float32bits(in.Barcoding.BottomScore))
		} else {
			b.WriteByte(0)
		}
		b.WriteByte(byte(in.Orientation))
		putStr(&b, in.UMI)
		putStr(&b, in.ParentID)
		putU32(&b, in.SplitPoint)
		if in.RNA != nil {
			b.WriteByte(1)
			putU32(&b, uint32(in.RNA.Length))
			putU32(&b, uint32(in.RNA.Anchor))
			putU32(&b, uint32(in.RNA.B0Start))
			putU32(&b, uint32(in.RNA.B0End))
			putU32(&b, uint32(in.RNA.B1Start))
			putU32(&b, uint32(in.RNA.B1End))
		} else {
			b.WriteByte(0)
		}
		putBool(&b, in.IsDuplex)
		putBool(&b, in.IsRNAModel)
		putU32(&b, in.MeanQStart)
		putBool(&b, in.DuplexParent)
		putBool(&b, in.EmitMoves)
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

func putU32(b *bytes.Buffer, v uint32) {
	var w [4]byte
	binary.LittleEndian.PutUint32(w[:], v)
	b.Write(w[:])
}

func putU64(b *bytes.Buffer, v uint64) {
	var w [8]byte
	binary.LittleEndian.PutUint64(w[:], v)
	b.Write(w[:])
}

func putStr(b *bytes.Buffer, s string) {
	putU32(b, uint32(len(s)))
	b.WriteString(s)
}

func putBool(b *bytes.Buffer, v bool) {
	if v {
		b.WriteByte(1)
	} else {
		b.WriteByte(0)
	}
}

// RunTagsOracle executes the compiled oracle and returns one aux blob per case.
func RunTagsOracle(bin, fixture, out string) ([][]byte, error) {
	cmd := exec.Command(bin, fixture, out)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("tags oracle: %w: %.4096s", err, stderr.String())
	}
	data, err := os.ReadFile(out)
	if err != nil {
		return nil, err
	}
	if len(data) < 12 || string(data[:8]) != "BTAGR001" {
		return nil, errors.New("tags oracle output magic is missing")
	}
	n := binary.LittleEndian.Uint32(data[8:])
	if n == 0 || n > 4096 {
		return nil, fmt.Errorf("tags oracle emitted %d cases", n)
	}
	off := 12
	out2 := make([][]byte, 0, n)
	for i := 0; i < int(n); i++ {
		if off+4 > len(data) {
			return nil, errors.New("tags oracle output is truncated")
		}
		l := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		if l < 0 || off+l > len(data) {
			return nil, errors.New("tags oracle aux blob overruns the output")
		}
		out2 = append(out2, append([]byte(nil), data[off:off+l]...))
		off += l
	}
	if off != len(data) {
		return nil, errors.New("tags oracle output has trailing bytes")
	}
	return out2, nil
}

// VerifyTagsOracle builds and runs the oracle, compares every case, and writes
// a receipt. Ported cases must be byte-identical; gap cases compare tag names
// and record the unported tags instead of claiming parity.
func VerifyTagsOracle(srcRoot, workDir, cxx string) (TagsOracleReceipt, error) {
	started := time.Now().UTC().Format(time.RFC3339)
	if err := os.Mkdir(workDir, 0o755); err != nil {
		return TagsOracleReceipt{}, fmt.Errorf("fresh output required: %w", err)
	}
	receipt := TagsOracleReceipt{Schema: TagsOracleSchema, Started: started}
	cases := TagsOracleCases()
	fixture := filepath.Join(workDir, "tags.fixture")
	if err := WriteTagsOracleFixture(fixture, cases); err != nil {
		return TagsOracleReceipt{}, err
	}
	fixtureBytes, err := os.ReadFile(fixture)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.FixtureSHA = HashBytes(fixtureBytes)

	bin, slices, cpp, err := BuildTagsOracle(srcRoot, workDir, cxx)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.Slices = slices
	receipt.Compiler = cxx
	binBytes, err := os.ReadFile(bin)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.BinarySHA = HashBytes(binBytes)
	spliced, err := os.ReadFile(cpp)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.SplicedSHA = HashBytes(spliced)
	if receipt.Htslib, err = htslibVersion(); err != nil {
		return TagsOracleReceipt{}, err
	}

	outPath := filepath.Join(workDir, "tags.out")
	native, err := RunTagsOracle(bin, fixture, outPath)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	outBytes, err := os.ReadFile(outPath)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.OutputSHA = HashBytes(outBytes)
	if len(native) != len(cases) {
		return TagsOracleReceipt{}, fmt.Errorf("oracle returned %d cases, want %d", len(native), len(cases))
	}

	header := oracleHeader()
	allPassed := true
	for i, c := range cases {
		res, err := compareTagsCase(c, native[i], header)
		if err != nil {
			return TagsOracleReceipt{}, fmt.Errorf("%s: %w", c.Name, err)
		}
		if !res.Passed {
			allPassed = false
		}
		receipt.Cases = append(receipt.Cases, res)
	}
	controls, err := tagsControls(cases, native, header)
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	receipt.Controls = controls
	for _, c := range controls {
		if !c.Passed {
			allPassed = false
		}
	}
	receipt.Passed = allPassed
	receipt.Finished = time.Now().UTC().Format(time.RFC3339)
	body, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return TagsOracleReceipt{}, err
	}
	if err := os.WriteFile(filepath.Join(workDir, "tags-receipt.json"), append(body, '\n'), 0o644); err != nil {
		return TagsOracleReceipt{}, err
	}
	if !allPassed {
		return receipt, errors.New("tags oracle case failed")
	}
	return receipt, nil
}

// tagsControl proves a one-field change moves the Go aux bytes, so a pass
// cannot come from comparing a blob with itself.
func tagsControl(name, caseName string, cases []TagsOracleCase, native [][]byte, header bam.Header,
	mutate func(TagsOracleInput, *bam.Metadata)) (TagsOracleControl, error) {
	ctrl := TagsOracleControl{Name: name, Expected: "go aux differs from native"}
	idx := -1
	for i, c := range cases {
		if c.Name == caseName {
			idx = i
			break
		}
	}
	if idx < 0 {
		return ctrl, fmt.Errorf("control %s needs the %s case", name, caseName)
	}
	base := cases[idx].In
	meta, err := tagsMetadata(base)
	if err != nil {
		return ctrl, err
	}
	tags, err := bam.Tags(meta)
	if err != nil {
		return ctrl, err
	}
	unchanged, err := goAuxBytes(header, base, tags)
	if err != nil {
		return ctrl, err
	}
	if !bytes.Equal(unchanged, native[idx]) {
		return ctrl, fmt.Errorf("control %s baseline is not identical", name)
	}
	changedMeta := meta
	mutate(base, &changedMeta)
	tags, err = bam.Tags(changedMeta)
	if err != nil {
		return ctrl, err
	}
	changed, err := goAuxBytes(header, base, tags)
	if err != nil {
		return ctrl, err
	}
	ctrl.Observed = firstAuxDifference(native[idx], changed)
	ctrl.Passed = !bytes.Equal(changed, native[idx])
	return ctrl, nil
}

func tagsControls(cases []TagsOracleCase, native [][]byte, header bam.Header) ([]TagsOracleControl, error) {
	split, err := tagsControl("split-point-1", "simplex-full", cases, native, header,
		func(_ TagsOracleInput, m *bam.Metadata) { m.SplitPoint++ })
	if err != nil {
		return nil, err
	}
	rna, err := tagsControl("rna-window", "rna-polytail-window", cases, native, header,
		func(in TagsOracleInput, m *bam.Metadata) {
			// Score the whole qstring instead of the poly-A prefix. If the
			// window were not exercised, the aux bytes would not move.
			if full, err := decode.MeanQScore(in.Qstring); err == nil {
				m.MeanQ = full
			}
		})
	if err != nil {
		return nil, err
	}
	return []TagsOracleControl{split, rna}, nil
}

func compareTagsCase(c TagsOracleCase, native []byte, header bam.Header) (TagsOracleCaseResult, error) {
	res := TagsOracleCaseResult{Name: c.Name, Ported: c.Ported, NativeBytes: len(native),
		NativeSHA256: HashBytes(native), Gap: c.Gap}
	nativeTags, err := parseBAMAux(native)
	if err != nil {
		return res, fmt.Errorf("native aux: %w", err)
	}
	res.NativeTags = auxNames(nativeTags)

	meta, err := tagsMetadata(c.In)
	if err != nil {
		return res, err
	}
	tags, tagErr := bam.Tags(meta)
	if c.Reject != "" {
		if tagErr == nil {
			res.Detail = "bam.Tags accepted a request the pinned source cannot represent"
			return res, nil
		}
		if !strings.Contains(tagErr.Error(), c.Reject) {
			res.RejectDetail = tagErr.Error()
			res.Detail = fmt.Sprintf("rejection %q does not mention %q", tagErr.Error(), c.Reject)
			return res, nil
		}
		res.Rejected = true
		res.RejectDetail = tagErr.Error()
		res.Passed = true
		res.Detail = "rejected as declared"
		return res, nil
	}
	if tagErr != nil {
		res.Detail = tagErr.Error()
		return res, nil
	}
	goAux, err := goAuxBytes(header, c.In, tags)
	if err != nil {
		return res, err
	}
	goTags, err := parseBAMAux(goAux)
	if err != nil {
		return res, fmt.Errorf("go aux: %w", err)
	}
	res.GoTags = auxNames(goTags)
	res.GoSHA256 = HashBytes(goAux)

	if len(c.Gap) > 0 {
		want := removeNames(res.NativeTags, c.Gap)
		if !equalStrings(res.GoTags, want) {
			res.Detail = fmt.Sprintf("go tags %v, want native %v without %v", res.GoTags, res.NativeTags, c.Gap)
			return res, nil
		}
		res.Passed = true
		res.Detail = fmt.Sprintf("gap recorded: native emits %v, Go omits them", c.Gap)
		return res, nil
	}

	res.ByteIdentical = bytes.Equal(goAux, native)
	res.Passed = res.ByteIdentical
	if res.Passed {
		res.Detail = "native aux bytes identical"
	} else {
		res.Detail = firstAuxDifference(native, goAux)
	}
	return res, nil
}

// tagsMetadata maps the synthetic read onto the writer's typed input. MeanQ is
// recomputed in Go from the accepted quality rule so qs is a real differential.
func tagsMetadata(in TagsOracleInput) (bam.Metadata, error) {
	q := in.Qstring
	// calculate_mean_qscore: an RNA model scores the prefix before the poly-A
	// run, a DNA model drops mean_qscore_start_pos. The native side computes
	// this itself, so qs stays a real differential.
	if in.IsRNAModel {
		if start := findRNAPolyA(in.Seq); start > 0 && start < len(q) {
			q = q[:start]
		}
	} else if len(q) > int(in.MeanQStart) {
		q = q[in.MeanQStart:]
	}
	meanQ, err := decode.MeanQScore(q)
	if err != nil {
		return bam.Metadata{}, err
	}
	ns := int32(in.RawSamples + in.NumTrimmed)
	strand := byte(0)
	switch in.Orientation {
	case 1:
		strand = '+'
	case -1:
		strand = '-'
	}
	var moves []uint8
	if in.EmitMoves {
		moves = append([]uint8(nil), in.Moves...)
	}
	return bam.Metadata{
		MeanQ:         meanQ,
		Duration:      float32(in.RawSamples+in.NumTrimmed) / float32(in.SampleRate),
		NumSamples:    ns,
		Trimmed:       int32(in.NumTrimmed),
		Mux:           int32(in.Mux),
		Channel:       in.Channel,
		StartTime:     in.StartTime,
		Strand:        strand,
		UMI:           in.UMI,
		ReadNumber:    in.ReadNumber,
		Filename:      in.Filename,
		Shift:         in.Shift,
		Scale:         in.Scale,
		ScalingMethod: in.Scaling,
		DuplexParent:  in.DuplexParent,
		ReadGroup:     bam.ReadGroupID(in.RunID, in.ModelName, in.Barcode),
		ParentID:      in.ParentID,
		SplitPoint:    int32(in.SplitPoint),
		PoreType:      in.PoreType,
		EndReason:     in.EndReason,
		MinknowEvents: uint32(in.MinknowEvent),
		Moves:         moves,
		ModelStride:   uint8(in.ModelStride),
		SampleRate:    in.SampleRate,
		IsDuplex:      in.IsDuplex,
		Barcode:       in.Barcode,
		BarcodeVariant: func() string {
			if in.Barcoding != nil {
				return in.Barcoding.Variant
			}
			return ""
		}(),
		BarcodeScore: in.barcodeScore(),
		PolyTail:     in.polyTail(),
	}, nil
}

func (in TagsOracleInput) barcodeScore() *bam.BarcodeScore {
	if in.Barcoding == nil {
		return nil
	}
	return &bam.BarcodeScore{
		Score:       in.Barcoding.BarcodeScore,
		TopStart:    in.Barcoding.TopStart,
		TopEnd:      in.Barcoding.TopEnd,
		TopScore:    in.Barcoding.TopScore,
		BottomStart: in.Barcoding.BottomStart,
		BottomEnd:   in.Barcoding.BottomEnd,
		BottomScore: in.Barcoding.BottomScore,
	}
}

func (in TagsOracleInput) polyTail() *bam.PolyTail {
	if in.RNA == nil {
		return nil
	}
	return &bam.PolyTail{
		Length: in.RNA.Length,
		Anchor: in.RNA.Anchor,
		Boundaries: [2][2]int32{
			{in.RNA.B0Start, in.RNA.B0End},
			{in.RNA.B1Start, in.RNA.B1End},
		},
	}
}

// findRNAPolyA is a harness helper for the qs input window, not part of the
// writer. It mirrors sequence_utils.cpp:142-166.
func findRNAPolyA(seq string) int {
	const search, minimum = 200, 5
	size := len(seq)
	end := 0
	if size > search {
		end = size - search
	}
	polyaSize, polyaEnd := 0, size
	for i := size; i > end; i-- {
		switch {
		case seq[i-1] == 'A':
			polyaSize++
			if polyaSize >= minimum {
				polyaEnd = i - 1
			}
		case polyaEnd != size:
			return polyaEnd
		default:
			polyaSize = 0
		}
	}
	return polyaEnd
}

func oracleHeader() bam.Header {
	return bam.Header{
		Version: "1.6", SortOrder: "unknown",
		Program: bam.Program{ID: "basecaller", Name: "dorado", Version: "2.1.2+8b8fc5d"},
	}
}

// goAuxBytes writes one record with the Go writer and returns its aux region,
// so the comparison is against real writer bytes, not a second encoder.
func goAuxBytes(header bam.Header, in TagsOracleInput, tags []bam.Aux) ([]byte, error) {
	qual := make([]byte, len(in.Seq))
	for i := range qual {
		qual[i] = 40
	}
	rec := bam.Record{Name: in.ReadID, Flag: 4, Sequence: in.Seq, Quality: qual, Aux: tags}
	var buf bytes.Buffer
	bw := bam.NewWriter(&buf)
	if err := bw.WriteHeader(header); err != nil {
		return nil, err
	}
	if err := bw.WriteRecord(rec); err != nil {
		return nil, err
	}
	if err := bw.Close(); err != nil {
		return nil, err
	}
	payload, _, err := decodeBGZF(buf.Bytes())
	if err != nil {
		return nil, err
	}
	return firstRecordAux(payload)
}

// firstRecordAux returns the aux region of the first BAM record.
func firstRecordAux(payload []byte) ([]byte, error) {
	if len(payload) < 12 || string(payload[:4]) != "BAM\x01" {
		return nil, errors.New("BAM magic is missing")
	}
	lText := int(int32(binary.LittleEndian.Uint32(payload[4:])))
	off := 8 + lText
	nRef := int(int32(binary.LittleEndian.Uint32(payload[off:])))
	off += 4
	for i := 0; i < nRef; i++ {
		if off+4 > len(payload) {
			return nil, errors.New("BAM reference table is truncated")
		}
		lName := int(int32(binary.LittleEndian.Uint32(payload[off:])))
		off += 4 + lName + 4
	}
	if off+4 > len(payload) {
		return nil, errors.New("BAM record is missing")
	}
	blockSize := int(int32(binary.LittleEndian.Uint32(payload[off:])))
	off += 4
	if blockSize < 32 || off+blockSize > len(payload) {
		return nil, errors.New("BAM record size is invalid")
	}
	rec := payload[off : off+blockSize]
	lReadName := int(rec[8])
	nCigar := int(binary.LittleEndian.Uint16(rec[12:]))
	lSeq := int(int32(binary.LittleEndian.Uint32(rec[16:])))
	auxStart := 32 + lReadName + nCigar*4 + (lSeq+1)/2 + lSeq
	if auxStart > len(rec) {
		return nil, errors.New("BAM record aux offset overruns the record")
	}
	return append([]byte(nil), rec[auxStart:]...), nil
}

func auxNames(tags []Aux) []string {
	out := make([]string, len(tags))
	for i, t := range tags {
		out[i] = t.Name
	}
	return out
}

func removeNames(names, drop []string) []string {
	dropSet := make(map[string]bool, len(drop))
	for _, d := range drop {
		dropSet[d] = true
	}
	out := make([]string, 0, len(names))
	for _, n := range names {
		if !dropSet[n] {
			out = append(out, n)
		}
	}
	return out
}

func firstAuxDifference(want, got []byte) string {
	n := len(want)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if want[i] != got[i] {
			lo := i - 8
			if lo < 0 {
				lo = 0
			}
			hi := i + 8
			if hi > n {
				hi = n
			}
			return fmt.Sprintf("first aux byte difference at %d: native %s go %s",
				i, hex.EncodeToString(want[lo:hi]), hex.EncodeToString(got[lo:hi]))
		}
	}
	return fmt.Sprintf("aux length differs: native %d go %d", len(want), len(got))
}

func htslibVersion() (string, error) {
	out, err := exec.Command("pkg-config", "--modversion", "htslib").Output()
	if err != nil {
		return "", fmt.Errorf("pkg-config htslib version: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}
