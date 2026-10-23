package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/caller"
)

type record struct{ Header, ID, Sequence, Qstring string }

func read(path string) (record, error) {
	f, err := os.Open(path)
	if err != nil {
		return record{}, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 4096), 4<<20)
	var lines []string
	for s.Scan() {
		lines = append(lines, s.Text())
	}
	if err := s.Err(); err != nil {
		return record{}, err
	}
	if len(lines) != 4 || !strings.HasPrefix(lines[0], "@") || !strings.HasPrefix(lines[2], "+") || len(lines[1]) != len(lines[3]) {
		return record{}, errors.New("expected exactly one complete FASTQ record")
	}
	fields := strings.Fields(lines[0][1:])
	if len(fields) == 0 {
		return record{}, errors.New("empty UUID")
	}
	return record{lines[0], fields[0], lines[1], lines[3]}, nil
}
func hash(data []byte) string { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }
func main() {
	actual := flag.String("go-fastq", "", "retained Go FASTQ")
	stock := flag.String("stock-fastq", "", "retained official CPU FASTQ")
	manifest := flag.String("manifest", "", "retained caller capture manifest")
	flag.Parse()
	if *actual == "" || *stock == "" || *manifest == "" {
		fmt.Fprintln(os.Stderr, "all three paths are required")
		os.Exit(2)
	}
	a, err := read(*actual)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	b, err := read(*stock)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	replay, err := caller.Replay(context.Background(), *manifest)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	exact := a.ID == b.ID && a.Sequence == b.Sequence && a.Qstring == b.Qstring
	replayExact := a.ID == replay.ID && a.Sequence == replay.Sequence && a.Qstring == replay.Qstring
	actualBytes, err := os.ReadFile(*actual)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	stockBytes, err := os.ReadFile(*stock)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var replayBytes bytes.Buffer
	if err := replay.WriteFASTQ(&replayBytes); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	rawExact := bytes.Equal(actualBytes, stockBytes)
	rawReplayExact := bytes.Equal(actualBytes, replayBytes.Bytes())
	report := map[string]any{"schema": 1, "endpoint": "raw FASTQ byte identity", "records_go": 1, "records_stock": 1, "id": a.ID, "bases_go": len(a.Sequence), "bases_stock": len(b.Sequence), "uuid_exact": a.ID == b.ID, "sequence_exact": a.Sequence == b.Sequence, "quality_exact": a.Qstring == b.Qstring, "canonical_endpoint_exact": exact, "capture_replay_exact": replayExact, "raw_header_exact": a.Header == b.Header, "go_header": a.Header, "stock_header": b.Header, "go_sequence_sha256": hash([]byte(a.Sequence)), "stock_sequence_sha256": hash([]byte(b.Sequence)), "go_quality_sha256": hash([]byte(a.Qstring)), "stock_quality_sha256": hash([]byte(b.Qstring)), "capture_moves_sha256": hash(replay.Moves), "native_moves_compared": false}
	report["schema"] = 2
	report["raw_fastq_exact"] = rawExact
	report["raw_capture_replay_exact"] = rawReplayExact
	report["go_fastq_sha256"] = hash(actualBytes)
	report["stock_fastq_sha256"] = hash(stockBytes)
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !rawExact || !rawReplayExact || !exact || !replayExact {
		os.Exit(1)
	}
}
