# Native default simplex DNA pipeline

This source investigation covers the post-stitch path for 5000 Hz DNA, FLO-PRO114M, and SQK-LSK114. It does not implement runtime behavior or qualify default-mode parity.

Observed on 2026-10-22 America/Denver. The Go checkout HEAD is `086f2361b7b4900ddcf0ceb1cdbfad1cf7a9c1e2`, dated `2026-10-22T21:02:53-06:00`, with existing uncommitted caller changes. Native source is `external/dorado-source` at `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`, dated `2026-08-26T11:54:52Z`. Its working tree was clean. Native paths below are relative to that source root. Go paths are relative to this checkout.

Throughput checkpoint is n/a for this read-only investigation. The assigned document is the only write.

## Overview

The missing default DNA behavior comprises six computational subsystems. In native execution order they are mux-change tail trimming, pore detection and adapter-supported splitting, child materialization, adapter and primer detection, synchronized sequence and move trimming, and final filtering. FASTQ metadata integration is a seventh delivery unit that consumes their outputs.

The native path is scaling, chunk inference and stitching, mux-change tail trimming, DNA splitting, optional modified-base inference, adapter and primer detection, optional barcode classification and poly-A estimation, trimming, filtering, and BAM conversion followed by the writer. `api/pipeline_creation.cpp:76-118` establishes scaling through post-call splitting. `read_pipeline/nodes/BasecallerNode.cpp:253-270` places mux-change trimming after stitching. `cli/cli_lib/basecaller.cpp:625-680` constructs the downstream graph backwards from its sink.

DNA splitting does not rerun basecalling. RNA splits raw signal before scaling and inference instead. `api/pipeline_creation.cpp:66-102` makes that distinction explicit.

The current Go caller ends after `stitch.Stitch` in `internal/caller/caller.go:245-255`. Its CLI requires `--no-trim --disable-read-splitting`, as documented in `docs/caller.md:108-116`. Those flags are a valid bounded contract, but native `--no-trim` does not disable mux-change trimming.

## Defaults and kit selection

`cli/cli_lib/basecaller.cpp:330-344` enables adapter, primer, and barcode trimming unless `--no-trim` or `--trim none` is supplied. `--trim adapters` disables primers and barcodes. `--disable-read-splitting` is false unless supplied, so splitting is enabled by default at line 1190. `--min-qscore` defaults to zero at lines 170-174. The minimum sequence length is five in `utils/include/utils/parameters.h:14`.

Barcoding does not activate merely because the POD5 has a sequencing kit. `validate_infos` creates `BarcodingInfo` only when `--kit-name` is used, at `cli/cli_lib/basecaller.cpp:348-378`. The target nonbarcoded SQK-LSK114 path therefore has no barcode classifier. Barcode-both-ends behavior is outside this target.

The adapter detector reads `ReadCommon.sequencing_kit`, uppercases it, and substitutes `ANY` for an empty value. `read_pipeline/nodes/AdapterDetectorNode.cpp:153-171` owns this selection. `ANY` does not mean all stock adapters. Only custom entries without kit restrictions populate it, as described and implemented in `demux/adapter_primer_kits.cpp:291-328`. Missing-kit reads must not silently use SQK-LSK114 sequences.

`models/kits.cpp:198-206,377-379` maps FLO-PRO114M and SQK-LSK114 into the 5000 Hz DNA chemistry. SQK-LSK114 is not rapid chemistry. Its `KitInfo` uses the default `ProtocolInfo`, whose `rapid_chemistry` is `NONE` in `models/include/models/kits.h:152-165`.

### Stock sequences for SQK-LSK114

| Purpose | Front sequence | Rear sequence | Source |
| --- | --- | --- | --- |
| End adapter LSK110 | `CCTGTACTTCGTTCAGTTACGTATTGC` | `AGCAATACGTAACTGAAC` | `demux/adapter_primer_kits.cpp:29-47` |
| cDNA primer | `TTTCTGTTGGTGCTGATATTGCTGGG` | `ACTTGCCTGTCGCTCTATCTTCTTT` | `demux/adapter_primer_kits.cpp:57-63,91-95` |
| Internal split adapter | `TACTTCGTTCAGTTACGTATTGCT` | Not used | `splitter/include/splitter/ReadSplitter.h:85-95` |

The end adapter sequence differs from the internal splitting sequence. They must have separate constants and pins. The cDNA primer really is mapped to SQK-LSK114, even for this genomic DNA target. Default matching searches it. GEN10X is also in the kit table, but `demux/adapter_primer_kits.cpp:332-347` excludes it unless the caller requests `PrimerAux::GEN10X`. `demux/adapter_info.cpp:15-29` supplies that special mapping. The default is `PrimerAux::DEFAULT` in `demux/include/demux/adapter_info.h:26`.

For primers, the detector generates forward queries using front and reverse-complemented rear sequences, and reverse queries using rear and reverse-complemented front sequences. `demux/AdapterDetector.cpp:178-201` owns this transformation. End adapters are used directly.

## Six computational subsystems

### 1. Mux-change tail trimming

The loader sets `is_end_reason_mux_change` for both `POD5_END_REASON_MUX_CHANGE` and `POD5_END_REASON_UNBLOCK_MUX_CHANGE` in `data_loader/DataLoader.cpp:226-239`. `BasecallerNode` always invokes `mux_change_trim_read`, including when adapter trimming is disabled.

`read_pipeline/base/read_utils.cpp:68-165` implements the algorithm. It scans quality backwards with scores -1 for Q at most 7, +1 for Q above 7 through 12, and +10 for higher Q. It tracks the last minimum with `<=`, uses the source's `i - 1` index, and has an early exit. Reads shorter than 100 bases are unchanged. Trimming is skipped when the candidate index is below `floor(length * 0.3f)` or is at least `length - 5`. Preserve these exact index conventions rather than interpreting the comments as a new algorithm.

The accepted index slices sequence and qstring to `[0,index)`. `utils::sequence_to_move_table_index` at `utils/sequence_utils.cpp:199-246` locates the corresponding move boundary. The algorithm shrinks moves and normalized signal, updates `attributes.num_samples` and `pre_trim_seq_length`, and zeros `num_minknow_events`. It does not increase `num_trimmed_samples` for this right trim.

The current Go `caller.go` has no mux-change tail call. This is an observed source gap. The two retained captures use end reason `signal_positive`, so this gap does not imply their FASTQ mismatch.

### 2. Simplex pore and internal-adapter detection

`api/pipeline_creation.cpp:92-101` constructs `DuplexReadSplitter` with `simplex_mode = true`. For PA scaling, its pore threshold is 2.8, rather than the other scaling default 2.4. Both retained HAC and SUP model configs specify PA scaling.

The active settings are in `splitter/include/splitter/ReadSplitter.h:46-95`.

| Setting | Value |
| --- | --- |
| Signal type | Native binary16 normalized samples |
| Ignore prefix | 5000 samples |
| Pore comparison | Strictly greater than binary16 threshold 2.8 |
| Cluster merge distance | 500 samples |
| Minimum spike duration | 0 samples |
| Maximum pore region | 500 bases before argmax reduction |
| Candidate coordinate | Peak argmax base, interval of one base |
| Quality window | Five bases, clipped at the sequence end |
| Maximum local quality | Comparison rejects values greater than `10 - float epsilon` |
| Maximum candidates | Ten, selected by peak signal and native pair ordering |
| Adapter edit distance | At most four |
| Adapter search span | Pore interval start through interval end plus 50 bases, clipped to the read |

`splitter/splitter_utils.cpp:125-300` detects contiguous above-threshold runs and merges nearby runs. Equal signal peaks choose the later argmax, including when clusters merge, at lines 256-258 and 293-295. Preserve the binary16 comparison path and tie behavior.

`splitter/DuplexReadSplitter.cpp:128-199` converts sample regions through inclusive move cumulative sums, rejects regions whose end move index is out of range or whose start has no called base, then uses `cum_sum - 1` for the argmax base. Local quality uses the arithmetic mean of numeric Phred scores at lines 106-117. It does not use the read's error-probability mean quality.

`check_nearby_adapter` at lines 331-340 calls the original edlib HW, LOC match at lines 29-53. It accepts the first native alignment location when edit distance is within four. `apply_split_finders` at lines 534-596 runs `PORE_ADAPTER` for this target. `muA_adapter` runs only for rapid V1 chemistry. `PORE_FLANK`, `PORE_ALL`, `ADAPTER_FLANK`, `ADAPTER_MIDDLE`, and `SPLIT_MIDDLE` are guarded by `!simplex_mode` and are outside the target. The class name does not mean those duplex heuristics are required for simplex support.

### 3. Child materialization and identity

`DuplexReadSplitter::subreads` at `splitter/DuplexReadSplitter.cpp:496-531` builds a base-to-signal map with the original `utils::moves_to_map` at `utils/sequence_utils.cpp:250-266`. Each move equal to one contributes its signal index times stride, and the last map entry is the original normalized signal length. Child sequence intervals exclude spacer intervals. Empty base or whole-stride signal intervals are not emitted. The last child can end on a partial stride.

`splitter/splitter_utils.cpp:34-109` owns child state. It slices sequence, qstring, moves, and signal. It sets `read_number = -1`, `num_minknow_events = 0`, and `pre_trim_seq_length` to the child sequence length. It accumulates `split_point` as prior split point plus prior leading trim plus child signal start. It resets `num_trimmed_samples` to zero and advances absolute `start_sample` by that same leading trim plus child signal start. It updates `end_sample`, sample count, and start time. Child start time uses rounded relative milliseconds before adding acquisition start, unlike the loader's integer floor for unsplit reads.

The child inherits the root `parent_read_id` if one exists, otherwise the input read ID. Its end reason becomes `unknown`, with the original end reason restored on the final child by `DuplexReadSplitter.cpp:617-627`. `split` assigns sequential child indices and final child count at lines 650-661 and reconnects neighbor IDs at lines 666-680. No-split reads retain their original identity and state.

`utils/uuid_utils.cpp:12-38` derives each child UUID from SHA256 of the parent UUID string concatenated directly with the decimal child index. It takes the first 16 digest bytes, sets version bits to 4 and RFC variant bits to 10, then emits lowercase UUID text. It is deterministic and is not UUIDv5 or a random UUID generator. Linux hashing is OpenSSL SHA256 in `utils/crypto_utils.cpp:11-24`.

Pinned behavior to preserve explicitly is that `utils::shallow_copy_read` at `read_pipeline/base/read_utils.cpp:18-65` does not copy `mean_qscore_start_pos`. A newly constructed child's value is zero, per `read_pipeline/base/include/read_pipeline/base/messages/ReadCommon.h:98`. The original read has the model's offset, set by `BasecallerNode.cpp:255`. Both retained model run pins have `MeanQStart = 60`. Split children therefore use zero for mean-quality calculation at this source revision. This conclusion is a source inference from construction and assignment, not a runtime differential result.

### 4. End adapter and primer detection

`demux/AdapterDetector.cpp:40-58,74-130,204-235` uses edlib HW mode and LOC task with N equality to A, T, C, and G. Adapter windows are the first and last 75 bases. Primer windows are the first and last 150 bases. Score is binary32 `1 - editDistance / queryLength`. Alignment positions are inclusive. Native matching chooses the first edlib location. Across query candidates, a score improvement greater than 0.1 wins outright. Within 0.1, a longer alignment span wins.

`demux/Trimmer.cpp:93-124` accepts scores at least 0.8. The retained interval begins after the front match's inclusive end and ends at the rear match's start. An empty or inverted interval resets to the full sequence and unclassifies both ends. `AdapterDetectorNode.cpp:163-187` computes adapter and primer intervals separately and intersects them. If their intersection is invalid, it leaves the read untrimmed and increments the short-read count.

`AdapterDetector::classify_primers` at `demux/AdapterDetector.cpp:238-285` classifies forward or reverse orientation when the accepted end names agree, or when only one end is classified. cDNA defaults have no UMI extraction. PCS110 UMI and GEN10X extensions are outside the target.

### 5. Synchronized trim application

`read_pipeline/nodes/TrimmerNode.cpp:24-74,152-171` combines requested intervals. Without barcoding, the adapter and primer intersection is the only interval. Invalid intersections skip trimming.

`demux/Trimmer.cpp:218-265` slices sequence and qstring together. `torch_utils/trim.cpp:68-90` slices moves while retaining zero moves associated with retained bases. Leading removed move positions times stride increase `num_trimmed_samples`. The remaining normalized signal begins at that leading sample offset and has `remaining_moves * stride` samples, subject to the original tensor slice behavior at a partial final stride. Plain DNA does not reverse move coordinates. RNA does.

Trimming does not shift the original read's `start_sample` or `start_time`. Split materialization already established child start time, and later adapter trimming adds to the child's leading trim count. The source leaves `pre_trim_seq_length` unchanged during this adapter trim. Do not reuse it as the current sequence length.

### 6. Final filtering and quality

`read_pipeline/nodes/ReadFilterNode.cpp:31-38` drops reads with sequence length below five, or with mean quality strictly below a positive requested threshold. Equality passes. The default Q threshold zero disables Q filtering. There is no default Q7 or Q10 cutoff. The Q10 comparison in the splitter is a separate local candidate test.

`ReadCommon::calculate_mean_qscore` at `read_pipeline/base/messages.cpp:419-422` skips `mean_qscore_start_pos` bases only when the sequence is longer than that offset. For a shorter or equal sequence it uses the whole qstring. `utils/sequence_utils.cpp:169-190` averages error probabilities in float32, converts using `-10*log10`, and clamps nonempty means to [1,50]. Empty strings yield zero.

With `--output-dir`, the native filter receives Q threshold zero, and the converter records pass or fail using the requested threshold instead. See `cli/cli_lib/basecaller.cpp:630-635` and `read_pipeline/nodes/ReadToBamTypeNode.cpp:30-32`. That output-directory routing is not needed to add a default streaming FASTQ endpoint, but the API must not conflate dropping with fail classification.

## Metadata and candidate Go interfaces

These interfaces are a recommendation, not an implementation decision. Keep post-call transformations pure Go, with the native executable confined to verification. The post-stitch read needs the sequence, quality, binary move table, normalized binary16 signal or its typed view, stride, current signal length, leading sample trim count, kit, end reason, root and child identity, sample clocks, primer classification, and quality-start offset. `CalledRead` alone currently lacks normalized signal and child state.

A minimal typed API can expose `Postprocess(read, policy) ([]Read, error)`, plus pure independently testable `MuxChangeTrim`, `FindSplits`, `MaterializeChildren`, `DetectEnds`, `ApplyTrim`, and `Filter` functions. `Read` owns the current signal interval and metadata. `Interval` uses half-open coordinates with distinct base and sample types. `Policy` contains the pinned DNA mode, adapter and primer flags, split flag, and optional quality threshold. End detection returns scores, names, inclusive alignment coordinates, retained interval, and primer orientation so differential tests can inspect decisions before final FASTQ formatting.

Retain the existing caller stage for an untrimmed and unsplit result. Compose the new pipeline immediately after stitching and before FASTQ metadata construction, without duplicating POD5 decoding, inference, CRF, beam search, or chunk stitching. `internal/pod5/reader.go:45-73` already exposes the sequencing kit, end reason, acquisition clock, start and end samples, read number, sample rate, and run metadata needed for the target. `internal/caller/capture.go:140-165` already retains the raw and normalized signal and those metadata.

`read_pipeline/base/messages.cpp:108-182` shows the metadata contract. Mean quality uses the transformed qstring. `ns` is current normalized signal length plus leading trim count. `ts` is leading trim count. `mv` starts with model stride. Split reads add `pi` and `sp`, use `rn = -1`, and have their own start time. Primer orientation adds `TS`. Only the selected FASTQ tags are emitted by `hts_utils/hts_file.cpp:29-30`, so not every BAM tag appears in FASTQ. Default trimming must also update the writer's trim flags and resulting `tm` tag through the existing native writer contract.

## Reusable native oracles without inference

Existing pattern is `verification/caller/oracle.go:58-156`, which validates source hashes, extracts whole source functions verbatim, compiles them with test-only container shims, records build receipts, and uses native HTSlib. Extend that pattern in a separate verification package rather than calling the stock basecaller or copying algorithms into an independent handwritten oracle.

| Oracle unit | Reusable native source | Dependency boundary |
| --- | --- | --- |
| Mux tail | `find_mux_change_trim_seq_index`, `mux_change_trim_read`, `sequence_to_move_table_index`, and trim helpers | Small read container and real ATen signal slicing, or an audited shim for slice-only storage |
| Pore candidates | `detect_pore_signal<c10::Half>`, `move_cum_sums`, and `possible_pore_regions` | Real ATen binary16 tensor preserves compare and argmax semantics |
| Split decisions | Original `DuplexReadSplitter.cpp` with simplex settings, plus `splitter_utils.cpp` | ATen, edlib, read construction, sequence helpers, logger, UUID and SHA256 helpers |
| Child identity and slicing | `subreads`, `subread`, `shallow_copy_read`, `derive_uuid`, `time_utils.cpp` | ATen, OpenSSL SHA256, exact metadata containers |
| End detection | `AdapterDetector` alignment, scoring, detection, classification, default candidate tables, and `determine_trim_interval` | edlib. Whole detector files include HTS headers but sequence alignment itself needs no reference or BAM I/O |
| Trim and filter | Original SimplexRead `trim_sequence`, `trim_move_table`, quality functions, and exact filter predicate | ATen slicing, source-sliced read container, existing mean-quality oracle |
| FASTQ output | Extend `verification/caller/oracle.cpp` with real trim flags and transformed metadata | System HTSlib and original writer source slices |

Full unchanged translation units provide stronger reuse where practical. Whole `DuplexReadSplitter.cpp` also references rapid and duplex functions even when runtime settings disable them, so build/link must satisfy those functions or use exact source slices with recorded extraction boundaries. Whole `demux/Trimmer.cpp` contains BAM and modified-base branches, and whole `sequence_utils.cpp` includes minimap functionality. Those file-level dependencies do not imply that reference alignment or modified-base inference belongs in the target runtime. A source-slice oracle must retain each original function byte-for-byte, hash the source, slice, shim, compiler flags, and dependency inputs, and make unsupported branches fail explicitly. Do not silently rewrite the method to remove unwanted branches.

Local dependency observations were read-only. The Go source submodule has empty edlib, spdlog, and HTSlib submodule directories. `pkg-config --modversion htslib` succeeds with 1.24. `pkg-config --modversion edlib` fails because no system edlib pkg-config entry is installed. The pinned native edlib gitlink is `42aa8fa7051fb2aa1c4f144c5eda22a0e753f026`.

An existing read-only edlib checkout at `/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/source-input/dorado/3rdparty/edlib` reports that exact revision. Its original `edlib/src/edlib.cpp` and `edlib/include/edlib.h` are available. Compile those into a new ignored oracle output if needed. A retained archive also exists at `software/source-build-qualification-20261023/build-gcc15/lib/libedlib.a`, but its filename alone does not establish input hashes or compiler identity. Native `dorado/3rdparty/patches/edlib.patch` changes the minimum CMake version only.

The qualified CPU LibTorch root is recorded in `artifacts/tensor/attempt-005-sup-abi3/manifest.json:4`. Its path is `/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch`. `libtorch_cpu.so` and `libc10.so` are present. Reuse the manifest's pins and the build flags in `verification/tensor/build/main.go:189-191`. No GPU, model weights, download, native basecaller process, or new inference run is needed to build these post-call oracles.

### Retained input fixtures and evidence gaps

Both existing caller captures have UUID `bb76c31a-36f0-48a0-abd2-107a10e1ade0`, 19,856 raw samples, ten leading scaling-trimmed samples, and 19,846 normalized binary16 samples. Their model run pins identify HAC v6.0.0 and SUP v5.2.0. Their end reason is `signal_positive`, kit is `sqk-lsk114`, and flowcell product is `FLO-PRO114M`.

- HAC is `artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json`.
- SUP is `artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json`.

Both decoded `output.json` move tables contain 3307 bytes. Go encodes `[]uint8` as base64 in JSON, so string length is not move count. Their native FASTQ controls do not contain moves. `internal/caller/native_test.go:63` and `verification/caller/cmd/main.go:89` expressly leave native moves unqualified. Before claiming native split parity, run the original decoder and stitch source oracles on retained chunk data to compare moves, then run the split oracle on the normalized signal. This is a prerequisite in evidence, not an observed move defect.

These two genuine captures alone do not prove positive splitting, mux-change trimming, short-read filtering, or every adapter and primer orientation. Synthetic controls must drive those branches. The post-call oracle has not been built or run in this investigation, so no default-mode output or default-mode parity is claimed.

## Ordered verifiable delivery units

The seven units below deliver the six computations and their consumer contract. Each unit requires original-source differential output and mutation rejection before the next integration claim.

1. Establish the post-stitch input shape and native move parity from retained chunk captures. Verify normalized bits and payload hashes. Reject altered signal, stride, moves, and source pin. Keep unsupported kit and sample type explicit.
2. Port mux-change trimming and compare exact index, sequence, quality, moves, signal range, and metadata. Include non-mux identity, short reads, threshold boundaries, a positive low-quality tail, excessive trim rejection, and ties.
3. Port simplex PA pore detection and adapter-supported split decisions. Include binary16 threshold equality, prefix boundary, equal peaks, cluster-gap boundaries, quality threshold, top-ten ties, and adapter edit-distance four versus five. Prove rapid and duplex-only branches remain outside the accepted domain.
4. Port child materialization and deterministic UUIDs. Compare all child ranges, remaining samples, root parent IDs, cumulative split points, qscore offset zero, final end reason, neighbor links, rounded clocks, and partial-stride tails.
5. Port default SQK-LSK114 adapter and cDNA primer detection. Compare raw scores, coordinates, selected query names, orientation, and retained interval. Include no-match, N equality, threshold, forward and reverse primers, overlapping windows, conflicting intervals, missing kit, and GEN10X rejection by default.
6. Port synchronized trimming and final length and quality filtering. Compare raw signal slice, moves, sequence, quality, `ns`, `ts`, and mean-quality bits. Test lengths four and five, threshold equality, model offset fallback, child offset zero, and default Q filtering disabled.
7. Compose default streaming FASTQ and CLI modes. Preserve the disabled-mode endpoint, add real trim flags and child metadata to the pinned formatter oracle, and compare full FASTQ bytes from both retained HAC and SUP captures. Check zero, one, and multiple output records, plus filtered children. Keep structured output-directory pass/fail routing outside the claim until separately implemented.

### Immediate mux-change package proposal

The smallest next owned package can be `internal/readtrim`, initially containing only mux-change behavior. A candidate input is `MuxRead{Sequence string, Qstring string, Moves []uint8, Signal []uint16, Stride uint32, IsMuxChange bool, NumMinknowEvents uint64, PreTrimSequenceLength uint64}`. The signal holds normalized binary16 bits, not raw POD5 samples. Return `MuxResult{Read MuxRead, Trimmed bool, SequenceEnd int, MoveEnd int, SignalEnd int}`. Sample clocks, leading scaling trim, UUID, and quality-start offset remain caller-owned because this transform does not alter them. Validate malformed inputs at the exported boundary, and preserve source no-op cases for valid reads.

The initial oracle needs four complete source functions. Extract `find_mux_change_trim_seq_index` and `mux_change_trim_read` from `read_pipeline/base/read_utils.cpp:68-165`, `sequence_to_move_table_index` from `utils/sequence_utils.cpp:201-247`, and string `trim_sequence` from `torch_utils/trim.cpp:58-66`. Provide only the read fields those functions access, a no-op trace logger, and either genuine CPU ATen tensors or a recorded slice-only tensor shim. Genuine ATen avoids having to prove the shim's slice semantics and can reuse the installed CPU library. This unit needs no edlib, HTSlib, modified-base code, reference alignment, model graph, or weights. Hash full source files before extracting exact functions, then hash extracted bytes and the driver.

The genuine HAC and SUP captures supply non-mux identity controls only. Their `signal_positive` end reasons do not exercise native mux-change trimming. Positive and threshold controls must construct explicit synthetic read metadata and compare the original source result with Go. The package's first acceptance should be a focused `go test` command once that package exists, followed by source mutation rejection and retained-capture identity checks. No new inference is justified for this unit.

Cheap current commands for the retained endpoint are the existing `verification/caller/verify.sh` invocations with `CALLER_CAPTURE_MANIFEST`, `CALLER_STOCK_FASTQ`, and `DORADO_SOURCE` set. That script performs replay, source-pinned metadata and decoder checks, mutation rejection, and full FASTQ comparison without inference. Its present success cannot prove the missing default path.

Future post-call acceptance should be a new source-pinned `go test` package and a replay-only command that accepts a retained capture manifest. This report does not invent an executable name that exists today. Each new test must record its exact native source slices, compiler and dependency hashes, original output, Go output, command clocks, and exit statuses under a fresh ignored attempt directory. Use `go test ./...` after integration, plus the maintained retained HAC and SUP consumer checks. No producer rerun is needed.

## Investigation checks

Read-only checks included `git status --short`, native and Go `git rev-parse HEAD` and `git log -1 --format='%H %cI'`, targeted `rg` and numbered source reads, `git -C external/dorado-source status --short`, `git -C external/dorado-source ls-tree HEAD dorado/3rdparty/edlib`, read-only edlib `rev-parse HEAD`, the two pkg-config probes, retained JSON field inspection, and `jq -r .moves .../output.json | base64 -d | wc -c` for each model capture. Both native source status and revision checks succeeded. The system edlib pkg-config probe failed as recorded above, and an exact-revision local source alternative was found. No runtime tests, inference, source edits, downloads, or commits ran.
