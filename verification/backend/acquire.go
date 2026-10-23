package backend

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const ReserveBytes uint64 = 30 << 30
const MaxExtractionBytes uint64 = 12 << 30

func freeBytes(path string) (uint64, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, err
	}
	return stat.Bavail * uint64(stat.Bsize), nil
}
func requireSpace(path string, extra uint64) error {
	free, err := freeBytes(path)
	if err != nil {
		return err
	}
	if free < ReserveBytes+extra {
		return fmt.Errorf("free bytes %d below reserve %d plus planned %d", free, ReserveBytes, extra)
	}
	return nil
}

type Acquisition struct {
	Family, StartedUTC, FinishedUTC  string
	ReserveBytes, MaxExtractionBytes uint64
	Artifacts                        []Acquired
}
type Acquired struct {
	Artifact        Artifact
	ExtractedRoot   string
	ExpandedBytes   uint64
	VerifiedArchive Check
	TreeSHA256      string
}

// Acquire retrieves the exact CUDA 12.8 stock-family packages unless another
// family is explicitly selected. It never modifies the stock experiment.
func Acquire(root, family string) (receipt Acquisition, err error) {
	return AcquireSelected(root, family, []string{"pod5", "koi", "torch"})
}

func AcquireSelected(root, family string, names []string) (receipt Acquisition, err error) {
	receipt = Acquisition{Family: family, StartedUTC: time.Now().UTC().Format(time.RFC3339), ReserveBytes: ReserveBytes, MaxExtractionBytes: MaxExtractionBytes}
	artifacts, err := Archives(family)
	if err != nil {
		return receipt, err
	}
	// Make the smaller input and kernel packages available before Torch.
	artifacts = []Artifact{artifacts[2], artifacts[1], artifacts[0]}
	selected := make(map[string]bool)
	for _, name := range names {
		if name != "pod5" && name != "koi" && name != "torch" {
			return receipt, fmt.Errorf("unsupported package %q", name)
		}
		selected[name] = true
	}
	if len(selected) == 0 {
		return receipt, fmt.Errorf("at least one package is required")
	}
	filtered := artifacts[:0]
	for _, item := range artifacts {
		if selected[item.Name] {
			filtered = append(filtered, item)
		}
	}
	artifacts = filtered
	if err = os.MkdirAll(filepath.Join(root, "archives"), 0755); err != nil {
		return receipt, err
	}
	var total uint64
	var plannedExtraction uint64
	for _, item := range artifacts {
		if CheckFile(filepath.Join(root, "archives", item.Path), item).Status != "verified" {
			total += uint64(item.Bytes)
		}
		if _, err := os.Stat(filepath.Join(root, item.Name)); err != nil {
			plannedExtraction = MaxExtractionBytes
		}
	}
	if err = requireSpace(root, total+plannedExtraction); err != nil {
		return receipt, err
	}
	defer func() {
		receipt.FinishedUTC = time.Now().UTC().Format(time.RFC3339)
		data, _ := json.MarshalIndent(receipt, "", "  ")
		_ = os.WriteFile(filepath.Join(root, "acquisition.json"), data, 0600)
	}()
	for _, item := range artifacts {
		archive := filepath.Join(root, "archives", item.Path)
		if check := CheckFile(archive, item); check.Status != "verified" {
			if _, statErr := os.Stat(archive); statErr == nil {
				return receipt, fmt.Errorf("existing archive failed pin: %s", archive)
			}
			partial := archive + ".partial"
			if err = requireSpace(root, uint64(item.Bytes)+MaxExtractionBytes); err != nil {
				return receipt, err
			}
			command := exec.Command("curl", "--fail", "--location", "--max-time", "180", "--continue-at", "-", "--output", partial, item.URL)
			log, openErr := os.OpenFile(filepath.Join(root, item.Name+"-curl.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
			if openErr != nil {
				return receipt, openErr
			}
			fmt.Fprintf(log, "%s command %q\n", time.Now().UTC().Format(time.RFC3339), command.Args)
			command.Stdout = log
			command.Stderr = log
			runErr := command.Run()
			fmt.Fprintf(log, "%s result %v\n", time.Now().UTC().Format(time.RFC3339), runErr)
			log.Close()
			if runErr != nil {
				return receipt, runErr
			}
			if check := CheckFile(partial, item); check.Status != "verified" {
				return receipt, fmt.Errorf("download pin rejected: %+v", check)
			}
			if err = os.Rename(partial, archive); err != nil {
				return receipt, err
			}
		}
		verified := CheckFile(archive, item)
		destination := filepath.Join(root, item.Name)
		extractionReceipt := filepath.Join(root, item.Name+"-extraction.json")
		if _, statErr := os.Stat(destination); statErr == nil {
			previous, readErr := readExtraction(destination, extractionReceipt, item)
			if readErr != nil {
				return receipt, readErr
			}
			receipt.Artifacts = append(receipt.Artifacts, previous)
			continue
		}
		expanded, sizeErr := ExpandedBytes(archive)
		if sizeErr != nil {
			return receipt, sizeErr
		}
		if expanded > MaxExtractionBytes {
			return receipt, fmt.Errorf("extraction exceeds declared maximum: %d", expanded)
		}
		if err = requireSpace(root, expanded+(1<<30)); err != nil {
			return receipt, err
		}
		stage, stageErr := os.MkdirTemp(root, "extract-"+item.Name+"-")
		if stageErr != nil {
			return receipt, stageErr
		}
		command := exec.Command("bsdtar", "-xf", archive, "-C", stage)
		output, runErr := command.CombinedOutput()
		if runErr != nil {
			return receipt, fmt.Errorf("extract %s: %w: %s", item.Name, runErr, output)
		}
		marker := map[string]string{"torch": "include/ATen/ATen.h", "koi": "include/koi.h", "pod5": "include/pod5_format/c_api.h"}[item.Name]
		selected := ""
		walkErr := filepath.WalkDir(stage, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if _, err := os.Stat(filepath.Join(path, marker)); err == nil {
					if selected != "" {
						return fmt.Errorf("ambiguous package roots")
					}
					selected = path
				}
			}
			return nil
		})
		if walkErr != nil {
			return receipt, walkErr
		}
		if selected == "" {
			return receipt, fmt.Errorf("missing package root marker %s", marker)
		}
		digest, treeErr := TreeDigest(selected)
		if treeErr != nil {
			return receipt, treeErr
		}
		acquired := Acquired{item, destination, expanded, verified, digest}
		if err = publishExtraction(selected, destination, acquired); err != nil {
			return receipt, err
		}
		if err = requireSpace(root, 0); err != nil {
			return receipt, err
		}
		data, marshalErr := json.MarshalIndent(acquired, "", "  ")
		if marshalErr != nil {
			return receipt, marshalErr
		}
		if err = os.WriteFile(extractionReceipt, data, 0600); err != nil {
			return receipt, err
		}
		receipt.Artifacts = append(receipt.Artifacts, acquired)
	}
	return receipt, nil
}

const embeddedReceipt = ".acquisition-receipt.json"

// The receipt travels with the staged payload in the same atomic rename.
// The sibling receipt is a projection; reuse never requires its publication.
func publishExtraction(stage, destination string, acquired Acquired) error {
	data, err := json.MarshalIndent(acquired, "", "  ")
	if err != nil {
		return err
	}
	file, err := os.OpenFile(filepath.Join(stage, embeddedReceipt), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err = file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	return os.Rename(stage, destination)
}

func readExtraction(destination, legacyReceipt string, item Artifact) (Acquired, error) {
	var acquired Acquired
	data, err := os.ReadFile(filepath.Join(destination, embeddedReceipt))
	if os.IsNotExist(err) {
		data, err = os.ReadFile(legacyReceipt)
	}
	if err != nil {
		return acquired, fmt.Errorf("existing extraction lacks receipt: %w", err)
	}
	if err = json.Unmarshal(data, &acquired); err != nil {
		return acquired, err
	}
	digest, err := TreeDigest(destination)
	if err != nil {
		return acquired, err
	}
	if acquired.Artifact.SHA256 != item.SHA256 || acquired.TreeSHA256 != digest {
		return acquired, fmt.Errorf("existing extracted tree differs from verified receipt")
	}
	return acquired, nil
}

func TreeDigest(root string) (string, error) {
	hash := sha256.New()
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if relative == embeddedReceipt {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		fmt.Fprintf(hash, "%q %v\n", relative, info.Mode())
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			fmt.Fprintf(hash, "link %q\n", target)
			return nil
		}
		if info.Mode().IsRegular() {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(hash, file)
			file.Close()
			return copyErr
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// ExpandedBytes inspects archive metadata before extraction and rejects unsafe
// names. Vendor content must first pass its complete archive checksum.
func ExpandedBytes(path string) (uint64, error) {
	var total uint64
	add := func(name string, size int64) error {
		clean := filepath.Clean(name)
		if filepath.IsAbs(name) || clean == ".." || strings.HasPrefix(clean, "../") || size < 0 {
			return fmt.Errorf("unsafe archive member %q", name)
		}
		total += uint64(size)
		if total > MaxExtractionBytes {
			return fmt.Errorf("expanded archive exceeds maximum")
		}
		return nil
	}
	if strings.HasSuffix(path, ".zip") {
		reader, err := zip.OpenReader(path)
		if err != nil {
			return 0, err
		}
		defer reader.Close()
		for _, entry := range reader.File {
			if err := add(entry.Name, int64(entry.UncompressedSize64)); err != nil {
				return 0, err
			}
		}
		return total, nil
	}
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return 0, err
	}
	defer gzipReader.Close()
	reader := tar.NewReader(gzipReader)
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, err
		}
		if err := add(header.Name, header.Size); err != nil {
			return 0, err
		}
		if header.Typeflag == tar.TypeSymlink || header.Typeflag == tar.TypeLink {
			target := filepath.Clean(filepath.Join(filepath.Dir(header.Name), header.Linkname))
			if filepath.IsAbs(header.Linkname) || target == ".." || strings.HasPrefix(target, "../") {
				return 0, fmt.Errorf("unsafe archive link %q", header.Linkname)
			}
		}
	}
	return total, nil
}
