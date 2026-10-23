package backend

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedFileRejectsMissingMutationAndWrongSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture")
	hash := sha256.Sum256([]byte("genuine producer bytes"))
	item := Artifact{Name: "fixture", SHA256: hex.EncodeToString(hash[:]), Bytes: 22}
	if got := CheckFile(path, item); got.Status != "missing" {
		t.Fatal(got)
	}
	if err := os.WriteFile(path, []byte("genuine producer bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := CheckFile(path, item); got.Status != "verified" {
		t.Fatal(got)
	}
	item.Bytes++
	if got := CheckFile(path, item); got.Status != "invalid" {
		t.Fatal(got)
	}
	item.Bytes--
	if err := os.WriteFile(path, []byte("mutated producer bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := CheckFile(path, item); got.Status != "invalid" {
		t.Fatal(got)
	}
}

func TestMissingBackendRejected(t *testing.T) {
	root := t.TempDir()
	report := Inspect(root, root, root, "12.8")
	if report.MissingOrInvalid == 0 || report.BackendQualified || report.Status != "missing_or_invalid_prerequisites" {
		t.Fatalf("false acceptance %+v", report)
	}
	if _, err := Archives("13.4"); err == nil {
		t.Fatal("accepted unpinned family")
	}
	for _, family := range []string{"12.8", "13.0"} {
		items, err := Archives(family)
		if err != nil || len(items) != 3 {
			t.Fatal(items, err)
		}
		for _, item := range items {
			if len(item.SHA256) != 64 || item.Bytes <= 0 || item.URL == "" {
				t.Fatal(item)
			}
		}
	}
}

func TestArchiveTraversalRejectedBeforeExtraction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unsafe.zip")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	entry, err := writer.Create("../escape")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := entry.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	file.Close()
	if _, err := ExpandedBytes(path); err == nil {
		t.Fatal("accepted traversal")
	}
}

func TestExtractionDigestDetectsMutationAndLinkChange(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "header.h")
	if err := os.WriteFile(file, []byte("header"), 0600); err != nil {
		t.Fatal(err)
	}
	before, err := TreeDigest(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("mutant"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := TreeDigest(root)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("tree digest ignored mutation")
	}
	if err := os.Symlink("header.h", filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	linked, err := TreeDigest(root)
	if err != nil {
		t.Fatal(err)
	}
	if linked == after {
		t.Fatal("tree digest ignored link")
	}
}

func TestPublishedExtractionResumesWithoutSidecar(t *testing.T) {
	root := t.TempDir()
	stage := filepath.Join(root, "stage")
	if err := os.Mkdir(stage, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stage, "header.h"), []byte("genuine header"), 0600); err != nil {
		t.Fatal(err)
	}
	digest, err := TreeDigest(stage)
	if err != nil {
		t.Fatal(err)
	}
	item := Artifact{Name: "fixture", SHA256: "verified-archive-pin"}
	destination := filepath.Join(root, "published")
	acquired := Acquired{Artifact: item, ExtractedRoot: destination, TreeSHA256: digest}
	if err := publishExtraction(stage, destination, acquired); err != nil {
		t.Fatal(err)
	}
	sidecar := filepath.Join(root, "fixture-extraction.json")
	if _, err := os.Stat(sidecar); !os.IsNotExist(err) {
		t.Fatalf("test must simulate interrupted sidecar publication: %v", err)
	}
	reused, err := readExtraction(destination, sidecar, item)
	if err != nil || reused.TreeSHA256 != digest {
		t.Fatalf("published payload could not resume without sidecar: %+v %v", reused, err)
	}
	if err := os.WriteFile(filepath.Join(destination, "header.h"), []byte("mutated header"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readExtraction(destination, sidecar, item); err == nil {
		t.Fatal("published mutation bypassed embedded receipt")
	}
}

func TestInterruptedStagingNeverPublishesReceiptlessTree(t *testing.T) {
	root := t.TempDir()
	prepare := func(name string) (string, Acquired) {
		t.Helper()
		stage := filepath.Join(root, name)
		if err := os.Mkdir(stage, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(stage, "header.h"), []byte("genuine header"), 0600); err != nil {
			t.Fatal(err)
		}
		digest, err := TreeDigest(stage)
		if err != nil {
			t.Fatal(err)
		}
		return stage, Acquired{Artifact: Artifact{Name: "fixture", SHA256: "verified-archive-pin"}, TreeSHA256: digest}
	}
	stage, acquired := prepare("interrupted-stage")
	impossible := filepath.Join(root, "missing-parent", "published")
	if err := publishExtraction(stage, impossible, acquired); err == nil {
		t.Fatal("expected rename failure")
	}
	if _, err := os.Stat(impossible); !os.IsNotExist(err) {
		t.Fatalf("failed publication exposed destination: %v", err)
	}
	if _, err := os.Stat(filepath.Join(stage, embeddedReceipt)); err != nil {
		t.Fatalf("failed stage lost receipt evidence: %v", err)
	}
	retryStage, retryAcquired := prepare("retry-stage")
	destination := filepath.Join(root, "published")
	if err := publishExtraction(retryStage, destination, retryAcquired); err != nil {
		t.Fatal(err)
	}
	if _, err := readExtraction(destination, filepath.Join(root, "absent-sidecar"), retryAcquired.Artifact); err != nil {
		t.Fatalf("fresh attempt could not reuse publication: %v", err)
	}
	if _, err := os.Stat(stage); err != nil {
		t.Fatal("retry removed interrupted evidence", err)
	}
}
