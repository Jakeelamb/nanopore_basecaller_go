package caller

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

var ErrGPUUnqualified = errors.New("GPU caller domain is not qualified")

type gpuBundleFile struct {
	path   string
	digest [32]byte
}

type gpuProfileKey struct {
	lane         string
	chunk, batch int
	variable     bool
}

type gpuBundleRecord struct {
	tensor, pod5           gpuBundleFile
	guide, beam, posterior [32]byte
	profiles               map[gpuProfileKey]bool
	partialProfiles        map[gpuProfileKey]bool
}

// Partial publication needs independent evidence in addition to full-profile
// approval. Neither an omitted entry nor ordinary profile approval admits it.
func (r gpuBundleRecord) validatePartialProfiles() error {
	for key, approved := range r.partialProfiles {
		if !approved {
			continue
		}
		fixed := !key.variable && (key.chunk == 12288 || key.chunk == 6144)
		physical := key.lane == "hac-flstm-v6" && key.batch == 64 || key.lane == "sup-transformer-v5.2" && key.batch == 32
		if !r.profiles[key] || !fixed || !physical {
			return fmt.Errorf("%w: partial approval requires an independently approved fixed physical profile", ErrGPUUnqualified)
		}
	}
	return nil
}

// Entries require independently accepted ABI5, POD5 and numerical evidence.
// No complete production bundle has that evidence yet.
var approvedGPUBundles = map[[32]byte]gpuBundleRecord{}

type gpuBundle struct {
	tensorLibrary, pod5Library string
	record                     gpuBundleRecord
}

func loadGPUBundle(directory string) (gpuBundle, error) {
	if directory == "" {
		return gpuBundle{}, fmt.Errorf("%w: --native-bundle is required", ErrGPUUnqualified)
	}
	root, err := filepath.EvalSymlinks(directory)
	if err != nil {
		return gpuBundle{}, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return gpuBundle{}, err
	}
	f, err := os.Open(filepath.Join(root, "gpu-bundle.json"))
	if err != nil {
		return gpuBundle{}, err
	}
	info, statErr := f.Stat()
	if statErr != nil {
		return gpuBundle{}, errors.Join(statErr, f.Close())
	}
	if !info.Mode().IsRegular() {
		return gpuBundle{}, errors.Join(errors.New("GPU bundle manifest is not a regular file"), f.Close())
	}
	body, readErr := io.ReadAll(io.LimitReader(f, 65537))
	if err = errors.Join(readErr, f.Close()); err != nil {
		return gpuBundle{}, err
	}
	if len(body) > 65536 {
		return gpuBundle{}, errors.New("GPU bundle manifest exceeds 64 KiB")
	}
	record, ok := approvedGPUBundles[sha256.Sum256(body)]
	if !ok {
		return gpuBundle{}, fmt.Errorf("%w: bundle manifest has no independently approved release", ErrGPUUnqualified)
	}
	if record.guide == [32]byte{} || record.beam == [32]byte{} || record.posterior == [32]byte{} || len(record.profiles) == 0 {
		return gpuBundle{}, fmt.Errorf("%w: incomplete approved bundle", ErrGPUUnqualified)
	}
	if err := record.validatePartialProfiles(); err != nil {
		return gpuBundle{}, err
	}
	tensorPath, err := verifyGPUBundleFile(root, record.tensor)
	if err != nil {
		return gpuBundle{}, err
	}
	pod5Path, err := verifyGPUBundleFile(root, record.pod5)
	if err != nil {
		return gpuBundle{}, err
	}
	return gpuBundle{tensorPath, pod5Path, record}, nil
}

func verifyGPUBundleFile(root string, file gpuBundleFile) (string, error) {
	if !filepath.IsLocal(file.path) || file.digest == [32]byte{} {
		return "", fmt.Errorf("%w: invalid approved library entry", ErrGPUUnqualified)
	}
	path, err := filepath.EvalSymlinks(filepath.Join(root, file.path))
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(root, path)
	if err != nil || !filepath.IsLocal(relative) {
		return "", errors.New("GPU bundle library escapes installation")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	info, statErr := f.Stat()
	if statErr != nil {
		return "", errors.Join(statErr, f.Close())
	}
	if !info.Mode().IsRegular() {
		return "", errors.Join(errors.New("GPU bundle library is not a regular file"), f.Close())
	}
	h := sha256.New()
	_, copyErr := io.Copy(h, f)
	if err = errors.Join(copyErr, f.Close()); err != nil {
		return "", err
	}
	if string(h.Sum(nil)) != string(file.digest[:]) {
		return "", fmt.Errorf("GPU bundle library digest differs: %s", file.path)
	}
	return path, nil
}
