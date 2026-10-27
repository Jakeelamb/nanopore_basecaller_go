package h04

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func replayControls(out string, old Receipt, capture hacgpu.FLSTMCaches, root string) ([]ReplayControl, error) {
	var records []ReplayControl
	for _, name := range []string{"native-payload", "shape", "source-revision", "quantizer", "direction"} {
		dir := filepath.Join(out, "controls", name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return records, err
		}
		raw, err := json.Marshal(capture)
		if err != nil {
			return records, err
		}
		var changed hacgpu.FLSTMCaches
		if err = json.Unmarshal(raw, &changed); err != nil {
			return records, err
		}
		file := filepath.Join(dir, "caches.json")
		before, err := runner.SHA(filepath.Join(old.Capture, "caches.json"))
		if err != nil {
			return records, err
		}
		for _, layer := range capture.Layers {
			for _, t := range layer.Tensors {
				src, err := filepath.Abs(filepath.Join(old.Capture, t.File))
				if err != nil {
					return records, err
				}
				if name == "native-payload" && t.File == capture.Layers[0].Tensors[0].File {
					body, err := os.ReadFile(src)
					if err != nil {
						return records, err
					}
					body[0] ^= 1
					file, before = filepath.Join(dir, t.File), t.SHA256
					if err = os.WriteFile(file, body, 0644); err != nil {
						return records, err
					}
				} else if err = os.Symlink(src, filepath.Join(dir, t.File)); err != nil {
					return records, err
				}
			}
		}
		switch name {
		case "shape":
			changed.Layers[0].Tensors[0].Shape[0]++
		case "source-revision":
			changed.SourceCommit = "changed-source"
		case "quantizer":
			changed.Quantize.QuantMax = 128
		case "direction":
			changed.Layers[0].Reverse = !changed.Layers[0].Reverse
		}
		body, err := json.MarshalIndent(changed, "", "  ")
		if err != nil {
			return records, err
		}
		if err = os.WriteFile(filepath.Join(dir, "caches.json"), append(body, '\n'), 0644); err != nil {
			return records, err
		}
		_, rejected := hacgpu.ImportFLSTMCaches(dir)
		rec, err := rejectedCopy(name, file, before, rejected)
		if err != nil {
			return records, err
		}
		records = append(records, rec)
	}

	raw, err := json.Marshal(old)
	if err != nil {
		return records, err
	}
	var changed Receipt
	if err = json.Unmarshal(raw, &changed); err != nil {
		return records, err
	}
	changed.Layers[0].Tensors[0].GoSHA256 = hashBytes([]byte("changed historical Go digest"))
	path := filepath.Join(out, "controls", "historical-go-digest.json")
	body, err := json.MarshalIndent(changed, "", "  ")
	if err != nil {
		return records, err
	}
	if err = os.WriteFile(path, append(body, '\n'), 0644); err != nil {
		return records, err
	}
	_, rejected := retainedPairs(changed, capture)
	rec, err := rejectedCopy("historical-go-digest", path, hashBytes(raw), rejected)
	if err != nil {
		return records, err
	}
	records = append(records, rec)
	for _, control := range []struct{ name, src, pin string }{
		{"oracle-build-manifest", old.BuildManifest, old.BuildSHA},
		{"oracle-source", old.OracleSource, old.OracleSrcSHA},
		{"consumer-source", filepath.Join(root, "verification/hacgpu/h04/replay.go"), ""},
	} {
		body, err := os.ReadFile(control.src)
		if err != nil {
			return records, err
		}
		before := control.pin
		if before == "" {
			before = hashBytes(body)
		}
		path := filepath.Join(out, "controls", control.name+".changed")
		body = append(body, '\n')
		if err = os.WriteFile(path, body, 0644); err != nil {
			return records, err
		}
		rec, err := rejectedCopy(control.name, path, before, replayPin(path, before))
		if err != nil {
			return records, err
		}
		records = append(records, rec)
	}
	return records, nil
}

func rejectedCopy(name, path, before string, rejection error) (ReplayControl, error) {
	r := ReplayControl{Name: name, ChangedFile: path, BeforeSHA256: before}
	var err error
	if r.AfterSHA256, err = runner.SHA(path); err != nil {
		return r, err
	}
	if rejection == nil || r.AfterSHA256 == before {
		return r, fmt.Errorf("control %s did not reject a changed input", name)
	}
	r.Rejection = rejection.Error()
	return r, nil
}
