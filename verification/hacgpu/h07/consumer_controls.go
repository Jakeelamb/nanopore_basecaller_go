package h07

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type ConsumerControl struct {
	Name, Status, Error, Started, Finished string
	Comparison                             *OutputComparison
	Parity                                 int32
	Scale                                  float32
	Stage                                  uint32
	NativeCode                             int32
	BeforeSHA, AfterSHA, AfterFile         string
}

// These are comparator controls over authenticated source observations, without
// launching an alternative projection or changing the production caches.
func cpuConsumerControls(o ConsumerOptions, w Witness, native OracleCaptureReceipt, r *ConsumerReceipt) error {
	const rowBytes = 64 * 1024
	const inputBytes = 32 * rowBytes
	if w.Input.OffsetBytes < 0 || int64(len(w.InputParent)) < w.Input.OffsetBytes+3*rowBytes+inputBytes || len(native.Caches) != 3 {
		return fmt.Errorf("incomplete input-row or cache control witness")
	}
	start := int(w.Input.OffsetBytes)
	expected := w.InputParent[start+2*rowBytes : start+2*rowBytes+inputBytes]
	path := filepath.Join(r.Attempt, "go-input-row2.bin")
	if err := os.WriteFile(path, expected, 0644); err != nil {
		return err
	}
	r.Outputs[path] = sum(expected)
	row := OutputComparison{Role: "input-row2", GoFile: path, GoSHA: sum(expected), NativeFile: filepath.Join(o.Capture, w.Input.DataFile), NativeSHA: sum(expected), Bytes: inputBytes, NativeOffset: 2 * rowBytes, Equal: true}
	if err := verifyOutputComparison(row); err != nil {
		return err
	}
	r.Comparisons = append(r.Comparisons, row)
	wrongRow := w.InputParent[start+rowBytes : start+rowBytes+inputBytes]
	if err := retainComparisonControl(r, "wrong-input-row1", wrongRow, row); err != nil {
		return err
	}
	for _, cache := range native.Caches {
		path := filepath.Join(native.Output, cache.File)
		body, err := readPinned(path, cache.SHA256)
		if err != nil {
			return err
		}
		changed := slices.Clone(body)
		changed[len(changed)/2] ^= 1
		comparison := OutputComparison{Role: "cache-" + cache.Role, NativeFile: path, NativeSHA: cache.SHA256, Bytes: len(body), Equal: true}
		if err = retainComparisonControl(r, "changed-cache-"+cache.Role, changed, comparison); err != nil {
			return err
		}
	}
	return nil
}

func retainComparisonControl(r *ConsumerReceipt, name string, changed []byte, comparison OutputComparison) error {
	c := ConsumerControl{Name: name, Started: time.Now().UTC().Format(time.RFC3339Nano), Status: "failed"}
	comparison.GoFile = filepath.Join(r.Attempt, "control-"+name+".bin")
	comparison.GoSHA = sum(changed)
	if err := os.WriteFile(comparison.GoFile, changed, 0644); err != nil {
		return err
	}
	r.Outputs[comparison.GoFile] = comparison.GoSHA
	c.Comparison = &comparison
	// Deliberately recompute the candidate hash and keep Equal true: rejection
	// must depend on the independent source bytes, not a stale candidate pin.
	err := verifyOutputComparison(comparison)
	if err != nil {
		c.Status, c.Error = "rejected", err.Error()
	}
	c.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	r.Controls = append(r.Controls, c)
	if err == nil {
		return fmt.Errorf("ineffective consumer control: %s", name)
	}
	return nil
}

func verifyConsumerControl(c ConsumerControl) error {
	start, err := time.Parse(time.RFC3339Nano, c.Started)
	if err != nil {
		return err
	}
	finish, err := time.Parse(time.RFC3339Nano, c.Finished)
	if err != nil || finish.Before(start) || c.Status != "rejected" || c.Error == "" {
		return fmt.Errorf("incomplete rejection control: %s", c.Name)
	}
	if c.Comparison != nil {
		if err = requirePin(c.Comparison.GoFile, c.Comparison.GoSHA); err != nil {
			return err
		}
		if err = verifyOutputComparison(*c.Comparison); err == nil {
			return fmt.Errorf("retained negative control now accepted: %s", c.Name)
		}
		return nil
	}
	if c.Stage != 0 || c.NativeCode != 0 || c.BeforeSHA == "" || c.BeforeSHA != c.AfterSHA {
		return fmt.Errorf("native scalar control reached execution or changed arena: %s", c.Name)
	}
	return requirePin(c.AfterFile, c.AfterSHA)
}

// Both invalid scalars are rejected by the native guard before stream/storage
// lookup and CUDAWork. All owners and geometry are otherwise valid, and the
// complete backing is downloaded after each call to prove it remains unchanged.
func nativeScalarControls(d *tensor.Device, s *tensor.Stream, backing, full *tensor.Storage, cache model.HACOutputCacheSnapshot, r *ConsumerReceipt) (err error) {
	var owners []*tensor.Storage
	defer func() {
		for i := len(owners) - 1; i >= 0; i-- {
			err = errors.Join(err, owners[i].Close())
		}
	}()
	keep := func(h *tensor.Storage, e error) (*tensor.Storage, error) {
		if e == nil {
			owners = append(owners, h)
		}
		return h, e
	}
	var operands tensor.FactorisedLinearOperands
	operands.Input, err = keep(full.View(2*64*1024, []int64{32, 64, 1024}, []int64{64 * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}))
	if err != nil {
		return err
	}
	operands.Down, err = keep(backing.Reinterpret(tensor.DTypeF16, 0, []int64{64 * 32 * 128}, []int64{1}, []tensor.KoiTag{tensor.TagC}))
	if err != nil {
		return err
	}
	operands.Output, err = keep(backing.Reinterpret(tensor.DTypeF16, DownBytes/2, []int64{64, 32, 1024}, []int64{32 * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}))
	if err != nil {
		return err
	}
	encodeHalf := func(v []uint16) []byte {
		body := make([]byte, 2*len(v))
		for i, bits := range v {
			binary.LittleEndian.PutUint16(body[2*i:], bits)
		}
		return body
	}
	down := make([]byte, len(cache.Down))
	for i, v := range cache.Down {
		down[i] = byte(v)
	}
	operands.DownWeight, err = keep(d.NewStorage(s, tensor.DTypeI8, []int64{128, 1024}, []tensor.KoiTag{'K', tensor.TagC}, down))
	if err != nil {
		return err
	}
	operands.DownScale, err = keep(d.NewStorage(s, tensor.DTypeF16, []int64{128}, []tensor.KoiTag{'K'}, encodeHalf(cache.DownScale)))
	if err != nil {
		return err
	}
	operands.UpWeight, err = keep(d.NewStorage(s, tensor.DTypeF16, []int64{1024, 128}, []tensor.KoiTag{tensor.TagC, 'K'}, encodeHalf(cache.Up)))
	if err != nil {
		return err
	}
	g := tensor.FactorisedLinearGeometry{N: 64, T: 32}
	if err = d.ValidateFactorisedLinear(s, operands, g); err != nil {
		return err
	}
	before, err := backing.Download(s)
	if err != nil {
		return err
	}
	for _, scalars := range []struct {
		name   string
		parity int32
		scale  float32
	}{{"native-parity0", 0, 5}, {"native-scale4", 1, 4}} {
		c := ConsumerControl{Name: scalars.name, Parity: scalars.parity, Scale: scalars.scale, Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), BeforeSHA: sum(before)}
		rejected := d.FactorisedLinearUnchecked(s, operands, g, scalars.parity, scalars.scale)
		var native *tensor.CUDAError
		valid := errors.As(rejected, &native) && native.Op == "factorised_linear" && native.Subsystem == "boundary" && native.Detail.Stage == 0 && native.Detail.NativeCode == 0 && native.Reason == "factorised linear requires parity 1 and scale 5"
		if rejected != nil {
			c.Error = rejected.Error()
		}
		if native != nil {
			c.Stage, c.NativeCode = native.Detail.Stage, native.Detail.NativeCode
		}
		after, readErr := backing.Download(s)
		if readErr != nil {
			return errors.Join(rejected, readErr)
		}
		c.AfterSHA, c.AfterFile = sum(after), filepath.Join(r.Attempt, "control-"+scalars.name+"-parent.bin")
		if err = os.WriteFile(c.AfterFile, after, 0644); err != nil {
			return err
		}
		r.Outputs[c.AfterFile] = c.AfterSHA
		if valid && bytes.Equal(before, after) {
			c.Status = "rejected"
		}
		c.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		r.Controls = append(r.Controls, c)
		if c.Status != "rejected" {
			return fmt.Errorf("native scalar control failed: %s: %v", c.Name, rejected)
		}
	}
	return nil
}
