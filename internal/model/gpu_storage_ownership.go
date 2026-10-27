package model

import (
	"errors"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func closeGPUStorageOwners(handles []*tensor.Storage) (failed []*tensor.Storage, err error) {
	for i := len(handles) - 1; i >= 0; i-- {
		if h := handles[i]; h != nil {
			if e := h.Close(); e != nil {
				err = errors.Join(err, e)
				failed = append(failed, h)
			}
		}
	}
	slices.Reverse(failed)
	return failed, err
}
