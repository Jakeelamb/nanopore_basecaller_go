package decodergpu

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

const maxPinnedMetadataBytes int64 = 64 << 20

type numericProofMode uint8

const historicalAcceptedLoad numericProofMode = 1

type numericProofEpoch uint8

const (
	beforeExternal numericProofEpoch = iota
	duringExternal
	afterExternal
)

type numericProofKey struct {
	Anchor File
	Mode   numericProofMode
	Epoch  numericProofEpoch
}

type numericProofs struct {
	reader   func(File) (numericLineage, error)
	epoch    numericProofEpoch
	key      numericProofKey
	bound    bool
	snapshot []byte
	denied   error
}

func newNumericProofs() *numericProofs {
	return &numericProofs{reader: readNumericLineage}
}

func (p *numericProofs) historical(anchor File) (numericLineage, error) {
	return p.read(anchor, historicalAcceptedLoad)
}

func (p *numericProofs) read(anchor File, mode numericProofMode) (numericLineage, error) {
	if mode != historicalAcceptedLoad {
		return numericLineage{}, fmt.Errorf("unsupported numeric proof validation mode")
	}
	if p == nil {
		return readNumericLineage(anchor)
	}
	if p.epoch == duringExternal {
		return numericLineage{}, fmt.Errorf("numeric proof is unavailable during external execution")
	}
	if p.denied != nil {
		return numericLineage{}, p.denied
	}
	key := numericProofKey{anchor, mode, p.epoch}
	if p.bound && p.key != key {
		return numericLineage{}, fmt.Errorf("numeric proof anchor/mode changed within authentication epoch")
	}
	if !p.bound {
		proof, err := p.reader(anchor)
		if err != nil {
			return numericLineage{}, err
		}
		raw, err := json.Marshal(proof)
		if err != nil {
			return numericLineage{}, err
		}
		if int64(len(raw)) > maxPinnedMetadataBytes {
			return numericLineage{}, fmt.Errorf("numeric proof metadata exceeds bounded snapshot")
		}
		p.snapshot, p.key, p.bound = raw, key, true
	}
	var copy numericLineage
	err := json.Unmarshal(p.snapshot, &copy)
	return copy, err
}

func (p *numericProofs) postEpoch() error {
	if p == nil {
		return nil
	}
	if p.epoch == afterExternal {
		return p.denied
	}
	previous, bound := p.key, p.bound
	p.epoch, p.bound, p.snapshot = afterExternal, false, nil
	p.key = numericProofKey{}
	if bound {
		_, p.denied = p.read(previous.Anchor, previous.Mode)
	}
	return p.denied
}

func (p *numericProofs) native(fn func() (supgpu.GuardedRun, error), authenticate func() error) (run supgpu.GuardedRun, err error) {
	if p == nil {
		return run, fmt.Errorf("numeric proof invocation required for external execution")
	}
	if p.epoch != beforeExternal {
		return run, fmt.Errorf("numeric proof invocation permits only one external child")
	}
	p.epoch = duringExternal
	defer func() {
		err = errors.Join(err, p.postEpoch())
		if authenticate != nil {
			err = errors.Join(err, authenticate())
		}
	}()
	return fn()
}

func (p *numericProofs) finish() error {
	if p == nil {
		return nil
	}
	if p.epoch == duringExternal {
		return fmt.Errorf("cannot finish numeric proof while external child is active")
	}
	return p.postEpoch()
}
