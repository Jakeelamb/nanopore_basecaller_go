package decodergpu

import (
	"fmt"
	"path/filepath"
)

// A historical producer authenticates its compiled source copies. It never
// substitutes later source bytes, and it does not authenticate a new consumer.
func checkProducerInputs(inputs map[string]string, p ClosedBuildPlan, current bool) error {
	if current {
		return checkPins(inputs)
	}
	for path, want := range inputs {
		actual := path
		if copy, ok := p.SourceBindings[path]; ok {
			if p.SnapshotPins[copy] != want || p.SnapshotPins[filepath.Join(p.Directory, "native-inputs", filepath.Base(path))] != want {
				return fmt.Errorf("historical producer source binding differs: %s", path)
			}
			actual = copy
		}
		if err := (File{actual, want}).verify(); err != nil {
			return err
		}
	}
	return nil
}

type ProducerConsumer struct {
	Schema                string
	Producer, Preparation File
	ProducerController    File
	ConsumerPreparation   File
	ConsumerController    File
}

func CheckProducerConsumer(build, consumer File) (a ProducerConsumer, err error) {
	return checkProducerConsumerScoped(build, consumer, nil)
}

func checkProducerConsumerScoped(build, consumer File, proofs *numericProofs) (a ProducerConsumer, err error) {
	build, err = normalizeAnchor(build)
	if err != nil {
		return a, err
	}
	consumer, err = normalizeAnchor(consumer)
	if err != nil {
		return a, err
	}
	b, p, err := readBuiltPhaseScoped(build, false, proofs)
	if err != nil {
		return a, err
	}
	old, err := readPreparation(p.Preparation, false, proofs)
	if err != nil {
		return a, err
	}
	if err = checkProducerInputs(old.Inputs, p, false); err != nil {
		return a, err
	}
	current, err := readPreparation(consumer, true, proofs)
	if err != nil {
		return a, err
	}
	if b.Plan != (File{p.Directory + "/plan.json", b.Plan.SHA256}) {
		return a, fmt.Errorf("producer plan directory changed")
	}
	return ProducerConsumer{"nanopore-decoder-producer-consumer/v1", build, p.Preparation, p.Controller, consumer, File{current.Go.Controller, current.Go.ControllerSHA}}, nil
}
