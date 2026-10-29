package supgpu

// ReadRuntimePolicy authenticates the original selected policy bytes before
// applying the shared closed wire contract and full runtime policy validator.
func ReadRuntimePolicy(selected Pin) (RuntimePolicy, error) {
	var policy RuntimePolicy
	if err := verifyPin("", selected); err != nil {
		return policy, err
	}
	if err := readPinnedJSON(selected.Path, selected.SHA256, &policy); err != nil {
		return policy, err
	}
	if err := validateGuardPolicy(policy); err != nil {
		return policy, err
	}
	return policy, nil
}

// ReadGuardPlan authenticates the selected plan, phase contract, runtime policy
// and custody inputs without executing a target or acquiring a host lock.
func ReadGuardPlan(planPin, contractPin Pin) (GuardPlan, PhaseContract, error) {
	return readGuardPlan(planPin, contractPin)
}
