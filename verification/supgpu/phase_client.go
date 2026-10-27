package supgpu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// PhaseClient is the Go-child side of the existing private FD5/FD6 protocol.
// Its parent, not this client, supplies the independently expected phase list.
type PhaseClient struct {
	mu       sync.Mutex
	request  *os.File
	approval *os.File
	reader   *bufio.Reader
	sequence uint64
	failed   bool
}

// OpenPhaseClient requires the pipes installed by ExecuteGuarded. A standalone
// child gets an error before its loader/device/model boundary. Use one client
// for the complete child process and call Checkpoint before each named stage.
func OpenPhaseClient() (*PhaseClient, error) {
	out, in := os.NewFile(5, "audited-phase-request"), os.NewFile(6, "audited-phase-approval")
	for _, f := range []*os.File{out, in} {
		info, err := f.Stat()
		if err != nil || info.Mode()&os.ModeNamedPipe == 0 {
			out.Close()
			in.Close()
			return nil, fmt.Errorf("private audited phase pipes are absent/invalid")
		}
	}
	return &PhaseClient{request: out, approval: in, reader: bufio.NewReaderSize(in, 128)}, nil
}

// Checkpoint blocks until the parent approves this exact sequence/name after
// observing mapped code. An error must stop the child before entering its stage.
func (p *PhaseClient) Checkpoint(name string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.failed || p.request == nil || !phaseNameRE.MatchString(name) || (p.sequence == 0 && name != "pre-model") {
		return fmt.Errorf("invalid/closed audited phase client or first phase")
	}
	p.sequence++
	q := GateRequest{Schema: "nanopore-supgpu-loader-gate/v1", Kind: "phase", Sequence: p.sequence, Path: name, PID: os.Getpid()}
	raw, err := json.Marshal(q)
	if err == nil {
		_, err = p.request.Write(append(raw, '\n'))
	}
	var reply []byte
	if err == nil {
		reply, err = p.reader.ReadSlice('\n')
	}
	if err != nil || string(reply) != fmt.Sprintf("allow phase %d\n", p.sequence) {
		p.failed = true
		return fmt.Errorf("audited phase %s rejected/unavailable: %v", name, err)
	}
	return nil
}

func (p *PhaseClient) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.request == nil {
		return nil
	}
	err := p.request.Close()
	if e := p.approval.Close(); err == nil {
		err = e
	}
	p.request, p.approval = nil, nil
	return err
}
