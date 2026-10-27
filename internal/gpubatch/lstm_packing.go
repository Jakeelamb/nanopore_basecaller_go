package gpubatch

import (
	"fmt"
	"sort"
)

// LSTMPacking owns the initialized int32 upload spans. OutLayoutCapacity also
// includes the native destination's final N words, which are never uploaded.
type LSTMPacking struct {
	InLayout          []int32
	OutLayout         []int32
	ForwardEncoding   []int32
	BackwardEncoding  []int32
	OutLayoutCapacity int
}

type lstmChain struct {
	offset, size, total, next int
	identifier                uint32
}

type lstmPart struct {
	chains     []lstmChain
	roots      int
	laneOffset int
}

type lstmCursor struct{ lane, time int }

const lstmSubbatch = 32

func prepareLSTMPart(chains []lstmChain, fullT, identifierOffset, laneOffset int) (lstmPart, error) {
	sort.SliceStable(chains, func(i, j int) bool { return chains[i].total > chains[j].total })
	roots := len(chains)
	for roots > lstmSubbatch {
		left, right := roots-2, roots-1
		if chains[left].total+chains[right].total > fullT {
			break
		}
		added := chains[right].total
		for node := left; ; node = chains[node].next {
			chains[node].total += added
			if chains[node].next < 0 {
				chains[node].next = right
				break
			}
		}
		roots--
		position := sort.Search(left, func(i int) bool { return chains[i].total < chains[left].total })
		root := chains[left]
		copy(chains[position+1:left+1], chains[position:left])
		chains[position] = root
	}
	if roots >= 2*lstmSubbatch {
		return lstmPart{}, fmt.Errorf("LSTM part has %d roots, must be below %d", roots, 2*lstmSubbatch)
	}
	if identifierOffset+roots > 65535 {
		return lstmPart{}, fmt.Errorf("LSTM root identifier end %d exceeds 65535", identifierOffset+roots)
	}
	for root := 0; root < roots; root++ {
		identifier := uint32(identifierOffset+root) << 16
		for node := root; node >= 0; node = chains[node].next {
			chains[node].identifier = identifier
		}
	}
	return lstmPart{chains: chains, roots: roots, laneOffset: laneOffset}, nil
}

func lstmParts(n, fullT int, sizes []int) ([]lstmPart, int, error) {
	var parts []lstmPart
	chunk, offset, identifiers := 0, 0, 0
	for part := 0; chunk < len(sizes); part++ {
		if part >= n/lstmSubbatch {
			return nil, 0, fmt.Errorf("LSTM chunks exceed ordered subbatch capacity")
		}
		used := 0
		var chains []lstmChain
		for chunk < len(sizes) && used+sizes[chunk]+2 <= lstmSubbatch*fullT {
			length := sizes[chunk] + 2
			chains = append(chains, lstmChain{offset: offset, size: length, total: length, next: -1})
			chunk++
			offset += length
			used += length
		}
		if len(chains) == 0 {
			return nil, 0, fmt.Errorf("LSTM chunk does not fit one subbatch")
		}
		p, err := prepareLSTMPart(chains, fullT, identifiers, part*lstmSubbatch)
		if err != nil {
			return nil, 0, err
		}
		identifiers += p.roots
		parts = append(parts, p)
	}
	return parts, offset, nil
}

func (p lstmPart) forwardLayout(n, fullT int, layout []int, encoding []int32) error {
	if p.roots <= lstmSubbatch {
		for root := 0; root < p.roots; root++ {
			position := p.laneOffset + root
			for node := root; node >= 0; node = p.chains[node].next {
				chain := p.chains[node]
				for i := 0; i < chain.size; i++ {
					layout[chain.offset+i] = position
					encoding[position] = int32(chain.identifier)
					position += n
				}
			}
		}
		return nil
	}
	cursors := [2]lstmCursor{{0, 0}, {0, 1}}
	for root := 0; root < p.roots; root++ {
		previous := cursors
		parity, other := root&1, (root&1)^1
		cursors[parity].lane++
		excess := p.chains[root].total - fullT/2
		if excess > 0 {
			cursors[other].time += 2 * excess
		}
		if cursors[parity].lane == lstmSubbatch {
			cursors[other].time += cursors[parity].time - parity
			cursors[parity].time = parity
		}
		if cursors[other].time >= fullT {
			cursors[other].lane++
			cursors[other].time -= fullT
			if cursors[other].lane == lstmSubbatch {
				cursors[other].time = other
			}
		}
		node, written := root, 0
		for timestep := 0; timestep < fullT; timestep++ {
			parity := timestep & 1
			before, after := previous[parity], cursors[parity]
			lane := after.lane
			if after.lane == before.lane {
				if before.time > timestep || after.time <= timestep {
					continue
				}
			} else if after.time <= timestep {
				if before.time > timestep {
					continue
				}
				lane = before.lane
			}
			if lane < 0 || node < 0 {
				continue
			}
			if lane >= lstmSubbatch {
				return fmt.Errorf("LSTM interleave lane %d exceeds subbatch", lane)
			}
			layout[p.chains[node].offset+written] = timestep*n + p.laneOffset + lane
			written++
			if written == p.chains[node].size {
				node, written = p.chains[node].next, 0
			}
		}
		if node >= 0 {
			return fmt.Errorf("LSTM interleave leaves chain data unwritten")
		}
	}
	for _, chain := range p.chains {
		for i := 0; i < chain.size; i++ {
			position := layout[chain.offset+i]
			if i+1 == chain.size && chain.next < 0 {
				encoding[position] = int32(chain.identifier | 16)
				continue
			}
			next := chain.offset + i + 1
			if i+1 == chain.size {
				next = p.chains[chain.next].offset
			}
			word, err := lstmStepWord(position, layout[next], n, chain.identifier, false)
			if err != nil {
				return err
			}
			encoding[position] = word
		}
	}
	return p.resolveEnds(n, fullT, encoding)
}

func lstmStepWord(previous, current, n int, identifier uint32, backward bool) (int32, error) {
	deltaTime := current/n - previous/n
	deltaLane := previous%n - current%n
	if deltaTime < 1 || deltaTime > 2 || deltaLane < 0 || deltaLane > 1 {
		return 0, fmt.Errorf("unsupported LSTM step (%d,%d) to (%d,%d)", previous/n, previous%n, current/n, current%n)
	}
	timeBit := deltaTime >> 1
	if backward {
		timeBit = deltaTime & 1
	}
	return int32(identifier | uint32(timeBit) | uint32(2*deltaLane)), nil
}

func (p lstmPart) resolveEnds(n, fullT int, encoding []int32) error {
	low := func(position int) uint32 { return uint32(encoding[position]) & 65535 }
	for timestep := 0; timestep < fullT; timestep++ {
		for lane := lstmSubbatch - 1; lane >= 0; lane-- {
			position := timestep*n + p.laneOffset + lane
			if low(position) != 16 {
				continue
			}
			right := timestep > 0 && low(position-n) == 1
			if lane+1 < lstmSubbatch {
				right = right || timestep > 0 && low(position-n+1) == 3 || low(position+1) == 2
			}
			nextRight := lane+1 < lstmSubbatch && low(position+1) == 3
			if timestep+1 < fullT {
				nextRight = nextRight || low(position+n) == 0 || lane+1 < lstmSubbatch && low(position+n+1) == 2
			}
			word := uint32(encoding[position]) ^ 16
			if lane == 0 {
				if nextRight && right {
					return fmt.Errorf("LSTM end has conflicting right dependencies")
				}
				if right {
					word |= 1
				}
				encoding[position] = int32(word)
				continue
			}
			left := timestep > 0 && (low(position-n-1) == 1 || low(position-n) == 3) || low(position-1) == 0
			nextLeft := low(position-1) == 1
			if timestep+1 < fullT {
				nextLeft = nextLeft || low(position+n-1) == 0 || low(position+n) == 2
			}
			if nextLeft && nextRight && right && left {
				return fmt.Errorf("LSTM end has conflicting left dependencies")
			}
			if right {
				switch {
				case !nextRight:
					word |= 1
				case !left:
					word |= 2
				case nextLeft:
					word |= 4
				default:
					word |= 3
				}
			}
			encoding[position] = int32(word)
		}
	}
	return nil
}

func (p lstmPart) backwardLayout(n int, forward, backward []int, fwdEncoding, bwdEncoding []int32) error {
	for _, chain := range p.chains {
		previous := 0
		for i := 0; i < chain.size; i++ {
			position := forward[chain.offset+i]
			flags := uint32(fwdEncoding[position]) & 65535
			if flags > 3 {
				return fmt.Errorf("unsupported forward LSTM flags %d", flags)
			}
			position += int(flags&1)*n - int(flags>>1)
			if position < 0 || position >= len(bwdEncoding) {
				return fmt.Errorf("backward LSTM position outside capacity")
			}
			backward[chain.offset+i] = position
			if i > 0 {
				word, err := lstmStepWord(previous, position, n, chain.identifier, true)
				if err != nil {
					return err
				}
				bwdEncoding[position] = word
			}
			previous = position
		}
	}
	for _, chain := range p.chains {
		fwdEncoding[forward[chain.offset]] |= 4
		fwdEncoding[forward[chain.offset+chain.size-1]] = 8
		bwdEncoding[backward[chain.offset+chain.size-1]] |= 4
	}
	return nil
}

func packLSTM(n, fullT, workspace int, sizes []int) (LSTMPacking, error) {
	if n <= 0 || n%lstmSubbatch != 0 || fullT <= 0 || fullT&1 != 0 || n > MaxCoordinate/(fullT+1) {
		return LSTMPacking{}, fmt.Errorf("invalid or overflowing LSTM geometry")
	}
	sum, withSentinels := 0, 0
	for _, size := range sizes {
		if size <= 0 || size > fullT-2 || sum > MaxCoordinate-size || withSentinels > MaxCoordinate-size-2 {
			return LSTMPacking{}, fmt.Errorf("invalid or overflowing LSTM chunk sizes")
		}
		sum += size
		withSentinels += size + 2
	}
	if len(sizes) == 0 || int64(2)*int64(withSentinels)+int64(4*fullT+2)*int64(n) > int64(workspace) {
		return LSTMPacking{}, fmt.Errorf("LSTM chunks exceed source workspace capacity")
	}
	parts, count, err := lstmParts(n, fullT, sizes)
	if err != nil {
		return LSTMPacking{}, err
	}
	forward, backward := make([]int, count), make([]int, count)
	result := LSTMPacking{InLayout: make([]int32, sum), OutLayout: make([]int32, n*fullT), ForwardEncoding: make([]int32, n*fullT), BackwardEncoding: make([]int32, n*(fullT+1)), OutLayoutCapacity: n * (fullT + 1)}
	for i := range result.OutLayout {
		result.OutLayout[i] = -1
		result.ForwardEncoding[i] = 8
	}
	for i := range result.BackwardEncoding {
		result.BackwardEncoding[i] = 8
	}
	for _, part := range parts {
		if err := part.forwardLayout(n, fullT, forward, result.ForwardEncoding); err != nil {
			return LSTMPacking{}, err
		}
		if err := part.backwardLayout(n, forward, backward, result.ForwardEncoding, result.BackwardEncoding); err != nil {
			return LSTMPacking{}, err
		}
	}
	original, compacted := 0, 0
	for _, size := range sizes {
		for i := 0; i < size; i++ {
			position := forward[original+i+1]
			if position < 0 || position >= len(result.OutLayout) || result.OutLayout[position] != -1 {
				return LSTMPacking{}, fmt.Errorf("invalid or duplicate forward LSTM position")
			}
			result.OutLayout[position] = int32(compacted + i)
			result.InLayout[compacted+i] = int32(backward[original+i+1])
		}
		original += size + 2
		compacted += size
	}
	return result, nil
}
