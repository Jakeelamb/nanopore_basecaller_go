//go:build cgo

// Source: Dorado 8b8fc5d36a9c0baab262a743cb175b5878e38ca6,
// dorado/basecall/decode/beam_search.cpp, lines 18-520.
package decode

import (
	"fmt"
	"math"
	"sort"
)

type beamElement struct {
	state    uint16
	previous uint8
	stay     bool
}

type frontElement struct {
	beamElement
	hash uint32
}

const lowest = -math.MaxFloat32
const maxElements = 1 << 27

// Decode returns sequence, quality and moves for one score tensor. Input and
// workspace lengths are bounded to 2^27 elements each before allocating.
func Decode(in Input, opt Options) (Result, error) {
	if err := validate(&in, opt); err != nil {
		return Result{}, err
	}
	result := search(in, opt)
	result.Sequence, result.Qstring = generateSequence(result, opt)
	return result, nil
}

func finite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

func validate(in *Input, opt Options) error {
	if in.Blocks < 1 || in.Blocks > maxElements {
		return fmt.Errorf("blocks must be in [1,%d]", maxElements)
	}
	if in.States < 4 || in.States > 16384 || in.States&(in.States-1) != 0 {
		return fmt.Errorf("states must be a power of four in [4,16384]")
	}
	bits := 0
	for n := in.States; n > 1; n >>= 1 {
		bits++
	}
	if bits%2 != 0 {
		return fmt.Errorf("states must be a power of four")
	}
	channels := 4 * in.States
	if in.ScoreStride == 0 {
		in.ScoreStride = channels
	}
	if in.ScoreStride < channels || in.ScoreStride > maxElements {
		return fmt.Errorf("score stride must be in [%d,%d]", channels, maxElements)
	}
	// Divide before multiplying. A single row never uses its stride.
	if in.Blocks-1 > (maxElements-channels)/in.ScoreStride {
		return fmt.Errorf("score span exceeds %d elements", maxElements)
	}
	span := (in.Blocks-1)*in.ScoreStride + channels
	if opt.BeamWidth < 1 || opt.BeamWidth > 256 {
		return fmt.Errorf("beam width must be in [1,256]")
	}
	for _, v := range []float32{opt.BeamCut, opt.BlankScore, opt.QShift, opt.QScale} {
		if !finite(v) {
			return fmt.Errorf("options must be finite")
		}
	}
	if opt.QScale <= 0 {
		return fmt.Errorf("quality scale must be positive")
	}
	if opt.BeamCut != 0 && opt.BeamCut < 1 {
		return fmt.Errorf("beam cut must be zero or at least one")
	}
	// The score bound prevents overflow in path accumulation, guide addition,
	// cutoff bisection and log-sum-exp, including very long accepted tensors.
	bound := float32(math.MaxFloat32 / (16 * float64(in.Blocks+1)))
	if !finite(bound) || float32(math.Abs(float64(opt.BlankScore))) > bound {
		return fmt.Errorf("blank score exceeds safe accumulation range")
	}
	if in.Blocks > maxElements/(4*in.States) || in.Blocks+1 > maxElements/in.States || in.Blocks+1 > maxElements/opt.BeamWidth {
		return fmt.Errorf("tensor or beam workspace exceeds %d elements", maxElements)
	}
	checks := []struct {
		name string
		data []float32
		size int
	}{
		{"scores", in.Scores, span},
		{"back guides", in.BackGuides, (in.Blocks + 1) * in.States},
		{"posts", in.Posts, (in.Blocks + 1) * in.States},
	}
	for _, c := range checks {
		if len(c.data) != c.size {
			return fmt.Errorf("%s length %d, want %d", c.name, len(c.data), c.size)
		}
		count := c.size
		if c.name == "scores" {
			count = in.Blocks * channels
		}
		for logical := 0; logical < count; logical++ {
			i := logical
			if c.name == "scores" {
				i = (logical/channels)*in.ScoreStride + logical%channels
			}
			v := c.data[i]
			if !finite(v) {
				return fmt.Errorf("%s[%d] must be finite", c.name, i)
			}
			if c.name == "posts" {
				if v < 0 || v > 1 {
					return fmt.Errorf("posts[%d] must be in [0,1]", i)
				}
			} else if float32(math.Abs(float64(v))) > bound {
				return fmt.Errorf("%s[%d] exceeds safe accumulation range", c.name, i)
			}
		}
	}
	return nil
}

func crc32c(crc, value uint32, bits int) uint32 {
	for i := 0; i < bits; i++ {
		b := (value ^ crc) & 1
		crc >>= 1
		if b != 0 {
			crc ^= 0x82f63b78
		}
		value >>= 1
	}
	return crc
}

func logSumExp(x, y float32) float32 {
	diff := float32(math.Abs(float64(float32(x - y))))
	extra := float32(0)
	if diff < 17 {
		extra = floatLog1p(floatExp(-diff))
	}
	if x < y {
		x = y
	}
	return x + extra
}

func search(in Input, opt Options) Result {
	statesMask := uint16(in.States - 1)
	stateBits := 0
	for n := in.States; n > 1; n >>= 1 {
		stateBits++
	}
	width := opt.BeamWidth
	persistent := make([]beamElement, width*(in.Blocks+1))
	current := make([]frontElement, 5*width)
	previous := make([]frontElement, 5*width)
	currentScores := make([]float32, 5*width)
	previousScores := make([]float32, 5*width)
	threshold := float32(lowest)
	if width < in.States {
		guides := append([]float32(nil), in.BackGuides[:in.States]...)
		sort.Slice(guides, func(i, j int) bool { return guides[i] > guides[j] })
		threshold = guides[width-1]
	}
	for state, element := 0, 0; state < in.States && element < width; state++ {
		if in.BackGuides[state] >= threshold {
			previous[element] = frontElement{beamElement{uint16(state), 0, false}, crc32c(0x12345678, uint32(state), 32)}
			element++
		}
	}
	beamWidth := min(width, in.States)
	for i := 0; i < beamWidth; i++ {
		persistent[i] = previous[i].beamElement
	}
	logCut := float32(math.MaxFloat32)
	if opt.BeamCut > 0 {
		logCut = floatLog(opt.BeamCut)
	}
	for block := 0; block < in.Blocks; block++ {
		start := block * in.ScoreStride
		scores := in.Scores[start : start+4*in.States]
		back := in.BackGuides[(block+1)*in.States : (block+2)*in.States]
		maxScore := float32(lowest)
		var hashPresent [4096]bool
		count := 0
		for p := 0; p < beamWidth; p++ {
			prev := previous[p]
			for base := 0; base < 4; base++ {
				state := uint16((int(prev.state)<<2)&int(statesMask)) | uint16(base)
				moveIndex := uint16((int(state) << 2) + ((int(prev.state) << 2) >> stateBits))
				score := float32(previousScores[p]+scores[moveIndex]) + back[state]
				hash := crc32c(prev.hash, uint32(base), 2)
				hashPresent[hash&4095] = true
				current[count] = frontElement{beamElement{state, uint8(p), false}, hash}
				currentScores[count] = score
				if score > maxScore {
					maxScore = score
				}
				count++
			}
		}
		for p := 0; p < beamWidth; p++ {
			prev := previous[p]
			score := float32(previousScores[p]+opt.BlankScore) + back[prev.state]
			current[count] = frontElement{beamElement{prev.state, uint8(p), true}, prev.hash}
			currentScores[count] = score
			if score > maxScore {
				maxScore = score
			}
			if hashPresent[prev.hash&4095] {
				stayIndex := 4*beamWidth + p
				base := int(prev.state & 3)
				for other := 0; other < beamWidth; other++ {
					stepIndex := 4*other + base
					if current[stayIndex].hash == current[stepIndex].hash {
						folded := logSumExp(currentScores[stayIndex], currentScores[stepIndex])
						if currentScores[stayIndex] > currentScores[stepIndex] {
							currentScores[stayIndex] = folded
							currentScores[stepIndex] = lowest
						} else {
							currentScores[stepIndex] = folded
							currentScores[stayIndex] = lowest
						}
						if folded > maxScore {
							maxScore = folded
						}
					}
				}
			}
			count++
		}
		cutoff := maxScore - logCut
		countAbove := func() int {
			n := 0
			for _, score := range currentScores[:count] {
				if score >= cutoff {
					n++
				}
			}
			return n
		}
		n := countAbove()
		if n > width {
			minimum := width * 8 / 10
			low, high := cutoff, maxScore
			guesses := 1
			for (n > width || n < minimum) && guesses < 10 {
				if n > width {
					low = cutoff
					cutoff = float32(cutoff+high) / 2
				} else {
					high = cutoff
					cutoff = float32(cutoff+low) / 2
				}
				n = countAbove()
				guesses++
			}
			if guesses == 10 {
				cutoff = high
				n = countAbove()
			}
			n = min(n, width)
		}
		write := 0
		for read := 0; read < count; read++ {
			if currentScores[read] >= cutoff {
				if write >= width {
					break
				}
				previous[write] = current[read]
				previousScores[write] = currentScores[read]
				write++
			}
		}
		if block == in.Blocks-1 {
			best := float32(lowest)
			bestIndex := 0
			for i := 0; i < n; i++ {
				if previousScores[i] > best {
					best = previousScores[i]
					bestIndex = i
				}
			}
			previous[0], previous[bestIndex] = previous[bestIndex], previous[0]
			previousScores[0], previousScores[bestIndex] = previousScores[bestIndex], previousScores[0]
		}
		for i := 0; i < n; i++ {
			previousScores[i] -= back[previous[i].state]
			persistent[(block+1)*width+i] = previous[i].beamElement
		}
		beamWidth = n
	}
	result := Result{Moves: make([]uint8, in.Blocks), States: make([]int32, in.Blocks), QualData: make([]float32, 4*in.Blocks), FinalScore: previousScores[0]}
	element := uint8(0)
	for b := in.Blocks; b > 0; b-- {
		chosen := persistent[b*width+int(element)]
		result.States[b-1] = int32(chosen.state)
		if !chosen.stay {
			result.Moves[b-1] = 1
		}
		element = chosen.previous
	}
	result.Moves[0] = 1
	for block := 0; block < in.Blocks; block++ {
		state := int(result.States[block])
		result.States[block] %= 4
		posts := in.Posts[(block+1)*in.States : (block+2)*in.States]
		probability := posts[state]
		var shifted [8]int
		for base := 0; base < 4; base++ {
			shifted[2*base] = (state >> 2) + (in.States>>2)*base
			shifted[2*base+1] = ((state << 2) % in.States) + base
		}
		for i, candidate := range shifted {
			unique := candidate != state
			for j := 0; j < i && unique; j++ {
				if shifted[j] == candidate {
					unique = false
				}
			}
			if unique {
				probability += posts[candidate]
			}
		}
		probability = min(max(probability, 0), 1)
		probability = floatPow(probability, 0.4)
		wrong := float32(1-probability) / 3
		for base := 0; base < 4; base++ {
			value := wrong
			if base == int(result.States[block]) {
				value = probability
			}
			result.QualData[4*block+base] = value
		}
	}
	return result
}

func generateSequence(result Result, opt Options) (string, string) {
	length := 0
	for _, move := range result.Moves {
		length += int(move)
	}
	sequence := make([]byte, length)
	quality := make([]byte, length)
	baseProbs, totalProbs := make([]float32, length), make([]float32, length)
	pos := 0
	for block, move := range result.Moves {
		base := int(result.States[block]) & 3
		offset := int(move) - 1
		if block == 0 {
			offset = 0
		}
		probPos := pos + offset
		baseProbs[probPos] += result.QualData[block*4+base]
		for k := 0; k < 4; k++ {
			totalProbs[probPos] += result.QualData[block*4+k]
		}
		if block == 0 {
			sequence[pos] = byte(base)
			pos++
		} else {
			for j := 0; j < int(move); j++ {
				sequence[pos] = byte(base)
				pos++
			}
		}
	}
	for i := range sequence {
		sequence[i] = "ACGT"[sequence[i]]
		errorProb := float32(1 - baseProbs[i]/totalProbs[i])
		phred := float32(-10) * floatLog10(errorProb)
		q := float32(phred*opt.QScale) + opt.QShift
		q = min(max(q, 1), 50)
		quality[i] = byte(float32(33.5) + q)
	}
	return string(sequence), string(quality)
}
