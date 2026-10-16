package gojxl

import (
	"errors"
	"math"
)

// Pure-Go lossless Modular JPEG XL encoder. Scope: single frame, grayscale or
// RGB, 1..16-bit, YCoCg RCT for 3-channel RGB, an adaptive multi-context MA
// decision tree with per-leaf predictor selection (including Weighted,
// Gradient, Select, Average4, Average0, Left, Top), and context-clustered
// frequency-derived ANS histograms. Images larger than one group are split into
// byte-aligned group sections sharing the global tree and clustered histograms.
// It is the inverse of decodeModularFrame, validated by round-tripping through
// the decoder and byte-exact against djxl.

func packSigned(v int32) uint32 { return (uint32(v) << 1) ^ uint32(v>>31) }

// maTreeTokens is a single Gradient-predictor leaf MA tree (used by VarDCT DC
// and AC-metadata streams).
var maTreeTokens = []encToken{
	{ctxProperty, 0}, // property+1 = 0 → leaf
	{ctxPredictor, predGradient},
	{ctxOffset, 0},
	{ctxMultiplierLog, 0},
	{ctxMultiplierBit, 0},
}

// gradientTokens Gradient-predicts the [x0,x0+rw) x [y0,y0+rh) rect of plane
// (full-width w) as an independent sub-image — borders are at the rect edges,
// matching how the decoder predicts each modular group — and emits one residual
// token per pixel in raster order.
func gradientTokens(plane []int32, w, x0, y0, rw, rh int, dst []encToken) []encToken {
	get := func(lx, ly int) int64 { return int64(plane[(y0+ly)*w+(x0+lx)]) }
	for ly := 0; ly < rh; ly++ {
		for lx := 0; lx < rw; lx++ {
			var left, top, topleft int64
			if lx > 0 {
				left = get(lx-1, ly)
			} else if ly > 0 {
				left = get(lx, ly-1)
			}
			top = left
			if ly > 0 {
				top = get(lx, ly-1)
			}
			topleft = left
			if lx > 0 && ly > 0 {
				topleft = get(lx-1, ly-1)
			}
			guess := clampedGradient(left, top, topleft)
			residual := int32(get(lx, ly) - guess)
			dst = append(dst, encToken{ctx: 0, value: packSigned(residual)})
		}
	}
	return dst
}

// fwdRCTYCoCg applies the forward reversible YCoCg transform (rctType = 6) in
// place on 3 equal-sized planes, matching invRCT(rctType = 6) in transforms.go.
func fwdRCTYCoCg(planes [][]int32) {
	c0, c1, c2 := planes[0], planes[1], planes[2]
	for i := range c0 {
		R, G, B := c0[i], c1[i], c2[i]
		Co := R - B
		tmp := B + (Co >> 1)
		Cg := G - tmp
		Y := tmp + (Cg >> 1)
		c0[i], c1[i], c2[i] = Y, Co, Cg
	}
}

// encodeChannel encodes channel ci within the sub-rect [x0,x0+rw) x [y0,y0+rh)
// of planes[ci] (full stride w), mirroring decodeChannel in modular_channel.go.
func encodeChannel(plane []int32, w, x0, y0, rw, rh, ci, groupID int, tree []treeNode, wpH wpHeader, dst []encToken) []encToken {
	if rw == 0 || rh == 0 {
		return dst
	}
	props := make([]int64, kNumNonrefProperties)
	props[0] = int64(ci)
	props[1] = int64(groupID)
	wp := newWPState(wpH, rw)

	get := func(lx, ly int) int64 { return int64(plane[(y0+ly)*w+(x0+lx)]) }

	for ly := 0; ly < rh; ly++ {
		props[2] = int64(ly)
		props[9] = 0
		for lx := 0; lx < rw; lx++ {
			var left, top, topleft, topright, leftleft, toptop, toprightright int64
			if lx > 0 {
				left = get(lx-1, ly)
			} else if ly > 0 {
				left = get(lx, ly-1)
			}
			top = left
			if ly > 0 {
				top = get(lx, ly-1)
			}
			topleft = left
			if lx > 0 && ly > 0 {
				topleft = get(lx-1, ly-1)
			}
			topright = top
			if lx+1 < rw && ly > 0 {
				topright = get(lx+1, ly-1)
			}
			leftleft = left
			if lx > 1 {
				leftleft = get(lx-2, ly)
			}
			toptop = top
			if ly > 1 {
				toptop = get(lx, ly-2)
			}
			toprightright = topright
			if lx+2 < rw && ly > 0 {
				toprightright = get(lx+2, ly-1)
			}

			props[3] = int64(lx)
			props[4] = absI64(top)
			props[5] = absI64(left)
			props[6] = top
			props[7] = left
			props[8] = left - props[9]
			props[9] = left + top - topleft
			props[10] = left - topleft
			props[11] = topleft - top
			props[12] = top - topright
			props[13] = top - toptop
			props[14] = left - leftleft

			wpPred := wp.predictProp(lx, ly, rw, top, left, topright, topleft, toptop, props, kWPProp)
			leaf := traverseTree(tree, props)
			guess := predictOne(leaf.predictor, left, top, toptop, topleft, topright, leftleft, toprightright, wpPred)
			val := get(lx, ly)
			residual := int32(val - guess)
			dst = append(dst, encToken{ctx: leaf.lchild, value: packSigned(residual)})
			wp.updateErrors(val, lx, ly, rw)
		}
	}
	return dst
}

// ---- Adaptive MA Decision Tree Builder ----

const (
	maxMALeaves      = 32
	maxMATrainPix    = 32768
	minSplitGainBits = 160.0
	numCandProps     = 12
	numCandPreds     = 7
	maxTrainTok      = 80
)

var treeCandProps = [numCandProps]int{
	0,       // channel index ci
	kWPProp, // 15: weighted predictor signed max neighbor error
	8,       // left - prev_grad
	4,       // |top|
	5,       // |left|
	10,      // left - topleft
	11,      // topleft - top
	12,      // top - topright
	6,       // top
	7,       // left
	13,      // top - toptop
	14,      // left - leftleft
}

var treeCandPreds = [numCandPreds]uint32{
	predWeighted,
	predGradient,
	predSelect,
	predAverage4,
	predAverage0,
	predLeft,
	predTop,
}

type maTrainSample struct {
	props     [numCandProps]int32
	predTok   [numCandPreds]uint8
	predExtra [numCandPreds]uint8
}

type maBuilderNode struct {
	property  int
	splitval  int32
	predictor uint32
	left      *maBuilderNode // property > splitval
	right     *maBuilderNode // property <= splitval

	// Builder working state for unsplit leaves:
	samples   []int
	baseCost  float64
	bestProp  int
	bestSplit int32
	bestGain  float64
	leftIdxs  []int
	rightIdxs []int
}

func clampI32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

// collectMATrainSamples runs property + predictor evaluation on a representative
// group region of the image and collects up to maxMATrainPix training samples.
func collectMATrainSamples(planes [][]int32, w, h int, fd frameDimensions) []maTrainSample {
	channels := len(planes)
	groupDim := int(fd.groupDim)
	gx := int(fd.xsizeGroups) / 2
	gy := int(fd.ysizeGroups) / 2
	x0 := gx * groupDim
	y0 := gy * groupDim
	rw := minInt(groupDim, w-x0)
	rh := minInt(groupDim, h-y0)
	if rw <= 0 || rh <= 0 {
		x0, y0, rw, rh = 0, 0, w, h
	}

	totalPix := rw * rh * channels
	stride := 1
	if totalPix > maxMATrainPix {
		stride = totalPix / maxMATrainPix
	}
	samples := make([]maTrainSample, 0, minInt(totalPix, maxMATrainPix+1024))
	wpH := defaultWPHeader()
	var props [kNumNonrefProperties]int64
	step := 0

	for ci := 0; ci < channels; ci++ {
		plane := planes[ci]
		props[0] = int64(ci)
		props[1] = 0
		wp := newWPState(wpH, rw)
		get := func(lx, ly int) int64 { return int64(plane[(y0+ly)*w+(x0+lx)]) }

		for ly := 0; ly < rh; ly++ {
			props[2] = int64(ly)
			props[9] = 0
			for lx := 0; lx < rw; lx++ {
				var left, top, topleft, topright, leftleft, toptop, toprightright int64
				if lx > 0 {
					left = get(lx-1, ly)
				} else if ly > 0 {
					left = get(lx, ly-1)
				}
				top = left
				if ly > 0 {
					top = get(lx, ly-1)
				}
				topleft = left
				if lx > 0 && ly > 0 {
					topleft = get(lx-1, ly-1)
				}
				topright = top
				if lx+1 < rw && ly > 0 {
					topright = get(lx+1, ly-1)
				}
				leftleft = left
				if lx > 1 {
					leftleft = get(lx-2, ly)
				}
				toptop = top
				if ly > 1 {
					toptop = get(lx, ly-2)
				}
				toprightright = topright
				if lx+2 < rw && ly > 0 {
					toprightright = get(lx+2, ly-1)
				}

				props[3] = int64(lx)
				props[4] = absI64(top)
				props[5] = absI64(left)
				props[6] = top
				props[7] = left
				props[8] = left - props[9]
				props[9] = left + top - topleft
				props[10] = left - topleft
				props[11] = topleft - top
				props[12] = top - topright
				props[13] = top - toptop
				props[14] = left - leftleft

				wpPred := wp.predictProp(lx, ly, rw, top, left, topright, topleft, toptop, props[:], kWPProp)
				val := get(lx, ly)

				if step%stride == 0 {
					var s maTrainSample
					for k, pIdx := range treeCandProps {
						s.props[k] = clampI32(props[pIdx])
					}
					for p, pred := range treeCandPreds {
						guess := predictOne(pred, left, top, toptop, topleft, topright, leftleft, toprightright, wpPred)
						tok, nb, _ := encodeHybridUint(encUintConfig, packSigned(int32(val-guess)))
						if tok >= maxTrainTok {
							tok = maxTrainTok - 1
						}
						s.predTok[p] = uint8(tok)
						s.predExtra[p] = uint8(nb)
					}
					samples = append(samples, s)
				}
				step++
				wp.updateErrors(val, lx, ly, rw)
			}
		}
	}
	return samples
}

func shannonBits(hist *[maxTrainTok]int32, total int32, extraBits int64) float64 {
	if total <= 0 {
		return 0
	}
	totF := float64(total)
	logTot := math.Log2(totF)
	bits := float64(extraBits)
	for _, c := range hist {
		if c > 0 {
			bits += float64(c) * (logTot - math.Log2(float64(c)))
		}
	}
	return bits
}

func bestPredictorForSamples(samples []maTrainSample, idxs []int) (uint32, int, float64) {
	if len(idxs) == 0 {
		return predWeighted, 0, 0
	}
	var hists [numCandPreds][maxTrainTok]int32
	var extras [numCandPreds]int64
	for _, idx := range idxs {
		s := &samples[idx]
		for p := 0; p < numCandPreds; p++ {
			hists[p][s.predTok[p]]++
			extras[p] += int64(s.predExtra[p])
		}
	}
	bestP := 0
	bestCost := math.Inf(1)
	total := int32(len(idxs))
	for p := 0; p < numCandPreds; p++ {
		c := shannonBits(&hists[p], total, extras[p])
		if c < bestCost {
			bestCost = c
			bestP = p
		}
	}
	return treeCandPreds[bestP], bestP, bestCost
}

func evaluateNodeSplit(samples []maTrainSample, n *maBuilderNode) {
	pred, predIdx, baseCost := bestPredictorForSamples(samples, n.samples)
	n.property = -1
	n.predictor = pred
	n.baseCost = baseCost
	n.bestGain = 0
	if len(n.samples) < 64 {
		return
	}

	bestGain := 0.0
	bestPropIdx := -1
	var bestThresh int32

	var qbuf [64]int32
	var candT [10]int32
	for k := 0; k < numCandProps; k++ {
		// Gather up to 64 evenly-spaced property values to estimate quantiles.
		nq := 64
		if len(n.samples) < nq {
			nq = len(n.samples)
		}
		minV, maxV := int32(math.MaxInt32), int32(math.MinInt32)
		for i := 0; i < nq; i++ {
			v := samples[n.samples[i*len(n.samples)/nq]].props[k]
			qbuf[i] = v
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
		if minV == maxV {
			continue
		}
		// Insertion-sort qbuf[:nq].
		for i := 1; i < nq; i++ {
			for j := i; j > 0 && qbuf[j] < qbuf[j-1]; j-- {
				qbuf[j], qbuf[j-1] = qbuf[j-1], qbuf[j]
			}
		}
		// Collect candidate split thresholds: -1, 0, plus 7 quantiles.
		nt := 0
		addThresh := func(t int32) {
			if t >= minV && t < maxV {
				candT[nt] = t
				nt++
			}
		}
		addThresh(-1)
		addThresh(0)
		for q := 1; q <= 7; q++ {
			addThresh(qbuf[q*nq/8])
		}
		if nt == 0 {
			continue
		}
		// Sort and deduplicate candT[:nt].
		for i := 1; i < nt; i++ {
			for j := i; j > 0 && candT[j] < candT[j-1]; j-- {
				candT[j], candT[j-1] = candT[j-1], candT[j]
			}
		}
		u := 1
		for i := 1; i < nt; i++ {
			if candT[i] != candT[u-1] {
				candT[u] = candT[i]
				u++
			}
		}
		nt = u

		// Single pass over n.samples to populate the nt+1 threshold intervals.
		var bHist [11][maxTrainTok]int32
		var bExtra [11]int64
		var bCount [11]int32
		var totHist [maxTrainTok]int32
		var totExtra int64
		for _, idx := range n.samples {
			s := &samples[idx]
			v := s.props[k]
			tok := s.predTok[predIdx]
			ext := int64(s.predExtra[predIdx])
			b := 0
			for b < nt && v > candT[b] {
				b++
			}
			bHist[b][tok]++
			bExtra[b] += ext
			bCount[b]++
			totHist[tok]++
			totExtra += ext
		}

		// Sweep prefix (right: <= candT[t]) vs complement (left: > candT[t]).
		var rHist [maxTrainTok]int32
		var rExtra int64
		var rCount int32
		totalCount := int32(len(n.samples))
		for t := 0; t < nt; t++ {
			for sym := 0; sym < maxTrainTok; sym++ {
				rHist[sym] += bHist[t][sym]
			}
			rExtra += bExtra[t]
			rCount += bCount[t]
			lCount := totalCount - rCount
			if rCount < 16 || lCount < 16 {
				continue
			}
			var lHist [maxTrainTok]int32
			for sym := 0; sym < maxTrainTok; sym++ {
				lHist[sym] = totHist[sym] - rHist[sym]
			}
			lExtra := totExtra - rExtra
			splitCost := shannonBits(&lHist, lCount, lExtra) + shannonBits(&rHist, rCount, rExtra)
			gain := baseCost - splitCost
			if gain > bestGain {
				bestGain = gain
				bestPropIdx = k
				bestThresh = candT[t]
			}
		}
	}

	if bestPropIdx < 0 || bestGain < minSplitGainBits {
		return
	}

	var leftIdxs, rightIdxs []int
	for _, idx := range n.samples {
		if samples[idx].props[bestPropIdx] > bestThresh {
			leftIdxs = append(leftIdxs, idx)
		} else {
			rightIdxs = append(rightIdxs, idx)
		}
	}
	if len(leftIdxs) < 16 || len(rightIdxs) < 16 {
		return
	}
	_, _, lCost := bestPredictorForSamples(samples, leftIdxs)
	_, _, rCost := bestPredictorForSamples(samples, rightIdxs)
	realGain := baseCost - (lCost + rCost)
	if realGain < minSplitGainBits {
		return
	}
	n.bestProp = treeCandProps[bestPropIdx]
	n.bestSplit = bestThresh
	n.bestGain = realGain
	n.leftIdxs = leftIdxs
	n.rightIdxs = rightIdxs
}

// buildAdaptiveMATree learns a compact MA tree and serializes it in BFS order.
func buildAdaptiveMATree(planes [][]int32, w, h int, fd frameDimensions) ([]treeNode, []encToken, int) {
	samples := collectMATrainSamples(planes, w, h, fd)
	allIdxs := make([]int, len(samples))
	for i := range allIdxs {
		allIdxs[i] = i
	}
	root := &maBuilderNode{property: -1, samples: allIdxs}
	evaluateNodeSplit(samples, root)

	leaves := []*maBuilderNode{root}
	for len(leaves) < maxMALeaves {
		bestIdx := -1
		bestGain := minSplitGainBits
		for i, lf := range leaves {
			if lf.bestGain > bestGain {
				bestGain = lf.bestGain
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		target := leaves[bestIdx]
		target.property = target.bestProp
		target.splitval = target.bestSplit
		leftNode := &maBuilderNode{property: -1, samples: target.leftIdxs}
		rightNode := &maBuilderNode{property: -1, samples: target.rightIdxs}
		target.left = leftNode
		target.right = rightNode
		target.samples = nil
		target.leftIdxs = nil
		target.rightIdxs = nil

		evaluateNodeSplit(samples, leftNode)
		evaluateNodeSplit(samples, rightNode)

		leaves[bestIdx] = leftNode
		leaves = append(leaves, rightNode)
	}

	// Flatten in BFS order matching decodeTree in modular.go.
	var tree []treeNode
	var toks []encToken
	queue := []*maBuilderNode{root}
	leafID := 0
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if cur.property == -1 {
			tree = append(tree, treeNode{
				property:   -1,
				lchild:     leafID,
				predictor:  cur.predictor,
				predOffset: 0,
				multiplier: 1,
			})
			toks = append(toks,
				encToken{ctxProperty, 0},
				encToken{ctxPredictor, cur.predictor},
				encToken{ctxOffset, 0},
				encToken{ctxMultiplierLog, 0},
				encToken{ctxMultiplierBit, 0},
			)
			leafID++
		} else {
			lchild := len(tree) + len(queue) + 1
			rchild := len(tree) + len(queue) + 2
			tree = append(tree, treeNode{
				property:   cur.property,
				splitval:   cur.splitval,
				lchild:     lchild,
				rchild:     rchild,
				multiplier: 1,
			})
			toks = append(toks,
				encToken{ctxProperty, uint32(cur.property + 1)},
				encToken{ctxSplitVal, packSigned(cur.splitval)},
			)
			queue = append(queue, cur.left, cur.right)
		}
	}
	return tree, toks, leafID
}

// writeGlobalModularGroupHeader writes the LfGlobal modular group header,
// emitting a global YCoCg RCT transform (rctType=6) when useRCT is true.
func writeGlobalModularGroupHeader(w *bitWriter, useRCT bool) {
	w.WriteBool(true) // use_global_tree
	w.WriteBool(true) // wp_header all_default
	if !useRCT {
		w.WriteU32(0, u32Val(0), u32Val(1), u32Off(4, 2), u32Off(8, 18)) // num_transforms = 0
		return
	}
	w.WriteU32(1, u32Val(0), u32Val(1), u32Off(4, 2), u32Off(8, 18))            // num_transforms = 1
	w.WriteU32(transformRCT, u32Val(0), u32Val(1), u32Val(2), u32Val(3))         // transform id = RCT (0)
	w.WriteU32(0, u32Bits(3), u32Off(6, 8), u32Off(10, 72), u32Off(13, 1096))    // begin_c = 0
	w.WriteU32(6, u32Val(6), u32Bits(2), u32Off(4, 2), u32Off(6, 10))            // rct_type = 6 (YCoCg)
}

// Encode compresses interleaved samples (matching Decode's output layout: row-
// major, channel-interleaved, little-endian for >8-bit) into a lossless Modular
// JPEG XL codestream. Images larger than one group are split into groups.
func Encode(pixels []byte, w, h, channels, bitdepth int) ([]byte, error) {
	if w <= 0 || h <= 0 || w > 1<<16 || h > 1<<16 {
		return nil, errors.New("gojxl: encode supports 1..65536 px per dimension")
	}
	if channels != 1 && channels != 3 {
		return nil, errors.New("gojxl: encode supports 1 or 3 channels")
	}
	if bitdepth < 1 || bitdepth > 16 {
		return nil, errors.New("gojxl: encode supports 1..16 bit depth")
	}
	bps := 1
	if bitdepth > 8 {
		bps = 2
	}
	if len(pixels) != w*h*channels*bps {
		return nil, errors.New("gojxl: pixel buffer size mismatch")
	}

	// Deinterleave into channel planes.
	planes := make([][]int32, channels)
	for c := range planes {
		planes[c] = make([]int32, w*h)
	}
	for i := 0; i < w*h; i++ {
		for c := 0; c < channels; c++ {
			idx := (i*channels + c) * bps
			v := uint32(pixels[idx])
			if bps == 2 {
				v |= uint32(pixels[idx+1]) << 8
			}
			planes[c][i] = int32(v)
		}
	}

	useRCT := channels == 3
	if useRCT {
		fwdRCTYCoCg(planes)
	}

	gss := groupSizeShiftFor(w, h)
	fd := computeFrameDimensions(uint32(w), uint32(h), gss, 1)
	tree, treeTokens, numLeaves := buildAdaptiveMATree(planes, w, h, fd)
	if fd.numGroups == 1 {
		return encodeSingleGroup(planes, w, h, channels, bitdepth, gss, useRCT, tree, treeTokens, numLeaves)
	}
	return encodeMultiGroup(planes, w, h, channels, bitdepth, gss, fd, useRCT, tree, treeTokens, numLeaves)
}

// encodeSingleGroup writes the whole frame as one section (LfGlobal contains the
// tree, histograms and all channel data).
func encodeSingleGroup(planes [][]int32, w, h, channels, bitdepth int, gss uint32, useRCT bool, tree []treeNode, treeTokens []encToken, numLeaves int) ([]byte, error) {
	wpH := defaultWPHeader()
	var chanTokens []encToken
	for c := 0; c < channels; c++ {
		chanTokens = encodeChannel(planes[c], w, 0, 0, w, h, c, 0, tree, wpH, chanTokens)
	}
	tks, maxToken := tokenizeAll(chanTokens)
	perGroup := [][]encTokenized{tks}

	lf := newBitWriter()
	lf.WriteBool(true) // DequantMatrices DC all_default
	lf.WriteBits(1, 1) // has_tree
	encodeANSStream(lf, treeTokens, numTreeContexts)
	ctxToCluster, clRev, clFreq := writeACEntropyHeader(lf, perGroup, maxToken+1, numLeaves)
	writeGlobalModularGroupHeader(lf, useRCT)
	encodeACDataMulti(lf, tks, ctxToCluster, clRev, clFreq)
	lfBytes := lf.Bytes()

	m := newBitWriter()
	writeContainerHeaders(m, w, h, channels, bitdepth, gss)
	m.WriteBits(0, 1) // TOC: no permutation
	m.ZeroPadToByte()
	writeTocSize(m, len(lfBytes))
	m.ZeroPadToByte()

	return assemble(m.Bytes(), [][]byte{lfBytes}), nil
}

// encodeMultiGroup writes the LfGlobal section (tree + global histogram + group
// header, no pixel data) followed by one byte-aligned section per group, each
// predicted independently and rANS-coded with the shared clustered histograms.
func encodeMultiGroup(planes [][]int32, w, h, channels, bitdepth int, gss uint32, fd frameDimensions, useRCT bool, tree []treeNode, treeTokens []encToken, numLeaves int) ([]byte, error) {
	groupDim := int(fd.groupDim)
	numGroups := int(fd.numGroups)
	xg := int(fd.xsizeGroups)
	wpH := defaultWPHeader()

	// Per-group token lists, in decode order (channel 0..N within the group's
	// rect), and the global maximum token for the shared histograms.
	perGroup := make([][]encTokenized, numGroups)
	maxToken := 0
	for g := 0; g < numGroups; g++ {
		x0 := (g % xg) * groupDim
		y0 := (g / xg) * groupDim
		rw := minInt(groupDim, w-x0)
		rh := minInt(groupDim, h-y0)
		groupID := 1 + 3*int(fd.numDCGroups) + kDequantMatricesNum + g
		var toks []encToken
		for c := 0; c < channels; c++ {
			toks = encodeChannel(planes[c], w, x0, y0, rw, rh, c, groupID, tree, wpH, toks)
		}
		tks, mx := tokenizeAll(toks)
		perGroup[g] = tks
		if mx > maxToken {
			maxToken = mx
		}
	}

	// ----- LfGlobal section: tree + shared histogram header + group header -----
	lf := newBitWriter()
	lf.WriteBool(true) // DequantMatrices DC all_default
	lf.WriteBits(1, 1) // has_tree
	encodeANSStream(lf, treeTokens, numTreeContexts)
	ctxToCluster, clRev, clFreq := writeACEntropyHeader(lf, perGroup, maxToken+1, numLeaves)
	writeGlobalModularGroupHeader(lf, useRCT)
	lfBytes := lf.Bytes()

	// ----- per-group sections: group header + rANS data (shared histogram) -----
	groupBytes := make([][]byte, numGroups)
	for g := 0; g < numGroups; g++ {
		gw := newBitWriter()
		gw.WriteBool(true) // use_global_tree
		gw.WriteBool(true) // wp_header all_default
		gw.WriteU32(0, u32Val(0), u32Val(1), u32Off(4, 2), u32Off(8, 18))
		encodeACDataMulti(gw, perGroup[g], ctxToCluster, clRev, clFreq)
		groupBytes[g] = gw.Bytes()
	}

	// ----- main stream: container headers + multi-section TOC -----
	m := newBitWriter()
	writeContainerHeaders(m, w, h, channels, bitdepth, gss)
	m.WriteBits(0, 1) // TOC: no permutation
	m.ZeroPadToByte()
	// TOC order: LfGlobal, then numDCGroups+1 empty DC/HfGlobal sections, then the
	// modular groups (toc.cc / acGroupIndex = 2 + numDCGroups + g).
	writeTocSize(m, len(lfBytes))
	for i := 0; i < int(fd.numDCGroups)+1; i++ {
		writeTocSize(m, 0)
	}
	sections := make([][]byte, 0, 1+numGroups)
	sections = append(sections, lfBytes)
	for g := 0; g < numGroups; g++ {
		writeTocSize(m, len(groupBytes[g]))
		sections = append(sections, groupBytes[g])
	}
	m.ZeroPadToByte()

	return assemble(m.Bytes(), sections), nil
}

// writeContainerHeaders writes the size header, image metadata, transform data
// and frame header (everything before the TOC).
func writeContainerHeaders(m *bitWriter, w, h, channels, bitdepth int, gss uint32) {
	writeSizeHeader(m, w, h)
	writeImageMetadata(m, channels, bitdepth)
	m.WriteBool(true) // transform_data all_default
	m.ZeroPadToByte()
	writeFrameHeader(m, gss)
}

// writeTocSize writes one TOC entry size with the toc.cc U32 distribution.
func writeTocSize(m *bitWriter, n int) {
	m.WriteU32(uint32(n), u32Bits(10), u32Off(14, 1024), u32Off(22, 17408), u32Off(30, 4211712))
}

// assemble concatenates the codestream signature, main stream and the section
// bytes (in section-index order).
func assemble(mainBytes []byte, sections [][]byte) []byte {
	total := 2 + len(mainBytes)
	for _, s := range sections {
		total += len(s)
	}
	out := make([]byte, 0, total)
	out = append(out, 0xFF, 0x0A)
	out = append(out, mainBytes...)
	for _, s := range sections {
		out = append(out, s...)
	}
	return out
}

func groupSizeShiftFor(w, h int) uint32 {
	d := w
	if h > d {
		d = h
	}
	for gss := uint32(0); gss <= 3; gss++ {
		if (128 << gss) >= d {
			return gss
		}
	}
	return 3
}

func writeSizeHeader(w *bitWriter, xs, ys int) {
	off := []u32d{u32Off(9, 1), u32Off(13, 1), u32Off(18, 1), u32Off(30, 1)}
	w.WriteBool(false) // small
	w.WriteU32(uint32(ys), off[0], off[1], off[2], off[3])
	w.WriteBits(0, 3) // ratio = 0
	w.WriteU32(uint32(xs), off[0], off[1], off[2], off[3])
}

func writeImageMetadata(w *bitWriter, channels, bitdepth int) {
	w.WriteBool(false) // all_default
	w.WriteBool(false) // extra_fields (orientation 1, no preview/animation)
	// BitDepth: integer.
	w.WriteBool(false) // floating_point_sample
	w.WriteU32(uint32(bitdepth), u32Val(8), u32Val(10), u32Val(12), u32Off(6, 1))
	w.WriteBool(bitdepth <= 14)                                      // modular_16bit_buffer_sufficient
	w.WriteU32(0, u32Val(0), u32Val(1), u32Off(4, 2), u32Off(12, 1)) // num_extra_channels = 0
	w.WriteBool(false)                                               // xyb_encoded
	writeColorEncoding(w, channels == 1)
	// extra_fields is false → no tone_mapping.
	w.WriteU64(0) // extensions
}

func writeColorEncoding(w *bitWriter, gray bool) {
	w.WriteBool(false) // all_default
	w.WriteBool(false) // want_icc
	if gray {
		w.WriteEnum(csGray)
	} else {
		w.WriteEnum(csRGB)
	}
	// !want_icc:
	w.WriteEnum(1) // white_point = kD65 (not implicit since not XYB)
	if !gray {
		w.WriteEnum(1) // primaries = kSRGB
	}
	// transfer function (not XYB):
	w.WriteBool(false) // have_gamma
	w.WriteEnum(13)    // transfer_function = kSRGB
	w.WriteEnum(1)     // rendering_intent = kRelative
}

func writeFrameHeader(w *bitWriter, gss uint32) {
	w.WriteBool(false)                                        // all_default
	w.WriteU32(0, u32Val(0), u32Val(1), u32Val(2), u32Val(3)) // frame_type = kRegularFrame
	w.WriteBool(true)                                         // is_modular
	w.WriteU64(0)                                             // flags
	w.WriteBool(false)                                        // color_transform alternate (kNone; xyb_encoded is false)
	w.WriteU32(1, u32Val(1), u32Val(2), u32Val(4), u32Val(8)) // upsampling = 1
	w.WriteBits(uint64(gss), 2)                               // group_size_shift (modular)
	// passes:
	w.WriteU32(1, u32Val(1), u32Val(2), u32Val(3), u32Off(3, 4)) // num_passes = 1
	w.WriteBool(false)                                           // custom_size_or_origin
	// blending (regular frame):
	w.WriteU32(0, u32Val(0), u32Val(1), u32Val(2), u32Off(2, 3)) // BlendMode kReplace
	w.WriteBool(true)                                            // is_last
	// name:
	w.WriteU32(0, u32Val(0), u32Bits(4), u32Off(5, 16), u32Off(10, 48)) // empty name
	// loop_filter: must be explicitly disabled — the default (Gaborish on, EPF=2)
	// would be applied by a conforming decoder and is not lossless.
	w.WriteBool(false) // loop_filter all_default = false
	w.WriteBool(false) // gab (Gaborish) = false
	w.WriteBits(0, 2)  // epf_iters = 0
	w.WriteU64(0)      // loop_filter extensions
	w.WriteU64(0)      // frame extensions
}
