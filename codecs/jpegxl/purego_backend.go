package jpegxl

import (
	"errors"

	"github.com/innovative-io/io-dicom/codecs/jpegxl/gojxl"
)

// gojpegxlBackend is the built-in pure-Go JPEG XL decoder. It handles the
// lossless Modular subset (single frame, RCT/Palette/Squeeze, single- and
// multi-group) and the lossy VarDCT subset for XYB-encoded RGB/grayscale: the
// full common transform set (square/rectangular/large DCTs, DCT2x2/4x4/4x8/8x4,
// IDENTITY, AFV0-3), multi-group, multi-DC-group, permuted TOC, local/global
// modular trees, one or more AC histogram sets, multiple passes (progressive),
// CfL, Gaborish and EPF — at any image size. Inputs outside those subsets
// (VarDCT with the rarely-used DCT128/256, non-XYB color, extra channels, JPEG
// recompression, animation, ICC, non-identity orientation) return an error.
// Encoding supports the lossless Modular subset via gojxl.Encode.
type gojpegxlBackend struct{}

func (gojpegxlBackend) Name() string { return "gojpegxl" }

func (gojpegxlBackend) SupportedTransferSyntaxUIDs() []string {
	return SupportedTransferSyntaxUIDs()
}

// Decode decodes a JPEG XL codestream into output. A panic on malformed input
// is converted to an error so the decoder is safe on hostile/fuzzed data.
func (gojpegxlBackend) Decode(encoded []byte, output []byte) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New("jpegxl: malformed codestream")
		}
	}()
	if len(encoded) == 0 || len(output) == 0 {
		return errInvalidJXLPayload
	}
	img, derr := gojxl.Decode(encoded)
	if derr != nil {
		return derr
	}
	if len(img.Pixels) > len(output) {
		return errInvalidJXLPayload
	}
	copy(output, img.Pixels)
	return nil
}

// kLossyGlobalScale is the base VarDCT quantizer scale used for lossy encoding
// (8192 = qf 1.0 in JPEG XL's 65536-denominated globalScale).
const kLossyGlobalScale = 8192

// Encode compresses raw interleaved samples to a JPEG XL codestream. Lossless
// uses the pure-Go Modular encoder (any size, 8/16-bit, 1/3 channels); lossy
// uses the pure-Go VarDCT encoder (8/16-bit, 1/3 channels, up to 16384px). Requests
// outside those subsets error so a native backend can take over.
func (gojpegxlBackend) Encode(raw []byte, width uint16, height uint16, samples uint16, bitsa uint16, lossless bool) (out []byte, err error) {
	defer func() {
		if r := recover(); r != nil {
			out, err = nil, errors.New("jpegxl: encode failed")
		}
	}()
	if !lossless {
		if bitsa != 8 && bitsa != 16 {
			return nil, errors.New("jpegxl: pure-Go lossy encode supports 8-bit and 16-bit only")
		}
		scale := kLossyGlobalScale
		if bitsa == 16 && len(raw) >= 2 {
			// Medical 16-bit DICOM images (e.g. 10–12 bits stored in 16-bit
			// containers) often use only a small fraction of [0, 65535]. Scale
			// globalScale so the active signal range gets comparable DCT
			// quantization precision to a full-range image.
			var peak uint16
			for i := 0; i+1 < len(raw); i += 2 {
				v := uint16(raw[i]) | uint16(raw[i+1])<<8
				if v > peak {
					peak = v
				}
			}
			if peak > 0 && peak < 32768 {
				// Boost quantizer precision for narrow-dynamic-range 16-bit frames,
				// capped at 65536 (maximum 16-bit U32 globalScale).
				boost := 65535 / int(peak)
				if boost > 8 {
					boost = 8
				}
				scale *= boost
				if scale > 65536 {
					scale = 65536
				}
			}
		}
		return gojxl.EncodeVarDCTWithBitDepth(raw, int(width), int(height), int(samples), int(bitsa), scale)
	}
	return gojxl.Encode(raw, int(width), int(height), int(samples), int(bitsa))
}

func registerPureGoBackend() {
	// gojpegxl is the only built-in backend; an external one can still be plugged
	// in at a higher priority via RegisterBackend.
	_ = mgr.RegisterWithPriority("gojpegxl", func() Backend { return gojpegxlBackend{} }, -1)
}
