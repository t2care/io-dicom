  I'm currently reviewing the latest changes to ensure everything is clearly summarized for you. I've decided to keep your `TestTranscode` funct...
  ### Why JPEG XL Compression Ratio Was Poor

  1. Degenerate ANS Histograms (writeANSDegenerateHeader):
      • Both the lossless Modular encoder (Encode) and the lossy VarDCT encoder (EncodeVarDCT) were using a single degenerate ANS histogram that
      pinned every rANS state strictly to token 0.
      • Any residual that mapped to tok != 0 had to be forced through tok = 0 (nbits = 4), wasting 4 bits on small non-zero residuals and failing
      on |residual| >= 8.
      • To avoid exceeding |residual| >= 8, the Modular encoder set splitBits = 8 (encUintConfig = {8, 0, 0}) so every single pixel
      unconditionally emitted 8 raw extra bits, guaranteeing ≥1 byte/pixel on 8-bit images and ≥1.19 bytes/pixel on 16-bit images (expanding
      testdata/MR from 512 KB raw pixel data to 623 KB).
  2. Single-Leaf MA Tree & No RCT in Modular (enc_encode.go):
      • Every pixel shared a single unclustered context with no adaptation to local gradients/smooth vs. edge regions, and 3-channel RGB lacked
      the reversible YCoCg color transform (rctType = 6).
  3. 16-Bit Rejection in .112 (gojpegxlBackend.Encode) & Destructive Error Handling in .111 (transcoder.go):
      • .112 (JPEGXL lossy VarDCT) rejected 16-bit DICOM images outright (bitsa != 8).
      • .111 (JPEGXLJPEGRecompression) called uncompress before checking whether the source was encapsulated JPEGBaseline8Bit (.50), and
      uncompress mutated the DICOMObject (stripping encapsulated fragments) before compress ran.

  ──────
  ### Changes Made

  1. **Lossless Modular Upgrade (.110 JPEGXLLossless) — enc_encode.go**:
      • Switched to standard Modular hybrid-uint configuration (splitExponent = 4, msbInToken = 2, lsbInToken = 0) with real frequency-based ANS
      histograms and context clustering via writeACEntropyHeader and encodeACDataMulti.
      • Added buildAdaptiveMATree and encodeChannel, training a compact global MA decision tree (up to 32 leaves over channel, kWPProp, gradient,
      and neighbor properties) that selects the optimal predictor per leaf (predWeighted, predGradient, predSelect, predAverage4, predAverage0,
      predLeft, predTop).
      • Added forward lossless YCoCg RCT (fwdRCTYCoCg and writeGlobalModularGroupHeader) for 3-channel RGB images.
  2. **Lossy VarDCT Upgrade & 16-Bit Support (.112 JPEGXL) — vardct_encode.go, vardct_reconstruct.go, purego_backend.go**:
      • Added EncodeVarDCTWithBitDepth supporting both 8-bit and 16-bit inputs and updated reconstructVarDCT to reconstruct 16-bit little-endian
      samples when BitDepth > 8.
      • Replaced degenerate ANS streams in assembleVarDCT and finishVarDCT with real ANS histograms (writeANSRealHeader) and context-clustered AC
      coefficient coding (acTokensByGroupJPEGCtx).
      • Updated gojpegxlBackend.Encode to support 8-bit and 16-bit lossy VarDCT with dynamic-range-aware quantization scaling for 16-bit medical
      images.
  3. **Strict, Non-Destructive .111 (JPEGXLJPEGRecompression) — transcoder.go**:
      • Validated up front in ChangeTransferSyntax and compress that .111 requires an encapsulated JPEGBaseline8Bit (.50) source.
      • Deferred stripping encapsulated fragments to stripEncapsulatedItems until after all target frames are encoded, ensuring failed transcodes
      never mutate the DICOMObject.

  ──────
  ### Compression Results on testdata/MR (512×512, 16-bit grayscale, 524,288 B raw pixels)

   Transfer Syntax                       |     Encapsulated Frame Size |       Full DICOM File Size |          Compression Ratio (Pixel Payload)
  ---------------------------------------|-----------------------------|----------------------------|--------------------------------------------
   Uncompressed (ExplicitVRLittleEndian) |                   524,288 B |                  542,892 B |                                      1.00×
   JPEGLosslessSV1 (.70)                 |                   214,922 B |                  233,524 B |                                      2.44×
   JPEGXLLossless (.110) (before)        |                   619,492 B |                  638,096 B |                           0.85× (expanded)
   JPEGXLLossless (.110) (after)         |                   172,224 B |                  190,828 B | 3.04× (19.9% smaller than JPEGLosslessSV1)
   JPEGXL lossy (.112) (after)           |                     6,178 B |                   24,782 B |                                     84.86×