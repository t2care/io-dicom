package transcoder_test

import (
	"bytes"
	"os"
	"testing"

	"github.com/innovative-io/io-dicom/dictionary/tags"
	"github.com/innovative-io/io-dicom/dictionary/transfersyntax"
	"github.com/innovative-io/io-dicom/media"
	"github.com/innovative-io/io-dicom/transcoder"
)

func TestRGBLosslessSV1SyntheticFixtureRoundTrip(t *testing.T) {
	const samplePath = "../testdata/synthetic-cr-color.dcm"
	data, err := os.ReadFile(samplePath)
	if err != nil {
		t.Skipf("sample %s unavailable: %v", samplePath, err)
	}
	obj, err := media.NewDCMObjFromBytes(data)
	if err != nil {
		t.Fatalf("NewDCMObjFromBytes: %v", err)
	}

	origPx, err := obj.GetPixelData(0)
	if err != nil {
		t.Fatalf("GetPixelData original: %v", err)
	}
	if len(origPx) == 0 {
		t.Fatal("empty original pixel data")
	}

	// Forward transcode to JPEGLosslessSV1
	if err := transcoder.ChangeTransferSyntax(obj, transfersyntax.JPEGLosslessSV1); err != nil {
		t.Fatalf("ChangeTransferSyntax to JPEGLosslessSV1 failed: %v", err)
	}
	if obj.GetTransferSyntax().UID != transfersyntax.JPEGLosslessSV1.UID {
		t.Fatalf("expected transfer syntax %s, got %s", transfersyntax.JPEGLosslessSV1.UID, obj.GetTransferSyntax().UID)
	}

	// Decompress back to ExplicitVRLittleEndian
	if err := transcoder.ChangeTransferSyntax(obj, transfersyntax.ExplicitVRLittleEndian); err != nil {
		t.Fatalf("ChangeTransferSyntax back to ExplicitVRLittleEndian failed: %v", err)
	}

	roundtripPx, err := obj.GetPixelData(0)
	if err != nil {
		t.Fatalf("GetPixelData decoded: %v", err)
	}

	if !bytes.Equal(origPx, roundtripPx) {
		t.Fatalf("lossless roundtrip mismatch: orig len %d vs roundtrip len %d", len(origPx), len(roundtripPx))
	}
}

func TestRGBMultiPixelLosslessSV1RoundTrip(t *testing.T) {
	obj := media.NewEmptyDCMObj()
	obj.SetTransferSyntax(transfersyntax.ExplicitVRLittleEndian)
	obj.SetExplicitVR(true)
	obj.SetBigEndian(false)

	obj.Write(tags.SOPClassUID, "1.2.840.10008.5.1.4.1.1.7")
	obj.Write(tags.SOPInstanceUID, "1.2.826.0.1.3680043.10.90.3")
	obj.Write(tags.SamplesPerPixel, 3)
	obj.Write(tags.PhotometricInterpretation, "RGB")
	obj.Write(tags.PlanarConfiguration, 0)
	obj.Write(tags.NumberOfFrames, "1")
	obj.Write(tags.Rows, 4)
	obj.Write(tags.Columns, 4)
	obj.Write(tags.BitsAllocated, 8)
	obj.Write(tags.BitsStored, 8)
	obj.Write(tags.PixelRepresentation, 0)

	// 4x4 RGB = 16 pixels * 3 = 48 bytes with distinct gradients
	pixelData := make([]byte, 48)
	for i := 0; i < len(pixelData); i++ {
		pixelData[i] = byte((i * 17) % 256)
	}

	pixel := &media.DICOMTag{
		Group:     0x7FE0,
		Element:   0x0010,
		Length:    uint32(len(pixelData)),
		VR:        "OB",
		Data:      pixelData,
		BigEndian: false,
	}
	media.FillTag(pixel)
	obj.Add(pixel)

	for _, ts := range []*transfersyntax.TransferSyntax{
		transfersyntax.JPEGLosslessSV1,
		transfersyntax.JPEGLossless,
	} {
		t.Run(ts.Name, func(t *testing.T) {
			testObj := obj.Clone()
			if err := transcoder.ChangeTransferSyntax(testObj, ts); err != nil {
				t.Fatalf("ChangeTransferSyntax to %s: %v", ts.Name, err)
			}
			if err := transcoder.ChangeTransferSyntax(testObj, transfersyntax.ExplicitVRLittleEndian); err != nil {
				t.Fatalf("ChangeTransferSyntax back to ExplicitVRLittleEndian: %v", err)
			}
			out, err := testObj.GetPixelData(0)
			if err != nil {
				t.Fatalf("GetPixelData: %v", err)
			}
			if !bytes.Equal(pixelData, out) {
				t.Fatalf("roundtrip mismatch for %s: got %v, want %v", ts.Name, out, pixelData)
			}
		})
	}
}
