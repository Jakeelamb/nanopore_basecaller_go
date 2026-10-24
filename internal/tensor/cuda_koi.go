package tensor

import "fmt"

// KoiDType mirrors the pinned KoiTypeId enum (koi.h:3).
type KoiDType uint32

const (
	KoiNone KoiDType = 0
	KoiE2M1 KoiDType = 1
	KoiI8   KoiDType = 2
	KoiI32  KoiDType = 3
	KoiE4M3 KoiDType = 4
	KoiF16  KoiDType = 5
	KoiF32  KoiDType = 6
)

func (t KoiDType) String() string {
	switch t {
	case KoiNone:
		return "KOI_NONE"
	case KoiE2M1:
		return "KOI_E2M1"
	case KoiI8:
		return "KOI_I8"
	case KoiI32:
		return "KOI_I32"
	case KoiE4M3:
		return "KOI_E4M3"
	case KoiF16:
		return "KOI_F16"
	case KoiF32:
		return "KOI_F32"
	}
	return fmt.Sprintf("KoiTypeId(%d)", uint32(t))
}

// KoiDim is one tagged KoiTensor dimension.
type KoiDim struct {
	Tag          KoiTag
	Size, Stride int64
}

// KoiScale is the typed optional scale owner of an I8 Koi tensor. The pinned
// source permits a scale only on an I8 tensor and requires F32 scale data
// (TxModules.cpp:36-47).
type KoiScale struct {
	DType KoiDType
	Tag   KoiTag
}

// KoiTensor is the checked descriptor Go hands to a Koi kernel boundary. It
// carries no pointers: the native side binds it to the live storage handle.
type KoiTensor struct {
	DType KoiDType
	Dims  []KoiDim
	Scale *KoiScale
}

func koiType(d DType) (KoiDType, error) {
	switch d {
	case DTypeF16:
		return KoiF16, nil
	case DTypeI8:
		return KoiI8, nil
	case DTypeI32:
		return KoiI32, nil
	case DTypeF32:
		return KoiF32, nil
	case DTypeE4M3:
		return KoiE4M3, nil
	}
	return KoiNone, &LayoutError{Op: "koi_descriptor", Reason: fmt.Sprintf("%s has no Koi type", d)}
}

// KoiTensor validates the live storage and returns its typed Koi descriptor.
func (s *Storage) KoiTensor() (KoiTensor, error) {
	var k KoiTensor
	m, err := s.Meta()
	if err != nil {
		return k, err
	}
	dtype, err := koiType(m.DType)
	if err != nil {
		return k, err
	}
	k.DType = dtype
	k.Dims = make([]KoiDim, len(m.Shape))
	for i := range m.Shape {
		if !m.Tags[i].valid() {
			return k, &LayoutError{Op: "koi_descriptor", Reason: fmt.Sprintf("unknown Koi dimension tag %d", int32(m.Tags[i]))}
		}
		if m.Strides[i] < 0 {
			return k, &LayoutError{Op: "koi_descriptor", Reason: "negative Koi dimension stride"}
		}
		k.Dims[i] = KoiDim{Tag: m.Tags[i], Size: m.Shape[i], Stride: m.Strides[i]}
	}
	if m.ScalePresent {
		if dtype != KoiI8 {
			return k, &DTypeError{Op: "koi_descriptor", Reason: "Koi scale requires an I8 tensor", Detail: CUDADetail{ActualDType: ScalarType(m.DType), ExpectedDType: ScalarType(DTypeI8)}}
		}
		if m.ScaleDType != DTypeF32 {
			return k, &DTypeError{Op: "koi_descriptor", Reason: "Koi scale must be F32", Detail: CUDADetail{ActualDType: ScalarType(m.ScaleDType), ExpectedDType: ScalarType(DTypeF32)}}
		}
		if !m.ScaleTag.valid() {
			return k, &LayoutError{Op: "koi_descriptor", Reason: "unknown Koi scale tag"}
		}
		k.Scale = &KoiScale{DType: KoiF32, Tag: m.ScaleTag}
	}
	return k, nil
}
