//go:build !cgo

package decode

// Decode reports that native float libm operations are unavailable in this build.
func Decode(Input, Options) (Result, error) { return Result{}, ErrCGORequired }

// MeanQScore requires the same native float libm semantics as Decode.
func MeanQScore(string) (float32, error) { return 0, ErrCGORequired }
