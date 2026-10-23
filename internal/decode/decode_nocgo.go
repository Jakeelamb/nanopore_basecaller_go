//go:build !cgo

package decode

// Decode reports that native float libm operations are unavailable in this build.
func Decode(Input, Options) (Result, error) { return Result{}, ErrCGORequired }
