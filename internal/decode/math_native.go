//go:build cgo

package decode

/*
#cgo LDFLAGS: -lm
#include <math.h>

static float decoder_log(float x) { return logf(x); }
static float decoder_log1p(float x) { return log1pf(x); }
static float decoder_exp(float x) { return expf(x); }
static float decoder_pow(float x, float y) { return powf(x, y); }
static float decoder_log10(float x) { return log10f(x); }
*/
import "C"

func floatLog(x float32) float32    { return float32(C.decoder_log(C.float(x))) }
func floatLog1p(x float32) float32  { return float32(C.decoder_log1p(C.float(x))) }
func floatExp(x float32) float32    { return float32(C.decoder_exp(C.float(x))) }
func floatPow(x, y float32) float32 { return float32(C.decoder_pow(C.float(x), C.float(y))) }
func floatLog10(x float32) float32  { return float32(C.decoder_log10(C.float(x))) }
