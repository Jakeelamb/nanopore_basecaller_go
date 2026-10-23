# Go model graph contract

Reference: Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. This document
describes the CPU fallback graphs for the two local stock models. Go owns config
recognition, named weight binding, layer ordering, recurrent state, reversals,
attention splits and output layouts. The native module exposes tensor primitives;
it does not construct or execute a Dorado model or layer graph.

CPU reference tensors are float32. `CPUDecoder::dtype()` returns Float;
[`ModelRunner.cpp:9-25`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/ModelRunner.cpp#L9-L25)
uses that type for model and input. The model returns `[N,T,C]`; the CPU runner
then transposes to contiguous `[T,N,C]` before decoding (lines 30-37). CUDA's
binary16, quantized and tiled paths require separate evidence.

## Config recognition

Initially recognize the exact config bytes by SHA-256 and return immutable typed
specs. Reject unknown configs before loading weights. A generic TOML interpreter
is unnecessary for these two admitted inputs. Model directory names alone are
insufficient identities. Validate all consumed weight files against an immutable
manifest as well as their expected names, shapes, dtype and tensor count.

| Field | HAC v6.0.0 | SUP v5.2.0 |
| --- | --- | --- |
| Directory | `dna_r10.4.1_e8.2_400bps_hac@v6.0.0` | `dna_r10.4.1_e8.2_400bps_sup@v5.2.0` |
| `config.toml` SHA-256 | `e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f` | `ad82d7c369deddc0d1e69367302b8e002c306a9770ae4304a3cbb70715fcce49` |
| Input features / sample rate | 1 / 5000 | 1 / 5000 |
| Scaling | pA, standardise | pA, standardise |
| Mean / stdev | 94 / 24 | float32(93.69239463939118) / float32(23.506745239082388) |
| Q scale / bias | float32(1.02) / float32(-0.3) | float32(1.05) / float32(0.3) |
| Mean Q start position | 60 | 60 |
| State length / transition channels | 4 / 1024 | 5 / 4096 |
| Blank score | 2 | 2 |
| Convolution stride / upsample | 6 / 1 | 12 / 2 |
| Effective model stride | 6 | 6 |
| Inner stride / chunk granularity | 6 / 6 | 12 / 768 |
| Config chunk / overlap | 12288 / 600 | 12288 / 600 |

`BasecallModelConfig.cpp` parses floats into float32, defaults mean-Q start to 60,
uses HAC's `global_norm.state_len` (ignoring its duplicate CRF state length), and
computes channels as `4^(state_len+1)` without explicit stay channels.
Sources: [qscore and normalisation, lines 64-197](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/config/BasecallModelConfig.cpp#L64-L197),
[HAC parsing, lines 214-325](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/config/BasecallModelConfig.cpp#L214-L325),
[SUP parsing, lines 373-472](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/config/BasecallModelConfig.cpp#L373-L472).

Config values do not imply running arbitrary TOML sublayers. The pinned parser
ignores convolution `padding`, `bias`, `norm`, explicit `permute` sublayers and
FLSTM `reverse`; constructors select the implemented behavior. Padding is
`winlen/2`, convolution biases are present, the stack starts reverse, and the
convolution module always transposes NCT to NTC. Both exact configs agree with
these choices. SUP has no runtime batch-normalization layers or BN weights in
the consumed state dict; do not add one from its `norm="batchnorm"` labels.

`stride_inner = stride * scale_factor` and transformer chunk granularity is
`stride_inner * 64` ([header lines 151-162](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/config/include/config/BasecallModelConfig.h#L151-L162)).
[`BatchParams.cpp:89-104`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/config/BatchParams.cpp#L89-L104)
normalizes overlap to `max(1, overlap/innerStride)*innerStride`, and chunk size to
`floor(max(overlap+granularity-1, chunk)/granularity)*granularity`. Both supplied
defaults already satisfy this. Production input lengths should obey the selected
planner's granularity, rather than rely on arbitrary convolution ceil behavior.

## Weight loading

[`crf_utils.cpp:30-153`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/crf_utils.cpp#L30-L153)
defines a deterministic filename vector. Each file is a Torch vector archive;
[`tensor_utils.cpp:152-160`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/torch_utils/tensor_utils.cpp#L152-L160)
appends its tensor through `torch::load(vector, path)`. Native
[`module_utils.h:13-31`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/torch_utils/include/torch_utils/module_utils.h#L13-L31)
binds the resulting vector by parameter registration order, rather than filename.
Go should bind named fields explicitly and require exactly one tensor per file.

Read-only ZIP metadata inspection of every local archive established that all
207 files contain one contiguous float32 tensor with storage offset zero. HAC
consumes all 50 files (9,808,240 elements; 39,232,960 storage bytes). SUP consumes
139 of 157 files (78,718,144 elements; 314,872,576 storage bytes). Its other 18
files are `transformer_encoder.<i>.deepnorm_alpha.tensor`, scalar float32 values
2.44921875. The native loader ignores those files and initializes alpha from the
config, approximately 2.4494897. Never substitute the unused scalar files.

These are raw tensor storage sizes, not archive sizes or runtime memory claims.
This inspection decoded only the constrained archive metadata and raw alpha
scalar bytes; it did not import Torch, execute a model, or prove native loading.
The first metadata check failed by assuming SUP had only seven filename patterns;
the corrected check explicitly accounted for the eighteen unconsumed scalars.

### HAC ordered binding

| Order | Filename | Shape |
| --- | --- | --- |
| 0,1 | `0.conv.weight.tensor`, `0.conv.bias.tensor` | `[16,1,5]`, `[16]` |
| 2,3 | `1.conv.weight.tensor`, `1.conv.bias.tensor` | `[16,16,5]`, `[16]` |
| 4,5 | `2.conv.weight.tensor`, `2.conv.bias.tensor` | `[1024,16,19]`, `[1024]` |
| `6+6*l` through `11+6*l`, `l=0..6` | `<4+l>.rnn.dn_weight_ih.tensor`, `.dn_weight_hh.tensor`, `.up_weight_ih.tensor`, `.up_weight_hh.tensor`, `.up_bias_ih.tensor`, `.up_bias_hh.tensor` | `[128,1024]`, `[128,1024]`, `[4096,128]`, `[4096,128]`, `[4096]`, `[4096]` |
| 48 | `11.linear.weight.tensor` | `[128,1024]` |
| 49 | `12.linear.weight.tensor` | `[1024,128]` |

For the recurrent row each suffix replaces `.dn_weight_ih.tensor` after the
shared `<4+l>.rnn` prefix. Native registered names are `convs.conv1..3.weight/bias`,
`rnns.rnn1..7.{dn_weight_ih,dn_weight_hh,up_weight_ih,up_weight_hh,up_bias_ih,up_bias_hh}`,
`linear1.linear.weight`, and `linear2.linear.weight` on the CPU path.

### SUP ordered binding

First load convolution weight then bias for indices 0 through 4:

| Prefix | Weight shape | Bias shape |
| --- | --- | --- |
| `conv.0.conv` | `[64,1,5]` | `[64]` |
| `conv.1.conv` | `[64,64,5]` | `[64]` |
| `conv.2.conv` | `[128,64,9]` | `[128]` |
| `conv.3.conv` | `[128,128,9]` | `[128]` |
| `conv.4.conv` | `[512,128,5]` | `[512]` |

For each `i=0..17`, load the following seven suffixes after
`transformer_encoder.<i>.`, in this order:

| Suffix | Shape |
| --- | --- |
| `self_attn.Wqkv.weight.tensor` | `[1536,512]` |
| `self_attn.out_proj.weight.tensor` | `[512,512]` |
| `self_attn.out_proj.bias.tensor` | `[512]` |
| `ff.fc1.weight.tensor` | `[4096,512]` |
| `ff.fc2.weight.tensor` | `[512,2048]` |
| `norm1.weight.tensor` | `[512]` |
| `norm2.weight.tensor` | `[512]` |

Finish with `upsample.linear.weight.tensor [1024,512]`,
`upsample.linear.bias.tensor [1024]`, `crf.linear.weight.tensor [4096,512]`.
The 18 transformer layers have identical metadata patterns. Native names use
`self_attn.wqkv` (lowercase) and nested registered encoder names; explicit Go
fields avoid depending on their spelling or registration order.

## HAC forward graph

Convolutions consume NCT, with weights `[Cout,Cin,K]`, ordinary cross-correlation,
dilation 1 and groups 1. Apply `Conv1d -> SiLU` twice (kernels 5, stride 1,
padding 2), then `Conv1d -> Tanh` (kernel 19, stride 6, padding 9). Transpose
dimensions 1 and 2 to NTC. For arbitrary positive lengths the third convolution
returns `ceil(L/6)`; admitted aligned lengths return `L/6`.
Source: [`ConvStack.cpp:110-118,169-185`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/ConvStack.cpp#L110-L118).

Each FLSTM layer consumes NTC. Transpose to contiguous TNC; initialize `h` and
`c` to zeros `[N,C]`. Compute all input projections once:

```text
ih = ((x @ dn_weight_ih.T) @ up_weight_ih.T) + up_bias_ih
for t = 0..T-1:
    z = (((h @ dn_weight_hh.T) @ up_weight_hh.T) + up_bias_hh) + ih[t]
    i, f, g, o = split z into four equal channel groups, in that order
    i = clamp(i*0.2f + 0.5f, 0, 1)
    f = clamp(f*0.2f + 0.5f, 0, 1)
    g = clamp(g, -1, 1)
    o = clamp(o*0.2f + 0.5f, 0, 1)
    c = (f*c) + (i*g)
    h = o*tanh(c)
    output[t] = h
return output.transpose(0,1).contiguous()
```

All intermediate tensors use the input's dtype/device; the CPU path uses F32.
Keep the two matmuls and the two bias additions in source order. Folding matrices
or biases changes rounding. The source hard gate constants are float32; encode
the 0.2 scalar as `float64(float32(0.2))` when passing a scalar through Go's
float64 primitive interface. A fused ordinary LSTM is incorrect. Source:
[`FLSTMStack.cpp:19-65`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/FLSTMStack.cpp#L19-L65).

The stack flips dimension 1 before the first layer and before every subsequent
layer. It does not flip each layer back immediately. With seven layers and
`first_reverse=true`, it flips once after the last layer to restore time order.
This gives reverse/forward/reverse/forward/reverse/forward/reverse relative to
the original sequence. Preserve this exact loop for rounding and layout parity:
[`FLSTMStack.cpp:83-92`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/FLSTMStack.cpp#L83-L92).

Then apply bias-free `Linear(1024,128)`, bias-free `Linear(128,1024)`, `Tanh`,
multiply by 5. The config has no clamp sublayer, so the final `Clamp(-5,5)` module
is inactive. Output is contiguous `[N,L/6,1024]` for aligned input. Sources:
[`CRFModel.cpp:31-79`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/model/CRFModel.cpp#L31-L79),
[`CRFModules.cpp:15-34,125-134`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/CRFModules.cpp#L15-L34).

## SUP forward graph

Run five `Conv1d -> SiLU` layers on NCT with `(Cin,Cout,K,S,P)`:
`(1,64,5,1,2)`, `(64,64,5,1,2)`, `(64,128,9,3,4)`,
`(128,128,9,2,4)`, `(128,512,5,2,2)`; transpose to NTC. For aligned
input length L, transformer time is L/12. There are 18 identical encoder shapes:

1. Bias-free QKV linear `[N,T,512] -> [N,T,1536]`; reshape `[N,T,3,8,64]`.
2. Apply RoPE to Q and K only. Split their last dimension into the first and
   second 32 elements, rather than adjacent even/odd pairs. Let
   `invFreq[j] = 1 / float32(pow(10000.0, float32(2*j)/float32(64)))`.
   The source computes the exponentiation in double, casts to model options,
   then takes reciprocal. Build position frequencies `t*invFreq[j]`, cosine
   and sine in the model dtype. For halves a,b, emit
   `[cos*a-sin*b, sin*a+cos*b]`. V is copied unchanged. Attention layout is
   `[3,N,8,T,64]`. Defaults: theta 10000, max sequence length 4096; reject
   larger transformer T before slicing buffers.
3. Run scaled dot-product attention over the exact source splits described below,
   no dropout and noncausal. Scale is `1/sqrt(64)`. Assemble attention as NTC.
4. Out projection `Linear(512,512)` with bias.
5. `x = RMSNorm(attn + x*alpha, norm1Weight)`.
6. Bias-free `fc1:512->4096`; split last dimension `[y,gate]`, each 2048.
   Compute `SiLU(gate)*y`, then bias-free `fc2:2048->512`.
7. `x = RMSNorm(ff + x*alpha, norm2Weight)`.

`RMSNorm(z,w) = (z*rsqrt(mean(z*z, lastDim, keepDim=true)+1e-5f))*w`.
Alpha is float32(config 2.4494897), used at both residuals. `deepnorm_beta`
0.2886751 in TOML is not parsed or used at inference. Default `use_uswiglu` is
false; do not substitute the optional clamped 1.702 gate variant.
Sources: [`TxModules.cpp:128-228`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/TxModules.cpp#L128-L228),
[lines 257-399](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/TxModules.cpp#L257-L399),
[lines 756-834](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/TxModules.cpp#L756-L834),
[`RMSNorm.cpp:8-16`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/RMSNorm.cpp#L8-L16)
and its header's `eps=1e-5f`.

Attention uses default `num_splits=12`, `mha_use_mask=true`. Source names
`[win_upper,win_lower]=[127,128]`. Its full boolean mask is
`ones(T,T).triu(-127).tril(128)`, so an allowed key j for query i satisfies
`-127 <= j-i <= 128`. For each split:

```text
E = roundUp(ceil(T/12),4)
qb = splitIndex*E; qe=min(T,qb+E); stop if qb>=T
kvb=max(0,qb-128); kve=min(T,qe+127)
q=Q[:,:,qb:qe,:]; k=K[:,:,kvb:kve,:]; v=V[:,:,kvb:kve,:]
mask=fullMask[qb:qe,kvb:kve]
out[:,:,qb:qe,:]=scaled_dot_product_attention(q,k,v,mask)
```

The key-range expression and mask have different asymmetric end conventions.
Copy both literally: replacing this with one full masked attention call can
include a key excluded at a split boundary. A test must probe those boundaries
with T large enough to extend past the 127/128 window. Source:
[`TxModules.cpp:288-295,350-379`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/TxModules.cpp#L350-L379).

After the encoders, upsample with `Linear(512,1024)` including bias and reshape
`[N,T,1024] -> [N,2*T,512]`. This interleaves two adjacent time outputs per
input position. CRF output is bias-free `Linear(512,4096)` after multiplying
its weights by 5 **once**. It has no tanh and no score clamp. Pre-scale owned
weights at load time, preserving the native F32 multiply, so concurrent forward
calls never race on first-use mutation. Output `[N,L/6,4096]`. `permute` and
`expand_blanks` do not introduce a runtime permutation or explicit blank channel
in this model. Sources: [`LinearUpsample.cpp:7-22`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/LinearUpsample.cpp#L7-L22),
[`TxModules.cpp:946-957`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/nn/TxModules.cpp#L946-L957).

## Two graph-control shapes

| Shape | Concrete representation | Strength | Cost |
| --- | --- | --- | --- |
| Tagged module plans | `Layer{Kind,Conv,FLSTM,Transformer,Linear}` slice; a Go interpreter validates tags and runs native primitives; each complex tag expands to Go loops | Easy declarative traces and future config import | Optional fields admit invalid combinations; layer-specific invariants move into a runtime validator; dispatch obscures state ownership |
| Typed layer structs | `hacGraph{convs []convLayer, rnns []flstmLayer, bottleneck, crf}` and a future typed SUP graph, with ordinary Go methods; the admitted HAC manifest fixes counts to 3 and 7 | Each layer owns its exact named weights; hard gates and attention control remain visible; fewer illegal states | A new architecture needs new source rather than an extensible plan |

Choose typed structs. Two pinned architectures do not justify a general graph
interpreter. Keep a single small public model interface and concrete private
layers. Suggested caller usage and interface:

```go
eng, err := tensor.Open(nativeLibrary)
// caller owns eng; model owns only its loaded/transformed weight handles
m, err := model.Open(eng, modelDirectory) // hash-gated, named, shape-checked
scores, err := m.Forward(ctx, inputNCT)   // caller owns returned NTC tensor
// consume scores, then Close it; close m before eng

spec, err := model.Inspect(modelDirectory) // config metadata without loading weights
receipts := m.Weights()                   // copied names, hashes and shapes
```

`Model.Spec()` returns a value. `Open` succeeds only for the implemented HAC
graph. `Inspect` recognizes SUP, while `Open` returns `ErrUnsupported` for SUP
before allocating weights. The runtime keeps no metadata-only model that can
be mistaken for usable inference. The embedded HAC weight manifest pins all
50 archive hashes. Each load checks the hash before and after native loading,
requires one CPU F32 tensor and validates its exact shape.

The initial tensor Engine methods already selected with its owner are
`LoadWeight`, `NewFloat32`, `Zeros`, `Conv1d(x,w,b,stride,padding,dilation,groups)`,
`Linear`, `Transpose`, `Permute`, `Contiguous`, `Flip`, `MatMul`, `Add`, `Mul`,
`AddScalar`, `MulScalar`, `Clamp`, `Sigmoid`, `SiLU`, `Tanh`, `Narrow`, `Concat`,
`Stack`, `Reshape`. Methods return `(*Tensor,error)`; `Tensor.Shape/Data` return
copied slices and errors. Nil bias is supported. All operations are out of place.
This suffices for HAC. SUP additionally needs primitive mean, reciprocal square
root, sine/cosine and boolean-mask scaled-dot-product attention; Go must still
own rotary construction, split scheduling and residual order. Prefer these
ATen operations over exporting a native MHA or transformer layer.

Validate input rank, batch/channels/length, granularity, finite numeric input,
weight shapes and engine ownership at the external seam. Tensor operations
already reject closed/foreign-engine handles. A model forward owns all its
temporary handles, releases them on every return path, and returns independent
output storage ownership. ATen view handles retain shared underlying storage,
so closing a parent handle must not invalidate a live view. Do not retain every
step's gates: release step temporaries and previous h/c once their outputs exist.
Keep output h slices only until stack/concat has created the layer result.

Weights are immutable after loading. A read collector owns chunk results;
model code does not append into shared reads. Engine calls serialize registry
access, but graph forward calls may interleave only when their states and
temporaries are disjoint. Close must not race with Forward; simplest initial
policy is one model forward/close mutex and a caller-owned engine lifetime.
Cancellation checks between layers and recurrent steps release temporaries;
it cannot interrupt an active synchronous ATen call. No partial scores are
returned on error. Native code must catch C++ exceptions and return diagnostics
without crossing the C ABI or retaining Go memory.

## Implemented source-linked HAC oracle

`verification/model/oracle.go` checks six whole source-file hashes and extracts
verbatim definitions into the verification adapter. Its source slices are:

| Source | Included lines |
| --- | --- |
| `nn/FLSTMStack.cpp` | 19-93: layer/stack constructors and CPU forwards |
| `nn/ConvStack.cpp` | 110-128, 169-188: constructor, CPU forward, layer constructor |
| `nn/CRFModules.cpp` | 15-34, 125-134: linear and clamp constructors/forwards |
| `basecall/model/CRFModel.cpp` | 66: exact encoder sequence; 135-146: model forward |
| `torch_utils/torch_utils.cpp` | 15-34: initialization and deterministic controls |
| `config/include/config/common.h` | Included unchanged, whole-file hash checked |

The generated C++20 executable uses CPU-only branches and the official ONT
LibTorch 2.9.0.0-ont CUDA-13 development package. It calls the original
initialization functions (one CPU thread, uninitialized-memory fill disabled,
deterministic algorithms enabled) and runs under `InferenceMode`. The production
tensor backend guards each native call because Go calls can change OS threads.
Minimal verification class declarations replace module storage/ABI; profiling
is a no-op. The selected math functions and their source order are unchanged.
This is source-function algorithm parity, not validation of Dorado's original
module ABI, config parser or complete application. Original files are read only.

Fixtures cover 48 FLSTM stacks (`C=4,K=2`, N=1/2, T=1/2/7, depth=1/2/3/7,
both first-reverse values), six full small HAC graphs (N=1/2, raw L=6/12/42),
and one genuine pinned-stock-weight HAC graph with normalized synthetic input
`[1,1,96]`. Position-distinct synthetic values and weights exercise saturated
hard gates, candidate clamp, soft cell tanh, bias and gate order, factored
matmuls, odd/even reversals, convolution padding and activations, two linears,
final tanh-times-five and inactive clamp. Every captured conv, recurrent,
linear and score output is compared by float32 bits, with no tolerance.

Rejection controls cover altered config/weight/source hashes, missing weights
including a partial load, wrong shape, wrong dtype, multiple or zero tensors,
invalid input shape, nonfinite input, closed and foreign-engine handles, SUP
runtime opening and cancellation. A deterministic midgraph cancellation test
then successfully reuses the same model. Inputs and synthetic weight values
remain unchanged. Closing a model preserves the caller's input and score tensor;
closing it again is harmless. Temporary handles are scoped per recurrent step.

```sh
go test ./internal/model ./verification/model
go vet ./internal/model ./verification/model
CGO_ENABLED=0 go test ./internal/model ./verification/model

NANOPORE_TENSOR_BACKEND="$PWD/artifacts/tensor/attempt-004/libnanopore_tensor.so" \
DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller/software/dorado-src \
MODEL_ARTIFACT_DIR="$PWD/artifacts/model" \
go test -count=1 -v ./internal/model
```

The first command explicitly skips native work unless all three integration
environment variables are present. The native gate requires a qualified backend
receipt beside its library and creates a fresh immutable attempt directory. It
retains source bytes/range hashes, compiler identity/flags, CPU library hashes,
the backend qualification receipt, generated fixture hashes and raw compile/run
logs with timestamps and exit codes. `parity.json` records checked counts and
status; the stock fixture also retains all 50 weight hashes and a run receipt.

The initialization-correct ABI2 backend in tensor attempt-004 passed 55 fixtures,
309 stages and 208,988 F32 values with zero differing bits. The later strengthened
oracle includes the original CRF model forward and encoder-order statement;
its final receipt is
`artifacts/model/attempt-20261023T023028.377698342Z/{manifest.json,parity.json}`;
raw gate output is `artifacts/model/checks-005.log`. The captured NTC score
file `stock/stock_hac_n1_l96/scores.bin` has shape `[1,16,1024]`, little-endian
F32, and SHA-256
`36229b983d66c9db8db986ea968f4fc679f81e6da8cf9bff5f75178128c35657`.
For N=1 the same bytes also represent `[16,1,1024]` TNC, so downstream CRF
verification can reuse it without repeating model inference.
Earlier compiler failures are preserved:
the first extraction omitted the stack's closing brace, and the initial C++17
flag did not support upstream `std::ssize`. The corrected extraction includes
line 93 and uses C++20. No original source was changed.

## Remaining SUP and runtime qualification

For SUP, compile original `TxModules.cpp`, `RMSNorm.cpp` and
`LinearUpsample.cpp` CPU paths with the same pin discipline. First test RoPE
halves and V copying, RMSNorm epsilon, gated-MLP split order, DeepNorm residuals,
upsample interleaving, weight-side scale-once, and an attention sequence longer
than the window crossing multiple split boundaries. Test repeated forward
calls to catch accidental repeated scaling. Pin default dev options. Then
compare a reduced-dimension original TxEncoder graph and a bounded real-weight
prefix/full forward once native loading is available. Shape metadata alone
does not qualify weight values or graph numerical behavior.

HAC evidence now includes genuine loading of all pinned weights and a complete
CPU forward on a short synthetic signal, plus exact source-linked numerical
parity. SUP graph execution, full-sized chunks, real-read inference, GPU parity,
complete-input processing, biological accuracy and performance remain
unqualified. CPU13 development evidence does not qualify stock CUDA-12.8 GPU
behavior.
