# GPU batch geometry

Status is host-verified source implementation. Native oracle agreement, device
upload and model execution remain unqualified.

`internal/gpubatch` ports the pure geometry that the pinned GPU caller derives
before a model launch. It owns batch membership, packed offsets, padding slots,
chunk tables, intervals, layout sizes and granularity transitions for HAC and
SUP. It reads no model file and launches no native code.

## Source pins

- `basecall/CudaModelRunner.cpp:17-59` sets the first-convolution padding, packs
  variable chunks and resets the packed offset after a flush.
- `read_pipeline/nodes/BasecallerNode.cpp:320-405,420-491` fixes batch
  granularity, the HAC part accumulator, SUP per-chunk padding and the flush
  conditions.
- `nn/AuxiliaryData.cpp:24-61,129-150` builds `[start_block,length_blocks]`,
  scales transformer tables, and sizes the LSTM layouts.
- `nn/TxModules.cpp:840-864` asserts encoder `T=64` and pads the QKV RoPE LUT to
  the Koi MatMul multiple of 256 values.
- `nn/ConvStack.cpp:445-458` divides the working granularity by each convolution
  stride in source order.
- `basecall/CudaCaller.cpp:211-214` fixes the HAC batch multiple at 64 and SUP at
  32.

The pinned development shapes give HAC `stride=6`, `inner=6`, `granularity=6`,
and SUP `stride=6`, `inner=12`, `granularity=768`. SUP convolution strides
`[1,1,3,2,2]` reduce the working granularity to `[768,768,256,128,64]`.

## Ported rules

Fixed batches hold exactly `BatchSize` real chunks in source slots. A short chunk
is recorded at full chunk length because the source repeat-pads it before the
copy. Variable batches record the rounded raw length, the packed sample offset
and the real chunk count separately from padded capacity.

HAC variable batches use the source slice accounting
`len/stride + 2` with a 32-chunk part size. The part accumulator survives a
flush, which the pinned worker thread also does. SUP variable batches start the
packed offset at the first-convolution padding, insert the same zero window
after each chunk, and compare `used + chunk + max_conv_padding` against
`BatchSize*ChunkSize`.

Auxiliary geometry mirrors the source tables exactly. HAC stores post-stride
sizes, an even `T_lstm`, the four int32 layout lengths and the Koi workspace
count. SUP stores the scaled device table and the padded QKV RoPE LUT length.

## Host evidence

```sh
go test -json -count=1 ./internal/gpubatch
go test -race -count=1 ./internal/gpubatch
go test -count=1 ./...
go vet ./internal/gpubatch
```

The tests cover fixed slots, HAC part carry and stride rounding, SUP boundary
padding and flush, both auxiliary layouts, granularity transitions, table
copying and the declared negative controls. Receipts are under
`artifacts/gpu-plan/gpubatch-host-20261023T184658Z/`; 14 tests pass with no
failures or skips.

These checks are Go unit evidence. They are not a native oracle, not device
table bytes and not model parity. The card's extracted source oracle,
`verification/gpubatch`, and the H01/S00 retained captures remain open. No GPU,
lock, compiler or model file was touched.
