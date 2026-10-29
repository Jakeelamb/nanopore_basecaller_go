#pragma once

#include <ATen/core/TensorBody.h>
#include "nn/AuxiliaryData.h"
#include "nn/WorkingMemory.h"
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Source-only ABI. There are deliberately no no-op definitions. A later,
// independently closed producer must define every hook before linking.
namespace packed_hac_observer {
using dorado::nn::AuxiliaryData;
using dorado::nn::WorkingMemory;

struct TensorRef {
    const char* role;
    const at::Tensor& tensor;
};

// Output metadata contract for the future hook implementation. An absent
// device/stream/allocation denotes a source reservation/host event, not zero.
struct TensorMetadata {
    std::string role;
    std::optional<std::string> allocation_id;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::string dtype;
    std::int64_t storage_offset_bytes;
    std::int64_t whole_storage_bytes;
    std::string backing_payload_sha256;
    std::string logical_payload_sha256;
};

enum class WriteKnowledge {
    source_zero_assignment,
    source_copy_assignment,
    source_upload_span,
    observed_byte_change
};
struct WriteMetadata {
    std::string source_anchor;
    std::string role;
    std::int64_t offset_bytes;
    std::int64_t extent_bytes;
    WriteKnowledge knowledge;
};
// Exact selected Koi type operand, rather than a tensor dtype inference.
struct AccumulatorMode {
    int koi_type;
};
struct EventMetadata {
    std::uint64_t sequence;
    std::string case_id;
    std::string source_anchor;
    std::string phase;
    std::optional<int> layer;
    // Present for fast-result only; all other phases explicitly unavailable.
    std::optional<AccumulatorMode> accumulator_mode;
    std::optional<int> device_ordinal;
    std::optional<std::string> stream_id;
    std::optional<std::string> aux_id;
    std::int64_t n, t, c;
    std::string layout;
    bool input_to_reverse_lstm;
    std::int64_t reservation_bytes, current_bytes;
    std::vector<TensorMetadata> tensors;
    std::vector<WriteMetadata> known_writes;
    // Always unknown for opaque Koi calls in this source-only unit. Observed
    // equality/change must not be promoted to a read mask or write contract.
    std::optional<std::vector<std::string>> opaque_read_sets;
    std::optional<std::vector<std::string>> opaque_write_sets;
};

// Each hook must synchronously stage immutable same-stream snapshots before
// returning (including whole backing storage and logical views separately).
// A later writer must drain before hashing. Const references do not establish
// deep immutability of tensor storage; the implementation must never mutate it.
// Required common record: sequence, case, exact anchor/phase/layer, stable
// allocation identity (not pointer alone), device/stream identity, shape,
// strides, dtype, offset, storage bytes, reservation/current bytes, N/T/C,
// layout/orientation, Aux identity, known writes, and unknown read sets.
void graph(const char* phase, const WorkingMemory&, const AuxiliaryData*);
void tensor(const char* phase, const WorkingMemory&, const AuxiliaryData*,
            std::initializer_list<TensorRef>);
void conv(int layer, const char* phase, const WorkingMemory&, const AuxiliaryData*,
          const at::Tensor& input, const at::Tensor& output,
          const at::Tensor& weights, const at::Tensor& repair_weights,
          const at::Tensor& bias, bool has_status, int status,
          std::optional<AccumulatorMode> accumulator_mode);

// Encoding is selected through the typed Aux tensor, never dereferenced from
// the opaque production void*. call_t/direction/parity are actual call values.
void lstm(int layer, const char* phase, const WorkingMemory&, const AuxiliaryData*,
          const at::Tensor& inout, const at::Tensor& down,
          const at::Tensor& scales, const at::Tensor& workspace,
          const at::Tensor& encoding, int call_t, int direction, int parity,
          std::initializer_list<TensorRef> caches);
void projection(const char* phase, const WorkingMemory&, const AuxiliaryData*,
                const at::Tensor& input_row3, const at::Tensor& down,
                const at::Tensor& output, const at::Tensor& gather,
                int parity, float scale, std::initializer_list<TensorRef> caches);

// Record workspace and constructor sample lengths separately from score
// lengths. Six device arrays include intervals/table and four packing arrays.
// Pre-upload has no native status; post-preprocess includes actual status.
void aux(const char* phase, const AuxiliaryData&, const at::Tensor& workspace,
	     std::span<const std::int32_t> constructor_samples,
	     std::span<const std::int32_t> host_intervals,
	     std::span<const std::int32_t> host_table,
         bool has_status, int status);
} // namespace packed_hac_observer
