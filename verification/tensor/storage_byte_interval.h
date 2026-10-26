#pragma once
#include <cstdint>
#include <limits>

struct StorageByteInterval {
    uint64_t offset = 0;
    uint64_t end = 0;
};

inline bool checked_storage_byte_interval(uint64_t offset, uint64_t end,
                                          uint64_t element_bytes, StorageByteInterval& out) {
    if (!element_bytes || end < offset || end > std::numeric_limits<uint64_t>::max() / element_bytes) {
        return false;
    }
    out = {offset * element_bytes, end * element_bytes};
    return true;
}

inline bool storage_byte_intervals_disjoint(const StorageByteInterval& a,
                                            const StorageByteInterval& b) {
    return a.offset == a.end || b.offset == b.end || a.end <= b.offset || b.end <= a.offset;
}

inline bool checked_storage_reinterpret_offset(const StorageByteInterval& parent,
                                               uint64_t element_bytes, uint64_t offset_elements,
                                               uint64_t& out) {
    if (!element_bytes || parent.end < parent.offset ||
        parent.end / element_bytes > uint64_t(std::numeric_limits<int64_t>::max()) ||
        offset_elements > (parent.end - parent.offset) / element_bytes) {
        return false;
    }
    const auto start_byte = parent.offset + offset_elements * element_bytes;
    if (start_byte % element_bytes) return false;
    out = start_byte / element_bytes;
    return true;
}
