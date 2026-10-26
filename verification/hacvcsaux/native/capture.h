#pragma once

#include <cstddef>

int capture_h2d(void* destination, const void* source, std::size_t bytes, int kind, void* stream);
void capture_transport_begin(const char* directory, unsigned expected_copies);
