#include "capture.h"
#include "koi_thread_pool.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* matrix_sha = "d4602d3fd2ac2e45515a949087b82fe50fe5a0a92e3dcffc66b82979d0c9e906";
constexpr std::size_t guard_words = 32;
constexpr std::int32_t guard = 0x6b497d31;
constexpr std::uintptr_t stream_value = 0x456780;

struct Buffer {
    std::string name;
    std::size_t words;
    std::size_t capacity;
    std::vector<std::int32_t> storage;
    std::vector<std::int32_t> captured;

    Buffer(std::string n, std::size_t count, std::size_t cap, std::int32_t fill)
            : name(std::move(n)), words(count), capacity(cap), storage(cap + 2 * guard_words, guard) {
        std::fill_n(data(), capacity, fill);
    }
    std::int32_t* data() { return storage.data() + guard_words; }
    bool guards_ok() const {
        return std::all_of(storage.begin(), storage.begin() + guard_words,
                           [](auto value) { return value == guard; }) &&
               std::all_of(storage.end() - guard_words, storage.end(),
                           [](auto value) { return value == guard; });
    }
};

struct Context {
    std::array<Buffer*, 4> copies;
    Buffer* workspace;
    std::size_t next = 0;
    std::vector<std::size_t> source_offsets;
    std::string failure;
};
Context* active = nullptr;

struct Fixture {
    std::string id;
    int n;
    int t_out;
    std::vector<int> sizes;
};

void require(bool valid, const std::string& message) {
    if (!valid) {
        throw std::runtime_error(message);
    }
}

void save(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::out);
    require(out.good(), "cannot open output " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.close();
    require(out.good(), "cannot write output " + path.string());
}

void save_maps(const std::filesystem::path& path) {
    std::ifstream source("/proc/self/maps");
    std::ofstream target(path);
    require(source.good() && target.good(), "cannot open native mapping capture");
    target << source.rdbuf();
    target.close();
    require(!source.bad() && target.good(), "native mapping capture failed");
}

std::vector<Fixture> read_inputs(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string schema, sha;
    int count = 0;
    input >> schema >> sha >> count;
    require(input.good() && schema == "hac-vcs-aux-inputs-v1" && sha == matrix_sha && count == 21,
            "input matrix identity differs");
    std::vector<Fixture> fixtures;
    for (int i = 0; i < count; ++i) {
        Fixture f;
        int chunks = 0;
        input >> f.id >> f.n >> f.t_out >> chunks;
        require(input.good() && f.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") == std::string::npos &&
                        !f.id.empty() && (f.n == 64 || f.n == 32832) && f.t_out > 0 && f.t_out <= 2048 && chunks > 0 && chunks <= 32832,
                "invalid frozen native fixture");
        f.sizes.resize(chunks);
        int sum = 0;
        for (auto& size : f.sizes) {
            input >> size;
            require(!input.fail() && size > 0 && size <= f.t_out, "invalid frozen native chunk");
            sum += size;
        }
        require(sum <= f.n * f.t_out, "frozen native capacity exceeded");
        fixtures.push_back(std::move(f));
    }
    std::string trailing;
    require(!(input >> trailing), "trailing native inputs");
    return fixtures;
}

void run(const Fixture& fixture, int workers, std::int32_t workspace_fill,
         std::int32_t destination_fill, int fill_id, const std::filesystem::path& root) {
    const int n = fixture.n;
    const int l = (fixture.t_out + 3) & ~1;
    int sum = 0;
    for (auto size : fixture.sizes) {
        sum += size;
    }
    const std::size_t nl = static_cast<std::size_t>(n) * l;
    const std::size_t nd = nl + n;
    Buffer in("InLayout", sum, sum, destination_fill);
    Buffer out("OutLayout", nl, nd, destination_fill);
    Buffer forward("ForwardEncoding", nl, nl, destination_fill);
    Buffer backward("BackwardEncoding", nd, nd, destination_fill);
    Buffer workspace("Workspace", 6 * (fixture.t_out + 3) * n, 6 * (fixture.t_out + 3) * n, workspace_fill);
    auto sizes = fixture.sizes;
    Context context{{&out, &forward, &in, &backward}, &workspace, 0, {}, {}};
    KoiThreadPool* pool = nullptr;
    const int create_status = koi_create_thread_pool(&pool, workers);
    require(create_status == 0 && pool != nullptr, "original thread-pool creation failed");
    active = &context;
    const int status = host_lstm_preprocess(reinterpret_cast<void*>(stream_value), n, sizes.data(),
                                            static_cast<int>(sizes.size()), 32, l, workspace.data(),
                                            static_cast<int>(workspace.capacity), pool, nullptr,
                                            out.data(), forward.data(), in.data(), nullptr, backward.data());
    active = nullptr;
    const int destroy_status = koi_destroy_thread_pool(&pool);
    const auto run_name = fixture.id + "-pool" + std::to_string(workers) + "-fill" + std::to_string(fill_id);
    const auto directory = root / run_name;
    require(std::filesystem::create_directory(directory), "fresh native run directory required");
    std::ofstream metadata(directory / "capture.json");
    metadata << "{\"schema\":\"hac-vcs-aux-capture/v1\",\"case\":\"" << fixture.id
             << "\",\"matrix_sha256\":\"" << matrix_sha << "\",\"workers\":" << workers
             << ",\"workspace_fill\":" << workspace_fill << ",\"destination_fill\":" << destination_fill
             << ",\"batch_size\":" << n << ",\"full_t_lstm\":" << l
             << ",\"status\":" << status << ",\"thread_pool_create_status\":" << create_status
             << ",\"thread_pool_destroy_status\":" << destroy_status
             << ",\"copy_count\":" << context.next << ",\"copy_kind\":1,\"stream\":" << stream_value
             << ",\"input_unchanged\":" << (sizes == fixture.sizes ? "true" : "false")
             << ",\"workspace_guards_ok\":" << (workspace.guards_ok() ? "true" : "false")
             << ",\"copies\":[";
    bool first = true;
    for (auto* buffer : context.copies) {
        if (!first) {
            metadata << ',';
        }
        first = false;
        metadata << "{\"name\":\"" << buffer->name << "\",\"words\":" << buffer->words
                 << ",\"capacity\":" << buffer->capacity << ",\"captured_words\":" << buffer->captured.size()
                 << ",\"guards_ok\":" << (buffer->guards_ok() ? "true" : "false") << '}';
        save(directory / (buffer->name + ".i32"), buffer->captured.data(), buffer->captured.size() * 4);
        save(directory / (buffer->name + "-destination.i32"), buffer->data(), buffer->capacity * 4);
        save(directory / (buffer->name + "-guarded.i32"), buffer->storage.data(), buffer->storage.size() * 4);
    }
    metadata << "],\"source_workspace_offsets\":[";
    for (std::size_t i = 0; i < context.source_offsets.size(); ++i) {
        if (i) {
            metadata << ',';
        }
        metadata << context.source_offsets[i];
    }
    metadata << "]}\n";
    metadata.close();
    save(directory / "input-before.i32", fixture.sizes.data(), fixture.sizes.size() * 4);
    save(directory / "input-after.i32", sizes.data(), sizes.size() * 4);
    save(directory / "workspace-after.i32", workspace.data(), workspace.capacity * 4);
    save(directory / "workspace-guarded.i32", workspace.storage.data(), workspace.storage.size() * 4);
    require(metadata.good(), "capture metadata write failed");
    require(context.failure.empty(), context.failure);
    require(status == 0 && destroy_status == 0 && context.next == 4 && sizes == fixture.sizes && workspace.guards_ok(),
            "native call or mutation control failed");
    for (auto* buffer : context.copies) {
        require(buffer->guards_ok() && buffer->captured.size() == buffer->words,
                "native output guard or span failed");
        require(std::equal(buffer->captured.begin(), buffer->captured.end(), buffer->data()),
                "captured source differs from transported prefix");
    }
    require(std::all_of(out.data() + nl, out.data() + nd,
                        [destination_fill](auto value) { return value == destination_fill; }),
            "OutLayout unwritten destination tail changed");
    std::cout << run_name << " status=" << status << " copies=" << context.next << '\n';
}
}

int capture_h2d(void* destination, const void* source, std::size_t bytes, int kind, void* stream) {
    if (!active || active->next >= 4) {
        std::abort();
    }
    auto& c = *active;
    auto* buffer = c.copies[c.next];
    const auto source_address = reinterpret_cast<std::uintptr_t>(source);
    const auto workspace_address = reinterpret_cast<std::uintptr_t>(c.workspace->data());
    const auto workspace_bytes = c.workspace->capacity * 4;
    if (destination != buffer->data() || bytes != buffer->words * 4 || kind != 1 ||
        reinterpret_cast<std::uintptr_t>(stream) != stream_value || source_address < workspace_address ||
        source_address - workspace_address > workspace_bytes || bytes > workspace_bytes - (source_address - workspace_address) ||
        (source_address - workspace_address) % 4 != 0) {
        c.failure = "original upload metadata or source range differs";
        return 1;
    }
    c.source_offsets.push_back((source_address - workspace_address) / 4);
    buffer->captured.resize(buffer->words);
    std::memcpy(buffer->captured.data(), source, bytes);
    std::memcpy(destination, source, bytes);
    ++c.next;
    return 0;
}

int main(int argc, char** argv) {
    try {
        require(sizeof(int) == 4 && sizeof(std::int32_t) == 4, "native int32 ABI differs");
        require(argc == 4, "usage: oracle INPUTS NEW_OUTPUT_DIRECTORY first|all");
        auto fixtures = read_inputs(argv[1]);
        require(std::string(argv[3]) == "first" || std::string(argv[3]) == "all", "unsupported fixture selection");
        if (std::string(argv[3]) == "first") {
            fixtures.resize(1);
        }
        const std::filesystem::path output(argv[2]);
        require(std::filesystem::create_directory(output), "fresh native output directory required");
        capture_transport_begin(argv[2], static_cast<unsigned>(fixtures.size() * 16));
        save_maps(output / "maps-before");
        for (const auto& fixture : fixtures) {
            for (int workers : {1, 2}) {
                run(fixture, workers, 324508639, 1431655765, 0, output);
                run(fixture, workers, 610839776, -1431655766, 1, output);
            }
        }
        save_maps(output / "maps-after");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
