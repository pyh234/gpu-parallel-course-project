#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

struct PipelineOptions {
    bool validate = false;
    bool show_help = false;
    std::string stop_after;
};

inline bool isValidStopStage(const std::string& stage) {
    return stage == "load" || stage == "hvg" || stage == "dense" ||
           stage == "pca-fit" || stage == "pca-transform" ||
           stage == "kmeans";
}

inline PipelineOptions parsePipelineOptions(int argc, char** argv) {
    PipelineOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--stop-after") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--stop-after requires a stage name");
            }
            options.stop_after = argv[++i];
            if (!isValidStopStage(options.stop_after)) {
                throw std::invalid_argument("Unknown stop stage: " + options.stop_after);
            }
        } else if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    return options;
}

inline void printPipelineUsage(const char* executable_name) {
    std::cout
        << "Usage: " << executable_name
        << " [--validate] [--stop-after STAGE]\n"
        << "Stages: load, hvg, dense, pca-fit, pca-transform, kmeans\n";
}

inline bool shouldStopAfter(const PipelineOptions& options,
                            const std::string& stage) {
    return options.stop_after == stage;
}

inline std::uint64_t validationMix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

template <typename T>
inline std::uint64_t validationElementBits(const T& value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Validation fingerprints require trivially copyable values");
    static_assert(sizeof(T) <= sizeof(std::uint64_t),
                  "Validation fingerprints support values up to 64 bits");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

template <typename T>
inline void printValidationFingerprint(const std::string& name,
                                       const T* data,
                                       std::size_t count) {
    std::uint64_t xor_hash = 0;
    std::uint64_t sum_hash = 0;
    std::uint64_t nonzero = 0;

    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t bits = validationElementBits(data[i]);
        const std::uint64_t mixed = validationMix64(
            validationMix64(static_cast<std::uint64_t>(i)) ^ bits);
        xor_hash ^= mixed;
        sum_hash += mixed;
        if (data[i] != static_cast<T>(0)) {
            ++nonzero;
        }
    }

    std::cout << "[Validation] " << name
              << " count=" << count
              << " nonzero=" << nonzero
              << " xor=0x" << std::hex << std::setw(16)
              << std::setfill('0') << xor_hash
              << " sum=0x" << std::setw(16) << sum_hash
              << std::dec << std::setfill(' ') << '\n';
}
