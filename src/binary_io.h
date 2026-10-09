#pragma once

#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

inline void readBinaryExact(const std::string& path, void* destination,
                            std::size_t expected_bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + path);
    }

    const std::streamoff actual_bytes = file.tellg();
    if (actual_bytes < 0 || static_cast<std::size_t>(actual_bytes) != expected_bytes) {
        throw std::runtime_error(
            "Unexpected file size for " + path + ": expected " +
            std::to_string(expected_bytes) + " bytes, got " +
            std::to_string(actual_bytes) + " bytes");
    }
    if (expected_bytes >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Binary file is too large to read: " + path);
    }

    file.seekg(0, std::ios::beg);
    if (expected_bytes > 0 &&
        !file.read(static_cast<char*>(destination),
                   static_cast<std::streamsize>(expected_bytes))) {
        throw std::runtime_error("Incomplete read from binary file: " + path);
    }
}
