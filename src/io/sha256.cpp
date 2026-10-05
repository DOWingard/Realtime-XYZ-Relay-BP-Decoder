#include "rtd/io/sha256.hpp"

#include <openssl/evp.h>

#include <array>
#include <cerrno>
#include <format>
#include <fstream>
#include <memory>
#include <system_error>
#include <vector>

namespace rtd::io {

std::expected<std::string, IoError> sha256_file(const std::filesystem::path& path) {
    const auto fail = [&](IoError::Code code, std::string expected, std::string found) {
        return std::unexpected(IoError{.code = code,
                                       .path = path.string(),
                                       .expected = std::move(expected),
                                       .found = std::move(found)});
    };
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return fail(IoError::Code::open_failed, "a readable file",
                    std::error_code(errno, std::generic_category()).message());
    }
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                         &EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        return fail(IoError::Code::read_failed, "an initialised SHA-256 context", "an OpenSSL error");
    }
    std::vector<char> block(std::size_t{1} << 20U);
    while (in) {
        in.read(block.data(), static_cast<std::streamsize>(block.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got > 0 && EVP_DigestUpdate(context.get(), block.data(), got) != 1) {
            return fail(IoError::Code::read_failed, "a digest update", "an OpenSSL error");
        }
    }
    if (!in.eof()) {
        return fail(IoError::Code::read_failed, "the whole file", "a stream error");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) {
        return fail(IoError::Code::read_failed, "a final digest", "an OpenSSL error");
    }
    std::string hex;
    hex.reserve(2 * std::size_t{length});
    for (unsigned int k = 0; k < length; ++k) {
        hex += std::format("{:02x}", digest[k]);
    }
    return hex;
}

} // namespace rtd::io
