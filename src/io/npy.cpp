#include "rtd/io/npy.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

namespace rtd::io {

static_assert(std::endian::native == std::endian::little,
              "npy arrays are read in place, which assumes a little-endian host");

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
    case DType::u1:
        return "|u1";
    case DType::b1:
        return "|b1";
    case DType::i4:
        return "<i4";
    case DType::u4:
        return "<u4";
    case DType::i8:
        return "<i8";
    case DType::u8:
        return "<u8";
    case DType::f4:
        return "<f4";
    case DType::f8:
        return "<f8";
    }
    return "?";
}

std::size_t item_size(DType dtype) noexcept {
    switch (dtype) {
    case DType::u1:
    case DType::b1:
        return 1;
    case DType::i4:
    case DType::u4:
    case DType::f4:
        return 4;
    case DType::i8:
    case DType::u8:
    case DType::f8:
        return 8;
    }
    return 0;
}

std::size_t NpyHeader::count() const noexcept {
    std::size_t n = 1;
    for (const std::size_t d : shape) {
        n *= d;
    }
    return n;
}

namespace {

// Streams read and write chars; any object's bytes may be accessed through a char pointer.
template <class T>
char* as_chars(T* p) noexcept {
    return reinterpret_cast<char*>(p); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <class T>
const char* as_chars(const T* p) noexcept {
    return reinterpret_cast<const char*>(p); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

constexpr std::array<unsigned char, 6> magic{0x93, 'N', 'U', 'M', 'P', 'Y'};

std::unexpected<IoError> fail(IoError::Code code, std::string_view path, std::string expected,
                              std::string found) {
    return std::unexpected(IoError{.code = code,
                                   .path = std::string(path),
                                   .expected = std::move(expected),
                                   .found = std::move(found)});
}

// Minimal reader for the dict literal numpy writes, e.g.
// {'descr': '<u4', 'fortran_order': False, 'shape': (1873,), }
class HeaderParser {
public:
    explicit HeaderParser(std::string_view text) : text_(text) {}

    // Position just after "'key':" (or the double-quoted form), or npos.
    [[nodiscard]] std::size_t value_of(std::string_view key) const {
        for (const char quote : {'\'', '"'}) {
            const std::string needle = std::string(1, quote) + std::string(key) + quote;
            std::size_t at = text_.find(needle);
            if (at == std::string_view::npos) {
                continue;
            }
            at = skip_space(at + needle.size());
            if (at < text_.size() && text_[at] == ':') {
                return skip_space(at + 1);
            }
        }
        return std::string_view::npos;
    }

    [[nodiscard]] std::optional<std::string_view> quoted(std::size_t at) const {
        if (at >= text_.size() || (text_[at] != '\'' && text_[at] != '"')) {
            return std::nullopt;
        }
        const std::size_t end = text_.find(text_[at], at + 1);
        if (end == std::string_view::npos) {
            return std::nullopt;
        }
        return text_.substr(at + 1, end - at - 1);
    }

    [[nodiscard]] std::optional<bool> boolean(std::size_t at) const {
        if (text_.substr(at, 4) == "True") {
            return true;
        }
        if (text_.substr(at, 5) == "False") {
            return false;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::vector<std::size_t>> tuple(std::size_t at) const {
        if (at >= text_.size() || text_[at] != '(') {
            return std::nullopt;
        }
        std::vector<std::size_t> values;
        ++at;
        for (;;) {
            at = skip_space(at);
            if (at >= text_.size()) {
                return std::nullopt;
            }
            if (text_[at] == ')') {
                return values;
            }
            std::size_t value = 0;
            const auto [ptr, ec] = std::from_chars(text_.data() + at, text_.data() + text_.size(), value);
            if (ec != std::errc{}) {
                return std::nullopt;
            }
            values.push_back(value);
            at = skip_space(static_cast<std::size_t>(ptr - text_.data()));
            if (at < text_.size() && text_[at] == ',') {
                ++at;
            }
        }
    }

private:
    [[nodiscard]] std::size_t skip_space(std::size_t at) const {
        while (at < text_.size() && (text_[at] == ' ' || text_[at] == '\t')) {
            ++at;
        }
        return at;
    }

    std::string_view text_;
};

std::optional<DType> parse_descr(std::string_view descr) {
    if (descr.size() != 3) {
        return std::nullopt;
    }
    const char order = descr[0];
    const std::string_view kind = descr.substr(1);
    const bool little = order == '<' || order == '=';
    const bool single = order == '|' || little;
    if (kind == "u1" && single) {
        return DType::u1;
    }
    if (kind == "b1" && single) {
        return DType::b1;
    }
    if (!little) {
        return std::nullopt;
    }
    if (kind == "i4") {
        return DType::i4;
    }
    if (kind == "u4") {
        return DType::u4;
    }
    if (kind == "i8") {
        return DType::i8;
    }
    if (kind == "u8") {
        return DType::u8;
    }
    if (kind == "f4") {
        return DType::f4;
    }
    if (kind == "f8") {
        return DType::f8;
    }
    return std::nullopt;
}

std::string shape_string(std::span<const std::size_t> shape) {
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        s += std::format("{}{}", i == 0 ? "" : ", ", shape[i]);
    }
    s += shape.size() == 1 ? ",)" : ")";
    return s;
}

} // namespace

std::expected<NpyHeader, IoError> parse_npy_header(std::span<const std::byte> bytes,
                                                   std::string_view path) {
    using Code = IoError::Code;
    if (bytes.size() < 10 || std::memcmp(bytes.data(), magic.data(), magic.size()) != 0) {
        return fail(Code::bad_format, path, "the \\x93NUMPY magic", "other bytes");
    }
    const auto major = static_cast<unsigned>(bytes[6]);
    std::size_t header_length = 0;
    std::size_t start = 0;
    if (major == 1) {
        header_length = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8U);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12) {
            return fail(Code::bad_format, path, "a 12-byte preamble", std::format("{} bytes", bytes.size()));
        }
        header_length = 0;
        for (std::size_t k = 0; k < 4; ++k) {
            header_length |= static_cast<std::size_t>(bytes[8 + k]) << (8 * k);
        }
        start = 12;
    } else {
        return fail(Code::bad_format, path, "format version 1, 2 or 3", std::format("{}", major));
    }
    if (bytes.size() < start + header_length) {
        return fail(Code::bad_format, path, std::format("a {}-byte header", header_length),
                    std::format("{} bytes", bytes.size() - start));
    }
    const std::string_view text(as_chars(bytes.data() + start), header_length);
    const HeaderParser parser(text);

    const auto descr = parser.quoted(parser.value_of("descr"));
    if (!descr) {
        return fail(Code::bad_format, path, "a 'descr' entry", std::string(text));
    }
    const auto dtype = parse_descr(*descr);
    if (!dtype) {
        return fail(Code::unsupported_dtype, path, "one of |u1 |b1 <i4 <u4 <i8 <u8 <f4 <f8",
                    std::string(*descr));
    }
    const auto fortran = parser.boolean(parser.value_of("fortran_order"));
    if (!fortran) {
        return fail(Code::bad_format, path, "a 'fortran_order' entry", std::string(text));
    }
    if (*fortran) {
        return fail(Code::bad_format, path, "C order", "Fortran order");
    }
    auto shape = parser.tuple(parser.value_of("shape"));
    if (!shape) {
        return fail(Code::bad_format, path, "a 'shape' tuple", std::string(text));
    }
    return NpyHeader{
        .dtype = *dtype, .shape = std::move(*shape), .data_offset = start + header_length};
}

std::expected<NpyHeader, IoError> read_npy_header(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return fail(IoError::Code::open_failed, path.string(), "a readable file",
                    std::error_code(errno, std::generic_category()).message());
    }
    std::array<std::byte, 12> preamble{};
    in.read(as_chars(preamble.data()), preamble.size());
    const auto got = static_cast<std::size_t>(in.gcount());
    if (got < 10) {
        return fail(IoError::Code::bad_format, path.string(), "an npy preamble",
                    std::format("{} bytes", got));
    }
    const auto major = static_cast<unsigned>(preamble[6]);
    std::size_t header_length = 0;
    std::size_t start = 10;
    if (major == 1) {
        header_length = static_cast<std::size_t>(preamble[8]) |
                        (static_cast<std::size_t>(preamble[9]) << 8U);
    } else {
        for (std::size_t k = 0; k < 4; ++k) {
            header_length |= static_cast<std::size_t>(preamble[8 + k]) << (8 * k);
        }
        start = 12;
    }
    // Guard against a corrupt length before allocating for it.
    if (header_length > (1U << 24U)) {
        return fail(IoError::Code::bad_format, path.string(), "a header under 16 MiB",
                    std::format("{} bytes", header_length));
    }
    std::vector<std::byte> bytes(start + header_length);
    std::memcpy(bytes.data(), preamble.data(), std::min(got, bytes.size()));
    if (bytes.size() > got) {
        in.read(as_chars(bytes.data() + got),
                static_cast<std::streamsize>(bytes.size() - got));
        if (static_cast<std::size_t>(in.gcount()) != bytes.size() - got) {
            return fail(IoError::Code::read_failed, path.string(), "a complete header",
                        "end of file");
        }
    }
    return parse_npy_header(bytes, path.string());
}

template <NpyElement T>
std::expected<NpyArray<T>, IoError> read_npy(const std::filesystem::path& path,
                                             std::optional<std::size_t> rank,
                                             std::span<const std::optional<std::size_t>> extents) {
    using Code = IoError::Code;
    auto header = read_npy_header(path);
    if (!header) {
        return std::unexpected(std::move(header.error()));
    }
    const DType wanted = dtype_of<T>::value;
    const bool bool_as_u1 = wanted == DType::u1 && header->dtype == DType::b1;
    if (header->dtype != wanted && !bool_as_u1) {
        return fail(Code::dtype_mismatch, path.string(), std::string(to_string(wanted)),
                    std::string(to_string(header->dtype)));
    }
    if (rank && header->shape.size() != *rank) {
        return fail(Code::shape_mismatch, path.string(), std::format("{} dimensions", *rank),
                    shape_string(header->shape));
    }
    for (std::size_t d = 0; d < extents.size() && d < header->shape.size(); ++d) {
        if (const auto& extent = extents[d]; extent && header->shape[d] != *extent) {
            return fail(Code::shape_mismatch, path.string(),
                        std::format("dimension {} of size {}", d, *extent),
                        shape_string(header->shape));
        }
    }
    std::size_t count = 1;
    for (const std::size_t d : header->shape) {
        if (d != 0 && count > std::numeric_limits<std::size_t>::max() / sizeof(T) / d) {
            return fail(Code::bad_format, path.string(), "an addressable size",
                        shape_string(header->shape));
        }
        count *= d;
    }
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return fail(Code::read_failed, path.string(), "a file size", ec.message());
    }
    const std::size_t expected_size = header->data_offset + count * sizeof(T);
    if (file_size != expected_size) {
        return fail(Code::bad_format, path.string(),
                    std::format("{} bytes for shape {}", expected_size, shape_string(header->shape)),
                    std::format("{} bytes", file_size));
    }
    NpyArray<T> array{AlignedBuffer<T>(count), std::move(header->shape)};
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(header->data_offset));
    in.read(as_chars(array.data.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in || static_cast<std::size_t>(in.gcount()) != count * sizeof(T)) {
        return fail(Code::read_failed, path.string(), std::format("{} data bytes", count * sizeof(T)),
                    std::format("{} bytes", in.gcount()));
    }
    return array;
}

template <NpyElement T>
std::expected<void, IoError> write_npy(const std::filesystem::path& path, std::span<const T> data,
                                       std::span<const std::size_t> shape) {
    std::size_t count = 1;
    for (const std::size_t d : shape) {
        count *= d;
    }
    if (count != data.size()) {
        return fail(IoError::Code::shape_mismatch, path.string(),
                    std::format("{} values for shape {}", count, shape_string(shape)),
                    std::format("{} values", data.size()));
    }
    std::string header = std::format("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
                                     to_string(dtype_of<T>::value), shape_string(shape));
    // Pad so the array starts on a 64-byte boundary, as numpy does.
    const std::size_t unpadded = magic.size() + 2 + 2 + header.size() + 1;
    header.append((64 - unpadded % 64) % 64, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        return fail(IoError::Code::write_failed, path.string(), "a header under 64 KiB",
                    std::format("{} bytes", header.size()));
    }

    // Write to a temporary name and rename, so a reader never sees a partial file.
    std::filesystem::path temporary = path;
    temporary += ".partial";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) {
            return fail(IoError::Code::open_failed, temporary.string(), "a writable file",
                        std::error_code(errno, std::generic_category()).message());
        }
        out.write(as_chars(magic.data()), magic.size());
        const std::array<char, 4> version_and_length{
            1, 0, static_cast<char>(header.size() & 0xFFU), static_cast<char>(header.size() >> 8U)};
        out.write(version_and_length.data(), version_and_length.size());
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(as_chars(data.data()),
                  static_cast<std::streamsize>(data.size_bytes()));
        out.flush();
        if (!out) {
            return fail(IoError::Code::write_failed, temporary.string(),
                        std::format("{} bytes written", data.size_bytes()), "a stream error");
        }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        return fail(IoError::Code::write_failed, path.string(), "a successful rename", ec.message());
    }
    return {};
}

// Explicit instantiation has no form that takes a list of types.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RTD_INSTANTIATE_NPY(T)                                                                     \
    template std::expected<NpyArray<T>, IoError> read_npy<T>(                                      \
        const std::filesystem::path&, std::optional<std::size_t>,                                  \
        std::span<const std::optional<std::size_t>>);                                             \
    template std::expected<void, IoError> write_npy<T>(const std::filesystem::path&,               \
                                                       std::span<const T>,                         \
                                                       std::span<const std::size_t>);

RTD_INSTANTIATE_NPY(std::uint8_t)
RTD_INSTANTIATE_NPY(std::int32_t)
RTD_INSTANTIATE_NPY(std::uint32_t)
RTD_INSTANTIATE_NPY(std::int64_t)
RTD_INSTANTIATE_NPY(std::uint64_t)
RTD_INSTANTIATE_NPY(float)
RTD_INSTANTIATE_NPY(double)

#undef RTD_INSTANTIATE_NPY

} // namespace rtd::io
