#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rtd/io/npy.hpp"

namespace {

using namespace rtd::io;
namespace fs = std::filesystem;

class NpyTest : public testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("rtd_npy_" + std::to_string(std::bit_cast<std::uintptr_t>(this)));
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    // Writes an .npy file by hand with the given header dict and payload.
    fs::path raw(const std::string& name, const std::string& dict, std::size_t payload_bytes,
                 char major = 1) {
        std::string header = dict;
        const std::size_t preamble = major == 1 ? 10 : 12;
        while ((preamble + header.size() + 1) % 64 != 0) {
            header.push_back(' ');
        }
        header.push_back('\n');
        const fs::path path = dir_ / name;
        std::ofstream out(path, std::ios::binary);
        out.write("\x93NUMPY", 6);
        out.put(major);
        out.put(0);
        if (major == 1) {
            out.put(static_cast<char>(header.size() & 0xFFU));
            out.put(static_cast<char>(header.size() >> 8U));
        } else {
            for (int k = 0; k < 4; ++k) {
                out.put(static_cast<char>((header.size() >> (8 * k)) & 0xFFU));
            }
        }
        out << header;
        out << std::string(payload_bytes, '\0');
        return path;
    }

    fs::path dir_;
};

TEST_F(NpyTest, RoundTripsEveryElementType) {
    const std::vector<std::uint32_t> u4{1, 2, 3, 4, 5, 6};
    const std::array<std::size_t, 2> shape{2, 3};
    ASSERT_TRUE(write_npy<std::uint32_t>(dir_ / "a.npy", u4, shape));
    auto a = read_npy<std::uint32_t>(dir_ / "a.npy", 2);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->shape, (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(a->row(1)[2], 6U);
    EXPECT_EQ((a->matrix()[1, 0]), 4U);

    const std::vector<double> f8{0.5, -1.25};
    const std::array<std::size_t, 1> one{2};
    ASSERT_TRUE(write_npy<double>(dir_ / "b.npy", f8, one));
    auto b = read_npy<double>(dir_ / "b.npy", 1);
    ASSERT_TRUE(b);
    EXPECT_EQ(b->data[1], -1.25);

    const std::vector<std::uint8_t> u1{};
    const std::array<std::size_t, 2> empty{0, 7};
    ASSERT_TRUE(write_npy<std::uint8_t>(dir_ / "c.npy", u1, empty));
    auto c = read_npy<std::uint8_t>(dir_ / "c.npy", 2);
    ASSERT_TRUE(c);
    EXPECT_EQ(c->size(), 0U);
    EXPECT_EQ(c->cols(), 7U);
}

TEST_F(NpyTest, ParsesVersionTwoHeadersAndBoolArrays) {
    const auto v2 = raw("v2.npy", "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }", 24, 2);
    auto a = read_npy<std::int64_t>(v2);
    ASSERT_TRUE(a) << describe(a.error());
    EXPECT_EQ(a->size(), 3U);
    const auto b1 = raw("b1.npy", "{'descr': '|b1', 'fortran_order': False, 'shape': (4, 2), }", 8);
    auto b = read_npy<std::uint8_t>(b1, 2);
    ASSERT_TRUE(b);
    EXPECT_EQ(b->rows(), 4U);
}

TEST_F(NpyTest, RejectsWhatItCannotRepresentExactly) {
    using Code = IoError::Code;
    const auto code = [](const auto& result) { return result ? std::optional<Code>{} : result.error().code; };
    const auto fortran =
        raw("f.npy", "{'descr': '<u4', 'fortran_order': True, 'shape': (2, 2), }", 16);
    EXPECT_EQ(code(read_npy<std::uint32_t>(fortran)), Code::bad_format);
    const auto big_endian = raw("be.npy", "{'descr': '>u4', 'fortran_order': False, 'shape': (2,), }", 8);
    EXPECT_EQ(code(read_npy<std::uint32_t>(big_endian)), Code::unsupported_dtype);
    const auto wrong_type = raw("wt.npy", "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }", 8);
    EXPECT_EQ(code(read_npy<std::uint32_t>(wrong_type)), Code::dtype_mismatch);
    const auto truncated = raw("tr.npy", "{'descr': '<u4', 'fortran_order': False, 'shape': (5,), }", 8);
    EXPECT_EQ(code(read_npy<std::uint32_t>(truncated)), Code::bad_format);
    const auto ok = raw("ok.npy", "{'descr': '<u4', 'fortran_order': False, 'shape': (2, 3), }", 24);
    EXPECT_EQ(code(read_npy<std::uint32_t>(ok, 1)), Code::shape_mismatch);
    const std::array<std::optional<std::size_t>, 2> extents{std::nullopt, 4};
    EXPECT_EQ(code(read_npy<std::uint32_t>(ok, 2, extents)), Code::shape_mismatch);
    EXPECT_EQ(code(read_npy<std::uint32_t>(dir_ / "missing.npy")), Code::open_failed);
    {
        std::ofstream(dir_ / "junk.npy") << "not an npy file at all";
    }
    EXPECT_EQ(code(read_npy<std::uint32_t>(dir_ / "junk.npy")), Code::bad_format);
}

} // namespace
