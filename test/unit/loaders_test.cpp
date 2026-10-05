#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "rtd/core/kernels.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/sha256.hpp"

namespace {

using namespace rtd;
namespace fs = std::filesystem;

const fs::path fixture = fs::path(RTD_FIXTURE_DIR) / "bb18_choi";

TEST(Sha256, KnownDigest) {
    const fs::path path = fs::temp_directory_path() / "rtd_sha256_abc.txt";
    std::ofstream(path, std::ios::binary) << "abc";
    auto digest = io::sha256_file(path);
    fs::remove(path);
    ASSERT_TRUE(digest);
    EXPECT_EQ(*digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(ArtifactLoader, LoadsAndVerifiesTheFixture) {
    auto artifact = io::load_artifact(fixture / "artifact");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    EXPECT_EQ(artifact->num_detectors(), 72U);
    EXPECT_EQ(artifact->num_columns(), 2268U);
    EXPECT_EQ(artifact->detector_round.size(), 72U);
    EXPECT_GT(artifact->num_observables(), 0U);
    EXPECT_FALSE(artifact->syndrome_bias.has_value());
}

TEST(ArtifactLoader, DetectsATamperedFile) {
    const fs::path copy = fs::temp_directory_path() / "rtd_tampered_artifact";
    fs::remove_all(copy);
    fs::copy(fixture / "artifact", copy, fs::copy_options::recursive);
    {
        std::fstream f(copy / "priors.npy", std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(-1, std::ios::end);
        f.put('\x7f');
    }
    auto artifact = io::load_artifact(copy);
    ASSERT_FALSE(artifact);
    EXPECT_EQ(artifact.error().code, io::IoError::Code::checksum_mismatch);
    fs::remove_all(copy);
}

TEST(GoldenLoader, TranslatesTheGoldenConfiguration) {
    auto artifact = io::load_artifact(fixture / "artifact");
    ASSERT_TRUE(artifact);
    auto relay = io::load_golden(fixture / "relay_f32", artifact->num_detectors(),
                                 artifact->num_columns());
    ASSERT_TRUE(relay) << io::describe(relay.error());
    EXPECT_EQ(relay->decoder, "relay");
    EXPECT_EQ(relay->relay.pre_iter, 20U);
    EXPECT_EQ(relay->relay.num_sets, 40U);
    EXPECT_EQ(std::get<AfterNConverged>(relay->relay.stopping).count, 3U);
    EXPECT_EQ(relay->min_sum.gamma0, 0.125);
    ASSERT_TRUE(relay->gammas.has_value());
    EXPECT_EQ(relay->gammas->rows(), 8U);

    auto adaptive = io::load_golden(fixture / "min_sum_adaptive_f32", artifact->num_detectors(),
                                    artifact->num_columns());
    ASSERT_TRUE(adaptive);
    EXPECT_EQ(std::get<AdaptiveAlpha>(adaptive->min_sum.alpha).scaling, 2.5);
    EXPECT_FALSE(adaptive->min_sum.gamma0.has_value());

    // Every converged golden correction reproduces its syndrome through the loaded graph.
    for (std::size_t s = 0; s < relay->count(); ++s) {
        if (relay->success.data[s] != 0) {
            EXPECT_TRUE(kernels::syndrome_matches(artifact->graph, relay->decoding.row(s),
                                                  relay->detectors.row(s)));
        }
    }
}

TEST(GoldenLoader, RejectsAMismatchedProblemSize) {
    auto golden = io::load_golden(fixture / "min_sum_f32", 72, 2267);
    ASSERT_FALSE(golden);
    EXPECT_EQ(golden.error().code, io::IoError::Code::shape_mismatch);
}

} // namespace
