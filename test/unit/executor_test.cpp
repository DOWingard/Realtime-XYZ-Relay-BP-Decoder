#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "rtd/core/executor.hpp"

namespace {

using namespace rtd;

TEST(Serial, RunsOnceOnTheCallingThread) {
    const Serial serial;
    int calls = 0;
    std::thread::id id;
    serial.run([&](WorkerContext& w) noexcept {
        ++calls;
        id = std::this_thread::get_id();
        EXPECT_EQ(w.index(), 0U);
        EXPECT_EQ(w.count(), 1U);
        w.sync();
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(id, std::this_thread::get_id());
}

TEST(Team, EveryWorkerRunsEachJobAndBarriersOrderPhases) {
    for (const unsigned threads : {1U, 2U, 3U, 5U}) {
        std::atomic<unsigned> started{0};
        Team team(threads, [&](unsigned) { started.fetch_add(1); });
        EXPECT_EQ(team.size(), threads);
        std::vector<int> phase_a(threads, 0);
        std::vector<int> seen(threads, 0);
        for (int job = 0; job < 50; ++job) {
            team.run([&](WorkerContext& w) noexcept {
                phase_a[w.index()] = job + 1;
                w.sync();
                // After the barrier every worker's phase-A write is visible.
                int sum = 0;
                for (unsigned k = 0; k < w.count(); ++k) {
                    sum += phase_a[k] == job + 1 ? 1 : 0;
                }
                seen[w.index()] = sum;
                w.sync();
            });
            for (unsigned k = 0; k < threads; ++k) {
                EXPECT_EQ(seen[k], static_cast<int>(threads));
            }
        }
        EXPECT_EQ(started.load(), threads - 1);
    }
}

TEST(Team, MovedTeamKeepsWorking) {
    Team a(3);
    Team b(std::move(a));
    std::atomic<int> count{0};
    b.run([&](WorkerContext&) noexcept { count.fetch_add(1); });
    EXPECT_EQ(count.load(), 3);
}

} // namespace
