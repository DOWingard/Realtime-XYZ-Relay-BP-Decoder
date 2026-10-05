#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "rtd/core/types.hpp"

// Executors decide how the row and column ranges of one leg are distributed over threads.
//
// A leg is written once as a per-worker function: worker w runs its share of the check pass,
// synchronises, runs its share of the variable pass, synchronises, and every worker then reaches
// the same convergence verdict from the same shared data. Serial runs that function once with
// no-op synchronisation; Team runs it on a persistent, pinned-by-the-caller thread team with a
// spinning barrier, which is the latency configuration for a single decoding problem.

namespace rtd {

// Tells the CPU that this is a spin-wait loop, releasing pipeline resources to the SMT sibling.
inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elifdef __aarch64__
    asm volatile("yield" ::: "memory");
#endif
}

// A reusable barrier for a fixed number of threads that spins instead of sleeping. Crossing it
// costs on the order of 100 ns between cores that share a cache, against microseconds for a
// futex-based barrier, which matters at two barriers per iteration.
class SpinBarrier {
public:
    explicit SpinBarrier(unsigned count) noexcept : count_(count) {}

    void arrive_and_wait() noexcept {
        const unsigned generation = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.store(generation + 1, std::memory_order_release);
            return;
        }
        while (generation_.load(std::memory_order_acquire) == generation) {
            cpu_relax();
        }
    }

private:
    alignas(cache_line_bytes) std::atomic<unsigned> arrived_{0};
    alignas(cache_line_bytes) std::atomic<unsigned> generation_{0};
    unsigned count_;
};

// Handed to each worker of a job.
class WorkerContext {
public:
    WorkerContext(unsigned index, unsigned count, SpinBarrier* barrier) noexcept
        : index_(index), count_(count), barrier_(barrier) {}

    [[nodiscard]] unsigned index() const noexcept { return index_; }
    [[nodiscard]] unsigned count() const noexcept { return count_; }

    // Waits until every worker of the job has reached this point.
    void sync() const noexcept {
        if (barrier_ != nullptr) {
            barrier_->arrive_and_wait();
        }
    }

private:
    unsigned index_;
    unsigned count_;
    SpinBarrier* barrier_;
};

// One worker: the calling thread.
class Serial {
public:
    [[nodiscard]] static constexpr unsigned size() noexcept { return 1; }

    template <class F>
        requires std::is_nothrow_invocable_v<const F&, WorkerContext&>
    void run(const F& job) const noexcept {
        WorkerContext context(0, 1, nullptr);
        job(context);
    }
};

// A persistent team of `threads` workers. Worker 0 is the thread that calls run(); workers
// 1..threads−1 are owned by the team, created once and parked between jobs (spinning briefly,
// then sleeping on an atomic wait). on_thread_start(w) runs first on each owned worker w, which is
// where the caller pins it to a CPU; the core itself has no notion of CPUs.
//
// Jobs must not throw. A Team is movable but not copyable; it is not safe to call run() on the
// same team from two threads at once.
class Team {
public:
    explicit Team(unsigned threads, std::function<void(unsigned)> on_thread_start = {});
    Team(Team&&) noexcept = default;
    Team& operator=(Team&&) noexcept = default;
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    ~Team();

    [[nodiscard]] unsigned size() const noexcept { return state_->count; }

    template <class F>
        requires std::is_nothrow_invocable_v<const F&, WorkerContext&>
    void run(const F& job) noexcept {
        auto trampoline = [](const void* arg, WorkerContext& context) noexcept {
            (*static_cast<const F*>(arg))(context);
        };
        state_->run(trampoline, std::addressof(job));
    }

private:
    using JobFn = void (*)(const void*, WorkerContext&) noexcept;

    struct State {
        explicit State(unsigned workers) : count(workers), barrier(workers) {}

        void run(JobFn fn, const void* arg) noexcept;
        void worker_loop(unsigned index) noexcept;
        void stop() noexcept;

        unsigned count;
        SpinBarrier barrier;
        alignas(cache_line_bytes) std::atomic<std::uint64_t> epoch{0};
        alignas(cache_line_bytes) std::atomic<unsigned> finished{0};
        JobFn job_fn = nullptr;
        const void* job_arg = nullptr;
        std::atomic<bool> stopping{false};
        std::function<void(unsigned)> on_thread_start;
        std::vector<std::jthread> threads;
    };

    std::unique_ptr<State> state_;
};

} // namespace rtd
