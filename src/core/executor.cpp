#include "rtd/core/executor.hpp"

#include <algorithm>

namespace rtd {

namespace {
// Spin iterations before a parked worker falls back to sleeping on the epoch counter. Covers the
// gap between the legs of one decode and between consecutive decodes without a futex round trip.
constexpr unsigned spins_before_sleep = 1U << 14U;
} // namespace

Team::Team(unsigned threads, std::function<void(unsigned)> on_thread_start)
    : state_(std::make_unique<State>(std::max(threads, 1U))) {
    state_->on_thread_start = std::move(on_thread_start);
    try {
        state_->threads.reserve(state_->count - 1);
        for (unsigned w = 1; w < state_->count; ++w) {
            state_->threads.emplace_back([state = state_.get(), w] { state->worker_loop(w); });
        }
    } catch (...) {
        // Release the workers already started, or joining them would wait forever.
        state_->stop();
        throw;
    }
}

Team::~Team() {
    if (state_) {
        state_->stop();
    }
}

void Team::State::stop() noexcept {
    stopping.store(true, std::memory_order_release);
    epoch.fetch_add(1, std::memory_order_release);
    epoch.notify_all();
    threads.clear(); // joins
}

void Team::State::run(JobFn fn, const void* arg) noexcept {
    job_fn = fn;
    job_arg = arg;
    finished.store(0, std::memory_order_relaxed);
    epoch.fetch_add(1, std::memory_order_release);
    epoch.notify_all();
    WorkerContext context(0, count, &barrier);
    fn(arg, context);
    while (finished.load(std::memory_order_acquire) != count - 1) {
        cpu_relax();
    }
}

void Team::State::worker_loop(unsigned index) noexcept {
    if (on_thread_start) {
        on_thread_start(index);
    }
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t current = epoch.load(std::memory_order_acquire);
        for (unsigned spins = 0; current == seen; current = epoch.load(std::memory_order_acquire)) {
            if (++spins < spins_before_sleep) {
                cpu_relax();
            } else {
                epoch.wait(seen, std::memory_order_acquire);
            }
        }
        seen = current;
        if (stopping.load(std::memory_order_acquire)) {
            return;
        }
        WorkerContext context(index, count, &barrier);
        job_fn(job_arg, context);
        finished.fetch_add(1, std::memory_order_release);
    }
}

} // namespace rtd
