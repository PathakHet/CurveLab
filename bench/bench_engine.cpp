// Latency benchmark for the incremental curve engine, run natively.
//
// Timing from Python would mostly measure the cost of crossing into pybind11,
// which is about ten times bigger than the update itself, so this runs entirely
// in C++.
//
// It prints percentiles rather than just a mean, because the mean hides the
// slow tail, and the tail is what you care about on an update path.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "curve_engine.hpp"

using namespace curvelab;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const std::size_t n_contracts = (argc > 1) ? std::stoul(argv[1]) : 11;
    const std::size_t window = (argc > 2) ? std::stoul(argv[2]) : 48;
    const std::size_t n_ticks = (argc > 3) ? std::stoul(argv[3]) : 2000000;

    // Build the same structure universe the research pipeline uses: every
    // spread, fly and double fly on evenly spaced legs.
    std::vector<std::vector<std::pair<ContractIndex, double>>> structures;
    for (std::size_t gap = 1; gap <= 3; ++gap) {
        for (std::size_t i = 0; i + gap < n_contracts; ++i) {
            structures.push_back({{(ContractIndex)i, 1.0}, {(ContractIndex)(i + gap), -1.0}});
        }
        for (std::size_t i = 0; i + 2 * gap < n_contracts; ++i) {
            structures.push_back({{(ContractIndex)i, 1.0},
                                  {(ContractIndex)(i + gap), -2.0},
                                  {(ContractIndex)(i + 2 * gap), 1.0}});
        }
        for (std::size_t i = 0; i + 3 * gap < n_contracts; ++i) {
            structures.push_back({{(ContractIndex)i, 1.0},
                                  {(ContractIndex)(i + gap), -3.0},
                                  {(ContractIndex)(i + 2 * gap), 3.0},
                                  {(ContractIndex)(i + 3 * gap), -1.0}});
        }
    }

    CurveEngine engine(n_contracts, structures, window);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::uint32_t> pick(0, (std::uint32_t)n_contracts - 1);
    std::normal_distribution<double> bump(0.0, 0.02);

    std::vector<std::uint32_t> contracts(n_ticks);
    std::vector<double> prices(n_ticks);
    std::vector<double> level(n_contracts);
    for (std::size_t i = 0; i < n_contracts; ++i) level[i] = 100.0 - 0.5 * (double)i;
    for (std::size_t i = 0; i < n_ticks; ++i) {
        const std::uint32_t c = pick(rng);
        level[c] += bump(rng);
        contracts[i] = c;
        prices[i] = level[c];
    }

    // Warm the caches and the branch predictor before measuring.
    for (std::size_t i = 0; i < std::min<std::size_t>(n_ticks, 100000); ++i) {
        engine.on_tick(contracts[i], prices[i]);
    }

    std::vector<double> latency_ns;
    latency_ns.reserve(n_ticks);
    const auto wall_start = Clock::now();
    for (std::size_t i = 0; i < n_ticks; ++i) {
        const auto t0 = Clock::now();
        engine.on_tick(contracts[i], prices[i]);
        const auto t1 = Clock::now();
        latency_ns.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    const auto wall_end = Clock::now();
    const double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    // Throughput is measured without per-tick timing, since reading the clock
    // costs more than the update.
    const auto bulk_start = Clock::now();
    for (std::size_t i = 0; i < n_ticks; ++i) engine.on_tick(contracts[i], prices[i]);
    const double bulk_s = std::chrono::duration<double>(Clock::now() - bulk_start).count();

    std::sort(latency_ns.begin(), latency_ns.end());
    auto pct = [&](double q) { return latency_ns[(std::size_t)(q * (latency_ns.size() - 1))]; };

    std::printf("contracts            %zu\n", n_contracts);
    std::printf("structures           %zu\n", engine.n_structures());
    std::printf("window               %zu\n", window);
    std::printf("ticks                %zu\n", n_ticks);
    std::printf("mean fan-out         %.2f structures/tick\n", engine.fan_out());
    std::printf("--- per-update latency (instrumented) ---\n");
    std::printf("p50                  %.1f ns\n", pct(0.50));
    std::printf("p90                  %.1f ns\n", pct(0.90));
    std::printf("p99                  %.1f ns\n", pct(0.99));
    std::printf("p99.9                %.1f ns\n", pct(0.999));
    std::printf("max                  %.1f ns\n", pct(1.0));
    std::printf("--- throughput (uninstrumented) ---\n");
    std::printf("bulk                 %.2f M updates/sec (%.1f ns/update)\n",
                n_ticks / bulk_s / 1e6, bulk_s * 1e9 / n_ticks);
    std::printf("instrumented wall    %.2f M updates/sec\n", n_ticks / wall_s / 1e6);
    return 0;
}
