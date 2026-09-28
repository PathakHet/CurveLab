// Incremental curve-structure pricing engine.
//
// About a dozen Brent contracts give you seventy-odd tradeable structures, and
// each structure is a fixed weighted sum of contract prices. So when one
// contract's price changes, only the structures that contain it need updating.
//
// The NumPy research code recomputes the whole price matrix and every rolling
// statistic whenever anything changes, which is O(S * W) per update. That's
// fine for research on a fixed dataset, but wasteful when a single leg ticks
// and only a handful of structures actually move. This engine works per update
// instead:
//
//   * A CSR index maps each contract to the (structure, weight) pairs that use
//     it, so a tick only touches structures containing that contract. A
//     contract typically shows up in ~15 of ~70 structures.
//
//   * Structure prices are updated by the change, price[s] += w * (new - old),
//     rather than rebuilt from their legs. That's one multiply-add per
//     structure however many legs it has.
//
//   * Rolling mean and variance live in a fixed-size ring buffer with O(1)
//     insert and evict, so z-scores cost the same whatever the window length.
//
//   * State is kept as a few flat arrays allocated once up front. The update
//     path doesn't allocate or chase pointers.
//
// A note on the rolling variance. My first version used the textbook O(1)
// trick: keep `sum` and `sumsq`, subtract on evict, and compute
// (sumsq - n*mean^2)/(n-1). The parity test caught it drifting. When prices in
// the window are nearly equal, sumsq and n*mean^2 are nearly equal too, and
// subtracting them throws away most of the precision. Z-scores ended up ~3e-6
// off the NumPy reference, and got worse as the window went quiet, which is
// exactly when a mean-reversion signal is about to fire.
//
// So the window now keeps Welford's running mean and sum of squared
// deviations, with matching add/remove steps that never do that subtraction.
// Removing is still where Welford is weakest, so every `window` updates the
// accumulator is recomputed exactly from the ring buffer. That's O(W) work
// every W updates (O(1) amortised) and keeps any drift to one window's worth.
// tests/test_engine_parity.py runs the same ticks through this engine and a
// NumPy reference and requires them to agree to 1e-9.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace curvelab {

using StructureIndex = std::uint32_t;
using ContractIndex = std::uint32_t;

// One (contract, weight) term of one structure, stored leg-major for the
// inverted index.
struct Leg {
    StructureIndex structure;
    double weight;
};

class CurveEngine {
public:
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // `legs_per_structure` is the structure definition: for each structure, the
    // (contract index, weight) pairs that define it. `window` is the rolling
    // z-score lookback in updates.
    CurveEngine(std::size_t n_contracts,
                const std::vector<std::vector<std::pair<ContractIndex, double>>>& legs_per_structure,
                std::size_t window)
        : n_contracts_(n_contracts),
          n_structures_(legs_per_structure.size()),
          window_(window) {
        if (window_ < 2) {
            throw std::invalid_argument("window must be at least 2");
        }

        // Build the CSR inverted index: contract -> legs of structures using it.
        std::vector<std::size_t> counts(n_contracts_ + 1, 0);
        for (const auto& legs : legs_per_structure) {
            for (const auto& [contract, weight] : legs) {
                if (contract >= n_contracts_) {
                    throw std::invalid_argument("contract index out of range");
                }
                ++counts[contract + 1];
            }
        }
        offsets_.assign(n_contracts_ + 1, 0);
        for (std::size_t i = 0; i < n_contracts_; ++i) {
            offsets_[i + 1] = offsets_[i] + counts[i + 1];
        }
        entries_.resize(offsets_.back());

        std::vector<std::size_t> cursor(offsets_.begin(), offsets_.end() - 1);
        for (std::size_t s = 0; s < n_structures_; ++s) {
            for (const auto& [contract, weight] : legs_per_structure[s]) {
                entries_[cursor[contract]++] =
                    Leg{static_cast<StructureIndex>(s), weight};
            }
        }

        // A structure has no price until every one of its legs has printed.
        pending_legs_.resize(n_structures_, 0);
        for (std::size_t s = 0; s < n_structures_; ++s) {
            pending_legs_[s] = static_cast<std::uint32_t>(legs_per_structure[s].size());
        }

        last_price_.assign(n_contracts_, kNaN);
        seen_.assign(n_contracts_, 0);
        price_.assign(n_structures_, 0.0);
        zscore_.assign(n_structures_, kNaN);
        mean_.assign(n_structures_, 0.0);
        m2_.assign(n_structures_, 0.0);
        since_refresh_.assign(n_structures_, 0);
        ring_.assign(n_structures_ * window_, 0.0);
        ring_pos_.assign(n_structures_, 0);
        ring_count_.assign(n_structures_, 0);

        // Welford divides by how full the window is, which is always between 0
        // and `window`. Precomputing 1/n for each value swaps two slow
        // divisions per update for two cache loads, and once the window is
        // full it's the same entry every time.
        inv_n_.resize(window_ + 2, 0.0);
        for (std::size_t k = 1; k < inv_n_.size(); ++k) {
            inv_n_[k] = 1.0 / static_cast<double>(k);
        }
    }

    // Hot path. Applies one contract price update and refreshes every structure
    // that contains that contract. Returns the number of structures touched.
    std::size_t on_tick(ContractIndex contract, double price) noexcept {
        if (contract >= n_contracts_) {
            return 0;
        }
        const bool first = !seen_[contract];
        const double previous = first ? 0.0 : last_price_[contract];
        const double delta = first ? price : price - previous;
        last_price_[contract] = price;
        seen_[contract] = 1;

        const std::size_t begin = offsets_[contract];
        const std::size_t end = offsets_[contract + 1];
        const Leg* __restrict entries = entries_.data();

        for (std::size_t i = begin; i < end; ++i) {
            const StructureIndex s = entries[i].structure;
            price_[s] += entries[i].weight * delta;
            if (first && pending_legs_[s]) {
                --pending_legs_[s];
            }
            if (pending_legs_[s] == 0) {
                push(s, price_[s]);
            }
        }
        ++n_updates_;
        return end - begin;
    }

    double price(StructureIndex s) const noexcept {
        return pending_legs_[s] == 0 ? price_[s] : kNaN;
    }
    double zscore(StructureIndex s) const noexcept { return zscore_[s]; }

    const double* prices() const noexcept { return price_.data(); }
    const double* zscores() const noexcept { return zscore_.data(); }

    std::size_t n_structures() const noexcept { return n_structures_; }
    std::size_t n_contracts() const noexcept { return n_contracts_; }
    std::size_t window() const noexcept { return window_; }
    std::uint64_t n_updates() const noexcept { return n_updates_; }

    // Average number of structures touched per tick. The lower this is, the
    // more the incremental approach pays off.
    double fan_out() const noexcept {
        return n_contracts_ ? static_cast<double>(entries_.size()) /
                              static_cast<double>(n_contracts_)
                            : 0.0;
    }

private:
    // Insert one observation into a structure's rolling window, evicting the
    // oldest, and refresh its z-score. Constant time in the window length.
    void push(StructureIndex s, double value) noexcept {
        double* __restrict ring = ring_.data() + static_cast<std::size_t>(s) * window_;
        const std::size_t pos = ring_pos_[s];

        if (ring_count_[s] == window_) {
            remove(s, ring[pos]);
        }
        ring[pos] = value;
        add(s, value);
        ring_pos_[s] = (pos + 1 == window_) ? 0 : pos + 1;

        // Every `window` updates, recompute exactly to clear any drift from
        // repeated Welford removals.
        if (++since_refresh_[s] >= window_) {
            refresh(s, ring);
            since_refresh_[s] = 0;
        }

        const std::size_t count = ring_count_[s];
        if (count < 2) {
            zscore_[s] = kNaN;
            return;
        }
        const double variance = m2_[s] * inv_n_[count - 1];
        if (!(variance > 0.0)) {
            zscore_[s] = 0.0;
            return;
        }
        zscore_[s] = (value - mean_[s]) / std::sqrt(variance);
    }

    // Welford insert.
    void add(StructureIndex s, double x) noexcept {
        const std::size_t n = ++ring_count_[s];
        const double delta = x - mean_[s];
        mean_[s] += delta * inv_n_[n];
        m2_[s] += delta * (x - mean_[s]);
    }

    // Welford delete: the exact inverse of `add`, never forming sumsq - n*mean^2.
    void remove(StructureIndex s, double x) noexcept {
        const std::size_t n = --ring_count_[s];
        if (n == 0) {
            mean_[s] = 0.0;
            m2_[s] = 0.0;
            return;
        }
        const double delta = x - mean_[s];
        mean_[s] -= delta * inv_n_[n];
        m2_[s] -= delta * (x - mean_[s]);
        if (m2_[s] < 0.0) m2_[s] = 0.0;
    }

    // Exact two-pass recomputation from the ring buffer.
    void refresh(StructureIndex s, const double* __restrict ring) noexcept {
        const std::size_t count = ring_count_[s];
        if (count == 0) return;
        // Valid entries occupy [0, count): the ring either is full, or has not
        // wrapped yet and was filled from index 0.
        double mean = 0.0;
        for (std::size_t i = 0; i < count; ++i) mean += ring[i];
        mean /= static_cast<double>(count);
        double m2 = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double d = ring[i] - mean;
            m2 += d * d;
        }
        mean_[s] = mean;
        m2_[s] = m2;
    }

    std::size_t n_contracts_;
    std::size_t n_structures_;
    std::size_t window_;
    std::uint64_t n_updates_ = 0;

    std::vector<std::size_t> offsets_;   // CSR row offsets, per contract
    std::vector<Leg> entries_;           // CSR entries: (structure, weight)

    std::vector<double> last_price_;
    std::vector<std::uint8_t> seen_;
    std::vector<std::uint32_t> pending_legs_;

    std::vector<double> price_;
    std::vector<double> zscore_;
    std::vector<double> mean_;
    std::vector<double> m2_;
    std::vector<std::size_t> since_refresh_;
    std::vector<double> ring_;
    std::vector<std::size_t> ring_pos_;
    std::vector<std::size_t> ring_count_;
    std::vector<double> inv_n_;
};

}  // namespace curvelab
