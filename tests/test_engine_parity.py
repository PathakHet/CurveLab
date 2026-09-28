"""The C++ engine must agree with a plain NumPy implementation, tick for tick.

The engine is fast because it never recomputes what it can update: structure
prices move by deltas and rolling stats are kept as running values. That kind
of optimisation tends to work right up until it quietly doesn't. A missed
eviction or a dropped leg drifts slowly and you'd never spot it in the final
results.

So the reference here is deliberately simple. It rebuilds every structure price
from its legs and every z-score from the full window, the way the NumPy code
does. The fast path only gets used because the two agree to 1e-9 over a long
random run of ticks.
"""

import numpy as np
import pytest

from curvelab import _engine


def naive_replay(n_contracts, legs, window, contracts, prices):
    """Recompute everything from scratch at every tick."""
    n_structures = len(legs)
    last = np.full(n_contracts, np.nan)
    history: list[list[float]] = [[] for _ in range(n_structures)]
    out_price = np.full((len(contracts), n_structures), np.nan)
    out_z = np.full((len(contracts), n_structures), np.nan)

    for t, (contract, price) in enumerate(zip(contracts, prices)):
        last[contract] = price
        for s, definition in enumerate(legs):
            if any(np.isnan(last[c]) for c, _ in definition):
                continue
            value = float(sum(w * last[c] for c, w in definition))
            out_price[t, s] = value
            if contract in [c for c, _ in definition]:
                history[s].append(value)
                if len(history[s]) > window:
                    history[s].pop(0)
            if len(history[s]) >= 2:
                window_values = np.array(history[s])
                std = window_values.std(ddof=1)
                out_z[t, s] = (
                    0.0 if std == 0 else (history[s][-1] - window_values.mean()) / std
                )
    return out_price, out_z


@pytest.fixture
def universe():
    """Eleven contracts and every spread/fly/double-fly on them, as in production."""
    n_contracts = 11
    legs = []
    for gap in (1, 2, 3):
        for i in range(n_contracts - gap):
            legs.append([(i, 1.0), (i + gap, -1.0)])
        for i in range(n_contracts - 2 * gap):
            legs.append([(i, 1.0), (i + gap, -2.0), (i + 2 * gap, 1.0)])
        for i in range(n_contracts - 3 * gap):
            legs.append([(i, 1.0), (i + gap, -3.0), (i + 2 * gap, 3.0), (i + 3 * gap, -1.0)])
    return n_contracts, legs


def test_matches_naive_reference(universe):
    n_contracts, legs = universe
    window = 24
    rng = np.random.default_rng(11)
    n_ticks = 1500

    contracts = rng.integers(0, n_contracts, n_ticks).astype(np.uint32)
    level = 100.0 - 0.5 * np.arange(n_contracts)
    prices = np.empty(n_ticks)
    for i, c in enumerate(contracts):
        level[c] += rng.normal(0, 0.02)
        prices[i] = level[c]

    engine = _engine.CurveEngine(n_contracts, legs, window)
    fast_price, fast_z = engine.replay(contracts, prices, True)
    slow_price, slow_z = naive_replay(n_contracts, legs, window, contracts, prices)

    np.testing.assert_allclose(fast_price, slow_price, rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fast_z, slow_z, rtol=0, atol=1e-9, equal_nan=True)


def test_price_is_nan_until_every_leg_has_printed(universe):
    n_contracts, legs = universe
    engine = _engine.CurveEngine(n_contracts, legs, 8)
    engine.on_tick(0, 100.0)
    # Structure 0 is the (0, 1) spread; its second leg has not printed yet.
    assert np.isnan(engine.price(0))
    engine.on_tick(1, 99.0)
    assert engine.price(0) == pytest.approx(1.0)


def test_fan_out_reflects_sparsity(universe):
    n_contracts, legs = universe
    engine = _engine.CurveEngine(n_contracts, legs, 8)
    # Each tick should touch far fewer structures than the whole universe,
    # otherwise the incremental design isn't buying anything.
    assert 0 < engine.fan_out < engine.n_structures


def test_rejects_degenerate_window(universe):
    n_contracts, legs = universe
    with pytest.raises(ValueError):
        _engine.CurveEngine(n_contracts, legs, 1)
