"""Tests for the feature evaluation layer."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from curvelab.research.feature_study import (
    bucket_table,
    cross_sectional_ic,
    economics,
    feature_persistence,
    monotonicity,
    surviving_features,
)


def _panel(n_bars=200, n_names=20, seed=0):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-01", periods=n_bars, freq="h"),
         [f"S{i}" for i in range(n_names)]],
        names=["timestamp", "structure_id"],
    )
    rng = np.random.default_rng(seed)
    return index, rng


def test_vectorised_ic_matches_scipy_spearman():
    """The fast path must be the same statistic, not an approximation of it."""
    index, rng = _panel()
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index)
    fast = cross_sectional_ic(x, y)

    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    slow = frame.groupby(level="timestamp").apply(
        lambda g: scipy_stats.spearmanr(g["x"], g["y"]).statistic
    )
    pd.testing.assert_series_equal(
        fast.sort_index(), slow.reindex(fast.index).sort_index(),
        check_names=False, atol=1e-12,
    )


def test_ic_detects_a_planted_signal():
    index, rng = _panel(seed=3)
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = x * 0.5 + pd.Series(rng.normal(size=len(index)), index=index) * 0.1
    assert cross_sectional_ic(x, y).mean() > 0.8


def test_ic_is_near_zero_on_noise():
    index, rng = _panel(n_bars=400, seed=5)
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index)
    assert abs(cross_sectional_ic(x, y).mean()) < 0.05


def test_bars_with_too_few_names_are_dropped():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-01", periods=3, freq="h"), ["A", "B"]],
        names=["timestamp", "structure_id"],
    )
    x = pd.Series(np.arange(6, dtype=float), index=index)
    assert cross_sectional_ic(x, x, min_names=8).empty


def test_persistence_separates_static_from_dynamic_features():
    """A per-structure constant must be flagged; a price-driven feature must not."""
    index, rng = _panel(n_bars=100, n_names=10, seed=7)
    names = index.get_level_values("structure_id")
    static = pd.Series(
        [float(n[1:]) for n in names], index=index
    )  # fixed per structure, e.g. days to expiry
    dynamic = pd.Series(rng.normal(size=len(index)), index=index)

    assert feature_persistence(static) > 0.95
    assert feature_persistence(dynamic) < 0.20


def test_static_features_are_excluded_from_survivors():
    scorecard = pd.DataFrame(
        {
            "feature": ["static_one", "static_one", "real_one", "real_one"],
            "horizon": [6, 6, 6, 6],
            "skip": [0, 1, 0, 1],
            "mean_ic": [0.15, 0.15, -0.04, -0.04],
            "t_stat": [5.0, 5.0, -3.0, -3.0],
            "persistence": [0.99, 0.99, 0.01, 0.01],
        }
    )
    survivors = surviving_features(scorecard)
    flags = survivors.set_index("feature")["survives"]
    assert not flags["static_one"]     # significant, but one repeated bet
    assert flags["real_one"]


def test_sign_flip_between_skips_fails_the_delay_check():
    """An IC that reverses once you cannot trade the signal bar is bounce."""
    scorecard = pd.DataFrame(
        {
            "feature": ["bounce", "bounce"],
            "horizon": [3, 3],
            "skip": [0, 1],
            "mean_ic": [-0.35, 0.04],
            "t_stat": [-9.0, 3.0],
            "persistence": [0.01, 0.01],
        }
    )
    assert not surviving_features(scorecard)["survives"].any()


def test_bucket_table_and_monotonicity():
    index, rng = _panel(n_bars=300, seed=11)
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = -x * 0.02 + pd.Series(rng.normal(size=len(index)), index=index) * 0.001
    table = bucket_table(x, y, n_buckets=5)
    assert len(table) == 5
    assert monotonicity(table) == pytest.approx(-1.0)
    assert table["mean_move"].iloc[0] > table["mean_move"].iloc[-1]


def test_economics_compares_edge_to_leg_weighted_cost():
    """One unit of a fly trades four contracts, so it pays four spreads."""
    table = pd.DataFrame({"mean_move": [0.021, 0.005, -0.002, -0.010, -0.021]})
    result = economics(table, gross_leg_exposure=4.0, cost_per_leg=0.015)
    assert result["round_trip_cost"] == pytest.approx(0.06)
    assert result["edge_over_cost"] == pytest.approx(0.35, abs=0.01)
    assert result["edge_over_cost"] < 1.0     # real but uneconomic
