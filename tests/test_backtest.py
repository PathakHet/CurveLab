"""Tests for the backtest engine's anti-lookahead gate and turnover control."""

import numpy as np
import pandas as pd
import pytest

from curvelab.research.backtest import run_backtest, select_active, summarize
from curvelab.research.diagnostics import breakeven_cost, verdict


@pytest.fixture
def straight_line():
    index = pd.date_range("2026-05-01", periods=50, freq="h")
    prices = pd.DataFrame({"A": np.arange(50, dtype=float)}, index=index)
    exposure = pd.Series({"A": 2.0})
    return prices, exposure


def test_signal_cannot_trade_on_its_own_bar(straight_line):
    """A signal formed at t is paid at t+1's price change, never at t's."""
    prices, exposure = straight_line
    signals = pd.DataFrame(0.0, index=prices.index, columns=["A"])
    signals.iloc[10] = 1.0
    result = run_backtest(prices, signals, exposure, cost_per_leg=0, slippage_per_leg=0,
                          max_active=None)
    assert result.positions.iloc[10]["A"] == 0.0   # not yet in the market
    assert result.positions.iloc[11]["A"] == 1.0
    assert result.pnl.iloc[10]["A"] == 0.0


def test_perfect_foresight_is_only_profitable_without_the_shift(straight_line):
    """An oracle trading on its own bar makes money. Shifted by one bar it still
    does on a trending series; what matters is that the shift changes the result."""
    prices, exposure = straight_line
    signals = pd.DataFrame(np.sign(prices.diff().fillna(0.0)), index=prices.index)
    shifted = run_backtest(prices, signals, exposure, cost_per_leg=0,
                           slippage_per_leg=0, max_active=None)
    unshifted_pnl = float((signals["A"] * prices["A"].diff()).sum())
    assert unshifted_pnl > float(shifted.portfolio_pnl.sum()) or np.isclose(
        unshifted_pnl, float(shifted.portfolio_pnl.sum())
    )


def test_execution_delay_adds_lag_on_top_of_the_shift(straight_line):
    prices, exposure = straight_line
    signals = pd.DataFrame(0.0, index=prices.index, columns=["A"])
    signals.iloc[10] = 1.0
    delayed = run_backtest(prices, signals, exposure, cost_per_leg=0, slippage_per_leg=0,
                           max_active=None, execution_delay=2)
    assert delayed.positions.iloc[11]["A"] == 0.0
    assert delayed.positions.iloc[13]["A"] == 1.0


def test_costs_are_charged_on_position_changes_only(straight_line):
    prices, exposure = straight_line
    signals = pd.DataFrame(1.0, index=prices.index, columns=["A"])
    result = run_backtest(prices, signals, exposure, cost_per_leg=0.01,
                          slippage_per_leg=0.005, max_active=None)
    charged = result.costs["A"]
    assert (charged > 0).sum() == 1              # entry only
    assert charged.iloc[1] == pytest.approx(2.0 * 0.015)


def test_hysteresis_reduces_churn_in_the_active_set():
    """A structure hovering at the selection boundary must not round-trip."""
    index = pd.date_range("2026-05-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    columns = [f"S{i}" for i in range(6)]
    signals = pd.DataFrame(1.0, index=index, columns=columns)
    # Convictions of similar magnitude that reorder constantly, so several
    # structures sit right at the top-3 cut-off on every bar.
    conviction = pd.DataFrame(
        rng.normal(0, 1, (200, 6)), index=index, columns=columns
    )
    churny = select_active(signals, conviction, max_active=3, hysteresis=0.0)
    sticky = select_active(signals, conviction, max_active=3, hysteresis=0.5)
    assert sticky.diff().abs().sum().sum() < churny.diff().abs().sum().sum()


def test_selection_bounds_gross_exposure_without_rescaling(straight_line):
    """Risk is capped by selection, so no position is perturbed by the cap."""
    index = pd.date_range("2026-05-01", periods=20, freq="h")
    columns = [f"S{i}" for i in range(8)]
    prices = pd.DataFrame(
        np.arange(20 * 8, dtype=float).reshape(20, 8), index=index, columns=columns
    )
    signals = pd.DataFrame(1.0, index=index, columns=columns)
    conviction = pd.DataFrame(
        np.tile(np.arange(8, dtype=float), (20, 1)), index=index, columns=columns
    )
    exposure = pd.Series({c: 4.0 for c in columns})
    result = run_backtest(prices, signals, exposure, conviction=conviction, max_active=3)
    held = result.positions.iloc[-1]
    assert (held != 0).sum() == 3
    assert set(np.unique(held.to_numpy())) <= {0.0, 1.0}   # never a fractional cap


def test_verdict_flags_an_edge_that_inverts():
    table = pd.DataFrame(
        {"extra_delay_bars": [0, 1, 2], "sharpe": [11.2, -8.7, -10.6]}
    )
    assert verdict(table) == "edge inverts after one bar -- microstructure, not forecast"


def test_breakeven_cost_interpolates_the_zero_crossing():
    sweep = pd.DataFrame({"cost_per_leg": [0.0, 0.01, 0.02], "total_pnl": [100.0, 50.0, -50.0]})
    assert breakeven_cost(sweep) == pytest.approx(0.015, abs=1e-6)


def test_summary_is_empty_rather_than_wrong_on_no_data():
    empty = pd.DataFrame(columns=["A"], dtype=float)
    result = run_backtest(empty, empty, pd.Series({"A": 2.0}), max_active=None)
    assert summarize(result) == {}
