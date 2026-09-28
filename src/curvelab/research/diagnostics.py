"""Robustness checks for deciding whether a backtest result means anything.

With two months of hourly bars you can make almost any signal look profitable.
These two sweeps are the cheapest way to find out whether it really is, and
every strategy goes through both before its result gets reported.

Execution delay. Re-run the strategy trading one, two and three bars later
than it has to. A real forecast fades gradually, because it's about the next
several bars, not just the next one. If the edge flips sign as soon as you add
one bar, it was never a forecast. It was catching the bounce-back of the move
that triggered it, which on close-to-close data is just bid-ask bounce. You
can't trade that at any speed, since the price that created the signal is the
price you'd have had to trade at.

Cost. Sweep the per-leg cost and find where P&L hits zero. That breakeven cost,
next to what the desk actually pays, tells you more than a Sharpe ratio at one
assumed cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curvelab.research.backtest import run_backtest, summarize


def delay_sensitivity(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    conviction: pd.DataFrame,
    exposure: pd.Series,
    delays: tuple[int, ...] = (0, 1, 2, 3),
    periods_per_year: float = 6000.0,
    **backtest_kwargs,
) -> pd.DataFrame:
    """Sharpe and P&L as a function of extra execution lag."""
    rows = []
    for delay in delays:
        result = run_backtest(
            prices, signals, exposure, conviction=conviction,
            execution_delay=delay, **backtest_kwargs
        )
        stats = summarize(result, periods_per_year)
        rows.append(
            {
                "extra_delay_bars": delay,
                "total_pnl": stats["total_pnl"],
                "sharpe": stats["sharpe"],
                "hit_rate": stats["hit_rate"],
                "n_trades": stats["n_trades"],
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    conviction: pd.DataFrame,
    exposure: pd.Series,
    costs: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
    periods_per_year: float = 6000.0,
    **backtest_kwargs,
) -> pd.DataFrame:
    """P&L as a function of per-leg transaction cost."""
    rows = []
    for cost in costs:
        result = run_backtest(
            prices, signals, exposure, cost_per_leg=cost, slippage_per_leg=cost / 2,
            conviction=conviction, **backtest_kwargs
        )
        stats = summarize(result, periods_per_year)
        rows.append(
            {"cost_per_leg": cost, "total_pnl": stats["total_pnl"], "sharpe": stats["sharpe"]}
        )
    return pd.DataFrame(rows)


def breakeven_cost(sweep: pd.DataFrame) -> float:
    """Per-leg cost at which P&L crosses zero, by linear interpolation."""
    frame = sweep.dropna(subset=["total_pnl"]).sort_values("cost_per_leg")
    pnl = frame["total_pnl"].to_numpy()
    cost = frame["cost_per_leg"].to_numpy()
    sign_change = np.where(np.sign(pnl[:-1]) != np.sign(pnl[1:]))[0]
    if not len(sign_change):
        return float("nan")
    i = sign_change[0]
    span = pnl[i] - pnl[i + 1]
    if span == 0:
        return float(cost[i])
    return float(cost[i] + (cost[i + 1] - cost[i]) * pnl[i] / span)


def verdict(delay_table: pd.DataFrame) -> str:
    """One-line reading of a delay sweep."""
    if delay_table.empty or delay_table["sharpe"].isna().all():
        return "inconclusive"
    base = delay_table.loc[delay_table["extra_delay_bars"] == 0, "sharpe"]
    delayed = delay_table.loc[delay_table["extra_delay_bars"] == 1, "sharpe"]
    if base.empty or delayed.empty:
        return "inconclusive"
    b, d = float(base.iloc[0]), float(delayed.iloc[0])
    if b <= 0:
        return "no edge even at zero delay"
    if d <= 0:
        return "edge inverts after one bar -- microstructure, not forecast"
    if d < 0.5 * b:
        return "edge decays sharply with delay -- treat as fragile"
    return "edge survives execution delay"
