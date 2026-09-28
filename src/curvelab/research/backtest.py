"""Backtester for curve structures, working in dollar P&L.

Structure prices are in dollars per barrel and often sit at or below zero, so
percentage returns and compounding don't make sense for them. Everything here
is plain additive dollar P&L on a fixed unit size.

Why it's written this way: my first backtester reported 5,916 trades over 949
bars, every one lasting exactly one bar, with costs at 12.8x gross P&L. That
turned out to be the risk controls, not the strategies. Two bugs caused it.
Positions were capped by ranking on |position|, but with signals in
{-1, 0, +1} every active position tied at 1.0, so column order decided what
got traded. And a gross-exposure cap rescaled every position slightly on every
bar, so nothing was ever unchanged, and the cost model (correctly charging on
position changes) billed a round trip each bar.

This version drops the rescaling. Risk is limited by only holding at most k
unit positions, so exposure is bounded without ever nudging a position that
isn't breaking a limit. select_active ranks on conviction, the continuous
signal strength, so ties don't decide what trades, and it adds hysteresis so a
structure sitting right on the cut-off doesn't flip in and out.

execution_delay adds extra lag on top of the usual one-bar shift. It's the most
useful robustness test you can run on bar data: an edge that survives a
one-bar delay is a forecast, and one that dies was just the reversal of the
move that triggered it. See research/diagnostics.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    pnl: pd.DataFrame
    costs: pd.DataFrame
    portfolio_pnl: pd.Series
    equity: pd.Series
    trades: pd.DataFrame


def select_active(
    signals: pd.DataFrame,
    conviction: pd.DataFrame,
    max_active: int | None,
    hysteresis: float = 0.25,
) -> pd.DataFrame:
    """Limit how many positions are open at once without constantly swapping them.

    A position already held is only replaced if the new one's conviction beats
    it by more than hysteresis. With hysteresis = 0 this is just a top-k ranking
    each bar.
    """
    if not max_active or max_active >= signals.shape[1]:
        return signals

    strength = conviction.abs().fillna(0.0)
    active = np.zeros(signals.shape[1], dtype=bool)
    columns = list(signals.columns)
    out = np.zeros(signals.shape, dtype=float)
    signal_values = signals.fillna(0.0).to_numpy(dtype=float)
    strength_values = strength.to_numpy(dtype=float)

    for t in range(signal_values.shape[0]):
        wants = signal_values[t] != 0.0
        scores = np.where(wants, strength_values[t], -np.inf)
        # Incumbents that still want a position get a hysteresis bonus, so a
        # marginal challenger cannot displace them.
        scores = np.where(active & wants, scores + hysteresis, scores)
        eligible = np.flatnonzero(np.isfinite(scores))
        if eligible.size > max_active:
            keep = eligible[np.argsort(-scores[eligible], kind="stable")[:max_active]]
        else:
            keep = eligible
        active = np.zeros_like(active)
        active[keep] = True
        out[t] = np.where(active, signal_values[t], 0.0)

    return pd.DataFrame(out, index=signals.index, columns=columns)


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    gross_leg_exposure: pd.Series,
    cost_per_leg: float = 0.01,
    slippage_per_leg: float = 0.005,
    conviction: pd.DataFrame | None = None,
    max_active: int | None = 10,
    hysteresis: float = 0.25,
    execution_delay: int = 0,
) -> BacktestResult:
    """Run the backtest.

    The shift(1) here is the one place look-ahead is prevented: a signal built
    from data up to bar t's close becomes the position held through bar t+1, and
    earns bar t+1's price change.
    """
    signals = signals.reindex(columns=prices.columns).fillna(0.0)
    conviction = signals if conviction is None else conviction.reindex(
        columns=prices.columns, index=prices.index
    )

    selected = select_active(signals, conviction, max_active, hysteresis)
    # The anti-lookahead gate. `execution_delay` adds further lag on top of it,
    # simulating a slower path from signal to fill.
    positions = selected.shift(1 + execution_delay).fillna(0.0)

    exposure = gross_leg_exposure.reindex(prices.columns).fillna(0.0)
    per_unit_cost = exposure * (cost_per_leg + slippage_per_leg)

    traded = positions.diff().fillna(positions).abs()
    costs = traded.mul(per_unit_cost, axis=1)
    gross = positions * prices.diff()
    pnl = gross - costs

    portfolio = pnl.sum(axis=1, skipna=True)
    return BacktestResult(
        positions=positions,
        pnl=pnl,
        costs=costs,
        portfolio_pnl=portfolio,
        equity=portfolio.cumsum(),
        trades=extract_trades(positions, pnl),
    )


def extract_trades(positions: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    """Group contiguous same-sign holdings into discrete trades."""
    records = []
    for column in positions.columns:
        side = np.sign(positions[column].fillna(0.0))
        open_mask = side != 0
        if not open_mask.any():
            continue
        block = ((side != side.shift(1)) & open_mask).cumsum().where(open_mask)
        frame = pd.DataFrame(
            {"block": block, "side": side, "pnl": pnl[column]}
        ).dropna(subset=["block"])
        for _, group in frame.groupby("block"):
            records.append(
                {
                    "structure_id": column,
                    "entry_time": group.index[0],
                    "exit_time": group.index[-1],
                    "holding_bars": len(group),
                    "direction": float(group["side"].iloc[0]),
                    "trade_pnl": float(group["pnl"].sum()),
                }
            )
    return pd.DataFrame(records)


def summarize(result: BacktestResult, periods_per_year: float = 6000.0) -> dict[str, float]:
    """Bar-level performance summary in dollar space."""
    pnl = result.portfolio_pnl.dropna()
    if pnl.empty:
        return {}
    std = float(pnl.std(ddof=1))
    downside = pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = float((equity - equity.cummax()).min())
    gross_pnl = float(np.nansum(result.pnl.to_numpy()) + np.nansum(result.costs.to_numpy()))
    trades = result.trades
    wins = trades[trades["trade_pnl"] > 0] if len(trades) else trades
    losses = trades[trades["trade_pnl"] <= 0] if len(trades) else trades
    return {
        "total_pnl": float(pnl.sum()),
        "sharpe": float(pnl.mean() / std * np.sqrt(periods_per_year)) if std else np.nan,
        "sortino": float(
            pnl.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year)
        ) if len(downside) > 1 and downside.std(ddof=1) else np.nan,
        "max_drawdown": drawdown,
        "total_costs": float(np.nansum(result.costs.to_numpy())),
        "cost_ratio": (
            float(np.nansum(result.costs.to_numpy()) / abs(gross_pnl))
            if abs(gross_pnl) > 1e-9 else np.nan
        ),
        "n_trades": int(len(trades)),
        "avg_holding_bars": float(trades["holding_bars"].mean()) if len(trades) else np.nan,
        "hit_rate": float((trades["trade_pnl"] > 0).mean()) if len(trades) else np.nan,
        "profit_factor": float(
            wins["trade_pnl"].sum() / abs(losses["trade_pnl"].sum())
        ) if len(losses) and losses["trade_pnl"].sum() != 0 else np.nan,
    }
