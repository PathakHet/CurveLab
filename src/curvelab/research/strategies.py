"""Rule-based and cross-sectional strategies over the structure universe.

Each strategy returns (signals, conviction). signals are target positions in
{-1, 0, +1}; conviction is how strong each one is, which the backtest uses to
choose between them without churning.

This matters because two structures both saying "long" aren't equally
attractive. Throwing that away and keeping only the -1/0/+1 is what made my
earlier risk cap swap its positions every bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curvelab.research.features import structure_features


def _hysteresis_band(metric: pd.Series, entry: float, exit_: float) -> pd.Series:
    """Mean-reversion signal with memory: enter beyond entry, exit inside exit_.

    Holding between the two bands is the whole point. A strategy that decides
    from scratch every bar ends up paying the spread every bar.
    """
    values = metric.to_numpy(dtype=float)
    out = np.zeros(len(values))
    position = 0.0
    for i, value in enumerate(values):
        if np.isnan(value):
            out[i] = position
            continue
        if position == 0.0:
            if value < -entry:
                position = 1.0
            elif value > entry:
                position = -1.0
        elif abs(value) < exit_:
            position = 0.0
        out[i] = position
    return pd.Series(out, index=metric.index)


def zscore_mean_reversion(
    prices: pd.DataFrame, entry: float = 1.5, exit_: float = 0.3, window: int = 48
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fade structures trading far from their own recent mean."""
    signals, convictions = {}, {}
    for column in prices.columns:
        z = structure_features(prices[column], zscore_window=window)["zscore"]
        signals[column] = _hysteresis_band(z, entry, exit_)
        convictions[column] = z.abs()
    return pd.DataFrame(signals), pd.DataFrame(convictions)


def momentum_breakout(
    prices: pd.DataFrame, window: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow structures breaking out of their recent range."""
    signals, convictions = {}, {}
    for column in prices.columns:
        price = prices[column]
        high = price.rolling(window, min_periods=window // 2).max().shift(1)
        low = price.rolling(window, min_periods=window // 2).min().shift(1)
        signal = pd.Series(0.0, index=price.index)
        signal[price > high] = 1.0
        signal[price < low] = -1.0
        signals[column] = signal.replace(0.0, np.nan).ffill().fillna(0.0)
        vol = price.diff().rolling(window, min_periods=2).std()
        convictions[column] = (price - (high + low) / 2).abs() / vol.replace(0.0, np.nan)
    return pd.DataFrame(signals), pd.DataFrame(convictions)


def cross_sectional_reversal(
    prices: pd.DataFrame, lookback: int = 12, top_k: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buy the structures that have done worst recently and sell the best.

    A relative-value baseline. It doesn't care which way the curve goes, only
    which structures have moved too far compared with the rest.
    """
    changes = prices.diff(lookback)
    scaled = changes.sub(changes.mean(axis=1), axis=0).div(
        changes.std(axis=1).replace(0.0, np.nan), axis=0
    )
    ranks = scaled.rank(axis=1, ascending=True)
    count = scaled.notna().sum(axis=1)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signals[ranks.le(top_k, axis=0)] = 1.0
    signals[ranks.gt(count - top_k, axis=0)] = -1.0
    return signals, scaled.abs()


def random_baseline(
    prices: pd.DataFrame, seed: int = 0, flip_every: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A coin-flip strategy with similar turnover.

    This is the fair benchmark for the others: same costs, same holding times.
    Beating it says something about the signal, not just about trading less.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice([-1.0, 0.0, 1.0], size=(len(prices) // flip_every + 1, prices.shape[1]))
    signals = pd.DataFrame(
        np.repeat(draws, flip_every, axis=0)[: len(prices)],
        index=prices.index,
        columns=prices.columns,
    )
    return signals, pd.DataFrame(
        rng.random(signals.shape), index=signals.index, columns=signals.columns
    )


RULE_BASED = {
    "zscore_mean_reversion": zscore_mean_reversion,
    "momentum_breakout": momentum_breakout,
    "cross_sectional_reversal": cross_sectional_reversal,
    "random_baseline": random_baseline,
}
