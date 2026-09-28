"""Statistics that account for having tried more than one thing.

The Sharpe ratio of the best of several strategies, over 949 bars, isn't the
Sharpe ratio of that strategy. Every headline number gets two corrections.

Deflated Sharpe (Bailey & Lopez de Prado). The best Sharpe you'd expect from N
useless strategies goes up with N. The deflated Sharpe is the probability that
the observed Sharpe beats that benchmark, after adjusting for skew and
kurtosis. It's the difference between "the best of nine strategies made money"
and "this strategy has an edge".

Block bootstrap p-values. Bar-level P&L is autocorrelated, so a plain t-test
overstates significance. The stationary bootstrap resamples blocks of bars and
keeps that dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from curvelab.desk.skill import stationary_bootstrap_indices


def sharpe_ratio(pnl: pd.Series, periods_per_year: float) -> float:
    clean = pnl.dropna()
    std = clean.std(ddof=1)
    if not len(clean) or std == 0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Expected maximum Sharpe of ``n_trials`` independent zero-edge strategies."""
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649015329
    z_1 = scipy_stats.norm.ppf(1 - 1.0 / n_trials)
    z_2 = scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(((1 - euler) * z_1 + euler * z_2) / np.sqrt(max(n_obs - 1, 1)))


def deflated_sharpe(
    pnl: pd.Series, n_trials: int, periods_per_year: float
) -> dict[str, float]:
    """Probability the observed Sharpe beats the best-of-``n_trials`` null."""
    clean = pnl.dropna()
    n_obs = len(clean)
    if n_obs < 20 or clean.std(ddof=1) == 0:
        return {"sharpe": np.nan, "sharpe_per_bar": np.nan,
                "benchmark": np.nan, "deflated_sharpe": np.nan}

    per_bar = float(clean.mean() / clean.std(ddof=1))
    skew = float(scipy_stats.skew(clean))
    kurtosis = float(scipy_stats.kurtosis(clean, fisher=False))
    benchmark = expected_max_sharpe(n_trials, n_obs)

    denominator = np.sqrt(
        max(1 - skew * per_bar + (kurtosis - 1) / 4 * per_bar**2, 1e-12)
    )
    z = (per_bar - benchmark) * np.sqrt(n_obs - 1) / denominator
    return {
        "sharpe": per_bar * np.sqrt(periods_per_year),
        "sharpe_per_bar": per_bar,
        "benchmark": benchmark * np.sqrt(periods_per_year),
        "deflated_sharpe": float(scipy_stats.norm.cdf(z)),
    }


def bootstrap_pvalue(
    pnl: pd.Series, n_draws: int = 2000, mean_block: float = 12.0, seed: int = 0
) -> float:
    """Two-sided p-value for mean bar P&L under a stationary bootstrap null."""
    values = pnl.dropna().to_numpy(dtype=float)
    if len(values) < 20:
        return float("nan")
    observed = values.mean()
    centred = values - observed
    rng = np.random.default_rng(seed)
    null = np.empty(n_draws)
    for draw in range(n_draws):
        null[draw] = centred[stationary_bootstrap_indices(len(values), mean_block, rng)].mean()
    return float((np.abs(null) >= abs(observed)).mean())


def evaluate(
    pnl: pd.Series, n_trials: int, periods_per_year: float, seed: int = 0
) -> dict[str, float]:
    """Full statistical assessment of one strategy's P&L series."""
    out = deflated_sharpe(pnl, n_trials, periods_per_year)
    out["bootstrap_p"] = bootstrap_pvalue(pnl, seed=seed)
    out["n_bars"] = int(pnl.dropna().size)
    return out
