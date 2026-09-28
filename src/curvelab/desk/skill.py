"""Is anyone on the desk measurably good, or just lucky?

With dozens of traders, testing each one at 5% will throw up a couple of
"skilled" traders by chance alone, and whoever tops a big contest looks
impressive whether or not they can actually trade. So I correct for three
things:

Autocorrelation. Positions carry over between days, so daily P&L isn't
independent and a plain t-test overstates significance. I use a stationary
bootstrap (Politis-Romano), which resamples random-length blocks of days and
keeps that dependence intact.

Multiple testing. The per-trader p-values go through Benjamini-Hochberg, which
controls the false discovery rate: among the traders it flags, at most a
fraction q should be noise. I went with BH over Bonferroni because the aim is
to find anyone who's genuinely good, not to prove one hypothesis beyond doubt.

Selection. Even after that, the top trader's numbers are biased upward just
because they're the top. max_statistic_test simulates a desk where nobody has
any skill and asks how often its best trader beats the real best trader. That
compares the winner against the best of N coin-flippers rather than a single
coin flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BLOCK = 3.0     # mean geometric block length, in days
DEFAULT_DRAWS = 5000


@dataclass
class SkillResult:
    table: pd.DataFrame
    q_value: float
    n_significant: int
    max_t_observed: float
    max_t_null_p: float

    def summary(self) -> str:
        return (
            f"Testing {len(self.table)} traders at FDR q={self.q_value:.2f}, "
            f"{self.n_significant} show P&L distinguishable from luck. "
            f"The best observed t-statistic is {self.max_t_observed:.2f}; under the null "
            f"that no one has skill, the best of {len(self.table)} traders exceeds that "
            f"value {self.max_t_null_p:.0%} of the time."
        )


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks, wrapped."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    current = rng.integers(0, n)
    for i in range(n):
        idx[i] = current
        if rng.random() < p:
            current = rng.integers(0, n)
        else:
            current = (current + 1) % n
    return idx


def bootstrap_mean_pvalue(
    series: pd.Series,
    n_draws: int = DEFAULT_DRAWS,
    mean_block: float = DEFAULT_BLOCK,
    seed: int = 0,
) -> tuple[float, float, tuple[float, float]]:
    """Two-sided p-value and CI for a mean daily P&L under serial dependence.

    Subtracting the mean before resampling centres the bootstrap on "no edge",
    and the observed mean is compared against that.
    """
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 5:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    observed = float(values.mean())
    centred = values - observed
    rng = np.random.default_rng(seed)

    null_means = np.empty(n_draws)
    boot_means = np.empty(n_draws)
    for draw in range(n_draws):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        null_means[draw] = centred[idx].mean()
        boot_means[draw] = values[idx].mean()

    p_value = float((np.abs(null_means) >= abs(observed)).mean())
    scale = null_means.std(ddof=1)
    t_stat = observed / scale if scale > 0 else np.nan
    ci = (float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975)))
    return p_value, float(t_stat), ci


def benjamini_hochberg(p_values: pd.Series, q: float = 0.10) -> pd.Series:
    """BH step-up: True where the hypothesis is rejected at FDR ``q``."""
    clean = p_values.dropna().sort_values()
    m = len(clean)
    if m == 0:
        return pd.Series(dtype=bool)
    thresholds = q * np.arange(1, m + 1) / m
    passing = np.where(clean.to_numpy() <= thresholds)[0]
    cutoff = clean.iloc[passing[-1]] if len(passing) else -np.inf
    return (p_values <= cutoff).fillna(False)


def max_statistic_test(
    daily_pnl: pd.DataFrame,
    n_draws: int = 1000,
    mean_block: float = DEFAULT_BLOCK,
    seed: int = 7,
) -> tuple[float, float]:
    """Compare the best trader against the best of an all-null desk.

    Resamples everyone's de-meaned P&L on the same days together (they all
    traded the same curve, so their results are correlated), takes the best
    t-stat each time, and reports how often it beats the real best.
    """
    matrix = daily_pnl.copy()
    observed_t = []
    for trader in matrix.columns:
        series = matrix[trader].dropna()
        if len(series) < 5 or series.std(ddof=1) == 0:
            continue
        observed_t.append(series.mean() / (series.std(ddof=1) / np.sqrt(len(series))))
    if not observed_t:
        return float("nan"), float("nan")
    max_observed = float(np.nanmax(observed_t))

    centred = matrix - matrix.mean()
    values = centred.to_numpy(dtype=float)
    n_days = values.shape[0]
    rng = np.random.default_rng(seed)

    null_max = np.empty(n_draws)
    for draw in range(n_draws):
        idx = stationary_bootstrap_indices(n_days, mean_block, rng)
        sample = values[idx]
        with np.errstate(invalid="ignore"):
            counts = np.sum(~np.isnan(sample), axis=0)
            means = np.nanmean(sample, axis=0)
            stds = np.nanstd(sample, axis=0, ddof=1)
            t_stats = np.where(
                (counts >= 5) & (stds > 0), means / (stds / np.sqrt(np.maximum(counts, 1))), np.nan
            )
        null_max[draw] = np.nanmax(t_stats) if np.any(~np.isnan(t_stats)) else np.nan
    p_value = float(np.nanmean(null_max >= max_observed))
    return max_observed, p_value


def assess(
    daily_pnl: pd.DataFrame,
    q: float = 0.10,
    min_days: int = 8,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> SkillResult:
    """Full skill assessment across the desk."""
    rows = []
    for i, trader in enumerate(daily_pnl.columns):
        series = daily_pnl[trader].dropna()
        if len(series) < min_days:
            continue
        p_value, t_stat, ci = bootstrap_mean_pvalue(
            series, n_draws=n_draws, seed=seed + i
        )
        rows.append(
            {
                "trader": trader,
                "n_days": len(series),
                "total_pnl": float(series.sum()),
                "mean_daily_pnl": float(series.mean()),
                "daily_vol": float(series.std(ddof=1)),
                "t_stat": t_stat,
                "p_value": p_value,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )
    if not rows:
        return SkillResult(pd.DataFrame(), q, 0, float("nan"), float("nan"))

    table = pd.DataFrame(rows).set_index("trader").sort_values("t_stat", ascending=False)
    table["significant_raw"] = table["p_value"] < 0.05
    table["significant_fdr"] = benjamini_hochberg(table["p_value"], q=q)
    max_t, max_p = max_statistic_test(daily_pnl[table.index], seed=seed + 999)

    return SkillResult(
        table=table,
        q_value=q,
        n_significant=int(table["significant_fdr"].sum()),
        max_t_observed=max_t,
        max_t_null_p=max_p,
    )
