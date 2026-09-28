"""Which signals actually tell you where a curve structure goes next?

This is the desk's question in plain terms: when I'm looking at a board of
flies, what should I be watching? That's about evaluating features, not about
building a strategy, and it's worth keeping separate from the backtest. A
feature can carry real information and still lose money after costs, and it
helps to know which of the two is the problem.

Features are scored by information coefficient (IC). At each hour, rank the
structures on the board by the feature, rank them by what they did next, and
take the rank correlation. That gives one number per hour. The average is the
feature's edge, and its t-stat tells you whether that edge is distinguishable
from zero. Ranking across the board, rather than pooling everything, matches
what a relative-value trader actually asks (which of these is the best trade
right now?) and isn't thrown off by the whole curve drifting.

Two extra things get measured that a plain importance ranking would miss.

Delay. Each IC is computed twice: on the return from this bar's close
(skip=0), and on the return starting from the next bar (skip=1). Only the
second one is something you could actually trade. If the IC collapses between
the two, the feature was picking up bid-ask bounce.

Horizon. Predicting the next hour and predicting the next day are different
jobs. The scorecard shows each feature at several horizons, so you can see
whether it's useful for timing entries or for holding a view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFAULT_HORIZONS = (1, 3, 6, 12, 24)
#: Features that are context rather than signal; kept out of the ranking.
NON_SIGNAL = ("kind", "structure_id", "hour")


def cross_sectional_ic(feature: pd.Series, forward: pd.Series, min_names: int = 8) -> pd.Series:
    """Rank correlation between a feature and the next move, one value per bar.

    Bars with too few live structures are skipped rather than contributing a
    noisy correlation over three names.

    Spearman's rho is just Pearson's r on ranks, so I rank within each bar and
    do the correlation with grouped arithmetic. Calling scipy.stats.spearmanr
    bar by bar was about 40x slower, which adds up across 16 features, 5
    horizons, 2 skips and 900 bars.
    """
    frame = pd.concat([feature.rename("x"), forward.rename("y")], axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float)

    bar = frame.index.get_level_values("timestamp")
    grouped = frame.groupby(bar)
    ranks = grouped[["x", "y"]].rank()
    centred = ranks - ranks.groupby(bar).transform("mean")

    covariance = (centred["x"] * centred["y"]).groupby(bar).sum()
    scale = np.sqrt(
        (centred["x"] ** 2).groupby(bar).sum() * (centred["y"] ** 2).groupby(bar).sum()
    )
    ic = covariance / scale.replace(0.0, np.nan)

    # Drop bars with too few names, or with no spread in either ranking.
    enough = grouped.size() >= min_names
    varied = (grouped["x"].nunique() >= 3) & (grouped["y"].nunique() >= 3)
    return ic.where(enough & varied).dropna()


def feature_persistence(feature: pd.Series) -> float:
    """How much of a feature's variance is fixed per structure rather than moving
    over time.

    This one caught a real mistake. A feature that barely changes within a
    structure (days to the front leg's expiry, for example) ranks the board
    almost the same way every bar. Its IC can look big and very significant
    while really being one bet on the shape of the curve, repeated 900 times,
    rather than 900 independent observations. The t-stat has no way of knowing
    that, so this gets reported next to it.

    Close to 1.0 means static, close to 0.0 means driven by price moves.
    """
    clean = feature.dropna()
    if clean.empty:
        return float("nan")
    total = float(clean.var())
    if not total or not np.isfinite(total):
        return float("nan")
    between = float(clean.groupby(level="structure_id").mean().var())
    return float(np.clip(between / total, 0.0, 1.0))


def _hac_tstat(series: pd.Series, lags: int) -> float:
    """Newey-West t-stat for a mean when observations overlap.

    When the horizon is longer than one bar, neighbouring ICs share most of
    their forward window, so the naive standard error is far too small.
    """
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 10:
        return float("nan")
    demeaned = values - values.mean()
    variance = float(demeaned @ demeaned / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(demeaned[lag:] @ demeaned[:-lag] / n)
        variance += 2.0 * weight * cov
    if variance <= 0:
        return float("nan")
    return float(values.mean() / np.sqrt(variance / n))


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    """Mean IC, its spread, information ratio and HAC t-statistic."""
    clean = ic.dropna()
    if len(clean) < 10:
        return {"mean_ic": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "n_bars": len(clean)}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_ir": mean / std if std else np.nan,
        "t_stat": _hac_tstat(clean, lags=horizon),
        # How often the feature points the right way at all.
        "hit_rate": float((np.sign(clean) == np.sign(mean)).mean()),
        "n_bars": int(len(clean)),
    }


def feature_scorecard(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    skips: tuple[int, ...] = (0, 1),
    kind: str | None = None,
) -> pd.DataFrame:
    """Score every feature by IC, across horizons and skips.

    kind limits the board to one family ("fly", "spread", "dfly"), which is
    how the desk would look at it. A fly trader doesn't care how a signal
    ranks spreads.
    """
    frame = features if kind is None else features[features["kind"] == kind]
    columns = [
        c for c in frame.columns
        if c not in NON_SIGNAL and pd.api.types.is_numeric_dtype(frame[c])
    ]

    # Forward returns, computed once per (horizon, skip) and reused per feature.
    stacked_price = prices.stack(future_stack=True)
    stacked_price.index.names = ["timestamp", "structure_id"]
    forwards: dict[tuple[int, int], pd.Series] = {}
    for horizon in horizons:
        for skip in skips:
            shifted = prices.shift(-(horizon + skip)) - prices.shift(-skip)
            series = shifted.stack(future_stack=True)
            series.index.names = ["timestamp", "structure_id"]
            forwards[(horizon, skip)] = series.reindex(frame.index)

    rows = []
    for column in columns:
        feature = frame[column]
        persistence = feature_persistence(feature)
        for horizon in horizons:
            for skip in skips:
                summary = ic_summary(
                    cross_sectional_ic(feature, forwards[(horizon, skip)]), horizon
                )
                summary.update(
                    feature=column, horizon=horizon, skip=skip, persistence=persistence
                )
                rows.append(summary)

    table = pd.DataFrame(rows)
    return table[
        ["feature", "horizon", "skip", "mean_ic", "ic_std", "ic_ir",
         "t_stat", "hit_rate", "persistence", "n_bars"]
    ]


#: Above this share of fixed-per-structure variance, a feature's IC is treated
#: as one repeated cross-sectional bet rather than a time series of them.
STATIC_THRESHOLD = 0.90


def surviving_features(
    scorecard: pd.DataFrame, t_threshold: float = 2.0
) -> pd.DataFrame:
    """Features whose IC is significant, survives the delay, and isn't static.

    A feature has to rank the board significantly, keep doing so once you
    accept you can't trade the print that produced the signal, and actually
    change over time rather than encoding one fact about the curve's shape that
    happened to pay off over two months.
    """
    tradable = scorecard[scorecard["skip"] == 1].copy()
    immediate = scorecard[scorecard["skip"] == 0].set_index(["feature", "horizon"])["mean_ic"]
    tradable["ic_at_skip0"] = tradable.set_index(["feature", "horizon"]).index.map(immediate)
    tradable["ic_retained"] = tradable["mean_ic"] / tradable["ic_at_skip0"].replace(0, np.nan)
    tradable["static"] = tradable["persistence"] >= STATIC_THRESHOLD
    tradable["survives"] = (
        (tradable["t_stat"].abs() >= t_threshold)
        & (np.sign(tradable["mean_ic"]) == np.sign(tradable["ic_at_skip0"]))
        & ~tradable["static"]
    )
    return tradable.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)


def bucket_table(
    feature: pd.Series, forward: pd.Series, n_buckets: int = 5
) -> pd.DataFrame:
    """What happened next, by feature bucket. This is the version to show a trader.

    IC is the right summary number but not a useful thing to hand someone on
    the desk. This says: when the signal was in its bottom fifth, the structure
    moved this much on average over the next N hours, and went up this often.
    """
    frame = pd.concat([feature.rename("x"), forward.rename("y")], axis=1).dropna()
    if len(frame) < n_buckets * 20:
        return pd.DataFrame()
    frame["bucket"] = pd.qcut(frame["x"], n_buckets, labels=False, duplicates="drop")
    grouped = frame.groupby("bucket")
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "feature_min": grouped["x"].min(),
            "feature_max": grouped["x"].max(),
            "mean_move": grouped["y"].mean(),
            "median_move": grouped["y"].median(),
            "pct_up": grouped["y"].apply(lambda s: float((s > 0).mean())),
        }
    )
    out.index.name = "bucket"
    return out.round(4)


def monotonicity(bucket: pd.DataFrame) -> float:
    """Rank correlation between bucket and average move.

    A feature worth trading should be monotone: the strongest signal should
    come with the biggest move. One that only works in its extreme bucket is
    a threshold rule, not a factor, and much more likely to be noise.
    """
    if bucket.empty or len(bucket) < 3:
        return float("nan")
    return float(
        scipy_stats.spearmanr(bucket.index.to_numpy(), bucket["mean_move"].to_numpy()).statistic
    )


def economics(
    bucket: pd.DataFrame, gross_leg_exposure: float, cost_per_leg: float
) -> dict[str, float]:
    """Compare the size of the predicted move with the cost of trading it.

    This is the question that decides whether any of it is useful. A signal can
    be real, monotone and highly significant and still be worthless, because
    one unit of a fly trades four contracts and pays the spread on each.
    """
    if bucket.empty:
        return {"edge": np.nan, "round_trip_cost": np.nan, "edge_over_cost": np.nan}
    edge = float(max(abs(bucket["mean_move"].iloc[0]), abs(bucket["mean_move"].iloc[-1])))
    cost = gross_leg_exposure * cost_per_leg
    return {
        "edge": edge,
        "round_trip_cost": cost,
        "edge_over_cost": edge / cost if cost else np.nan,
    }


def study(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    kind: str | None = "fly",
    top_n: int = 3,
    gross_leg_exposure: float = 4.0,
    cost_per_leg: float = 0.015,
) -> dict[str, pd.DataFrame | str]:
    """Run the full feature study and summarise it in plain English."""
    scorecard = feature_scorecard(features, prices, horizons=horizons, kind=kind)
    survivors = surviving_features(scorecard)
    passed = survivors[survivors["survives"]]

    buckets: dict[str, pd.DataFrame] = {}
    frame = features if kind is None else features[features["kind"] == kind]
    for _, row in passed.head(top_n).iterrows():
        horizon, skip = int(row["horizon"]), int(row["skip"])
        shifted = prices.shift(-(horizon + skip)) - prices.shift(-skip)
        series = shifted.stack(future_stack=True)
        series.index.names = ["timestamp", "structure_id"]
        table = bucket_table(frame[row["feature"]], series.reindex(frame.index))
        if not table.empty:
            table.attrs["monotonicity"] = monotonicity(table)
            buckets[f"{row['feature']}_h{horizon}"] = table

    if passed.empty:
        finding = (
            f"No feature's cross-sectional information coefficient survives both a "
            f"|t| >= 2 test and the one-bar execution delay on {kind or 'all'} "
            f"structures. Signals that look predictive on same-bar returns are "
            f"measuring the bounce they were computed from."
        )
    else:
        best = passed.iloc[0]
        finding = (
            f"{len(passed)} of {len(survivors)} feature-horizon pairs survive both a "
            f"|t| >= 2 test and the one-bar execution delay on {kind or 'all'} "
            f"structures. The strongest is {best['feature']} at a {int(best['horizon'])}-bar "
            f"horizon (mean IC {best['mean_ic']:+.3f}, t = {best['t_stat']:.2f}, "
            f"retaining {best['ic_retained']:.0%} of its same-bar IC). "
            f"{int(survivors['static'].sum())} feature-horizon pairs were excluded as "
            f"near-static: their ranking barely changes between bars, so a high "
            f"t-statistic reflects one repeated bet on curve shape rather than "
            f"independent observations."
        )
    verdict = ""
    if buckets:
        top_name, top_bucket = next(iter(buckets.items()))
        econ = economics(top_bucket, gross_leg_exposure, cost_per_leg)
        verdict = (
            f"The strongest surviving signal is monotone across its range "
            f"(rank correlation {top_bucket.attrs['monotonicity']:+.2f} between "
            f"signal bucket and subsequent move), so it behaves like a factor rather "
            f"than a threshold rule. Its extreme bucket predicts a "
            f"${econ['edge']:.3f}/bbl move against a ${econ['round_trip_cost']:.3f}/bbl "
            f"round-trip cost on a {gross_leg_exposure:.0f}-leg structure -- "
            f"{econ['edge_over_cost']:.2f}x cost. "
            + (
                "The signal is real but too small to trade at this cost level."
                if econ["edge_over_cost"] < 1
                else "The signal clears costs and is worth pursuing."
            )
        )
    return {
        "scorecard": scorecard,
        "survivors": survivors,
        "buckets": buckets,
        "finding": finding,
        "verdict": verdict,
    }
