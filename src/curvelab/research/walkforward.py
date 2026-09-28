"""Walk-forward validation with purging and an embargo, plus an ML ranker.

Forward-return labels overlap, so a naive train/test split leaks. A label at
time t isn't known until t + horizon, which means a test set starting at t + 1
partly depends on data the training set has already seen. The standard fix
(Lopez de Prado) is used here:

- Purge: drop training rows whose label window overlaps the test window.
- Embargo: also drop training rows just after the test window, since
  autocorrelation makes them near-copies of the test data.

Without both, a model that has learned nothing still scores well out of
sample, and the walk-forward result is just decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=2000, C=0.5),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=40, random_state=seed, n_jobs=-1
    ),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed
    ),
}


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def make_folds(
    timestamps: pd.DatetimeIndex,
    train_bars: int = 400,
    test_bars: int = 120,
    step_bars: int = 120,
    horizon: int = 6,
    embargo_bars: int = 6,
) -> list[Fold]:
    """Expanding-window folds with purge and embargo gaps built in."""
    unique = pd.DatetimeIndex(sorted(set(timestamps)))
    folds: list[Fold] = []
    start = train_bars
    while start + test_bars <= len(unique):
        # Purge: the training window stops `horizon` bars before the test window
        # so no training label can peek into it.
        train_end_idx = max(start - horizon, 1)
        test_slice = unique[start : start + test_bars]
        folds.append(
            Fold(
                train_start=unique[0],
                train_end=unique[train_end_idx - 1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
                n_train=train_end_idx,
                n_test=len(test_slice),
            )
        )
        start += step_bars
    return folds


def _select(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    stamps = frame.index.get_level_values("timestamp")
    return frame[(stamps >= start) & (stamps <= end)]


def walk_forward_ranker(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    feature_columns: tuple[str, ...] | None = None,
    horizon: int = 6,
    top_k: int = 5,
    embargo_bars: int = 6,
    seed: int = 42,
    **fold_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Fold], pd.DataFrame]:
    """Train a cross-sectional ranker walk-forward and produce out-of-sample signals.

    At each test bar the model scores every structure. The top k by predicted
    chance of going up are bought and the bottom k sold. It's a relative
    question (which structures look best right now?), not a call on where the
    curve is heading, which two months of data couldn't support anyway.
    """
    columns = list(feature_columns or [c for c in features.columns if c not in ("kind",)])
    dataset = features[columns].join(labels, how="inner").dropna()
    binary = (dataset["fwd_return"] > 0).astype(int)
    timestamps = pd.DatetimeIndex(dataset.index.get_level_values("timestamp").unique())

    folds = make_folds(timestamps, horizon=horizon, embargo_bars=embargo_bars, **fold_kwargs)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    conviction = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    importances = []

    for fold in folds:
        train = _select(dataset, fold.train_start, fold.train_end)
        # Embargo: also drop training rows that sit just after the test block.
        after = dataset.index.get_level_values("timestamp") > fold.test_end
        embargo_end = fold.test_end + (timestamps[1] - timestamps[0]) * embargo_bars
        keep_after = dataset[after & (dataset.index.get_level_values("timestamp") > embargo_end)]
        train = pd.concat([train, keep_after])
        test = _select(dataset, fold.test_start, fold.test_end)
        if len(train) < 200 or test.empty:
            continue

        y_train = binary.loc[train.index]
        if y_train.nunique() < 2:
            continue

        scaler = StandardScaler().fit(train[columns])
        model = MODELS[model_name](seed)
        model.fit(scaler.transform(train[columns]), y_train)
        scores = pd.Series(
            model.predict_proba(scaler.transform(test[columns]))[:, 1], index=test.index
        )

        wide = scores.unstack("structure_id")
        ranks = wide.rank(axis=1, ascending=False)
        count = wide.notna().sum(axis=1)
        long_mask = ranks.le(top_k, axis=0)
        short_mask = ranks.gt(count - top_k, axis=0)

        block = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
        block[long_mask] = 1.0
        block[short_mask] = -1.0
        aligned = block.reindex(index=signals.index, columns=signals.columns).fillna(0.0)
        window = (signals.index >= fold.test_start) & (signals.index <= fold.test_end)
        signals.loc[window] = aligned.loc[window]
        conviction.loc[window] = (
            (wide - 0.5).abs().reindex(index=signals.index, columns=signals.columns).loc[window]
        )

        # Permutation importance is measured out of sample and is comparable
        # across model families, unlike impurity gains or raw coefficients.
        sample = test.sample(min(len(test), 4000), random_state=seed)
        result = permutation_importance(
            model, scaler.transform(sample[columns]), binary.loc[sample.index],
            n_repeats=3, random_state=seed, n_jobs=-1,
        )
        importances.append(pd.Series(result.importances_mean, index=columns))

    importance = (
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).to_frame("importance")
        if importances
        else pd.DataFrame()
    )
    return signals, conviction, folds, importance
