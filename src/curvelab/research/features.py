"""Per-structure features and forward-return labels.

Every feature only uses data available at or before its own bar's close. The
labels (forward returns) never get joined into the feature matrix:
build_dataset keeps them in separate frames and only lines them up when
fitting, so it isn't possible to leak them by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curvelab.core.book import CurveBook
from curvelab.core.structure import Structure

FEATURE_COLUMNS = (
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "zscore", "vol", "range_pos", "momentum", "carry",
    "vol_ratio", "hour", "session", "days_to_front", "xs_zscore", "xs_rank",
)

#: Hours (UTC) that split the Brent session into Asia, Europe and US.
SESSION_BOUNDS = (7, 13)


def structure_features(
    price: pd.Series, zscore_window: int = 48, vol_window: int = 24
) -> pd.DataFrame:
    """Technical features for one structure's price series.

    Structure prices are differences and can cross zero, so everything is built
    from absolute changes. A percentage return on a fly passing through zero
    doesn't mean anything.
    """
    out = pd.DataFrame(index=price.index)
    for horizon in (1, 3, 6, 12, 24):
        out[f"ret_{horizon}"] = price.diff(horizon)

    rolling = price.rolling(zscore_window, min_periods=zscore_window // 2)
    mean, std = rolling.mean(), rolling.std()
    out["zscore"] = (price - mean) / std.replace(0.0, np.nan)

    out["vol"] = price.diff().rolling(vol_window, min_periods=vol_window // 2).std()

    high = price.rolling(vol_window, min_periods=2).max()
    low = price.rolling(vol_window, min_periods=2).min()
    out["range_pos"] = (price - low) / (high - low).replace(0.0, np.nan)

    out["momentum"] = price.diff(12) / out["vol"].replace(0.0, np.nan)

    # Carry: how far the structure sits from zero in its own volatility units.
    # For a fly this is the "richness" a trader eyeballs off the board.
    out["carry"] = price / std.replace(0.0, np.nan)

    # Volatility regime: current realised vol against its own recent norm.
    # Mean reversion behaves differently in a quiet curve than a moving one.
    out["vol_ratio"] = out["vol"] / out["vol"].rolling(
        zscore_window * 2, min_periods=vol_window
    ).mean().replace(0.0, np.nan)

    # Time of day. Brent liquidity is not uniform across the 23-hour session,
    # and a bar's information content depends on which desks were awake.
    hour = pd.Series(price.index.hour, index=price.index, dtype=float)
    out["hour"] = hour
    out["session"] = np.select(
        [hour < SESSION_BOUNDS[0], hour < SESSION_BOUNDS[1]], [0.0, 1.0], default=2.0
    )
    return out


def add_structure_context(
    features: pd.DataFrame, structure: Structure, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Features about the structure itself rather than its price history."""
    out = features.copy()
    front = structure.front
    if front is not None:
        # Days until the front leg's delivery month begins. Curve structures
        # behave differently as the front leg approaches expiry and rolls.
        expiry = pd.Timestamp(year=front.year, month=front.month, day=1)
        out["days_to_front"] = (expiry - index.normalize()).days.astype(float)
    else:
        out["days_to_front"] = np.nan
    return out


def add_cross_sectional(features: pd.DataFrame) -> pd.DataFrame:
    """Rank each structure against everything else on the board at the same time.

    A z-score of -2 doesn't say much on its own. If every structure is at -2,
    the whole curve has moved and there's no relative-value trade. What a curve
    trader acts on is being cheap compared with the rest of the board.
    """
    out = features.copy()
    grouped = out.groupby(level="timestamp")["zscore"]
    out["xs_zscore"] = (out["zscore"] - grouped.transform("mean")) / grouped.transform(
        "std"
    ).replace(0.0, np.nan)
    out["xs_rank"] = grouped.rank(pct=True)
    return out


def forward_return(price: pd.Series, horizon: int, skip: int = 0) -> pd.Series:
    """Price change over horizon bars, starting skip bars from now.

    skip=0 is the return from this bar's close, which you'd only get if you
    could trade at the same price that produced the signal. skip=1 starts from
    the next bar's close, which is what a trader could actually have got.
    Comparing the two is how I tell a forecast apart from bid-ask bounce, so
    skip is a proper parameter rather than something bolted on afterwards.
    """
    return price.shift(-(horizon + skip)) - price.shift(-skip)


def build_dataset(
    book: CurveBook,
    structures: list[Structure],
    horizon: int = 6,
    zscore_window: int = 48,
    vol_window: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the price panel, the feature matrix and the labels.

    Returns (prices, features, labels). prices is wide (time x structure);
    features and labels are long format, indexed by (timestamp, structure_id),
    which is the shape a cross-sectional model wants and keeps X and y apart.
    """
    prices, feature_frames, label_frames = {}, [], []
    for structure in structures:
        price = book.price(structure).rename(structure.structure_id)
        if price.notna().sum() < zscore_window * 2:
            continue
        prices[structure.structure_id] = price

        features = structure_features(price, zscore_window, vol_window)
        features = add_structure_context(features, structure, price.index)
        features["structure_id"] = structure.structure_id
        features["kind"] = structure.kind
        feature_frames.append(features.reset_index().rename(columns={"index": "timestamp"}))

        label = forward_return(price, horizon).rename("fwd_return")
        label_frame = label.reset_index()
        label_frame.columns = ["timestamp", "fwd_return"]
        label_frame["structure_id"] = structure.structure_id
        label_frames.append(label_frame)

    price_panel = pd.DataFrame(prices)
    features = pd.concat(feature_frames, ignore_index=True).set_index(
        ["timestamp", "structure_id"]
    )
    features = add_cross_sectional(features)
    labels = pd.concat(label_frames, ignore_index=True).set_index(
        ["timestamp", "structure_id"]
    )
    return price_panel, features, labels
