"""Aligned curve panels and structure pricing.

A panel is a wide timestamp x contract frame holding one price field. Every
structure is priced from a panel the same way, as a weighted sum of its legs
(price_t = sum_i w_i * leg_i_t), so spreads, flies and anything fancier all go
through one code path.

Two rules worth knowing about:

- No partial legs. If any leg is missing at a timestamp, the structure price
  is NaN. Summing whatever legs happen to be there would quietly turn a fly
  into a spread whenever one contract didn't print.
- Limited forward-fill. Gaps are filled for at most max_ffill_bars bars, so a
  contract that stops trading doesn't leave a flat, tradeable-looking price
  behind it for the rest of the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from curvelab.core.contracts import Contract
from curvelab.core.loader import ContractSeries
from curvelab.core.structure import Structure


def build_panel(
    curve: dict[Contract, ContractSeries],
    field: str = "close",
    max_ffill_bars: int = 1,
) -> pd.DataFrame:
    """Wide ``timestamp x contract`` panel of one price field."""
    columns = {
        contract: series.bars[field].rename(contract)
        for contract, series in curve.items()
    }
    if not columns:
        return pd.DataFrame()
    panel = pd.concat(columns.values(), axis=1)
    panel.columns = list(columns.keys())
    panel = panel.sort_index()
    if max_ffill_bars > 0:
        panel = panel.ffill(limit=max_ffill_bars)
    return panel.reindex(sorted(panel.columns), axis=1)


def liquidity_filter(
    panel: pd.DataFrame, min_bars: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into a liquid universe and an audit of what was dropped."""
    counts = panel.notna().sum()
    keep = counts[counts >= min_bars].index.tolist()
    dropped = pd.DataFrame(
        [
            {
                "contract": c.code,
                "bars": int(counts[c]),
                "reason": f"only {int(counts[c])} bars (< min_bars={min_bars})",
            }
            for c in panel.columns
            if c not in keep
        ]
    )
    return panel[keep], dropped


def price_structure(panel: pd.DataFrame, structure: Structure) -> pd.Series:
    """Price one structure from a panel, NaN wherever any leg is missing."""
    if structure.is_empty:
        return pd.Series(np.nan, index=panel.index)
    missing = [c for c, _ in structure.legs if c not in panel.columns]
    if missing:
        return pd.Series(np.nan, index=panel.index, name=structure.structure_id)

    legs = panel[[c for c, _ in structure.legs]]
    weights = np.array([w for _, w in structure.legs], dtype=float)
    values = legs.to_numpy(dtype=float)
    priced = values @ weights
    priced[np.isnan(values).any(axis=1)] = np.nan  # all-legs-or-nothing
    return pd.Series(priced, index=panel.index, name=structure.structure_id)


def price_structures(panel: pd.DataFrame, structures: list[Structure]) -> pd.DataFrame:
    """Price many structures at once, one column per structure."""
    if not structures:
        return pd.DataFrame(index=panel.index)
    priced = {s.structure_id: price_structure(panel, s) for s in structures}
    return pd.DataFrame(priced, index=panel.index)


def structure_bounds(
    high_panel: pd.DataFrame, low_panel: pd.DataFrame, structure: Structure
) -> pd.DataFrame:
    """Attainable price range for a structure over each bar.

    For a basket with mixed signs the extremes are attained by taking the high
    of every positively weighted leg against the low of every negatively
    weighted one (and vice versa)::

        upper = sum_{w>0} w*high + sum_{w<0} w*low
        lower = sum_{w>0} w*low  + sum_{w<0} w*high

    These are bounds rather than prices you could actually get, because the
    legs don't have to hit their extremes at the same moment. That makes them
    a loose envelope, which is what I want: a fill reported outside it simply
    can't have happened, so it's a good way to catch journal typos.
    """
    if structure.is_empty:
        return pd.DataFrame(index=high_panel.index, columns=["lower", "upper"], dtype=float)
    upper = pd.Series(0.0, index=high_panel.index)
    lower = pd.Series(0.0, index=high_panel.index)
    for contract, weight in structure.legs:
        if contract not in high_panel.columns or contract not in low_panel.columns:
            return pd.DataFrame(
                np.nan, index=high_panel.index, columns=["lower", "upper"]
            )
        hi, lo = high_panel[contract], low_panel[contract]
        if weight > 0:
            upper += weight * hi
            lower += weight * lo
        else:
            upper += weight * lo
            lower += weight * hi
    return pd.DataFrame({"lower": lower, "upper": upper})


def daily_ohlc(panel_o: pd.DataFrame, panel_h: pd.DataFrame, panel_l: pd.DataFrame,
               panel_c: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collapse hourly panels to daily, for matching against date-only trades."""
    day = panel_c.index.normalize()
    return {
        "open": panel_o.groupby(day).first(),
        "high": panel_h.groupby(day).max(),
        "low": panel_l.groupby(day).min(),
        "close": panel_c.groupby(day).last(),
    }
