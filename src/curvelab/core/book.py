"""The curve book: aligned price panels plus cached structure prices.

Almost every later step (disambiguation, trade reconstruction, TCA, factor
attribution, signal research) keeps asking the same question: what was this
structure worth at this time? CurveBook holds the aligned panels, applies the
liquidity filter once, and caches structure prices so nothing gets priced twice.
"""

from __future__ import annotations

import pandas as pd

from curvelab.core.contracts import Contract
from curvelab.core.loader import ContractSeries
from curvelab.core.panel import (
    build_panel,
    liquidity_filter,
    price_structure,
    structure_bounds,
)
from curvelab.core.structure import Structure


class CurveBook:
    """Aligned OHLC panels over the liquid contract universe."""

    def __init__(
        self,
        curve: dict[Contract, ContractSeries],
        min_bars: int = 700,
        max_ffill_bars: int = 1,
    ) -> None:
        close = build_panel(curve, "close", max_ffill_bars)
        self.close, self.dropped = liquidity_filter(close, min_bars)
        keep = list(self.close.columns)
        self.open = build_panel(curve, "open", max_ffill_bars)[keep]
        self.high = build_panel(curve, "high", max_ffill_bars)[keep]
        self.low = build_panel(curve, "low", max_ffill_bars)[keep]
        self.volume = build_panel(curve, "volume", 0).reindex(columns=keep)
        self._day = self.close.index.normalize()

        self.daily_close = self.close.groupby(self._day).last()
        self.daily_open = self.open.groupby(self._day).first()
        self.daily_high = self.high.groupby(self._day).max()
        self.daily_low = self.low.groupby(self._day).min()

        self._price_cache: dict[Structure, pd.Series] = {}
        self._daily_cache: dict[Structure, pd.Series] = {}
        self._bounds_cache: dict[Structure, pd.DataFrame] = {}

    # ---- universe -------------------------------------------------------
    @property
    def contracts(self) -> list[Contract]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def covers(self, structure: Structure) -> bool:
        """True if every leg of the structure is in the liquid universe."""
        return bool(structure.legs) and all(c in self.close.columns for c, _ in structure.legs)

    # ---- pricing --------------------------------------------------------
    def price(self, structure: Structure) -> pd.Series:
        """Hourly mark for a structure (memoised)."""
        if structure not in self._price_cache:
            self._price_cache[structure] = price_structure(self.close, structure)
        return self._price_cache[structure]

    def daily_price(self, structure: Structure) -> pd.Series:
        """Daily settlement mark (last hourly close of each day)."""
        if structure not in self._daily_cache:
            self._daily_cache[structure] = self.price(structure).groupby(self._day).last()
        return self._daily_cache[structure]

    def daily_bounds(self, structure: Structure) -> pd.DataFrame:
        """Daily attainable price envelope, used as a fill-feasibility check."""
        if structure not in self._bounds_cache:
            self._bounds_cache[structure] = structure_bounds(
                self.daily_high, self.daily_low, structure
            )
        return self._bounds_cache[structure]

    def stats(self, structure: Structure) -> dict[str, float]:
        """Summary of a structure's observed price distribution."""
        series = self.price(structure).dropna()
        if series.empty:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        return {
            "n": int(series.size),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }
