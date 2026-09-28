"""Build the set of tradeable structures from the liquid contracts.

It's generated the way the desk thinks about the curve: every calendar spread,
fly and double fly on evenly spaced legs, at every spacing the data allows,
then filtered down to the ones with enough shared history to backtest.
"""

from __future__ import annotations

import pandas as pd

from curvelab.core.book import CurveBook
from curvelab.core.structure import Structure, dfly_from, fly_from, spread_from


def build_universe(
    book: CurveBook,
    spacings: tuple[int, ...] = (1, 2, 3),
    kinds: tuple[str, ...] = ("spread", "fly", "dfly"),
    min_bars: int = 500,
) -> list[Structure]:
    """Every spread/fly/dfly on the liquid curve with sufficient joint history."""
    builders = {"spread": spread_from, "fly": fly_from, "dfly": dfly_from}
    seen: dict[Structure, None] = {}
    for contract in book.contracts:
        for spacing in spacings:
            for kind in kinds:
                structure = builders[kind](contract, spacing)
                if not book.covers(structure) or structure in seen:
                    continue
                if int(book.price(structure).notna().sum()) >= min_bars:
                    seen[structure] = None
    return list(seen)


def catalogue(book: CurveBook, structures: list[Structure]) -> pd.DataFrame:
    """Descriptive table of the universe, for the report."""
    rows = []
    for structure in structures:
        series = book.price(structure).dropna()
        rows.append(
            {
                "structure_id": structure.structure_id,
                "kind": structure.kind,
                "display": structure.display,
                "front": structure.front.code,
                "spacing_months": structure.spacing,
                "n_legs": structure.n_legs,
                "gross_leg_exposure": structure.gross_leg_exposure,
                "bars": len(series),
                "mean": float(series.mean()) if len(series) else float("nan"),
                "std": float(series.std()) if len(series) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["kind", "front", "spacing_months"])
