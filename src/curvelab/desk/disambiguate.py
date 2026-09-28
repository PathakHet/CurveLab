"""Use market prices to pick between readings of an ambiguous strategy name.

Some names don't pin down a structure on their own. The usual case is a tenor
with no type: "dec26 3mo" could be the Dec26/Mar27 spread or the Dec26 3-month
fly, and people on the desk traded both.

The text can't tell them apart, but the price can, because they trade at very
different levels. In the real data the Dec26 3mo spread sat around $3.45 and
the fly around $0.28, so an entry of 2.19 is almost certainly the spread.

Each candidate gets checked against the prices the trader wrote down:

- Feasible: is the price inside what the structure could have traded at on
  that day? If not, that reading is out.
- Plausible: how many standard deviations is the price from where the
  structure usually trades? Closest wins.

If the candidates still can't be told apart, the trade is left unresolved
rather than handed to whichever reading is more common.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from curvelab.core.book import CurveBook
from curvelab.core.structure import Structure
from curvelab.desk.nameparser import ParseResult

#: A candidate must win by at least this much mean-|z| to be chosen.
SEPARATION_MARGIN = 0.5


@dataclass
class Resolution:
    structure: Structure | None
    method: str            # unique | price_feasible | price_plausible | unresolved
    margin: float = float("nan")
    n_candidates: int = 1


def _reported_prices(trade: pd.Series) -> list[float]:
    values = [trade.get("entry_price"), trade.get("exit_price")]
    return [float(v) for v in values if v is not None and not pd.isna(v)]


def _plausibility(book: CurveBook, structure: Structure, prices: list[float]) -> float:
    """Average |z-score| of the reported prices against the structure's usual
    range. Lower means a better fit; inf if the structure can't be priced."""
    stats = book.stats(structure)
    if not stats["n"] or not np.isfinite(stats["std"]) or stats["std"] == 0:
        return float("inf")
    return float(np.mean([abs(p - stats["mean"]) / stats["std"] for p in prices]))


def _feasible(book: CurveBook, structure: Structure, prices: list[float],
              date: pd.Timestamp | None) -> bool:
    """Could these prices have been traded, given the day's attainable range?"""
    if date is None or pd.isna(date):
        stats = book.stats(structure)
        if not stats["n"]:
            return False
        pad = 0.05 * (stats["max"] - stats["min"] + 1e-9)
        return all(stats["min"] - pad <= p <= stats["max"] + pad for p in prices)
    bounds = book.daily_bounds(structure)
    day = pd.Timestamp(date).normalize()
    if day not in bounds.index:
        return False
    lower, upper = bounds.loc[day, "lower"], bounds.loc[day, "upper"]
    if not np.isfinite(lower) or not np.isfinite(upper):
        return False
    pad = 0.02 * (abs(upper - lower) + 1e-9)
    return all(lower - pad <= p <= upper + pad for p in prices)


def resolve(
    parse: ParseResult, trade: pd.Series, book: CurveBook
) -> Resolution:
    """Pick one structure from a parse result's candidates."""
    candidates = [c for c in parse.candidates if book.covers(c)]
    if not candidates:
        return Resolution(parse.structure, "uncovered", n_candidates=len(parse.candidates))
    if len(candidates) == 1:
        return Resolution(candidates[0], "unique", n_candidates=1)

    prices = _reported_prices(trade)
    date = trade.get("trade_date")
    if not prices:
        return Resolution(None, "unresolved", n_candidates=len(candidates))

    feasible = [c for c in candidates if _feasible(book, c, prices, date)]
    if len(feasible) == 1:
        return Resolution(feasible[0], "price_feasible", n_candidates=len(candidates))

    pool = feasible or candidates
    scored = sorted(((_plausibility(book, c, prices), c) for c in pool), key=lambda t: t[0])
    if not np.isfinite(scored[0][0]):
        return Resolution(None, "unresolved", n_candidates=len(candidates))
    margin = scored[1][0] - scored[0][0] if len(scored) > 1 else float("inf")
    if margin < SEPARATION_MARGIN:
        return Resolution(None, "unresolved", margin=margin, n_candidates=len(candidates))
    return Resolution(scored[0][1], "price_plausible", margin=margin, n_candidates=len(candidates))


def attach_structures(
    trades: pd.DataFrame, parses: list[ParseResult], book: CurveBook
) -> pd.DataFrame:
    """Add resolved ``structure`` columns to the trade table."""
    out = trades.copy()
    structures: list[Structure | None] = []
    methods: list[str] = []
    for (_, trade), parse in zip(out.iterrows(), parses):
        if not parse.ok:
            structures.append(None)
            methods.append("unparsed")
            continue
        resolution = resolve(parse, trade, book)
        structures.append(resolution.structure)
        methods.append(resolution.method)
    out["structure"] = structures
    out["resolve_method"] = methods
    out["structure_id"] = [s.structure_id if s is not None else None for s in structures]
    out["structure_kind"] = [s.kind if s is not None else None for s in structures]
    out["structure_display"] = [s.display if s is not None else None for s in structures]
    out["gross_leg_exposure"] = [
        s.gross_leg_exposure if s is not None else np.nan for s in structures
    ]
    return out
