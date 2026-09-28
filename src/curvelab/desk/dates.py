"""Fix trade dates that got their day and month swapped.

The journals were typed day-first (12/06/2026 meaning 12 June) and then read
back month-first, which turns that into 6 December. Nothing errors. The trade
just lands six months away from the market data it needs to be checked
against.

Luckily the damage gives itself away. A date with a day above 12 can't be
read as a month, so those came through untouched, and they tell you when the
desk was actually trading. The ambiguous ones are then settled against that:

1. Unambiguous (day > 12): keep as is, and use these to find the window.
2. Window rule: of the two possible readings, pick the one inside the
   trading window. This handles most of them.
3. Monotonic rule: journals are written in order, so if both readings are
   plausible, pick the one closest to the neighbouring rows in the same sheet.
4. Unresolved: if neither rule decides it, drop the date instead of guessing,
   and count it in the audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DateAudit:
    """What the resolver did, so it can be reported and double-checked."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_unambiguous: int
    n_window_rule: int
    n_monotonic_rule: int
    n_unresolved: int
    n_swapped: int

    def summary(self) -> str:
        total = self.n_unambiguous + self.n_window_rule + self.n_monotonic_rule + self.n_unresolved
        return (
            f"Activity window inferred from {self.n_unambiguous:,} unambiguous dates: "
            f"{self.window_start.date()} to {self.window_end.date()}. "
            f"Of {total:,} dated trades, {self.n_window_rule:,} ambiguous dates were "
            f"resolved by the window rule and {self.n_monotonic_rule:,} by sheet "
            f"monotonicity; {self.n_unresolved:,} were left unresolved. "
            f"{self.n_swapped:,} dates ({self.n_swapped / max(total, 1):.0%}) were "
            f"day/month-swapped back to their true value."
        )


def _swap(stamp: pd.Timestamp) -> pd.Timestamp | None:
    """Reinterpret a timestamp with its day and month exchanged."""
    if stamp is None or pd.isna(stamp) or stamp.day > 12:
        return None
    try:
        return pd.Timestamp(year=stamp.year, month=stamp.day, day=stamp.month)
    except ValueError:
        return None


def infer_window(
    dates: pd.Series, pad_days: int = 21, quantiles: tuple[float, float] = (0.02, 0.98)
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the desk's activity window from unambiguous (day > 12) dates.

    Uses trimmed quantiles instead of min/max so one mistyped year can't
    stretch the window and break the rule.
    """
    stamps = pd.to_datetime(dates.dropna())
    certain = stamps[stamps.dt.day > 12]
    if certain.empty:
        certain = stamps
    if certain.empty:
        raise ValueError("no dates available to infer an activity window")
    lo = certain.quantile(quantiles[0])
    hi = certain.quantile(quantiles[1])
    return lo - pd.Timedelta(days=pad_days), hi + pd.Timedelta(days=pad_days)


def resolve_dates(
    trades: pd.DataFrame,
    date_col: str = "date_raw",
    sheet_col: str = "sheet",
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, DateAudit]:
    """Add ``trade_date``, ``date_rule`` and ``date_swapped`` columns.

    Returns the augmented frame and a :class:`DateAudit` describing the
    inference, both of which feed the data-quality section of the report.
    """
    out = trades.copy()
    raw = pd.to_datetime(out[date_col], errors="coerce")
    lo, hi = window if window is not None else infer_window(raw)

    resolved: list[pd.Timestamp | None] = [None] * len(out)
    rule: list[str] = ["missing"] * len(out)
    swapped: list[bool] = [False] * len(out)

    def in_window(stamp: pd.Timestamp | None) -> bool:
        return stamp is not None and lo <= stamp <= hi

    # Passes 1 and 2: unambiguous dates, then the window rule.
    for i, stamp in enumerate(raw):
        if pd.isna(stamp):
            continue
        if stamp.day > 12:
            resolved[i], rule[i] = stamp, "unambiguous"
            continue
        alternative = _swap(stamp)
        primary_ok, alt_ok = in_window(stamp), in_window(alternative)
        if primary_ok and not alt_ok:
            resolved[i], rule[i] = stamp, "window"
        elif alt_ok and not primary_ok:
            resolved[i], rule[i], swapped[i] = alternative, "window", True
        # both or neither in window -> defer to the monotonicity pass

    # Pass 3: within each sheet, break ties using chronological neighbours.
    sheets = out[sheet_col] if sheet_col in out.columns else pd.Series(["_"] * len(out))
    for _, positions in out.groupby(sheets.values, sort=False).groups.items():
        idx = [out.index.get_loc(p) for p in positions]
        anchors = [(j, resolved[j]) for j in idx if resolved[j] is not None]
        if not anchors:
            continue
        for j in idx:
            if resolved[j] is not None or pd.isna(raw.iloc[j]):
                continue
            stamp = raw.iloc[j]
            alternative = _swap(stamp)
            if alternative is None:
                continue
            neighbours = [a for a in anchors if abs(a[0] - j) <= 12]
            if not neighbours:
                continue
            centre = pd.Timestamp(np.mean([a[1].value for a in neighbours]))
            if abs(stamp - centre) <= abs(alternative - centre):
                resolved[j], rule[j] = stamp, "monotonic"
            else:
                resolved[j], rule[j], swapped[j] = alternative, "monotonic", True

    for i, stamp in enumerate(raw):
        if resolved[i] is None and not pd.isna(stamp):
            rule[i] = "unresolved"

    out["trade_date"] = pd.to_datetime(pd.Series(resolved, index=out.index))
    out["date_rule"] = rule
    out["date_swapped"] = swapped

    audit = DateAudit(
        window_start=lo,
        window_end=hi,
        n_unambiguous=rule.count("unambiguous"),
        n_window_rule=rule.count("window"),
        n_monotonic_rule=rule.count("monotonic"),
        n_unresolved=rule.count("unresolved"),
        n_swapped=int(sum(swapped)),
    )
    return out, audit
