"""Read the trade journals into a single tidy table of trades.

The journals are hand-kept Excel sheets, one per trader. They share the same
core columns ("Strategy Name", "Average Entry Price", "Average Exit Price",
"Stop loss", "No. of Lots", "Net Pnl") and not much else. The header might be
on row 0, 1 or 2. Some sheets have extra Remarks/Target/Date columns, some
have #REF! labels left over from broken formulas, and most don't have a date
column at all. Instead, a row with just a date (and the day's P&L) sits above
that day's trades.

So the reader finds the header by looking for it, matches column names
loosely, and carries each day-header date down onto the trades below it. Rows
that get rejected are counted, and ingest_summary reports them, so nothing
disappears without a trace.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Fuzzy header label -> canonical field name.
_COLUMN_ALIASES: dict[str, str] = {
    "strategy name": "strategy_raw",
    "average entry price": "entry_price",
    "avg. entry price": "entry_price",
    "avg entry price": "entry_price",
    "average exit price": "exit_price",
    "avg. exit price": "exit_price",
    "avg exit price": "exit_price",
    "stop loss": "stop_loss",
    "stoploss": "stop_loss",
    "no. of lots": "lots",
    "no of lots": "lots",
    "number of lots": "lots",
    "net pnl": "net_pnl",
    "pnl": "net_pnl",
    "target": "target",
    "date": "date_col",
    "open date": "open_date",
    "entry date": "open_date",
    "entry order filled date & time": "open_date",
    "close date": "close_date",
    "exit date": "close_date",
    "exit order filled date & time": "close_date",
    "remarks": "remarks",
    "remark": "remarks",
    "notes:": "remarks",
    "notes": "remarks",
    "details": "remarks",
    "reasoning": "remarks",
    "observation..": "remarks",
}

#: Canonical numeric fields, coerced with :func:`_to_float`.
NUMERIC_FIELDS = ("entry_price", "exit_price", "stop_loss", "lots", "net_pnl", "target")

TRADE_COLUMNS = [
    "trader", "source_file", "sheet", "row_index",
    "strategy_raw", "entry_price", "exit_price", "stop_loss",
    "lots", "net_pnl", "target", "date_raw", "remarks",
]


@dataclass
class IngestStats:
    """Per-sheet accounting so nothing disappears without a trace."""

    sheet: str
    rows_scanned: int = 0
    trades: int = 0
    date_header_rows: int = 0
    rejected_no_strategy: int = 0
    rejected_no_numbers: int = 0
    header_row: int | None = None


def _norm_label(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text.rstrip(":").strip() if text.endswith(":") else text


def _to_float(value: object) -> float:
    """Coerce a journal cell to a float, tolerating text like ``"-0.02 "``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "-", "na", "n/a", "#ref!", "#value!"}:
        return np.nan
    text = re.sub(r"[^\d.\-+eE]", "", text)
    try:
        return float(text)
    except ValueError:
        return np.nan


def _find_header_row(frame: pd.DataFrame, max_scan: int = 12) -> int | None:
    for i in range(min(max_scan, len(frame))):
        labels = {_norm_label(v) for v in frame.iloc[i].tolist()}
        if "strategy name" in labels:
            return i
    return None


def _resolve_columns(header: pd.Series) -> dict[int, str]:
    """Map column positions to canonical field names via fuzzy label match."""
    mapping: dict[int, str] = {}
    for pos, raw in enumerate(header.tolist()):
        label = _norm_label(raw)
        field = _COLUMN_ALIASES.get(label)
        if field is None:
            # tolerate trailing punctuation / stray whitespace variants
            for alias, candidate in _COLUMN_ALIASES.items():
                if label and (label.startswith(alias) or alias.startswith(label)) and len(label) > 3:
                    field = candidate
                    break
        if field and field not in mapping.values():
            mapping[pos] = field
    return mapping


def _as_timestamp(value: object) -> pd.Timestamp | None:
    """Coerce a date-like cell to a Timestamp.

    openpyxl gives back datetime.datetime on some sheets and pd.Timestamp on
    others. I originally only checked for Timestamp and lost the dates on most
    of the journals without noticing.
    """
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, _dt.datetime):
        return pd.Timestamp(value)
    if isinstance(value, _dt.date):
        return pd.Timestamp(value)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text):
            try:
                return pd.Timestamp(text)
            except (ValueError, TypeError):
                return None
    return None


def read_sheet(
    frame: pd.DataFrame, trader: str, source_file: str, sheet: str
) -> tuple[pd.DataFrame, IngestStats]:
    """Extract trades from one raw sheet."""
    stats = IngestStats(sheet=sheet, rows_scanned=len(frame))
    header_row = _find_header_row(frame)
    if header_row is None:
        return pd.DataFrame(columns=TRADE_COLUMNS), stats
    stats.header_row = header_row

    columns = _resolve_columns(frame.iloc[header_row])
    strategy_pos = next((p for p, f in columns.items() if f == "strategy_raw"), 0)

    records: list[dict] = []
    current_date: pd.Timestamp | None = None

    for idx in range(header_row + 1, len(frame)):
        row = frame.iloc[idx]
        values = row.tolist()

        strategy_cell = values[strategy_pos] if strategy_pos < len(values) else None
        as_date = _as_timestamp(strategy_cell)
        strategy_text = (
            "" if as_date is not None or strategy_cell is None
            else str(strategy_cell).strip()
        )
        # An empty cell turns into the string "nan", which would otherwise look
        # like a strategy name and fail to parse hundreds of times later on.
        if strategy_text.lower() in {"nan", "nat", "none", "#ref!", "-"}:
            strategy_text = ""

        # Day-header row: a date on its own in the strategy column applies to
        # every trade below it until the next one.
        if as_date is not None:
            current_date = as_date
            stats.date_header_rows += 1
            continue

        record: dict = {
            "trader": trader,
            "source_file": source_file,
            "sheet": sheet,
            "row_index": idx,
            "strategy_raw": strategy_text,
            "remarks": "",
            "target": np.nan,
        }
        explicit_date: pd.Timestamp | None = None
        for pos, field in columns.items():
            if pos >= len(values):
                continue
            value = values[pos]
            if field in NUMERIC_FIELDS:
                record[field] = _to_float(value)
            elif field in ("date_col", "open_date", "close_date"):
                stamp = _as_timestamp(value)
                if stamp is not None and explicit_date is None:
                    explicit_date = stamp
            elif field == "remarks":
                record["remarks"] = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()

        if not strategy_text or _norm_label(strategy_text) in _COLUMN_ALIASES:
            stats.rejected_no_strategy += 1
            continue
        numeric_present = sum(
            1 for f in ("entry_price", "exit_price", "net_pnl", "lots")
            if not np.isnan(record.get(f, np.nan))
        )
        if numeric_present < 2:
            stats.rejected_no_numbers += 1
            continue

        record["date_raw"] = explicit_date if explicit_date is not None else current_date
        records.append(record)
        stats.trades += 1

    trades = pd.DataFrame(records)
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), stats
    for column in TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = np.nan
    return trades[TRADE_COLUMNS], stats


def load_journals(paths: list[Path] | list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every sheet of every workbook into one trade table.

    Returns ``(trades, stats)`` where ``stats`` is the per-sheet ingest audit.
    """
    all_trades: list[pd.DataFrame] = []
    all_stats: list[IngestStats] = []
    for path in paths:
        path = Path(path)
        workbook = pd.ExcelFile(path)
        for sheet in workbook.sheet_names:
            raw = workbook.parse(sheet, header=None)
            trader = re.sub(r"\s+", " ", sheet).strip()
            trades, stats = read_sheet(raw, trader, path.name, sheet)
            if not trades.empty:
                all_trades.append(trades)
            all_stats.append(stats)

    trades = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    stats = pd.DataFrame([s.__dict__ for s in all_stats])
    return trades, stats


def ingest_summary(trades: pd.DataFrame, stats: pd.DataFrame) -> str:
    """One-paragraph accounting of what was read and what was rejected."""
    return (
        f"{len(trades):,} trades from {trades['trader'].nunique()} traders "
        f"across {len(stats)} sheets "
        f"({int(stats['rejected_no_strategy'].sum()):,} rows rejected with no strategy name, "
        f"{int(stats['rejected_no_numbers'].sum()):,} with too few numeric fields, "
        f"{int(stats['date_header_rows'].sum()):,} day-header rows consumed as date context)."
    )
