"""Load hourly futures exports from Trading Technologies (TT).

The exports come in two layouts:

    full:  Timestamp (UTC), Open, High, Low, Close, <sym>: Volume, <sym>: Bollinger ...
    bare:  Timestamp (UTC), Open, High, Low, Close

Some files were also exported with no header row at all. A plain read_csv
would quietly eat the first bar as column names, so I sniff the first line to
work out which layout I'm looking at.

The indicator columns (Bollinger, ATR, VWAP and so on) are overlays the TT
terminal draws, not market data, so they get dropped. Anything I need I can
compute from OHLC myself, without guessing at TT's settings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from curvelab.core.contracts import Contract

OHLC = ("open", "high", "low", "close")
_BARE_COLUMNS = ["timestamp", "open", "high", "low", "close"]
_FULL_COLUMNS = _BARE_COLUMNS + ["volume"]

#: "BRN Oct26_60min.csv" -> Oct26
_FILENAME_RE = re.compile(r"([A-Za-z]{3}\d{2})", re.IGNORECASE)


@dataclass
class ContractSeries:
    """One contract's bar series plus the provenance needed to audit it."""

    contract: Contract
    bars: pd.DataFrame          # DatetimeIndex x [open, high, low, close, volume]
    source: str
    had_header: bool
    has_volume: bool

    @property
    def n_bars(self) -> int:
        return len(self.bars)


def contract_from_filename(path: Path | str) -> Contract:
    """Extract the delivery month from a TT export filename."""
    match = _FILENAME_RE.search(Path(path).stem)
    if not match:
        raise ValueError(f"cannot infer contract from filename {path!r}")
    return Contract.parse(match.group(1))


def _has_header(path: Path) -> bool:
    with open(path, encoding="utf-8-sig") as handle:
        return handle.readline().lstrip().lower().startswith("timestamp")


def load_contract_csv(path: Path | str) -> ContractSeries:
    """Read one TT export, detecting its schema and header presence."""
    path = Path(path)
    contract = contract_from_filename(path)
    had_header = _has_header(path)

    if had_header:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        rename = {c: c.strip().lower() for c in frame.columns[:5]}
        volume_col = next(
            (c for c in frame.columns if str(c).strip().lower().endswith("volume")), None
        )
        if volume_col is not None:
            rename[volume_col] = "volume"
        frame = frame.rename(columns=rename)
    else:
        # Headerless export: positional columns, width tells us the schema.
        probe = pd.read_csv(path, header=None, nrows=1, encoding="utf-8-sig")
        names = _FULL_COLUMNS if probe.shape[1] > 5 else _BARE_COLUMNS
        names = names + [f"_x{i}" for i in range(probe.shape[1] - len(names))]
        frame = pd.read_csv(path, header=None, names=names, encoding="utf-8-sig")

    keep = ["timestamp", *OHLC] + (["volume"] if "volume" in frame.columns else [])
    frame = frame.loc[:, keep].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in OHLC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["close"])

    return ContractSeries(
        contract=contract,
        bars=frame[["open", "high", "low", "close", "volume"]],
        source=path.name,
        had_header=had_header,
        has_volume=bool(frame["volume"].notna().any()),
    )


def load_curve(data_dir: Path | str, pattern: str = "*.csv") -> dict[Contract, ContractSeries]:
    """Load every contract export in a directory, keyed by delivery month."""
    series: dict[Contract, ContractSeries] = {}
    for path in sorted(Path(data_dir).glob(pattern)):
        try:
            loaded = load_contract_csv(path)
        except (ValueError, KeyError):
            continue
        series[loaded.contract] = loaded
    return dict(sorted(series.items()))


def audit_frame(curve: dict[Contract, ContractSeries]) -> pd.DataFrame:
    """Per-contract coverage table for the data-quality section of the report."""
    rows = []
    for contract, series in curve.items():
        bars = series.bars
        rows.append(
            {
                "contract": contract.code,
                "delivery": f"{contract.year}-{contract.month:02d}",
                "bars": len(bars),
                "start": bars.index.min(),
                "end": bars.index.max(),
                "close_non_null": float(bars["close"].notna().mean()) if len(bars) else 0.0,
                "has_volume": series.has_volume,
                "header_present": series.had_header,
                "source": series.source,
            }
        )
    return pd.DataFrame(rows)
