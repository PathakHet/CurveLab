"""Replace trader names with codes.

The real journals belong to a live desk: people's names, positions and P&L.
None of that should leave my machine, and none of it is needed, since every
question here is about how results are spread across traders, not about who
anyone is.

Each trader gets a code (T01, T02, ...). The order comes from a salted hash of
the name, so codes stay the same between runs but say nothing about the names
behind them. The lookup table is saved outside the repo; everything that gets
published only ever shows codes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def build_mapping(traders: pd.Series, salt: str = "") -> dict[str, str]:
    """Map each name to a Tnn code, ordered by a salted hash.

    Sorting alphabetically would let you guess names from codes. Sorting by
    hash doesn't, and it's still stable from run to run.
    """
    unique = sorted({str(t).strip() for t in traders.dropna() if str(t).strip()})
    ordered = sorted(
        unique, key=lambda name: hashlib.sha256((salt + name).encode()).hexdigest()
    )
    width = max(2, len(str(len(ordered))))
    return {name: f"T{i + 1:0{width}d}" for i, name in enumerate(ordered)}


def apply_mapping(
    trades: pd.DataFrame, mapping: dict[str, str], column: str = "trader"
) -> pd.DataFrame:
    """Replace trader names with codes, dropping the identifying source columns."""
    out = trades.copy()
    out[column] = out[column].map(lambda name: mapping.get(str(name).strip(), "T??"))
    for identifying in ("sheet", "source_file"):
        if identifying in out.columns:
            out = out.drop(columns=identifying)
    return out


def save_mapping(mapping: dict[str, str], path: Path | str) -> None:
    """Write the lookup table outside the repository."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def anonymize(
    trades: pd.DataFrame, salt: str = "", mapping_path: Path | str | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pseudonymise a trade table, optionally persisting the mapping."""
    mapping = build_mapping(trades["trader"], salt=salt)
    if mapping_path is not None:
        save_mapping(mapping, mapping_path)
    return apply_mapping(trades, mapping), mapping
