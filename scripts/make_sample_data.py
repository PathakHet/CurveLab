"""Generate the synthetic sample dataset committed to the repository.

The real inputs (a Trading Technologies market-data subscription and a live
desk's trade journals) can't be published. This script makes stand-ins with
the same shape, so anyone who clones the repo can run the whole pipeline.

Everything here is random. The curve is built from the same three-factor
structure the real one has (a big level factor, a smaller slope factor and a
small curvature factor) plus some noise per contract. That part matters: if
each contract were an independent random walk, every fly would look tradeable
and the research results would be nonsense.

The journal copies the kinds of mess found in the real one, since that's what
the ingest code is there to deal with: strategy names written lots of
different ways, day-first dates that a month-first parser will mangle, swapped
entry/exit prices, and rows with missing fields.

Run: python scripts/make_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curvelab.core.contracts import Contract  # noqa: E402

OUT = ROOT / "data" / "sample"
N_CONTRACTS = 12
N_BARS = 950
START = pd.Timestamp("2026-05-01 00:00:00")
SEED = 20260501


def build_curve(rng: np.random.Generator) -> pd.DataFrame:
    """Three-factor Brent curve: level, slope, curvature, plus idiosyncratic noise."""
    tenor = np.arange(N_CONTRACTS)
    level_loading = np.exp(-0.05 * tenor)
    slope_loading = np.linspace(1.0, -1.0, N_CONTRACTS)
    curve_loading = np.cos(np.linspace(0, np.pi, N_CONTRACTS))

    level = np.cumsum(rng.normal(0, 0.09, N_BARS))
    slope = np.cumsum(rng.normal(0, 0.012, N_BARS))
    curvature = np.cumsum(rng.normal(0, 0.004, N_BARS))

    base = 92.0 - 0.55 * tenor          # backwardated, as Brent was in the sample
    noise = rng.normal(0, 0.010, (N_BARS, N_CONTRACTS)).cumsum(axis=0) * 0.25

    closes = (
        base
        + np.outer(level, level_loading)
        + np.outer(slope, slope_loading)
        + np.outer(curvature, curve_loading)
        + noise
    )
    index = pd.date_range(START, periods=N_BARS, freq="h")
    contracts = [Contract(2026, 7).plus_months(i) for i in range(N_CONTRACTS)]
    return pd.DataFrame(closes, index=index, columns=[c.code for c in contracts])


def write_market_data(closes: pd.DataFrame, rng: np.random.Generator) -> None:
    """Write per-contract TT-style CSVs, reproducing both real schemas."""
    OUT.mkdir(parents=True, exist_ok=True)
    for position, code in enumerate(closes.columns):
        close = closes[code]
        spread = rng.uniform(0.01, 0.04, len(close))
        high = close + spread * rng.uniform(0.5, 1.5, len(close))
        low = close - spread * rng.uniform(0.5, 1.5, len(close))
        open_ = close.shift(1).fillna(close.iloc[0]) + rng.normal(0, 0.01, len(close))
        frame = pd.DataFrame(
            {
                "Timestamp (UTC)": close.index.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": open_.round(2).to_numpy(),
                "High": np.maximum.reduce([high, open_, close]).round(2),
                "Low": np.minimum.reduce([low, open_, close]).round(2),
                "Close": close.round(2).to_numpy(),
            }
        )
        # Back contracts thin out, as in the real export.
        if position >= 9:
            frame = frame.iloc[: rng.integers(20, 400)]
        if position < 5:
            frame[f"BRN {code}: Volume"] = rng.integers(20, 5000, len(frame))

        path = OUT / f"BRN {code}_60min.csv"
        # Some of the real exports had no header row, so copy that too.
        if position in (2, 5):
            frame.to_csv(path, index=False, header=False)
        else:
            frame.to_csv(path, index=False)


NAME_TEMPLATES = (
    "{front} 1mo fly", "{front}_1mo_fly", "{front} 1mo butterfly",
    "BRN {front} 1mo fly(long)", "{front} 2mo fly", "{front} 2mo fly short",
    "{m0}{m1}{m2}", "BRN {front} - 2*{n1} + {n2} [{front} 1mofly]",
    "({front} - 2*{n1} + {n2}) ( {front} 1month fly)", "{front} 3mo",
    "{front} 3mo fly vs 2*{n3} 3mo fly", "{front} 6mo fly",
)


def write_journal(closes: pd.DataFrame, rng: np.random.Generator) -> None:
    """Write a synthetic multi-trader journal with the real one's defects."""
    contracts = [Contract.parse(c) for c in closes.columns]
    day_index = pd.Series(closes.index.normalize().unique())
    traders = [f"Trader {i + 1}" for i in range(12)]

    with pd.ExcelWriter(OUT / "Sample Trade Journal.xlsx", engine="openpyxl") as writer:
        for trader in traders:
            rows: list[list] = []
            header_offset = int(rng.integers(0, 3))
            for _ in range(header_offset):
                rows.append([None] * 8)
            rows.append(["Strategy Name", "Average Entry Price ", "Average Exit Price ",
                         "Stop loss ", "No. of Lots", "Net Pnl ", "Close Date", "Remarks"])

            for day in rng.choice(day_index, size=int(rng.integers(12, 26)), replace=False):
                day = pd.Timestamp(day)
                # Day-header row: the date, day P&L, and nothing else.
                rows.append([day, None, None, None, None, int(rng.integers(-500, 900)),
                             None, None])
                for _ in range(int(rng.integers(1, 6))):
                    i = int(rng.integers(0, len(contracts) - 4))
                    front = contracts[i]
                    template = str(rng.choice(NAME_TEMPLATES))
                    name = template.format(
                        front=front.code, n1=contracts[i + 1].code, n2=contracts[i + 2].code,
                        n3=contracts[i + 3].code,
                        m0=front.code[:3], m1=contracts[i + 1].code[:3],
                        m2=contracts[i + 2].code[:3],
                    )
                    entry = round(float(rng.normal(0, 0.25)), 2)
                    exit_ = round(entry + float(rng.normal(0, 0.06)), 2)
                    lots = int(rng.choice([-10, -5, -3, 2, 3, 5, 10]))
                    pnl = round((exit_ - entry) * lots * 1000, 0)
                    draw = rng.random()
                    if draw < 0.08:            # transposed entry/exit
                        pnl = -pnl
                    elif draw < 0.12:          # unexplainable row
                        pnl = round(pnl + rng.normal(0, 400), 0)
                    stop = round(entry - np.sign(lots) * abs(rng.normal(0.05, 0.02)), 2)
                    # Day-first date, which a month-first parser will mangle.
                    written = (day.strftime("%d/%m/%Y") if rng.random() < 0.45 else day)
                    rows.append([name, entry, exit_, stop, lots, pnl, written, None])

            width = max(len(r) for r in rows)
            frame = pd.DataFrame([r + [None] * (width - len(r)) for r in rows])
            frame.to_excel(writer, sheet_name=trader[:31], index=False, header=False)


def main() -> None:
    rng = np.random.default_rng(SEED)
    closes = build_curve(rng)
    write_market_data(closes, rng)
    write_journal(closes, rng)
    files = sorted(p.name for p in OUT.iterdir())
    print(f"wrote {len(files)} files to {OUT.relative_to(ROOT)}:")
    for name in files:
        print(f"  {name}")


if __name__ == "__main__":
    main()
