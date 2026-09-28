"""The incremental C++ engine against the vectorised NumPy approach.

I tried to be fair to NumPy here. The baseline isn't a slow Python loop, it's
the same recompute-the-whole-matrix approach the research code uses, running on
optimised BLAS. What it can't do is take advantage of the fact that when one
contract ticks, most of the matrix stays the same.

Run: python bench/bench_vs_numpy.py [market_data_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curvelab import _engine  # noqa: E402
from curvelab.core.book import CurveBook  # noqa: E402
from curvelab.core.loader import load_curve  # noqa: E402
from curvelab.research.universe import build_universe  # noqa: E402

WINDOW = 48


def numpy_replay(weights: np.ndarray, contracts: np.ndarray, prices: np.ndarray,
                 window: int) -> np.ndarray:
    """Vectorised baseline: rebuild prices and rolling z-scores from the matrix."""
    n_contracts = weights.shape[1]
    last = np.full(n_contracts, np.nan)
    n_ticks = len(contracts)
    priced = np.full((n_ticks, weights.shape[0]), np.nan)
    for t in range(n_ticks):
        last[contracts[t]] = prices[t]
        priced[t] = weights @ last
    # Rolling z-score over the whole matrix, the way pandas would do it.
    out = np.full_like(priced, np.nan)
    for t in range(window, n_ticks):
        block = priced[t - window : t + 1]
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = (priced[t] - mean) / std
    return out


def main() -> None:
    # Pass a directory of TT exports to benchmark on your own data.
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample"
    book = CurveBook(load_curve(data_dir))
    structures = build_universe(book)
    contracts_index = {c: i for i, c in enumerate(book.contracts)}

    legs = [
        [(contracts_index[c], w) for c, w in s.legs]
        for s in structures
        if book.covers(s)
    ]
    weights = np.zeros((len(legs), len(contracts_index)))
    for i, definition in enumerate(legs):
        for c, w in definition:
            weights[i, c] = w

    # Turn the hourly panel into a tick sequence: one update per contract per bar.
    panel = book.close
    ticks_c, ticks_p = [], []
    for _, row in panel.iterrows():
        for contract, value in row.items():
            if not np.isnan(value):
                ticks_c.append(contracts_index[contract])
                ticks_p.append(float(value))
    contracts_arr = np.array(ticks_c, dtype=np.uint32)
    prices_arr = np.array(ticks_p, dtype=np.float64)

    print(f"contracts   {len(contracts_index)}")
    print(f"structures  {len(legs)}")
    print(f"ticks       {len(contracts_arr):,} (from {len(panel):,} hourly bars)")
    print(f"window      {WINDOW}")

    engine = _engine.CurveEngine(len(contracts_index), legs, WINDOW)
    start = time.perf_counter()
    engine.replay(contracts_arr, prices_arr, False)
    cpp_seconds = time.perf_counter() - start

    start = time.perf_counter()
    numpy_replay(weights, contracts_arr, prices_arr, WINDOW)
    numpy_seconds = time.perf_counter() - start

    print(f"\nC++ incremental   {cpp_seconds * 1e3:9.2f} ms"
          f"   ({len(contracts_arr) / cpp_seconds / 1e6:6.2f} M updates/sec)")
    print(f"NumPy vectorised  {numpy_seconds * 1e3:9.2f} ms"
          f"   ({len(contracts_arr) / numpy_seconds / 1e6:6.2f} M updates/sec)")
    print(f"speed-up          {numpy_seconds / cpp_seconds:9.1f}x")
    print(f"mean fan-out      {engine.fan_out:9.2f} structures touched per tick "
          f"(of {engine.n_structures})")


if __name__ == "__main__":
    main()
