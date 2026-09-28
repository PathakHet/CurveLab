"""Check the journal trades against each other and against the market.

Journal entries were typed by hand, quickly, during the session, so I don't
take them at face value. Before using them to say anything about skill I
check three things.

Does the P&L add up? For a futures structure,
pnl = (exit - entry) * lots * multiplier. Rather than assume the multiplier,
fit_multiplier works it out from the journals, and it comes out at 1,000, the
size of an ICE Brent contract in barrels. Most trades match to within a
dollar. Of the rest, a good chunk match once you flip the sign, which usually
means entry and exit got swapped. Those are labelled sign_flip rather than
silently fixed.

Could the fill have happened? The range a structure could trade in on a given
day follows from its legs' highs and lows (see core/panel.structure_bounds).
A fill outside that range is impossible, so it points to a mistyped price or
the wrong structure.

How good was the fill? Timestamps are date-only, so real implementation
shortfall can't be measured. Instead I look at where in the day's range the
fill landed, scaled so 1.0 is the best price of the day and 0.0 the worst.
Over enough trades it still separates people who work their orders from
people who cross the spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from curvelab.core.book import CurveBook

#: Reconciliation tolerance in dollars.
PNL_TOL = 1.0


@dataclass
class MultiplierFit:
    multiplier: float
    support: float          # share of trades consistent with it
    n_used: int

    def summary(self) -> str:
        return (
            f"Contract multiplier recovered from the journals: "
            f"{self.multiplier:,.0f} barrels/lot "
            f"({self.support:.1%} of {self.n_used:,} reconcilable trades agree)."
        )


def fit_multiplier(trades: pd.DataFrame, candidates: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)) -> MultiplierFit:
    """Recover the contract multiplier implied by the journals.

    Chooses the candidate maximising the share of trades whose reported P&L
    matches ``(exit - entry) * lots * multiplier`` to within :data:`PNL_TOL`,
    rather than assuming the contract specification.
    """
    usable = trades.dropna(subset=["entry_price", "exit_price", "lots", "net_pnl"])
    delta = (usable["exit_price"] - usable["entry_price"]) * usable["lots"]
    usable = usable[delta.abs() > 1e-9]
    delta = delta[delta.abs() > 1e-9]
    if usable.empty:
        return MultiplierFit(1000.0, float("nan"), 0)
    best, best_support = candidates[-1], -1.0
    for candidate in candidates:
        support = float(((usable["net_pnl"] - delta * candidate).abs() < PNL_TOL).mean())
        if support > best_support:
            best, best_support = candidate, support
    return MultiplierFit(best, best_support, int(len(usable)))


def reconcile(trades: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Recompute each trade's P&L and classify any disagreement.

    Adds ``pnl_recomputed``, ``pnl_residual`` and ``pnl_status``, where status is

    ``match``        reported P&L reproduces to within :data:`PNL_TOL`
    ``sign_flip``    reproduces only after negation (transposed entry/exit or lot sign)
    ``unexplained``  neither; the row is excluded from P&L-based inference
    ``incomplete``   not enough fields to check
    """
    out = trades.copy()
    complete = out[["entry_price", "exit_price", "lots", "net_pnl"]].notna().all(axis=1)
    delta = (out["exit_price"] - out["entry_price"]) * out["lots"]
    recomputed = delta * multiplier

    residual = out["net_pnl"] - recomputed
    flipped = out["net_pnl"] + recomputed

    status = pd.Series("incomplete", index=out.index, dtype=object)
    status[complete & (residual.abs() < PNL_TOL)] = "match"
    status[complete & (residual.abs() >= PNL_TOL) & (flipped.abs() < PNL_TOL)] = "sign_flip"
    status[complete & (residual.abs() >= PNL_TOL) & (flipped.abs() >= PNL_TOL)] = "unexplained"

    out["pnl_recomputed"] = recomputed
    out["pnl_residual"] = residual
    out["pnl_status"] = status
    # The P&L everything downstream uses: the trader's number if it checks
    # out, the sign-flipped number if that's the only way it checks out, and
    # NaN if it doesn't check out at all.
    trusted = pd.Series(np.nan, index=out.index, dtype=float)
    trusted[status == "match"] = out.loc[status == "match", "net_pnl"]
    trusted[status == "sign_flip"] = -out.loc[status == "sign_flip", "net_pnl"]
    out["pnl_trusted"] = trusted
    return out


def mark_to_market(trades: pd.DataFrame, book: CurveBook) -> pd.DataFrame:
    """Check fills against the market and score their placement in the day's range.

    Adds the daily attainable envelope for each trade's structure, a
    feasibility flag per fill, and an execution-quality score in ``[0, 1]``
    where 1.0 is the best price the day offered on that side.
    """
    out = trades.copy()
    n = len(out)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    settle = np.full(n, np.nan)

    for position, (_, trade) in enumerate(out.iterrows()):
        structure = trade.get("structure")
        date = trade.get("trade_date")
        if structure is None or date is None or pd.isna(date) or not book.covers(structure):
            continue
        day = pd.Timestamp(date).normalize()
        bounds = book.daily_bounds(structure)
        if day not in bounds.index:
            continue
        lower[position] = bounds.at[day, "lower"]
        upper[position] = bounds.at[day, "upper"]
        daily = book.daily_price(structure)
        if day in daily.index:
            settle[position] = daily.loc[day]

    out["day_low"], out["day_high"], out["day_settle"] = lower, upper, settle
    width = out["day_high"] - out["day_low"]
    padding = 0.02 * width.abs()

    def _feasible(price: pd.Series) -> pd.Series:
        return (price >= out["day_low"] - padding) & (price <= out["day_high"] + padding)

    out["entry_feasible"] = _feasible(out["entry_price"])
    out["exit_feasible"] = _feasible(out["exit_price"])

    # Execution quality: 1.0 = best available price on that side of the trade.
    side = np.sign(out["lots"]).replace(0, np.nan)
    denominator = width.replace(0, np.nan)
    entry_pos = (out["entry_price"] - out["day_low"]) / denominator
    exit_pos = (out["exit_price"] - out["day_low"]) / denominator
    # Buying entry (side>0) is better low; selling exit is better high.
    out["entry_quality"] = np.where(side > 0, 1.0 - entry_pos, entry_pos)
    out["exit_quality"] = np.where(side > 0, exit_pos, 1.0 - exit_pos)
    for column in ("entry_quality", "exit_quality"):
        out.loc[~out[column].between(-0.5, 1.5), column] = np.nan
    out["fill_quality"] = out[["entry_quality", "exit_quality"]].mean(axis=1)
    return out


def reconciliation_summary(trades: pd.DataFrame) -> str:
    counts = trades["pnl_status"].value_counts()
    checkable = int(counts.drop(labels=["incomplete"], errors="ignore").sum())
    if not checkable:
        return "No trades had enough fields to reconcile."
    match = int(counts.get("match", 0))
    flip = int(counts.get("sign_flip", 0))
    bad = int(counts.get("unexplained", 0))
    return (
        f"Of {checkable:,} reconcilable trades, {match:,} ({match / checkable:.1%}) "
        f"reproduce the reported P&L to within ${PNL_TOL:.0f}, "
        f"{flip:,} ({flip / checkable:.1%}) reproduce it only after a sign flip, "
        f"and {bad:,} ({bad / checkable:.1%}) remain unexplained and are excluded "
        f"from P&L-based inference."
    )
