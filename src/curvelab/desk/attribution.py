"""Split desk P&L into curve-factor exposure and what's left over (alpha).

Making money on a curve desk doesn't automatically mean skill. If the whole
curve steepens and you happen to be long every calendar spread, you profit for
reasons that have nothing to do with picking the right structure. This module
tries to separate the two.

I run PCA on daily contract price changes to get the usual term-structure
factors: level (the whole curve shifts), slope (it tilts) and curvature (the
middle bends against the wings). Each trader's daily P&L is then regressed on
those factors:

    pnl_t = alpha + sum_k beta_k * factor_k,t + e_t

The betas are the part of P&L explained by curve exposure, and alpha is what's
left. Daily P&L is autocorrelated and its variance moves around, so I use
Newey-West (HAC) standard errors. Plain OLS would make things look more
significant than they are.

Fly traders ought to have level and slope betas near zero, since a 1:2:1 fly
is built to be neutral to both. It's better to check that than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # statsmodels gives HAC standard errors; fall back to OLS if absent.
    import statsmodels.api as sm
    _HAVE_SM = True
except ImportError:  # pragma: no cover
    _HAVE_SM = False

FACTOR_NAMES = ("level", "slope", "curvature")


@dataclass
class CurveFactors:
    """PCA term-structure factors and how much variance each explains."""

    scores: pd.DataFrame            # date x [level, slope, curvature]
    loadings: pd.DataFrame          # contract x factor
    explained: pd.Series            # variance share per factor

    def summary(self) -> str:
        parts = ", ".join(
            f"{name} {self.explained[name]:.1%}" for name in self.scores.columns
        )
        return f"Curve factors explain {self.explained.sum():.1%} of daily variance ({parts})."


def curve_factors(daily_close: pd.DataFrame, n_factors: int = 3) -> CurveFactors:
    """PCA on daily contract price changes, giving level, slope and curvature.

    PCA can return each component with either sign, so I pin them down: level
    loads positively overall and slope loads positively on the front month.
    Otherwise betas could flip sign between runs and the table would be
    meaningless.
    """
    changes = daily_close.diff().dropna(how="all")
    changes = changes.dropna(axis=1, how="any") if changes.isna().any().any() else changes
    changes = changes.dropna()
    if changes.shape[0] < 5 or changes.shape[1] < 3:
        raise ValueError("not enough clean daily changes to fit curve factors")

    centred = changes - changes.mean()
    matrix = centred.to_numpy(dtype=float)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular**2 / max(matrix.shape[0] - 1, 1)
    explained = variance / variance.sum()

    components = vt[:n_factors]
    for k in range(components.shape[0]):
        if k == 0 and components[k].sum() < 0 or k > 0 and components[k][0] < 0:
            components[k] *= -1

    names = list(FACTOR_NAMES[:n_factors])
    scores = pd.DataFrame(matrix @ components.T, index=changes.index, columns=names)
    loadings = pd.DataFrame(components.T, index=changes.columns, columns=names)
    return CurveFactors(scores, loadings, pd.Series(explained[:n_factors], index=names))


def daily_pnl_by_trader(trades: pd.DataFrame, pnl_col: str = "pnl_trusted") -> pd.DataFrame:
    """Trader x date matrix of daily P&L from reconciled trades."""
    usable = trades.dropna(subset=["trade_date", pnl_col])
    if usable.empty:
        return pd.DataFrame()
    grouped = usable.groupby(["trader", usable["trade_date"].dt.normalize()])[pnl_col].sum()
    return grouped.unstack(level=0).sort_index()


def _hac_regress(y: pd.Series, X: pd.DataFrame, lags: int = 3) -> dict[str, float]:
    """Regress with Newey-West standard errors; returns alpha, betas and t-stats."""
    aligned = pd.concat([y.rename("y"), X], axis=1, sort=False).dropna()
    if len(aligned) < len(X.columns) + 3:
        return {}
    y_vec = aligned["y"]
    X_mat = aligned[list(X.columns)]
    if _HAVE_SM:
        model = sm.OLS(y_vec, sm.add_constant(X_mat)).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        out = {
            "alpha": float(model.params.get("const", np.nan)),
            "alpha_t": float(model.tvalues.get("const", np.nan)),
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
        }
        for name in X.columns:
            out[f"beta_{name}"] = float(model.params.get(name, np.nan))
            out[f"t_{name}"] = float(model.tvalues.get(name, np.nan))
        return out
    design = np.column_stack([np.ones(len(aligned)), X_mat.to_numpy(float)])
    coefficients, *_ = np.linalg.lstsq(design, y_vec.to_numpy(float), rcond=None)
    out = {"alpha": float(coefficients[0]), "alpha_t": np.nan, "r_squared": np.nan,
           "n_obs": len(aligned)}
    for i, name in enumerate(X.columns, start=1):
        out[f"beta_{name}"] = float(coefficients[i])
        out[f"t_{name}"] = np.nan
    return out


def attribute(
    daily_pnl: pd.DataFrame, factors: CurveFactors, min_days: int = 8
) -> pd.DataFrame:
    """Per-trader factor attribution table, sorted by residual alpha."""
    rows = []
    for trader in daily_pnl.columns:
        series = daily_pnl[trader].dropna()
        if len(series) < min_days:
            continue
        stats = _hac_regress(series, factors.scores)
        if not stats:
            continue
        stats.update(
            trader=trader,
            n_days=int(len(series)),
            total_pnl=float(series.sum()),
            mean_daily_pnl=float(series.mean()),
        )
        rows.append(stats)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).set_index("trader")
    ordered = ["n_days", "total_pnl", "mean_daily_pnl", "alpha", "alpha_t", "r_squared", "n_obs"]
    ordered += [c for c in table.columns if c.startswith(("beta_", "t_"))]
    return table[ordered].sort_values("alpha", ascending=False)


def desk_attribution(daily_pnl: pd.DataFrame, factors: CurveFactors) -> dict[str, float]:
    """Attribution for the desk as a whole (summed P&L across traders)."""
    return _hac_regress(daily_pnl.sum(axis=1, min_count=1), factors.scores)
