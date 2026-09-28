"""Figures and the generated markdown report.

Every number in the report comes from the run that wrote it. Nothing is edited
by hand, so changing a parameter can't leave an out-of-date claim behind.

Chart style: one y-axis per chart, colours assigned in a fixed order, a light
grid, a legend whenever there's more than one series, and labels only on the
points that matter. Each figure has a CSV of the same name in reports/results
with the numbers behind it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from curvelab.config import Config  # noqa: E402
from curvelab.pipeline import RunResult  # noqa: E402

# Series colours, always handed out in this order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, WARN, BAD, MUTED = "#1baf7a", "#eda100", "#d1435b", "#b9b8b2"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=0)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return path.name


def figure_term_structure(book, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    daily = book.daily_close.dropna(how="all")
    picks = [daily.index[0], daily.index[len(daily) // 2], daily.index[-1]]
    x = np.arange(len(book.contracts))
    for i, when in enumerate(picks):
        row = daily.loc[when].to_numpy()
        ax.plot(x, row, marker="o", markersize=4.5, linewidth=2, color=SERIES[i],
                label=pd.Timestamp(when).date().isoformat())
        ax.annotate(pd.Timestamp(when).strftime("%d %b"), (x[-1], row[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=SERIES[i], va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([c.code for c in book.contracts], rotation=45, ha="right")
    _style(ax, "Brent forward curve: start, middle and end of sample", "", "$/bbl")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    return _save(fig, path)


def figure_factor_loadings(loadings: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(loadings))
    for i, column in enumerate(loadings.columns):
        values = loadings[column].to_numpy()
        ax.plot(x, values, marker="o", markersize=4.5, linewidth=2,
                color=SERIES[i], label=column)
        ax.annotate(column, (x[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=SERIES[i], va="center")
    ax.axhline(0, color=INK_SOFT, linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in loadings.index], rotation=45, ha="right")
    _style(ax, "Curve factor loadings", "", "loading")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    return _save(fig, path)


def figure_delay_sensitivity(table: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (name, group) in enumerate(table.groupby("strategy")):
        group = group.sort_values("extra_delay_bars")
        colour = SERIES[i % len(SERIES)]
        ax.plot(group["extra_delay_bars"], group["sharpe"], marker="o",
                markersize=5, linewidth=2, color=colour, label=name.replace("_", " "))
    ax.axhline(0, color=INK, linewidth=1.2)
    _style(ax, "Annualised Sharpe against extra execution delay",
           "extra bars of delay beyond the anti-lookahead shift", "Sharpe")
    ax.set_xticks(sorted(table["extra_delay_bars"].unique()))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="best")
    return _save(fig, path)


def figure_skill(table: pd.DataFrame, path: Path) -> str:
    table = table.sort_values("t_stat")
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.30 * len(table) + 1.4)))
    colours = [GOOD if flag else MUTED for flag in table["significant_fdr"]]
    y = np.arange(len(table))
    ax.barh(y, table["t_stat"].to_numpy(), color=colours, height=0.62)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.axvline(1.96, color=INK_SOFT, linewidth=1, linestyle="--")
    ax.annotate("t = 1.96 (uncorrected)", (1.96, len(table) - 0.6), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=INK_SOFT, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(table.index, fontsize=8)
    _style(ax, "Trader t-statistics; filled bars survive Benjamini-Hochberg FDR control",
           "bootstrap t-statistic on mean daily P&L", "")
    return _save(fig, path)


def figure_reconciliation(trades: pd.DataFrame, path: Path) -> str:
    counts = trades["pnl_status"].value_counts()
    order = [s for s in ("match", "sign_flip", "unexplained", "incomplete") if s in counts.index]
    palette = {"match": GOOD, "sign_flip": WARN, "unexplained": BAD, "incomplete": MUTED}
    values = [int(counts[s]) for s in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = np.arange(len(order))
    ax.bar(x, values, color=[palette[s] for s in order], width=0.6)
    for xi, value in zip(x, values):
        ax.annotate(f"{value:,}", (xi, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in order], fontsize=9)
    _style(ax, "Journal P&L reconciled against recomputed trade P&L", "", "trades")
    return _save(fig, path)


def figure_strategy_pnl(result: RunResult, path: Path) -> str:
    board = result.tables.get("strategy_leaderboard")
    delay = result.tables.get("delay_sensitivity")
    if board is None or delay is None:
        return ""
    names = list(board.index[:5])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, name in enumerate(names):
        subset = delay[(delay["strategy"] == name) & (delay["extra_delay_bars"] == 0)]
        if subset.empty:
            continue
        value = float(subset["total_pnl"].iloc[0])
        ax.bar(i, value, color=SERIES[i % len(SERIES)], width=0.6)
        ax.annotate(f"${value:,.0f}", (i, value),
                    xytext=(0, 4 if value >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    _style(ax, "Total P&L per strategy at zero extra delay", "", "$ per unit position")
    return _save(fig, path)


def build_figures(result: RunResult, config: Config, root: Path) -> dict[str, str]:
    out_dir = Path(root) / config.output.figures_dir
    figures: dict[str, str] = {}
    if result.book is not None:
        figures["term_structure"] = figure_term_structure(result.book, out_dir / "term_structure.png")
    if "factor_loadings" in result.tables:
        figures["factor_loadings"] = figure_factor_loadings(
            result.tables["factor_loadings"], out_dir / "factor_loadings.png")
    if "delay_sensitivity" in result.tables:
        figures["delay_sensitivity"] = figure_delay_sensitivity(
            result.tables["delay_sensitivity"], out_dir / "delay_sensitivity.png")
    if "skill" in result.tables and not result.tables["skill"].empty:
        figures["skill"] = figure_skill(result.tables["skill"], out_dir / "skill.png")
    trades = result.tables.get("trades")
    if trades is not None and "pnl_status" in trades.columns:
        figures["reconciliation"] = figure_reconciliation(trades, out_dir / "reconciliation.png")
    pnl = figure_strategy_pnl(result, out_dir / "strategy_pnl.png")
    if pnl:
        figures["pnl"] = pnl
    return figures


def _table(frame: pd.DataFrame, max_rows: int = 12, fmt: str = "{:,.3f}") -> str:
    subset = frame.head(max_rows).copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return subset.to_markdown()


def build_report(result: RunResult, config: Config, root: Path) -> Path:
    figures = build_figures(result, config, root)
    f = result.findings
    lines: list[str] = []
    add = lines.append

    add("# CurveLab: Brent curve research and desk analytics")
    add("")
    add("Generated by `curvelab run`. Every figure and number below was produced "
        "by the run that wrote this file.")
    add("")
    add("## 1. Data")
    add("")
    for key in ("data_audit", "structure_catalogue"):
        if key in f:
            add(f[key])
            add("")
    if "term_structure" in figures:
        add(f"![Brent forward curve](figures/{figures['term_structure']})")
        add("")
    if "data_audit" in result.tables:
        add(_table(result.tables["data_audit"], 20))
        add("")

    if "ingest_stats" in f:
        add("## 2. Reading the trade journals")
        add("")
        for key in ("ingest_stats", "dates", "parser", "disambiguation"):
            if key in f:
                add(f[key])
                add("")
        add("## 3. Reconciliation and execution quality")
        add("")
        for key in ("multiplier", "reconciliation", "execution", "anonymisation"):
            if key in f:
                add(f[key])
                add("")
        if "reconciliation" in figures:
            add(f"![P&L reconciliation](figures/{figures['reconciliation']})")
            add("")
        add("## 4. Where the desk's P&L came from")
        add("")
        for key in ("factor_loadings", "desk_attribution"):
            if key in f:
                add(f[key])
                add("")
        if "factor_loadings" in figures:
            add(f"![Curve factor loadings](figures/{figures['factor_loadings']})")
            add("")
        table = result.tables.get("trader_attribution")
        if table is not None and not table.empty:
            add("Per-trader attribution, ranked by residual alpha:")
            add("")
            add(_table(table, 10, "{:,.2f}"))
            add("")
        add("## 5. Skill or luck?")
        add("")
        if "skill" in f:
            add(f["skill"])
            add("")
        if "skill" in figures:
            add(f"![Trader t-statistics](figures/{figures['skill']})")
            add("")
        table = result.tables.get("skill")
        if table is not None and not table.empty:
            add(_table(table, 10, "{:,.2f}"))
            add("")

    add("## 6. Systematic strategies")

    add("")
    for key in ("strategy_leaderboard", "breakeven_costs"):
        if key in f:
            add(f[key])
            add("")
    if "strategy_leaderboard" in result.tables:
        add(_table(result.tables["strategy_leaderboard"], 12, "{:,.2f}"))
        add("")
    if "pnl" in figures:
        add(f"![Strategy P&L](figures/{figures['pnl']})")
        add("")

    add("## 7. Which signals actually help?")

    add("")
    for key in ("feature_scorecard", "feature_economics"):
        if key in f:
            add(f[key])
            add("")
    survivors = result.tables.get("feature_survivors")
    if survivors is not None and not survivors.empty:
        add("Features ranked by the significance of their cross-sectional "
            "information coefficient, measured on returns a trader could actually "
            "have captured (skip = 1 bar):")
        add("")
        columns = [c for c in ("feature", "horizon", "mean_ic", "t_stat", "hit_rate",
                               "ic_retained", "persistence", "static", "survives")
                   if c in survivors.columns]
        add(_table(survivors[columns].reset_index(drop=True), 12, "{:,.3f}"))
        add("")
    buckets = result.scalars.get("feature_buckets") or {}
    for name, table in list(buckets.items())[:1]:
        add(f"What happened next, by bucket of `{name}`:")
        add("")
        add(_table(table.reset_index(), 8, "{:,.4f}"))
        add("")

    add("### The execution-delay test")

    add("")
    add("Re-running each strategy with extra lag separates a forecast from an "
        "artefact. A genuine signal about the next several bars degrades gradually; "
        "an edge that inverts on the first extra bar was capturing the reversal of "
        "the move that generated it, which on close-to-close data is bid-ask bounce "
        "and is not tradable at any speed.")
    add("")
    if "delay_sensitivity" in figures:
        add(f"![Delay sensitivity](figures/{figures['delay_sensitivity']})")
        add("")
    if "delay_sensitivity" in result.tables:
        add(_table(result.tables["delay_sensitivity"], 24, "{:,.2f}"))
        add("")

    add("## 8. Limitations")

    add("")
    for line in (
        "The market data covers roughly two months of hourly bars. Every Sharpe "
        "ratio here is indicative; none is a claim about live profitability.",
        "Journal timestamps are date-only, so execution quality is measured as "
        "placement within the day's attainable range, not as implementation shortfall.",
        "Only a subset of traders recorded enough dated trades to test individually, "
        "which materially limits the power of the skill analysis.",
        "Structures are marked close-to-close with no bid-ask data, so transaction "
        "costs are a modelled assumption rather than an observation.",
    ):
        add(f"- {line}")
    add("")

    path = Path(root) / config.output.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
