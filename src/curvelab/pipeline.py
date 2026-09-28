"""The full run: market data and journals go in, CSVs and a report come out.

Each stage takes what the previous one produced and hands back a table plus a
one-line finding. I never edit the report by hand. Every number in
``reports/REPORT.md`` comes from the run that wrote it, so if a parameter
changes, the write-up changes with it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curvelab.config import Config
from curvelab.core.book import CurveBook
from curvelab.core.loader import audit_frame, load_curve
from curvelab.desk import attribution as attribution_mod
from curvelab.desk import skill as skill_mod
from curvelab.desk.anonymize import anonymize
from curvelab.desk.dates import resolve_dates
from curvelab.desk.disambiguate import attach_structures
from curvelab.desk.journal import ingest_summary, load_journals
from curvelab.desk.nameparser import parse_strategy_name
from curvelab.desk.reconstruct import (
    fit_multiplier,
    mark_to_market,
    reconcile,
    reconciliation_summary,
)
from curvelab.research import diagnostics, feature_study, stats
from curvelab.research.backtest import run_backtest, summarize
from curvelab.research.features import build_dataset
from curvelab.research.strategies import RULE_BASED
from curvelab.research.universe import build_universe, catalogue
from curvelab.research.walkforward import walk_forward_ranker


@dataclass
class RunResult:
    """Everything a run produced: frames for CSV, findings for the report."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    findings: dict[str, str] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    book: CurveBook | None = None

    def add(self, name: str, table: pd.DataFrame, finding: str = "") -> None:
        self.tables[name] = table
        if finding:
            self.findings[name] = finding


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def run(config: Config, root: Path | None = None) -> RunResult:
    """Run every stage and return the collected results."""
    root = Path(root or Path.cwd())
    result = RunResult()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # --- market data ----------------------------------------------------
    curve = load_curve(_resolve(root, config.data.market_dir))
    if not curve:
        raise FileNotFoundError(f"no market data found in {config.data.market_dir}")
    book = CurveBook(curve, config.data.min_bars, config.data.max_ffill_bars)
    result.book = book
    audit = audit_frame(curve)
    result.add(
        "data_audit",
        audit,
        f"{len(curve)} contracts loaded ({int(audit['header_present'].eq(False).sum())} "
        f"exported without a header row); {len(book.contracts)} pass the "
        f"{config.data.min_bars}-bar liquidity filter over {len(book.close):,} hourly bars.",
    )
    result.add("dropped_contracts", book.dropped)

    # --- structure universe ---------------------------------------------
    universe = build_universe(
        book,
        spacings=tuple(config.universe.spacings),
        kinds=tuple(config.universe.kinds),
        min_bars=config.universe.min_structure_bars,
    )
    result.add(
        "structure_catalogue",
        catalogue(book, universe),
        f"{len(universe)} tradable structures generated from "
        f"{len(book.contracts)} liquid contracts.",
    )

    # --- journals: ingest, dates, parsing, disambiguation ----------------
    journal_paths = [_resolve(root, p) for p in config.data.journals]
    journal_paths = [p for p in journal_paths if p.exists()]
    if journal_paths:
        trades, ingest_stats = load_journals(journal_paths)
        result.add("ingest_stats", ingest_stats, ingest_summary(trades, ingest_stats))

        trades, date_audit = resolve_dates(trades)
        result.findings["dates"] = date_audit.summary()
        result.scalars["date_audit"] = date_audit

        parses = [parse_strategy_name(name) for name in trades["strategy_raw"]]
        parsed_ok = sum(p.ok for p in parses)
        unique_names = {str(n).strip().lower(): p for n, p in zip(trades["strategy_raw"], parses)}
        checked = [p for p in parses if p.gloss_agrees is not None]
        result.findings["parser"] = (
            f"{parsed_ok:,} of {len(parses):,} trades ({parsed_ok / max(len(parses), 1):.1%}) "
            f"parsed into a canonical structure, covering "
            f"{sum(p.ok for p in unique_names.values())} of {len(unique_names)} distinct "
            f"name spellings. {len(checked)} names carried a redundant algebraic gloss; "
            f"{sum(bool(p.gloss_agrees) for p in checked)} agreed with the shorthand."
        )
        result.add(
            "parse_failures",
            pd.DataFrame(
                sorted(
                    {p.raw.strip() for p in parses if not p.ok},
                )[:100],
                columns=["unparsed_name"],
            ),
        )

        trades = attach_structures(trades, parses, book)
        methods = trades["resolve_method"].value_counts()
        result.findings["disambiguation"] = (
            f"{int(methods.get('price_feasible', 0) + methods.get('price_plausible', 0)):,} "
            f"ambiguous names were resolved against observed market prices; "
            f"{int(methods.get('unresolved', 0)):,} could not be separated and were left unassigned."
        )

        # --- reconciliation and execution quality ------------------------
        fit = fit_multiplier(trades)
        result.findings["multiplier"] = fit.summary()
        result.scalars["multiplier"] = fit.multiplier
        trades = reconcile(trades, fit.multiplier)
        result.findings["reconciliation"] = reconciliation_summary(trades)
        trades = mark_to_market(trades, book)

        marked = trades[trades["day_low"].notna()]
        if len(marked):
            result.findings["execution"] = (
                f"{len(marked):,} trades were marked against the market on "
                f"{marked['trade_date'].nunique()} trading days. "
                f"{marked['entry_feasible'].mean():.1%} of entry fills and "
                f"{marked['exit_feasible'].mean():.1%} of exits fall inside the "
                f"structure's attainable range for that day -- an independent check that "
                f"the parser assigned the right instrument. Mean fill placement is "
                f"{marked['fill_quality'].mean():.2f} on a 0-1 scale where 1.0 is the "
                f"best price the day offered."
            )

        if config.desk.anonymize:
            trades, mapping = anonymize(
                trades,
                salt=config.desk.anonymize_salt,
                mapping_path=Path(config.desk.mapping_path).expanduser(),
            )
            result.findings["anonymisation"] = (
                f"{len(mapping)} trader identities replaced with stable codes; "
                f"the lookup table is written outside the repository."
            )

        keep = [
            c for c in (
                "trader", "strategy_raw", "structure_id", "structure_kind",
                "structure_display", "resolve_method", "trade_date", "date_rule",
                "date_swapped", "entry_price", "exit_price", "lots", "net_pnl",
                "pnl_recomputed", "pnl_status", "pnl_trusted", "day_low", "day_high",
                "entry_feasible", "exit_feasible", "fill_quality",
            ) if c in trades.columns
        ]
        result.add("trades", trades[keep])

        # --- factor attribution and skill --------------------------------
        factors = attribution_mod.curve_factors(book.daily_close)
        result.add("factor_loadings", factors.loadings, factors.summary())
        daily_pnl = attribution_mod.daily_pnl_by_trader(trades)
        result.scalars["daily_pnl"] = daily_pnl

        if not daily_pnl.empty:
            desk = attribution_mod.desk_attribution(daily_pnl, factors)
            if desk:
                result.findings["desk_attribution"] = (
                    f"Desk-wide, curve factors explain {desk['r_squared']:.1%} of daily P&L "
                    f"variance; residual alpha is ${desk['alpha']:,.0f}/day "
                    f"(t={desk['alpha_t']:.2f}). Near-zero explanatory power is the expected "
                    f"signature of a book built from flies, which are constructed to be "
                    f"neutral to level and slope."
                )
                result.scalars["desk_attribution"] = desk
            result.add(
                "trader_attribution",
                attribution_mod.attribute(daily_pnl, factors, config.desk.min_days),
            )

            assessment = skill_mod.assess(
                daily_pnl, q=config.desk.fdr_q, min_days=config.desk.min_days,
                n_draws=config.desk.bootstrap_draws, seed=config.seed,
            )
            result.add("skill", assessment.table, assessment.summary())
            result.scalars["skill"] = assessment

    # --- systematic research ---------------------------------------------
    prices, features, labels = build_dataset(
        book, universe, config.research.horizon,
        config.research.zscore_window, config.research.vol_window,
    )
    exposure = pd.Series({s.structure_id: s.gross_leg_exposure for s in universe})
    backtest_kwargs = dict(
        cost_per_leg=config.cost.cost_per_leg,
        slippage_per_leg=config.cost.slippage_per_leg,
        max_active=config.research.max_active,
        hysteresis=config.research.hysteresis,
    )

    leaderboard, delay_tables, signal_store = {}, {}, {}
    for name, strategy in RULE_BASED.items():
        signals, conviction = strategy(prices)
        signal_store[name] = (signals, conviction)
        outcome = run_backtest(prices, signals, exposure, conviction=conviction, **backtest_kwargs)
        leaderboard[name] = summarize(outcome, config.research.periods_per_year)
        delay_tables[name] = diagnostics.delay_sensitivity(
            prices, signals, conviction, exposure,
            delays=tuple(config.research.delays),
            periods_per_year=config.research.periods_per_year, **backtest_kwargs,
        )

    ml_signals, ml_conviction, folds, importance = walk_forward_ranker(
        features, labels, prices,
        model_name=config.research.model, horizon=config.research.horizon,
        top_k=config.research.top_k, embargo_bars=config.research.embargo_bars,
        seed=config.seed, train_bars=config.research.train_bars,
        test_bars=config.research.test_bars, step_bars=config.research.step_bars,
    )
    signal_store["ml_ranker_oos"] = (ml_signals, ml_conviction)
    ml_result = run_backtest(
        prices, ml_signals, exposure, conviction=ml_conviction, **backtest_kwargs
    )
    leaderboard["ml_ranker_oos"] = summarize(ml_result, config.research.periods_per_year)
    delay_tables["ml_ranker_oos"] = diagnostics.delay_sensitivity(
        prices, ml_signals, ml_conviction, exposure,
        delays=tuple(config.research.delays),
        periods_per_year=config.research.periods_per_year, **backtest_kwargs,
    )

    board = pd.DataFrame(leaderboard).T.sort_values("sharpe", ascending=False)
    n_trials = len(board)
    deflated = []
    for name in board.index:
        signals, conviction = signal_store[name]
        outcome = run_backtest(prices, signals, exposure, conviction=conviction, **backtest_kwargs)
        row = stats.evaluate(
            outcome.portfolio_pnl, n_trials, config.research.periods_per_year, seed=config.seed
        )
        row["strategy"] = name
        row["delay_verdict"] = diagnostics.verdict(delay_tables[name])
        deflated.append(row)
    significance = pd.DataFrame(deflated).set_index("strategy")

    board = board.join(significance[["benchmark", "deflated_sharpe", "bootstrap_p", "delay_verdict"]])
    survivors = int((board["delay_verdict"] == "edge survives execution delay").sum())
    result.add(
        "strategy_leaderboard",
        board,
        f"{len(board)} strategies tested over {len(folds)} walk-forward folds. "
        f"The best in-sample Sharpe is {board['sharpe'].max():.2f} against a "
        f"best-of-{n_trials} null of {board['benchmark'].iloc[0]:.2f}. "
        f"{survivors} of {len(board)} survive a one-bar execution delay.",
    )
    result.add("delay_sensitivity", pd.concat(delay_tables, names=["strategy"]).reset_index(level=0))
    if not importance.empty:
        result.add("feature_importance", importance)

    # --- which signals actually carry information -------------------------
    fly_study = feature_study.study(
        features, prices,
        kind="fly",
        gross_leg_exposure=4.0,
        cost_per_leg=config.cost.cost_per_leg + config.cost.slippage_per_leg,
    )
    result.add("feature_scorecard", fly_study["scorecard"], fly_study["finding"])
    result.add("feature_survivors", fly_study["survivors"])
    if fly_study["verdict"]:
        result.findings["feature_economics"] = fly_study["verdict"]
    for name, table in fly_study["buckets"].items():
        result.add(f"buckets_{name}", table)
    result.scalars["feature_buckets"] = fly_study["buckets"]

    cost_sweep = {}
    for name, (signals, conviction) in signal_store.items():
        sweep = diagnostics.cost_sensitivity(
            prices, signals, conviction, exposure,
            periods_per_year=config.research.periods_per_year,
            max_active=config.research.max_active, hysteresis=config.research.hysteresis,
        )
        cost_sweep[name] = diagnostics.breakeven_cost(sweep)
    result.add(
        "breakeven_costs",
        pd.Series(cost_sweep, name="breakeven_cost_per_leg").to_frame(),
        f"Assumed round-trip cost is ${config.cost.cost_per_leg + config.cost.slippage_per_leg:.4f} "
        f"per leg; the best strategy breaks even at "
        f"${np.nanmax(list(cost_sweep.values())):.4f} per leg.",
    )

    result.scalars["prices"] = prices
    result.scalars["n_folds"] = len(folds)
    return result


def write_outputs(result: RunResult, config: Config, root: Path | None = None) -> Path:
    """Write every table to CSV and return the results directory."""
    root = Path(root or Path.cwd())
    results_dir = _resolve(root, config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        table.to_csv(results_dir / f"{name}.csv")
    return results_dir
