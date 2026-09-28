"""End-to-end run on the committed sample data.

This is the test that catches stages wired together wrongly. It runs the whole
pipeline (ingest, dates, parsing, disambiguation, reconciliation, attribution,
the skill test, the research layer and the report) on the synthetic data in
data/sample. That data is generated with the same problems as the real journals
(day-first dates, swapped prices, lots of spellings per instrument), so all the
clean-up code actually gets exercised.
"""

from pathlib import Path

import pytest

from curvelab.config import Config
from curvelab.pipeline import run, write_outputs
from curvelab.report import build_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    config = Config()
    config.desk.bootstrap_draws = 200      # keep the smoke test quick
    config.research.train_bars = 300
    return run(config, ROOT), config


def test_pipeline_produces_every_stage(result):
    outcome, _ = result
    for table in (
        "data_audit", "structure_catalogue", "ingest_stats", "trades",
        "factor_loadings", "strategy_leaderboard", "delay_sensitivity",
    ):
        assert table in outcome.tables, f"missing stage output: {table}"
    for finding in ("parser", "dates", "multiplier", "reconciliation"):
        assert outcome.findings.get(finding), f"missing finding: {finding}"


def test_sample_journal_parses_essentially_completely(result):
    outcome, _ = result
    trades = outcome.tables["trades"]
    assert trades["structure_id"].notna().mean() > 0.95


def test_injected_defects_are_detected(result):
    """The sample journal contains transposed prices; they must be caught."""
    outcome, _ = result
    statuses = outcome.tables["trades"]["pnl_status"].value_counts()
    assert statuses.get("match", 0) > 0
    assert statuses.get("sign_flip", 0) > 0


def test_day_first_dates_are_recovered(result):
    outcome, _ = result
    assert outcome.scalars["date_audit"].n_swapped > 0
    assert outcome.scalars["date_audit"].n_unresolved <= 5


def test_traders_are_anonymised(result):
    outcome, _ = result
    traders = set(outcome.tables["trades"]["trader"].unique())
    assert all(t.startswith("T") and t[1:].isdigit() for t in traders)


def test_every_strategy_gets_a_delay_verdict(result):
    outcome, _ = result
    board = outcome.tables["strategy_leaderboard"]
    assert board["delay_verdict"].notna().all()
    assert "deflated_sharpe" in board.columns


def test_report_and_csvs_are_written(result, tmp_path):
    outcome, config = result
    config.output.results_dir = str(tmp_path / "results")
    config.output.figures_dir = str(tmp_path / "figures")
    config.output.report_path = str(tmp_path / "REPORT.md")
    results_dir = write_outputs(outcome, config, ROOT)
    assert any(results_dir.glob("*.csv"))
    report = build_report(outcome, config, ROOT)
    text = report.read_text()
    assert "execution-delay test" in text
    assert len(text) > 2000
