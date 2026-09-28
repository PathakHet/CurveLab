"""Tests for the journal ingest, parser, date resolver and reconciliation."""

import numpy as np
import pandas as pd
import pytest

from curvelab.core.contracts import Contract
from curvelab.core.structure import fly_from, spread_from
from curvelab.desk.anonymize import build_mapping
from curvelab.desk.dates import resolve_dates
from curvelab.desk.journal import _to_float, read_sheet
from curvelab.desk.nameparser import parse_strategy_name
from curvelab.desk.reconstruct import fit_multiplier, reconcile
from curvelab.desk.skill import benjamini_hochberg

OCT26 = Contract(2026, 10)


@pytest.mark.parametrize(
    "name",
    [
        "oct26 1mo fly", "oct_1mo_fly", "OctNovDec", "oct26 1mo butterfly",
        "BRN Oct26 1mo fly(long)", "oct 26 1 mon fly", "oct26 1mo fly- short",
        "(oct26 - 2*nov26 + dec26) ( oct26 1month fly)",
        "BRN oct26 - 2*nov26 + dec26 [oct26 1mofly]",
    ],
)
def test_all_spellings_resolve_to_the_same_structure(name):
    """Nine spellings from the real journals, one instrument."""
    assert parse_strategy_name(name).structure == fly_from(OCT26, 1)


def test_direction_is_extracted_and_does_not_change_the_structure():
    long = parse_strategy_name("oct26 1mo fly long")
    short = parse_strategy_name("oct26 1mo fly short")
    assert long.direction == 1 and short.direction == -1
    assert long.structure == short.structure


def test_ice_month_codes():
    """(BZ26-BN27) - 2*(BN27-BZ27): Dec26 and Jul27 in ICE notation."""
    result = parse_strategy_name("(BZ26-BN27)-2*(BN27-BZ27)")
    assert result.ok
    assert Contract(2026, 12) in result.structure.contracts
    assert Contract(2027, 7) in result.structure.contracts


def test_gloss_is_cross_checked_not_multiplied():
    """Juxtaposed expressions are a value and its gloss, not a product."""
    result = parse_strategy_name("(oct26 - 2*nov26 + dec26) ( oct26 1month fly)")
    assert result.gloss_agrees is True
    assert result.structure.gross_leg_exposure == 4.0  # not 16.0


def test_untyped_tenor_is_reported_as_ambiguous():
    """"dec26 3mo" is a spread or a fly; the parser must not guess."""
    result = parse_strategy_name("dec26 3mo")
    assert result.ambiguous
    assert spread_from(Contract(2026, 12), 3) in result.candidates
    assert fly_from(Contract(2026, 12), 3) in result.candidates


def test_month_initials_need_a_consistent_spacing():
    assert parse_strategy_name("djd 6mo fly").ok      # dec -> jun -> dec
    assert not parse_strategy_name("djd 3mo fly").ok  # dec -> mar breaks 'j'


def test_unparseable_names_fail_loudly():
    for junk in ("r1", "", "short"):
        assert not parse_strategy_name(junk).ok


def test_day_first_dates_are_swapped_back_into_the_window():
    """12/06 read as 6 December must be recovered as 12 June."""
    trades = pd.DataFrame(
        {
            "sheet": ["a"] * 6,
            "date_raw": pd.to_datetime(
                ["2026-06-18", "2026-06-22", "2026-06-25",   # unambiguous anchors
                 "2026-12-06", "2026-10-06", "2026-06-05"],  # ambiguous
            ),
        }
    )
    resolved, audit = resolve_dates(trades)
    assert resolved["trade_date"].dt.month.eq(6).all()
    assert audit.n_swapped == 2
    assert resolved.loc[3, "trade_date"] == pd.Timestamp("2026-06-12")


def test_unambiguous_dates_are_never_swapped():
    trades = pd.DataFrame({"sheet": ["a"] * 3,
                           "date_raw": pd.to_datetime(["2026-05-18", "2026-05-19", "2026-05-20"])})
    resolved, audit = resolve_dates(trades)
    assert audit.n_swapped == 0
    assert (resolved["date_rule"] == "unambiguous").all()


def test_multiplier_is_recovered_not_assumed():
    trades = pd.DataFrame(
        {
            "entry_price": [0.10, -0.05, 0.20],
            "exit_price": [0.15, -0.08, 0.18],
            "lots": [10.0, -5.0, 4.0],
            "net_pnl": [500.0, 150.0, -80.0],
        }
    )
    assert fit_multiplier(trades).multiplier == 1000.0


def test_reconcile_classifies_matches_flips_and_junk():
    trades = pd.DataFrame(
        {
            "entry_price": [0.10, 0.10, 0.10, 0.10],
            "exit_price": [0.15, 0.15, 0.15, np.nan],
            "lots": [10.0, 10.0, 10.0, 10.0],
            "net_pnl": [500.0, -500.0, 123.0, 500.0],
        }
    )
    out = reconcile(trades, 1000.0)
    assert list(out["pnl_status"]) == ["match", "sign_flip", "unexplained", "incomplete"]
    # A sign-flipped row is trusted only after correction; junk is dropped.
    assert out.loc[1, "pnl_trusted"] == 500.0
    assert np.isnan(out.loc[2, "pnl_trusted"])


def test_benjamini_hochberg_is_stricter_than_raw_and_looser_than_bonferroni():
    p_values = pd.Series([0.001, 0.02, 0.04, 0.3, 0.5, 0.9])
    rejected = benjamini_hochberg(p_values, q=0.10)
    assert rejected.sum() >= 1
    assert rejected.sum() <= (p_values < 0.05).sum()


def test_anonymisation_is_stable_and_leaks_no_names():
    names = pd.Series(["Alice", "Bob", "Carol"])
    first = build_mapping(names, salt="s")
    assert first == build_mapping(names, salt="s")           # deterministic
    assert set(first.values()) == {"T01", "T02", "T03"}
    assert build_mapping(names, salt="other") != first        # salt changes order


def test_journal_reader_finds_a_header_below_blank_rows():
    raw = pd.DataFrame(
        [
            [None, None, None, None, None, None],
            ["Strategy Name", "Average Entry Price ", "Average Exit Price ",
             "Stop loss ", "No. of Lots", "Net Pnl "],
            [pd.Timestamp("2026-06-25"), None, None, None, None, 550],  # day header
            ["oct26 1mo fly", -0.02, 0.03, 0.0, -10, 100],
            ["nan", None, None, None, None, None],                      # NaN cell
        ]
    )
    trades, stats = read_sheet(raw, "t", "f.xlsx", "sheet")
    assert stats.header_row == 1
    assert len(trades) == 1
    assert trades.iloc[0]["date_raw"] == pd.Timestamp("2026-06-25")


def test_numeric_coercion_tolerates_journal_junk():
    assert _to_float(" -0.02 ") == pytest.approx(-0.02)
    assert np.isnan(_to_float("#REF!"))
    assert np.isnan(_to_float(""))
