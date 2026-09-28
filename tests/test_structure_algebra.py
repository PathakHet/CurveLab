"""Property tests for the curve-structure algebra.

These check the identities traders take for granted. If one of them breaks,
every P&L number downstream is wrong, in a way a test on a single example
probably wouldn't catch.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from curvelab.core.contracts import Contract
from curvelab.core.structure import (
    Structure,
    dfly,
    dfly_from,
    fly,
    fly_from,
    outright,
    spread,
    spread_from,
)

months = st.integers(min_value=1, max_value=12)
years = st.integers(min_value=2024, max_value=2030)
contracts = st.builds(Contract, year=years, month=months)
spacings = st.integers(min_value=1, max_value=6)


@given(contracts, spacings)
def test_fly_is_a_spread_of_spreads(front, gap):
    a = front
    b = front.plus_months(gap)
    c = front.plus_months(2 * gap)
    assert spread(a, b) - spread(b, c) == fly(a, b, c)


@given(contracts, spacings)
def test_dfly_is_a_fly_of_flies(front, gap):
    legs = [front.plus_months(i * gap) for i in range(4)]
    assert fly(*legs[:3]) - fly(*legs[1:]) == dfly(*legs)


@given(contracts, spacings)
def test_named_constructors_agree_with_arithmetic(front, gap):
    assert fly_from(front, gap) == fly(
        front, front.plus_months(gap), front.plus_months(2 * gap)
    )
    assert dfly_from(front, gap) == dfly(
        *[front.plus_months(i * gap) for i in range(4)]
    )


@given(contracts, spacings)
def test_structures_are_price_neutral(front, gap):
    """Spreads, flies and double flies carry no outright flat-price exposure."""
    for structure in (spread_from(front, gap), fly_from(front, gap), dfly_from(front, gap)):
        assert structure.is_price_neutral
    assert not outright(front).is_price_neutral


@given(contracts, spacings)
def test_order_and_kind_are_derived_from_weights(front, gap):
    assert outright(front).order == 0
    assert spread_from(front, gap).order == 1
    assert fly_from(front, gap).order == 2
    assert dfly_from(front, gap).order == 3
    assert fly_from(front, gap).kind == "fly"


@given(contracts, spacings)
def test_self_cancellation(front, gap):
    structure = fly_from(front, gap)
    assert (structure - structure).is_empty
    assert (structure + (-structure)).is_empty


@given(contracts, spacings, st.floats(min_value=0.1, max_value=10, allow_nan=False))
def test_scaling_scales_exposure(front, gap, k):
    """Exposure scales linearly, up to the 1e-6 grid weights are canonicalised to.

    Weights are rounded so that arithmetic results stay canonical and
    hashable. The tolerance is that rounding step times the number of legs,
    not an arbitrary fudge.
    """
    structure = fly_from(front, gap)
    assert (structure * k).gross_leg_exposure == pytest.approx(
        structure.gross_leg_exposure * k, abs=1e-6 * structure.n_legs
    )


@given(contracts, spacings)
def test_hashable_and_canonical(front, gap):
    """Equal structures must hash equally -- they are used as dict keys."""
    a = spread(front, front.plus_months(gap)) - spread(front.plus_months(gap), front.plus_months(2 * gap))
    b = fly_from(front, gap)
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


@settings(max_examples=50)
@given(contracts, spacings)
def test_gross_leg_exposure_matches_cost_basis(front, gap):
    assert spread_from(front, gap).gross_leg_exposure == 2.0
    assert fly_from(front, gap).gross_leg_exposure == 4.0
    assert dfly_from(front, gap).gross_leg_exposure == 8.0


def test_uneven_legs_are_not_classified_as_a_fly():
    """A 1:2:1 basket on unevenly spaced legs is not a fly."""
    a, b, c = Contract(2026, 10), Contract(2026, 11), Contract(2027, 3)
    assert fly(a, b, c).order is None
    assert fly(a, b, c).kind == "custom_neutral"


def test_empty_structure():
    assert Structure.from_weights({}).is_empty
    assert Structure.from_weights({}).kind == "empty"
