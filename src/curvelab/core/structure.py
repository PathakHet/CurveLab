"""Curve structures as weighted baskets of contracts.

A Structure is a set of (contract, weight) legs. That one type covers
everything the desk trades, from a single outright up to a double fly against
another double fly. It's frozen and hashable, so structures work as dict keys
and DataFrame columns.

I made it support arithmetic, rather than having separate spread/fly/dfly
classes, because the structures really do compose. The identities traders say
out loud then hold in code as well:

    spread(a, b) - spread(b, c) == fly(a, b, c)
    fly(a, b, c) - fly(b, c, d) == dfly(a, b, c, d)

tests/test_structure_algebra.py checks these with Hypothesis. The kind of a
structure (spread, fly, ...) is worked out from its weights rather than stored
as a label, so a fly you build by subtracting two spreads is still recognised
as a fly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from curvelab.core.contracts import Contract

#: Weights smaller than this count as zero and get dropped. Without it,
#: fly(a,b,c) - fly(a,b,c) would leave three legs of floating-point dust
#: instead of an empty structure.
WEIGHT_TOL = 1e-9


def _quantize(w: float) -> float:
    """Snap a weight to 1e-6 to keep arithmetic results canonical."""
    return round(w, 6) + 0.0  # +0.0 normalizes -0.0 -> 0.0


@dataclass(frozen=True, slots=True)
class Structure:
    """An immutable weighted basket of contracts.

    Parameters
    ----------
    legs
        ``(contract, weight)`` pairs, sorted by expiry with zero weights
        dropped. Construct via :meth:`from_weights` or the module-level
        constructors rather than passing this directly.
    """

    legs: tuple[tuple[Contract, float], ...]

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_weights(cls, weights: Mapping[Contract, float] | Iterable[tuple[Contract, float]]) -> Structure:
        """Build a canonical structure: summed duplicates, no zeros, expiry-sorted."""
        acc: dict[Contract, float] = {}
        items = weights.items() if isinstance(weights, Mapping) else weights
        for contract, weight in items:
            acc[contract] = acc.get(contract, 0.0) + float(weight)
        legs = tuple(
            (c, _quantize(w))
            for c, w in sorted(acc.items())
            if abs(w) > WEIGHT_TOL
        )
        return cls(legs=legs)

    # ---- views ----------------------------------------------------------
    @property
    def weights(self) -> dict[Contract, float]:
        return dict(self.legs)

    @property
    def contracts(self) -> tuple[Contract, ...]:
        return tuple(c for c, _ in self.legs)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def is_empty(self) -> bool:
        return not self.legs

    @property
    def gross_leg_exposure(self) -> float:
        """How many contracts one unit of the structure trades, which drives cost.

        A fly {1, -2, 1} trades 4 contracts per unit, so it costs twice as much
        as a spread {1, -1}.
        """
        return sum(abs(w) for _, w in self.legs)

    @property
    def net_weight(self) -> float:
        """Sum of weights. Zero means no outright flat-price exposure."""
        return _quantize(sum(w for _, w in self.legs))

    @property
    def is_price_neutral(self) -> bool:
        """True if the structure has no first-order flat-price exposure."""
        return abs(self.net_weight) < WEIGHT_TOL

    @property
    def front(self) -> Contract | None:
        return self.legs[0][0] if self.legs else None

    @property
    def tenor_span(self) -> int:
        """Months between the nearest and furthest leg."""
        if not self.legs:
            return 0
        return self.legs[-1][0] - self.legs[0][0]

    # ---- algebra --------------------------------------------------------
    def __add__(self, other: Structure) -> Structure:
        if not isinstance(other, Structure):
            return NotImplemented
        return Structure.from_weights(list(self.legs) + list(other.legs))

    def __sub__(self, other: Structure) -> Structure:
        if not isinstance(other, Structure):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> Structure:
        return Structure.from_weights([(c, -w) for c, w in self.legs])

    def __mul__(self, k: float) -> Structure:
        if not isinstance(k, (int, float)):
            return NotImplemented
        return Structure.from_weights([(c, w * k) for c, w in self.legs])

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> Structure:
        if not isinstance(k, (int, float)) or k == 0:
            return NotImplemented
        return self * (1.0 / k)

    # ---- classification -------------------------------------------------
    @property
    def order(self) -> int | None:
        """Finite-difference order, if the structure is a pure one.

        0 for an outright, 1 for a spread, 2 for a fly, 3 for a double fly.
        In general it's n when the legs are evenly spaced and the weights are
        (-1)^i * C(n, i), up to a positive scale. Anything else (ratio flies,
        fly vs fly, condors, uneven spacing) gives None.

        This is how a fly gets recognised as a fly however it was built.
        """
        if not self.legs:
            return None
        n = self.n_legs - 1
        spacings = {
            self.legs[i + 1][0] - self.legs[i][0] for i in range(n)
        }
        if len(spacings) > 1:  # legs must be evenly spaced on the curve
            return None
        scale = self.legs[0][1]
        if scale == 0:
            return None
        for i, (_, w) in enumerate(self.legs):
            expected = scale * ((-1) ** i) * math.comb(n, i)
            if abs(w - expected) > 1e-6:
                return None
        return n

    @property
    def spacing(self) -> int | None:
        """Uniform leg spacing in months, if the legs are evenly spaced."""
        if self.n_legs < 2:
            return None
        spacings = {
            self.legs[i + 1][0] - self.legs[i][0] for i in range(self.n_legs - 1)
        }
        return spacings.pop() if len(spacings) == 1 else None

    @property
    def kind(self) -> str:
        """Human-readable family name derived from the weights."""
        if self.is_empty:
            return "empty"
        match self.order:
            case 0:
                return "outright"
            case 1:
                return "spread"
            case 2:
                return "fly"
            case 3:
                return "dfly"
        if self.is_price_neutral:
            return "custom_neutral"
        return "custom"

    # ---- naming ---------------------------------------------------------
    @property
    def structure_id(self) -> str:
        """Stable, filesystem-safe identifier, e.g. ``FLY_Oct26_1m``."""
        if self.is_empty:
            return "EMPTY"
        order = self.order
        if order is not None and self.legs[0][1] > 0:
            spacing = self.spacing
            tag = {0: "OUT", 1: "SPR", 2: "FLY", 3: "DFLY"}[order]
            if order == 0:
                return f"OUT_{self.front.code}"
            unit = "" if self.legs[0][1] == 1.0 else f"x{self.legs[0][1]:g}_"
            return f"{tag}_{unit}{self.front.code}_{spacing}m"
        body = "_".join(f"{w:+g}{c.code}" for c, w in self.legs)
        return f"CUS_{body}".replace("+", "p").replace("-", "m").replace(".", "d")

    @property
    def display(self) -> str:
        """Algebraic rendering, e.g. ``Oct26 - 2*Nov26 + Dec26``."""
        if self.is_empty:
            return "<empty>"
        parts: list[str] = []
        for i, (c, w) in enumerate(self.legs):
            sign = "-" if w < 0 else ("+" if i else "")
            mag = abs(w)
            coeff = "" if abs(mag - 1.0) < WEIGHT_TOL else f"{mag:g}*"
            token = f"{coeff}{c.code}"
            parts.append(f"{sign} {token}" if i else f"{sign}{token}")
        return " ".join(parts)

    def __str__(self) -> str:
        return self.display

    def __repr__(self) -> str:
        return f"Structure({self.display})"


# ---- named constructors -------------------------------------------------
def outright(c: Contract) -> Structure:
    return Structure.from_weights({c: 1.0})


def spread(near: Contract, far: Contract) -> Structure:
    """``near - far``: the market convention for a calendar spread."""
    return Structure.from_weights({near: 1.0, far: -1.0})


def fly(a: Contract, b: Contract, c: Contract) -> Structure:
    """``a - 2b + c``: a 1:2:1 butterfly."""
    return Structure.from_weights({a: 1.0, b: -2.0, c: 1.0})


def dfly(a: Contract, b: Contract, c: Contract, d: Contract) -> Structure:
    """``a - 3b + 3c - d``: a double fly (fly of flies)."""
    return Structure.from_weights({a: 1.0, b: -3.0, c: 3.0, d: -1.0})


def condor(a: Contract, b: Contract, c: Contract, d: Contract) -> Structure:
    """``a - b - c + d``: a condor / spread-of-spreads with a gap."""
    return Structure.from_weights({a: 1.0, b: -1.0, c: -1.0, d: 1.0})


def ratio_fly(a: Contract, b: Contract, c: Contract, ratio: tuple[float, float] = (1.0, 2.0)) -> Structure:
    """A non-1:2:1 fly, e.g. the desk's ``(1:3)`` flies: ``x*a - (x+y)*b + y*c``."""
    x, y = ratio
    return Structure.from_weights({a: x, b: -(x + y), c: y})


def spread_from(front: Contract, spacing_months: int) -> Structure:
    """``front - (front + spacing)``, e.g. the "Dec26 3mo" spread."""
    return spread(front, front.plus_months(spacing_months))


def fly_from(front: Contract, spacing_months: int) -> Structure:
    """The evenly spaced fly off ``front``, e.g. "Oct26 1mo fly"."""
    return fly(front, front.plus_months(spacing_months), front.plus_months(2 * spacing_months))


def dfly_from(front: Contract, spacing_months: int) -> Structure:
    """The evenly spaced double fly off ``front``."""
    s = spacing_months
    return dfly(
        front,
        front.plus_months(s),
        front.plus_months(2 * s),
        front.plus_months(3 * s),
    )
