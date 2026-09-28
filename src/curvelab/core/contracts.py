"""Brent futures contracts: identity, ordering on the curve, and parsing.

A Contract is just a (year, month) delivery slot. Every structure is built out
of these, so it's frozen, hashable and ordered by expiry. The distance between
two contracts on the curve is the difference of their month ordinals.

The data uses three notations, and Contract.parse accepts all of them:

    "Oct26" / "oct 26" / "oct'26" / "October 2026"   month-name form (TT exports,
                                                     trader journals)
    "BV26" / "V26"                                   ICE month-code form
                                                     (B = Brent, V = October)
    "BRN Oct26"                                      TT symbol prefix form
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering

MONTH_ABBRS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
MONTH_TO_NUM = {abbr: i + 1 for i, abbr in enumerate(MONTH_ABBRS)}
NUM_TO_MONTH = {i + 1: abbr.capitalize() for i, abbr in enumerate(MONTH_ABBRS)}

#: ICE/CME delivery-month letter codes. Jan=F ... Dec=Z.
ICE_MONTH_CODES = {
    "f": 1, "g": 2, "h": 3, "j": 4, "k": 5, "m": 6,
    "n": 7, "q": 8, "u": 9, "v": 10, "x": 11, "z": 12,
}
NUM_TO_ICE = {v: k.upper() for k, v in ICE_MONTH_CODES.items()}

# "oct26", "oct 26", "oct'26", "oct-26", "october 2026"
_NAME_RE = re.compile(
    r"(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"\s*['’\-_]?\s*"
    r"(?P<year>\d{2,4})",
    re.IGNORECASE,
)
# "BZ26", "BN27", "Z26" -- the leading B is the ICE Brent product letter.
_CODE_RE = re.compile(
    r"\bb?(?P<code>[fghjkmnquvxz])\s*['’]?\s*(?P<year>\d{2,4})\b",
    re.IGNORECASE,
)

#: Two-digit years below this pivot roll into the 2100s; above, into the 2000s.
_YEAR_PIVOT = 70


def _normalize_year(raw: str) -> int:
    year = int(raw)
    if year >= 1000:
        return year
    if year < _YEAR_PIVOT:
        return 2000 + year
    return 1900 + year


class ContractParseError(ValueError):
    """Raised when a string cannot be resolved to a single contract."""


@total_ordering
@dataclass(frozen=True, slots=True)
class Contract:
    """A single Brent delivery month, e.g. ``Contract(2026, 10)`` -> ``Oct26``."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month must be in 1..12, got {self.month}")
        if not 1900 <= self.year <= 2200:
            raise ValueError(f"year out of supported range: {self.year}")

    # ---- identity -------------------------------------------------------
    @property
    def code(self) -> str:
        """Canonical short code used everywhere in this project: ``Oct26``."""
        return f"{NUM_TO_MONTH[self.month]}{self.year % 100:02d}"

    @property
    def ice_code(self) -> str:
        """ICE-style code: ``BV26`` for October 2026 Brent."""
        return f"B{NUM_TO_ICE[self.month]}{self.year % 100:02d}"

    @property
    def ordinal(self) -> int:
        """Months since year 0. Subtracting two of these gives the tenor gap."""
        return self.year * 12 + (self.month - 1)

    # ---- ordering -------------------------------------------------------
    def __lt__(self, other: Contract) -> bool:
        if not isinstance(other, Contract):
            return NotImplemented
        return self.ordinal < other.ordinal

    def __sub__(self, other: Contract) -> int:
        """Curve distance in months (``Dec26 - Oct26 == 2``)."""
        if not isinstance(other, Contract):
            return NotImplemented
        return self.ordinal - other.ordinal

    def plus_months(self, n: int) -> Contract:
        """The contract ``n`` months further out (or back, if negative)."""
        total = self.ordinal + n
        return Contract(year=total // 12, month=total % 12 + 1)

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"Contract({self.code})"

    # ---- parsing --------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> Contract:
        """Parse a single contract from free text, raising if none is found."""
        found = cls.find_all(text)
        if not found:
            raise ContractParseError(f"no contract found in {text!r}")
        if len(found) > 1:
            raise ContractParseError(
                f"{len(found)} contracts found in {text!r}; use find_all()"
            )
        return found[0][0]

    @classmethod
    def try_parse(cls, text: str) -> Contract | None:
        try:
            return cls.parse(text)
        except (ContractParseError, ValueError):
            return None

    @classmethod
    def find_all(cls, text: str) -> list[tuple[Contract, int, int]]:
        """Find every contract mentioned in the text, as (contract, start, end).

        Month names are matched first and blanked out before looking for ICE
        codes. Otherwise "BRN Oct26" would also produce a bogus N26 from the
        N at the end of BRN.
        """
        spans: list[tuple[Contract, int, int]] = []
        masked = list(text)

        for m in _NAME_RE.finditer(text):
            month = MONTH_TO_NUM[m.group("mon").lower()]
            try:
                contract = cls(year=_normalize_year(m.group("year")), month=month)
            except ValueError:
                continue
            spans.append((contract, m.start(), m.end()))
            for i in range(m.start(), m.end()):
                masked[i] = " "

        remaining = "".join(masked)
        for m in _CODE_RE.finditer(remaining):
            month = ICE_MONTH_CODES[m.group("code").lower()]
            try:
                contract = cls(year=_normalize_year(m.group("year")), month=month)
            except ValueError:
                continue
            spans.append((contract, m.start(), m.end()))

        spans.sort(key=lambda t: t[1])
        return spans


def contract_range(start: Contract, end: Contract) -> list[Contract]:
    """Every monthly contract from ``start`` to ``end`` inclusive."""
    if end < start:
        start, end = end, start
    return [start.plus_months(i) for i in range(end - start + 1)]
