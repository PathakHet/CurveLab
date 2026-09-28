"""Turn free-text strategy names from the journals into Structures.

Traders wrote the same instrument down in hundreds of different ways. All of
these mean Oct26 - 2*Nov26 + Dec26:

    "oct26 1mo fly"          "oct_1mo_fly"        "OctNovDec"
    "oct26 1mo butterfly"    "oct 26 1 mon fly"   "BRN Oct26 1mo fly(long)"
    "(oct26 - 2*nov26 + dec26) ( oct26 1month fly)"

Instead of piling up special cases, I treat the names as a tiny expression
language and parse them with a small recursive-descent parser:

    expr    := term (('+' | '-' | 'vs') term)*
    term    := factor ('*' factor)*
    factor  := NUMBER | STRUCT | '(' expr ')' | '-' factor
    STRUCT  := explicit contract, or a shorthand phrase such as
               "<contract> <spacing>mo fly", or a run of concatenated month
               abbreviations such as "OctNovDec"

Two details matter:

Glosses. When two expressions sit next to each other with no operator, as in
"(oct26 - 2*nov26 + dec26) (oct26 1month fly)", the second is a description of
the first, not something to multiply by. The parser stops there, parses the
second part on its own, and records whether the two agree. If they don't,
that's a data problem and it shows up in the report.

Ambiguity. Some names really are ambiguous: "dec26 3mo" could be a spread or a
fly, and a "(1:3)" fly can be weighted more than one way. The parser returns
every possible reading instead of guessing, and desk/disambiguate.py picks
between them using market prices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from curvelab.core.contracts import ICE_MONTH_CODES, MONTH_ABBRS, MONTH_TO_NUM, Contract
from curvelab.core.structure import (
    Structure,
    dfly_from,
    fly_from,
    outright,
    spread_from,
)

#: Names without a year are taken to mean the first matching month on or
#: after this (year, month).
DEFAULT_REF = (2026, 6)

_NOISE = re.compile(r"\b(brn|ice|bfoe|dated|fut|future|futures)\b", re.IGNORECASE)
_LONG = re.compile(r"\b(long|buy|bought|bot)\b", re.IGNORECASE)
_SHORT = re.compile(r"\b(short|sell|sold|sld)\b", re.IGNORECASE)

_TYPE_WORDS = {
    "fly": "fly",
    "flys": "fly",
    "flies": "fly",
    "butterfly": "fly",
    "butterflies": "fly",
    "bfly": "fly",
    "dfly": "dfly",
    "dblfly": "dfly",
    "doublefly": "dfly",
    "spread": "spread",
    "spreads": "spread",
    "sprd": "spread",
    "spr": "spread",
    "outright": "outright",
    "flat": "outright",
}

_SPACING_UNITS = r"(?:months?|mths?|mons?|mos?|m)"


class NameParseError(ValueError):
    """Raised when a strategy name yields no interpretable structure."""


@dataclass
class ParseResult:
    """Outcome of parsing one strategy-name string."""

    raw: str
    structure: Structure | None
    direction: int = 0          # +1 long, -1 short, 0 unstated
    candidates: list[Structure] = field(default_factory=list)
    rule: str = ""              # which parse path produced the result
    gloss: Structure | None = None
    gloss_agrees: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.structure is not None and not self.structure.is_empty

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, unify separators and spell out operators."""
    s = str(text).strip().lower()
    s = s.replace("’", "'").replace("‘", "'")
    # Unicode minus / en / em dashes are typed interchangeably with ASCII "-".
    for dash in ("\u2212", "\u2013", "\u2014", "\u2012"):
        s = s.replace(dash, "-")
    s = s.replace(":", " ") if not re.search(r"\d\s*:\s*\d", s) else s
    s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\bversus\b|\bvs\.?\b", " vs ", s)
    s = re.sub(r"\bminus\b", " - ", s)
    s = re.sub(r"\bplus\b", " + ", s)
    s = _NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_direction(text: str) -> tuple[str, int]:
    """Strip long/short words, returning ``(remaining_text, direction)``."""
    direction = 0
    if _SHORT.search(text):
        direction = -1
    elif _LONG.search(text):
        direction = +1
    cleaned = _SHORT.sub(" ", _LONG.sub(" ", text))
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Removing the direction word from "oct26 1mo fly- short" or
    # "long - 2*mar27 3mo fly" leaves a stray dash. It was a separator, not
    # a minus sign, so strip it.
    cleaned = cleaned.strip(" -+*/,;.")
    return cleaned, direction


def _resolve_year(month: int, ref: tuple[int, int]) -> Contract:
    """First delivery of ``month`` on or after the reference (year, month)."""
    ref_year, ref_month = ref
    year = ref_year if month >= ref_month else ref_year + 1
    return Contract(year=year, month=month)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
@dataclass
class Token:
    kind: str                       # NUM | STRUCT | OP | LPAREN | RPAREN
    text: str
    value: float | None = None
    structures: list[Structure] = field(default_factory=list)
    rule: str = ""


def _shorthand_at(s: str, pos: int, ref: tuple[int, int]) -> tuple[Token, int] | None:
    """Match a shorthand structure phrase starting at ``pos``.

    Handles ``"oct26 1mo fly"``, ``"dec'26 6mth fly (1:2)"``, ``"oct 2m"``,
    ``"nov26 fly"`` and bare contracts. Returns the token and the new position.
    """
    m = re.compile(
        r"""
        (?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*
        \s*'?\s*(?P<year>\d{2,4})?
        (?:\s*(?P<spacing>\d+)\s*(?:%s))?
        (?:\s*\(\s*(?P<rx>\d+)\s*:\s*(?P<ry>\d+)\s*\))?
        (?:\s*(?P<type>butterfly|butterflies|bfly|dblfly|doublefly|dfly|flies|flys|fly|spreads|spread|sprd|spr|outright|flat))?
        (?:\s*\(\s*(?P<rx2>\d+)\s*:\s*(?P<ry2>\d+)\s*\))?
        """ % _SPACING_UNITS,
        re.IGNORECASE | re.VERBOSE,
    ).match(s, pos)
    if not m:
        return None

    month = MONTH_TO_NUM[m.group("mon")[:3]]
    if m.group("year"):
        yr = int(m.group("year"))
        yr = yr + 2000 if yr < 70 else (yr + 1900 if yr < 1000 else yr)
        front = Contract(year=yr, month=month)
    else:
        front = _resolve_year(month, ref)

    spacing = int(m.group("spacing")) if m.group("spacing") else None
    stype = _TYPE_WORDS.get(m.group("type")) if m.group("type") else None
    ratio = m.group("rx") or m.group("rx2")

    candidates: list[Structure] = []
    rule = "shorthand"
    if stype == "fly":
        candidates = [fly_from(front, spacing or 1)]
        if ratio and (int(m.group("ry") or m.group("ry2")), int(ratio)) != (2, 1):
            # A non-standard "(x:y)" ratio. Keep the normal 1:2:1 fly as the
            # first guess and add the ratio version as an alternative for the
            # price-based check to choose between.
            x, y = int(ratio), int(m.group("ry") or m.group("ry2"))
            s_ = spacing or 1
            candidates.append(
                Structure.from_weights(
                    {
                        front: float(x),
                        front.plus_months(s_): -float(x + y - 1),
                        front.plus_months(2 * s_): float(y - 1),
                    }
                )
            )
            rule = "shorthand_ratio_fly"
    elif stype == "dfly":
        candidates = [dfly_from(front, spacing or 1)]
    elif stype == "spread":
        candidates = [spread_from(front, spacing or 1)]
    elif stype == "outright":
        candidates = [outright(front)]
    elif spacing is not None:
        # "dec26 3mo" with no type word. On the desk this usually meant a
        # spread, but it could be a fly, so return both and let prices decide.
        candidates = [spread_from(front, spacing), fly_from(front, spacing)]
        rule = "shorthand_untyped"
    else:
        candidates = [outright(front)]
        rule = "bare_contract"

    return Token("STRUCT", m.group(0), structures=candidates, rule=rule), m.end()


def _month_run_at(s: str, pos: int, ref: tuple[int, int]) -> tuple[Token, int] | None:
    """Match a run of concatenated month abbreviations: ``OctNovDec``.

    Three evenly spaced months mean a fly on those legs, two mean a spread,
    four a double fly. If the months aren't evenly spaced it's rejected.
    """
    pat = re.compile(r"(?:%s)" % "|".join(MONTH_ABBRS), re.IGNORECASE)
    months: list[int] = []
    p = pos
    while True:
        m = pat.match(s, p)
        if not m:
            break
        months.append(MONTH_TO_NUM[m.group(0).lower()])
        p = m.end()
        while p < len(s) and s[p] in " -_":
            p += 1
    if len(months) < 2:
        return None

    contracts: list[Contract] = []
    cursor = ref
    for mth in months:
        c = _resolve_year(mth, cursor)
        contracts.append(c)
        cursor = (c.year, c.month + 1) if c.month < 12 else (c.year + 1, 1)

    spacings = {contracts[i + 1] - contracts[i] for i in range(len(contracts) - 1)}
    if len(spacings) != 1:
        return None
    spacing = spacings.pop()

    if len(contracts) == 3:
        struct = fly_from(contracts[0], spacing)
    elif len(contracts) == 2:
        struct = spread_from(contracts[0], spacing)
    elif len(contracts) == 4:
        struct = dfly_from(contracts[0], spacing)
    else:
        return None
    return Token("STRUCT", s[pos:p], structures=[struct], rule="month_run"), p


NUM_TO_ABBR = {i + 1: a[0] for i, a in enumerate(MONTH_ABBRS)}

#: Month initials used in shorthand like "djd 6mo fly" (dec-jun-dec).
_INITIAL_TO_MONTHS: dict[str, list[int]] = {}
for _i, _abbr in enumerate(MONTH_ABBRS):
    _INITIAL_TO_MONTHS.setdefault(_abbr[0], []).append(_i + 1)


def _ice_at(s: str, pos: int, ref: tuple[int, int]) -> tuple[Token, int] | None:
    """Match an ICE month-code contract such as ``BZ26`` (Brent Dec 2026)."""
    m = re.compile(
        r"b?(?P<code>[fghjkmnquvxz])\s*'?\s*(?P<year>\d{2,4})", re.IGNORECASE
    ).match(s, pos)
    if not m:
        return None
    yr = int(m.group("year"))
    yr = yr + 2000 if yr < 70 else (yr + 1900 if yr < 1000 else yr)
    contract = Contract(year=yr, month=ICE_MONTH_CODES[m.group("code").lower()])
    return Token("STRUCT", m.group(0), structures=[outright(contract)], rule="ice_code"), m.end()


def _initials_at(s: str, pos: int, ref: tuple[int, int]) -> tuple[Token, int] | None:
    """Match month-initial shorthand: ``"djd 6mo fly"`` -> Dec26 - 2*Jun27 + Dec27.

    A single initial doesn't mean much ("j" could be Jan, Jun or Jul), so this
    only accepts readings where every leg's initial fits the spacing.
    """
    m = re.compile(
        r"(?P<init>[jfmasond]{2,4})"
        r"(?:\s*(?P<spacing>\d+)\s*(?:%s))?"
        r"\s*(?P<type>butterfly|bfly|dfly|flies|flys|fly|spread|sprd)?" % _SPACING_UNITS,
        re.IGNORECASE,
    ).match(s, pos)
    if not m:
        return None
    letters = m.group("init").lower()
    if not all(ch in _INITIAL_TO_MONTHS for ch in letters):
        return None
    if m.group("spacing"):
        spacings = [int(m.group("spacing"))]
    else:
        # No spacing given, so try the common ones and keep whatever fits
        # (e.g. "ddd" only works as a 12-month fly). Only do this when the
        # initials are the whole name, otherwise it matches too eagerly.
        if m.end() != len(s) and not m.group("type"):
            return None
        spacings = [1, 2, 3, 6, 12]

    consistent: list[Structure] = []
    for spacing in spacings:
      for first_month in _INITIAL_TO_MONTHS[letters[0]]:
        front = _resolve_year(first_month, ref)
        legs = [front.plus_months(i * spacing) for i in range(len(letters))]
        if all(NUM_TO_ABBR[c.month] == ch for c, ch in zip(legs, letters)):
            if len(letters) == 3:
                consistent.append(fly_from(front, spacing))
            elif len(letters) == 2:
                consistent.append(spread_from(front, spacing))
            elif len(letters) == 4:
                consistent.append(dfly_from(front, spacing))
    consistent = list(dict.fromkeys(consistent))
    if not consistent:
        return None
    return Token("STRUCT", m.group(0), structures=consistent, rule="initials"), m.end()


def tokenize(s: str, ref: tuple[int, int]) -> list[Token]:
    """Split a normalized name into structure, number and operator tokens."""
    tokens: list[Token] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(Token("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("RPAREN", ch))
            i += 1
            continue
        if ch in "+-*/":
            tokens.append(Token("OP", "-" if ch == "-" else ch))
            i += 1
            continue
        if s.startswith("vs", i) and (i + 2 >= n or not s[i + 2].isalnum()):
            tokens.append(Token("OP", "-"))
            i += 2
            continue
        m = re.compile(r"\d+(?:\.\d+)?").match(s, i)
        if m and not _shorthand_at(s, i, ref):
            tokens.append(Token("NUM", m.group(0), value=float(m.group(0))))
            i = m.end()
            continue
        hit = _shorthand_at(s, i, ref)
        if hit:
            token, end = hit
            # "OctNovDec" should be read as a fly, not as just "Oct".
            run = _month_run_at(s, i, ref)
            if run and run[1] >= end and len(run[0].structures) > 0:
                tokens.append(run[0])
                i = run[1]
            else:
                tokens.append(token)
                i = end
            continue
        for matcher in (_month_run_at, _initials_at, _ice_at):
            hit = matcher(s, i, ref)
            if hit:
                tokens.append(hit[0])
                i = hit[1]
                break
        else:
            i += 1
            if not (tokens and tokens[-1].kind == "SKIP"):
                tokens.append(Token("SKIP", ch))
        continue
    return [t for t in tokens if t.kind != "SKIP"]


# ---------------------------------------------------------------------------
# recursive-descent parser
# ---------------------------------------------------------------------------
class _Parser:
    """Parses the token stream into a Structure, stopping at a juxtaposition."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.rules: list[str] = []

    def peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_expr(self) -> Structure:
        left = self.parse_term()
        while (tok := self.peek()) is not None and tok.kind == "OP" and tok.text in "+-":
            mark = self.pos
            op = self.next().text
            try:
                right = self.parse_term()
            except (NameParseError, IndexError):
                self.pos = mark  # dangling trailing operator: end the expression
                break
            left = left + right if op == "+" else left - right
        return left

    def parse_term(self) -> Structure:
        scale = 1.0
        structure: Structure | None = None
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == "NUM":
                self.next()
                scale *= tok.value or 1.0
                if (nxt := self.peek()) is not None and nxt.kind == "OP" and nxt.text == "*":
                    self.next()
                continue
            if tok.kind in ("STRUCT", "LPAREN") or (tok.kind == "OP" and tok.text == "-"):
                if structure is not None:
                    break  # juxtaposition -> this is a gloss, stop here
                structure = self.parse_factor()
                if (nxt := self.peek()) is not None and nxt.kind == "OP" and nxt.text == "*":
                    self.next()
                    continue
                continue
            break
        if structure is None:
            raise NameParseError("expected a structure")
        return structure * scale

    def parse_factor(self) -> Structure:
        tok = self.peek()
        if tok is None:
            raise NameParseError("unexpected end of input")
        if tok.kind == "OP" and tok.text == "-":
            self.next()
            return -self.parse_factor()
        if tok.kind == "LPAREN":
            self.next()
            # Empty or purely descriptive brackets like "(hedged)" have no
            # legs in them, so skip ahead to the next factor.
            if (nxt := self.peek()) is not None and nxt.kind == "RPAREN":
                self.next()
                return self.parse_factor()
            inner = self.parse_expr()
            if (nxt := self.peek()) is not None and nxt.kind == "RPAREN":
                self.next()
            return inner
        if tok.kind == "STRUCT":
            self.next()
            self.rules.append(tok.rule)
            return tok.structures[0]
        raise NameParseError(f"unexpected token {tok.kind}:{tok.text!r}")


def parse_strategy_name(text: str, ref: tuple[int, int] = DEFAULT_REF) -> ParseResult:
    """Parse one free-text strategy name into a :class:`ParseResult`."""
    raw = str(text)
    cleaned, direction = extract_direction(normalize(raw))
    if not cleaned:
        return ParseResult(raw=raw, structure=None, direction=direction, error="empty after cleaning")

    tokens = tokenize(cleaned, ref)
    if not any(t.kind == "STRUCT" for t in tokens):
        return ParseResult(raw=raw, structure=None, direction=direction, error="no contract found")

    parser = _Parser(tokens)
    try:
        structure = parser.parse_expr()
    except (NameParseError, IndexError) as exc:
        return ParseResult(raw=raw, structure=None, direction=direction, error=str(exc))

    # Whatever's left is a gloss. Parse it and see if it agrees.
    gloss: Structure | None = None
    gloss_agrees: bool | None = None
    if parser.pos < len(tokens):
        tail = _Parser(tokens[parser.pos:])
        try:
            gloss = tail.parse_expr()
            gloss_agrees = gloss == structure
        except (NameParseError, IndexError):
            gloss = None

    # Only keep multiple candidates if the whole name was one ambiguous token.
    struct_tokens = [t for t in tokens if t.kind == "STRUCT"]
    candidates = [structure]
    if len(struct_tokens) == 1 and len(struct_tokens[0].structures) > 1:
        candidates = list(struct_tokens[0].structures)

    consumed = [t for t in tokens[: parser.pos] if t.kind == "STRUCT"]
    if len(consumed) > 1:
        rule = "algebra"
    elif consumed:
        rule = consumed[0].rule
    else:
        rule = "expr"

    return ParseResult(
        raw=raw,
        structure=structure,
        direction=direction,
        candidates=candidates,
        rule=rule,
        gloss=gloss,
        gloss_agrees=gloss_agrees,
    )
