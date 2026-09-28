# CurveLab

A two-month internship project on a Brent crude futures curve desk, turned into
a small research codebase.

The desk trades curve *structures* — calendar spreads and butterflies across
delivery months, rather than the oil price itself. Interns traded them and kept
hand-written journals. Afterwards I used those journals, plus the hourly market
data behind them, to answer three questions I could not answer while trading:

1. **What should a trader actually be watching?** Which signals carry
   information about where a fly goes next, and are any of them worth the spread?
2. **Where did the desk's P&L come from** — genuine calls, or curve exposure?
3. **Was anyone measurably good**, or was it a two-month coin-flipping contest?

```bash
make run      # runs the pipeline on the synthetic sample data, writes reports/REPORT.md
make test     # 63 tests
make bench    # C++ engine latency and throughput
```

**Inputs:** hourly OHLCV for 16 Brent delivery months (May–Jun 2026, 949 bars);
**3,315 real trades from 43 traders** across 45 Excel sheets.

---

## 1. What should a trader watch? (`research/feature_study.py`)

The main question, and the reason the rest exists. Each candidate signal is
scored by its **information coefficient**: at every hour, rank the flies on the
board by the signal, rank them by what they actually did next, take the rank
correlation. That's one number per hour; its mean is the signal's edge and its
t-statistic says whether the edge is real. Ranking across the board rather than
pooling asks the question a curve trader faces — *which of these is the best
trade right now* — and is unaffected by the whole curve drifting.

Three filters, because the first two ways of being fooled here are easy:

- **Significance** — Newey-West t-statistic, since overlapping forward windows
  make consecutive ICs share most of their information.
- **Delay** — every IC is computed twice, once on the return from the current
  bar's close and once starting from the *next* bar. Only the second is
  capturable; you can't trade the print that produced your signal.
- **Staleness** — a near-constant signal (days to the front leg's expiry, say)
  ranks the board the same way every hour. Its t-statistic can look excellent
  while representing one bet on curve shape repeated 900 times.

> **22 of 75 signal-horizon pairs survive all three.** The strongest is the
> last hour's move, predicting the next 3 hours (mean IC −0.041, t = −3.4).
> Five pairs were excluded as stale, including the best-looking one.
>
> The effect is **monotone across all five quintiles** (rank correlation −1.00
> between signal bucket and subsequent move), so it behaves like a factor rather
> than a threshold rule:

| last hour's move | n | mean move next 3h | % up |
|---|---|---|---|
| bottom quintile | 3,724 | **+0.008** | 49.9% |
| 2nd | 3,688 | −0.005 | 48.3% |
| 3rd | 3,626 | −0.008 | 47.1% |
| 4th | 3,707 | −0.009 | 46.2% |
| top quintile | 3,610 | **−0.021** | 46.2% |

**And then the part that matters.** That extreme bucket predicts a $0.021/bbl
move. One unit of a fly moves four contracts, so a round trip costs $0.060/bbl.
The signal is **0.35× its own transaction cost** — real, statistically solid,
monotone, and not worth trading. Only about 10% of its apparent strength
survives the one-bar delay; the other 90% was bid-ask bounce.

That is the useful answer for the desk: *short-horizon fly reversion is real but
smaller than the spread, so discretionary traders should not expect to earn it
by reacting quickly to hourly moves.*

A cross-sectional gradient-boosting ranker (top-K/bottom-K, purged and embargoed
walk-forward folds) reaches the same conclusion from the other direction — out
of sample it makes $26 and does not survive the delay test either.

## 2. Where did the desk's P&L come from? (`desk/attribution.py`)

Daily P&L per trader, regressed on PCA term-structure factors with Newey-West
standard errors. Level, slope and curvature explain **97.8% / 1.6% / 0.3%** of
daily curve variance.

> Those factors explain only **2.2%** of the desk's daily P&L variance. That is
> the expected signature of a book built from flies, which are constructed to be
> neutral to level and slope — measured here rather than assumed.

## 3. Was anyone good? (`desk/skill.py`)

With 43 traders, testing each at 5% guarantees false positives. Three
corrections: a stationary bootstrap for serial dependence, Benjamini-Hochberg
FDR control for multiplicity, and a max-statistic test for selection.

> Of 15 traders with enough dated trades, 4 clear an uncorrected p < 0.05 and
> **3 survive FDR control**. But the best t-statistic is 5.32, and under the null
> that nobody has skill, the best of 15 traders beats that **12% of the time** —
> the top performer is not distinguishable from the luckiest of fifteen.

## 4. Getting the journals into a usable state (`desk/`)

Most of the work, and none of it glamorous. The journals were written by hand,
at speed, during the session.

**710 spellings, one instrument.** All of these mean `Oct26 - 2*Nov26 + Dec26`:

```
oct26 1mo fly    OctNovDec    oct_1mo_fly    djd 6mo fly    (BZ26-BN27)-2*(BN27-BZ27)
```

A tokeniser and recursive-descent parser over a small expression grammar handles
ICE month codes, concatenated abbreviations and algebraic notation, reaching
**99.8% coverage**. Two decisions mattered: juxtaposed expressions are treated as
a value and its gloss and cross-checked, not multiplied; and genuinely ambiguous
names (`"dec26 3mo"` is a spread *or* a fly, and the desk traded both) emit every
candidate, resolved against observed prices — the spread traded near $3.45, the
fly near $0.28, so a reported entry of 2.19 is unambiguous. **251 resolved this
way; 12 that couldn't be separated were left unassigned.**

**A date bug worth catching.** The journals were typed day-first and read back
month-first, so `12/06/2026` became 6 December — six months from the data it
needed to be marked against. It's self-diagnosing: a date whose day exceeds 12
can't be a month, so those survived intact and pin down the real window.
**253 dates (19%) recovered; usable dated trades went from 573 to 1,377.**

**Reconciliation.** `pnl = (exit − entry) × lots × multiplier`, with the
multiplier recovered from the data rather than assumed — it comes out at exactly
**1,000**, the ICE Brent contract size. **82.9%** reconcile to within $1, a
further **8.3%** only after a sign flip (transposed entry/exit), and the
remaining 8.7% are excluded from every downstream inference.

**A useful check for free:** a structure's attainable daily range follows from
its legs' highs and lows, so a fill outside it is impossible. **95.5% of entry
fills land inside** — which independently confirms the parser assigned the right
instrument, since a misidentified structure would price nowhere near the fill.

## 5. Supporting pieces

**Structure algebra** (`core/`) — structures compose, so they're modelled as an
algebra. `spread(a,b) - spread(b,c) == fly(a,b,c)` is literally true in code and
enforced by property-based tests.

**Pricing engine** (`engine/`) — the feature study reprices 63 structures
repeatedly, so the inner loop went to C++. A tick on one contract only affects
the ~16 structures containing it, so a CSR index updates just those by delta
rather than rebuilding the matrix: **109 ns p50 per update, 369× the NumPy
path**. It's a research-loop speed-up, not a trading system — the data is hourly
bars. A differential test proves tick-for-tick parity with NumPy to 1e-9, and it
earned its keep: the naive O(1) rolling variance (`sum`/`sumsq`, subtract to
evict) drifted 3e-6 through catastrophic cancellation, worst exactly when a
window goes quiet and a mean-reversion signal is about to fire. Replaced with
Welford add/remove plus a periodic exact refresh.

---

## Data and privacy

The real inputs — a Trading Technologies subscription export and a live desk's
book — are not in this repository. Trader identities are replaced with codes and
the mapping is written outside the repo. `data/sample/` ships synthetic data
generated from the same three-factor curve structure and seeded with the same
defects (day-first dates, transposed prices, many spellings per instrument), so
`make run` exercises every recovery path.

To run on your own data, copy `configs/example.yaml`, point it at your files,
and run `curvelab run --config <your config>`.

## Limitations

Two months of hourly bars, 34 trading days with marked trades, 15 traders with
enough history to test. Every Sharpe here is indicative and none is a claim about
live profitability. Journal timestamps are date-only, so execution quality is
measured as placement within the day's range, not implementation shortfall.
Costs are a modelled assumption — there is no bid-ask data in the sample.

## Layout

```
src/curvelab/
  core/       contracts, structure algebra, TT loader, curve book
  desk/       journal ingest, name parser, date resolver, disambiguation,
              reconciliation, factor attribution, skill testing
  research/   features, feature study, backtest, strategies, walk-forward
  engine/     C++ incremental pricer + pybind11 bindings
tests/        63 tests: property-based algebra, C++/NumPy parity, end-to-end
```
