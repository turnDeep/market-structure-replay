# NQ selected-date model summary — 2026-10-16

## Defect and scope

Nikkei already had a selected-date result panel; the authored NQ view did not.
Added the NQ panel at the same position below the decision controls. This is a
presentation/accounting repair, not a new strategy or an edge validation.

- Population: all saved fills from the first collected bar through the selected
  bar close, independent of the candle window or the selected evidence case.
- NQ: one contract, USD 20/point, saved fills include 0.25-point slippage.
- Realized net: known SELL/STOP only, USD 5 round-trip commission included.
- Open P&L: selected close minus actual entry fill, USD 2.50 entry fee deducted;
  the not-yet-incurred exit fee is not deducted. Never use an eventual exit.
- Win rate: positive closes / all closes; breakeven is in the denominator.
- Concentration: largest positive known close, and cumulative realized net
  minus that single close. This is not an account return or drawdown statistic.

## Reconciliation

`node scripts/verify_nq_summary.mjs --saved-mtm <daily_mtm.jsonl>` passed
72,171 checks across all 6,560 daily cutoffs. Each cutoff matched independently
recomputed fill-price accounting and the existing saved daily MTM ledger.
Full-input and selected-date-truncated-input summaries matched at every cutoff.
Small fixtures also cover pending orders, future invalid rows, entry fees,
breakeven and explicit rejection of state/realized-net inconsistencies.
The saved machine result is [nq-summary-20261016.json](nq-summary-20261016.json).
Its final 2026-10-15 observation remains provisional under the unchanged manifest
quality warning; this QA does not certify Yahoo's last bar as completed.

All six reviewed query row arrays are unchanged. Nikkei's compressed HTML,
its manifest item, strategy parameters, cloud inputs and daily schedule are
unchanged. Only NQ's compiled UI and scoped metric metadata were repackaged.
Local Python suite: 32 tests passed.

## Actual browser checks

Tested the packaged HTML through local Streamlit, not only the source tree.

| Selected date | Checked behavior |
|---|---|
| 2000-09-18 | No fills: realized zero, no position, undefined win rate |
| 2026-02-05 | Pending SELL does not realize tomorrow's exit |
| 2026-02-06 | Executed SELL included; open P&L becomes no position |
| 2026-02-07 | Weekend selection uses preceding observed 2026-02-06 |
| 2026-03-31 | Pending BUY does not create a position |
| 2026-04-01 | Executed BUY produces entry-fee-adjusted open P&L |
| 2026-10-14 | Realized USD 449,245; 43 closed / 16 wins / 27 losses; open USD 125,867.50 |

Switching the chart window between 60 and 120 bars did not change the summary.
Desktop and 430px-wide screenshots were inspected; the new summary becomes one
column on narrow screens and has no horizontal overflow (299px content width
and 299px scroll width). The source drawer showed the component-specific metric
definitions. An actual JSON download included the matching `model_summary`
with cutoff 2026-10-14 and no later fills. The chart and summary rendered without
browser console errors. Existing runtime warnings are outside this repair.

Cloud Streamlit rendering and authentication have not been retested here;
local browser verification does not prove a deployed page has refreshed.
