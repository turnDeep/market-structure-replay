# NQ / ES switch — 2026-10-19

## Scope and provenance

Active Streamlit markets and the existing 09:00 JST GitHub daily updater now
use NQ=F and ES=F. No new schedule, strategy optimization, orders, sharing or
authentication changes. v0.12 remains radius-one alpha, NOT_VALIDATED.

NQ's compressed HTML is byte-identical to the previous generation (SHA-256
`23084efcef8306de362ea7328164edd9d03cad9096f743e810cb1c209009f876`).
The previous NKD view/manifest are retained under `assets/archive/nkd-v012/`;
its existing state, vendor sources and earlier research receipts are retained.
The immutable v0.12 migration receipt still describes its original NQ/NKD
release; it has not been rewritten to pretend that NKD was ES.

ES uses the saved ES study inputs and previously tested alpha configuration.
Only actual completed Yahoo observations dated October 15 and 16 were appended,
with yfinance / curl-cffi Chrome-session acquisition. Original frozen rows were
not revised. Invalid historical OHLC rows (53) remain explicitly excluded,
without interpolation. Provider current-day exclusion and the CME time boundary
are both applied. Yahoo continuous futures are not certified individual-contract
roll, settlement, or intraday-execution data.

ES: 2000-09-18 through 2026-10-16, 6,575 usable daily rows. One contract,
50 USD/point, 0.25-point one-way slippage, 2.50 USD commission per side.
NQ retains its own 20 USD/point accounting.

## Completed checks

- Full native ES replay completed with the fixed v0.12 adapter. Input hashes,
  strategy configuration and all frozen execution facts matched.
- Through 2026-10-14: all 44 closed trades (dates, prices and net USD),
  6,573 daily MTM values (absolute tolerance 1e-8) and exposure states matched
  the prior ES alpha experiment. This is reproduction, not independent evidence
  of predictive superiority.
- Initial ES view audit: 27,879 checks and 210 full-input/truncated-input
  comparisons covering actual decision cutoffs and the three display modes.
  Independent SMA200, fills, commissions, stops and evidence clocks checked.
- ES selected-date summary: 65,892 checks across all 6,575 cutoffs, including
  the saved native daily realized and open-P&L ledger. See
  `es-summary-20261019.json`. Final 44 closes / 18 wins / 26 losses; realized
  USD 263,792.50, separate open P&L after entry fee USD 34,997.50. These are
  fixed-one-contract model dollars, not account returns or liquidation results.
- NQ selected-date summary: 52,498 checks over 6,561 dates. Its unchanged
  reviewed data continue to reconcile with independent fill-based accounting;
  this check did not reload a separate saved MTM file.
- Ordinary future-refresh export paths were exercised from completed native
  ES and NQ logs: all existing reviewed query rows were reproduced exactly.
  This check did not rerun both engines from the preceding last-bar cutoff.
- Local Python/Streamlit suite: **52 passed**. Package code/input hash checks,
  active-market allowlist, NQ/ES AppTest switching, failure/no-publication
  paths, archived legacy pins and new ES prefix pins passed.

## Route-specific repairs (presentation only)

The ES 2003-10-06 BUY uses **IFVG**, not an ordinary FVG. Its actual type,
availability date, price band and R mark are displayed. `fvg_used=false`
remains truthful for this inversion-FVG case; `retest_used=true` is separate.
The original record is not transformed into a different trading decision.

The 2018-12-20 SELL is `major_protected_low_close_failure`, not H1/H2 top
failure. Its pre-existing 2,552 protection level was effective July 17, 2018;
December 18 and 19 closed below it. Evidence now renders that actual route,
using its logged activation and the existing two-close rule. Unknown protective
routes fail explicitly. The frozen strategy sources and exits were not changed.

## Rendered review and limits

Actual local Streamlit preview: `http://127.0.0.1:4190/?market=es`.
NQ/ES radio switching and both titles checked. ES summary, 2003 IFVG/OB bands,
BUY arrow, structure-protection SELL level, and its December 19 pending state
were inspected. The pending state does not expose the next day's fill or P&L.
Candles, SMA, bands, numbered anchors and arrows were visually checked in an
actual screenshot. No captured browser error logs.

390px responsive layout was also inspected: cards stack and controls wrap;
the shared information-button hit target has a small horizontal overflow
(body 333px versus scroll width 340px in that inspection). Dense annotations
are easier to inspect in landscape. This is not a pixel-perfect mobile claim.
The interactive JSON download event was attempted but the browser automation
timed out, so an actual downloaded file was **not** verified in this turn.

GitHub push/CI and actual Streamlit Cloud redeployment are separate gates.
The long offline NQ+ES replay will run in GitHub CI on the cloud-code push.
Local success does not establish Cloud authentication or completed deployment.
