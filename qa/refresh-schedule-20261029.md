# NQ / ES scheduled refresh: day after a CME equity session

- Requested update time: 09:00 Asia/Tokyo on the calendar day following a
  US CME equity-index futures trade date, not Japanese business days.
- GitHub Actions wakes Tuesday-Saturday at 00:00 UTC. `cloud.schedule` checks
  the preceding trade date against pinned `pandas_market_calendars==5.5.0`,
  calendar `CME_Equity`, before the data-fetch/replay/publish job starts.
- Fully closed CME holidays are skipped. Shortened sessions still qualify;
  an NYSE holiday alone does not mean NQ/ES futures were closed.
- Friday sessions update Saturday morning. No scheduled update Sunday/Monday.
- The Japanese time stays 09:00 across US daylight-saving changes. GitHub
  may delay scheduled jobs; 09:00 is the requested trigger, not a finish SLA.
- Calendar failures fail closed. Manual refresh and push-triggered read-only
  regression remain available. The first qualifying future scheduled run is
  still separate from the local deterministic scheduling tests.
- Market-data finality, append-only behavior, v0.12 BUY/SELL logic, costs,
  both-market verification and preservation of the previous good generation
  on failure are unchanged. The conservative provider-date cutoff is retained;
  this scheduler does not promise immediate admission of provisional daily bars.
- No stored market prices, replay assets or trades are edited by this change.
- Local acceptance: 80 tests passed, including strict previous-session gating,
  winter/summer and DST-boundary clocks, full holidays, shortened sessions,
  manual invocation, failure-closed behavior, existing frozen-data checks and
  Streamlit NQ/ES switching. A future scheduled cloud run has not yet occurred.
- This maintained library calendar is not an assurance against future emergency
  closures or exchange revisions. Official reference: https://www.cmegroup.com/trading-hours.html
- GitHub scheduling reference: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
