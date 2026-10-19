# Version-label follow-up — 2026-10-19

Scope: current NQ/NKD Streamlit titles, authored UI, source measurement
definitions, downloads and normal daily-export output. No price fetch, strategy
execution, parameter change or trade/evidence-row change is part of this repair.

## Findings and repair

- Current repository snapshots already had v0.12 titles and inline/encoded UI.
- NQ `executions.source.metricDefinitions[0].label` incorrectly said
  `v0.11 約定履歴`; changed to `v0.12 約定履歴`.
- NKD's corresponding label said `v0.11 約定モデル`; changed to v0.12.
- The visible legacy implementation-stage footer was clarified to
  `現行：v0.12 / BUY参照半径1`. Internal base-stage identifiers remain intact.
- Genuine pre-migration v0.11 refresh events keep their original detail and
  now have an explicit `旧v0.11時点（移行前）の記録` title. Renaming those past
  computations to v0.12 would falsify provenance.
- The normal exporter now normalizes active labels on every update; the
  packager rejects stale v0.11 active labels in v0.12 snapshots. Old v0.11
  snapshots cannot be relabelled as v0.12 by this normalizer.

The in-app browser also contained two old `file:///.../v011_...html` tabs.
Those are actual v0.11 local outputs and were not overwritten or retitled as
v0.12. The current local app is `http://127.0.0.1:4190/?market=nq`.
The user-reported title's exact originating tab/Cloud URL was not supplied;
the current local Streamlit title was directly observed as v0.12.

## Verification

- Full pytest: **47 passed**, including new label, archive, idempotence,
  provenance-preservation and packaging-rejection tests.
- Every existing price/trade/state/evidence query hash remains identical.
- Original v0.11 archive files remain byte-for-byte identical.
- The ordinary scheduled-update exporter was smoke-tested from the completed
  native v0.12 logs: NQ 26,738 and NKD 22,764 view/ledger checks passed, with
  identical query rows and current v0.12 measurement labels. This did not
  execute the trading strategy again or acquire prices.
- NQ current Streamlit header, footer and source-panel label were inspected
  in the existing in-app browser; NKD switching/title was also checked.
- [Current NQ screenshot](version-label-current-nq-20261019.png) shows the
  v0.12 title in the actual local Streamlit view.
- Machine-readable audit: [version-label-audit-20261019.json](version-label-audit-20261019.json).

Historic filenames, frozen original exporter/vendor sources, compatibility
defaults, migration explanations, prior reports and old archives intentionally
retain their real v0.11 names. This is not a new v0.12 strategy release and the
original v0.12 tag is not moved.
