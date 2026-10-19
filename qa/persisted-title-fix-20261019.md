# Persisted stock replay title correction — 2026-10-19

## Report and reproduction

The supplied phone screenshot showed the outer Streamlit application at v0.12,
but the inner NQ replay header at `NQ=F · v0.11 追体験`.
Both packaged snapshots already contained v0.12 titles. A fresh-browser-only
check therefore did not cover this failure.

Reproduced on the local reviewed NQ application using its ordinary title editor:
save the previous stock title, then reload. The saved presentation title wins
over the current snapshot title, reproducing the reported mismatch. No direct
browser-storage injection was used.

## Repair boundaries

- Use the public `useDataApp` context and `setAppTitle` action in authored NQ/ES
  content. The protected application shell/runtime is unchanged.
- Correct only the exact matching-market v0.11 stock title when the underlying
  snapshot is the canonical NQ/ES v0.12 replay.
- Preserve custom titles, archived versions, application identities, and other
  saved presentation preferences. Do not clear browser storage or replace IDs.
- Rebuild/repackage both active HTML assets. Daily data hydration retains the
  corrected compiled UI; no strategy run or price acquisition was performed.

## Verification

- Actual saved old title → reload → canonical v0.12 header: confirmed in browser.
- Public live-context comparison: the only changed presentation key was `title`.
  The NQ artifact ID remained `dashboard:7ae38490-be27-449b-bd5d-0ac468bd17b4`.
- Actual custom title `NQ 検証用メモ v0.11` → save → reload: preserved. Restored
  the canonical v0.12 title in that isolated local QA origin afterward.
- Local Streamlit: switched NQ → ES → NQ and confirmed both inner headers at
  v0.12. Final screenshot includes the outer and inner NQ v0.12 titles together.
- Independent comparison against Git commit
  `0091dad038fa10febcf3a6434fd818093feec761`: both complete embedded snapshot
  objects are identical, including all six reviewed query row sets and IDs.
  Compressed HTML bytes changed because the authored UI was rebuilt.
- `pytest -q`: **53 passed**. The new title-guard check includes 24 assertions
  covering correction, idempotence, custom titles, archive boundaries, unchanged
  data and preferences. Existing active-version scan tests remain enabled.

Local screenshots (ignored build evidence):

- `qa/screenshots/title-stale-before-20261019.png`
- `qa/screenshots/title-corrected-after-20261019.png`
- `qa/screenshots/streamlit-title-fixed-20261019.png`

Cloud deployment and the user's Safari session were not directly inspected.
GitHub delivery is not, by itself, proof that Streamlit Cloud has finished
redeploying. The intended user action after deployment is an ordinary page
reload, not clearing all browser settings.
