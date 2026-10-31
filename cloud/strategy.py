"""Release identity and the v0.13 strategy changes; vendor code stays frozen.

Changes relative to the inherited rev11 strategy:
- v0.12: radius-one strict pivot supply for the BUY reference (unchanged).
- v0.13: an NFCI persistence regime gate on BUY assessments. A BUY offer is
  rejected when financial conditions have been tight for at least four
  consecutive weeks (see REGIME_GATE). The gate reads only the stored weekly
  NFCI series in cloud/regime/, lagged by the publication delay, so no
  decision ever uses a print that was not yet public.
"""
import bisect
import csv
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

STRATEGY_VERSION = "v0.13"
BASE_STAGE = "rev11_without_continuation"

REGIME_GATE = {
    "id": "nfci_persistence_gate",
    "series": "FRED NFCI, weekly, week-ending Friday",
    "data_file": "regime/nfci_weekly.csv",
    "threshold": 0.0,
    "min_consecutive_weeks": 4,
    "publication_lag_days": 7,
    "reason_blocked": "nfci_tightening_persistent",
    "reason_clear": "nfci_regime_clear",
}

_REGIME_SERIES = None


def _regime_series():
    """Load the stored NFCI series once; fail closed if it is absent."""
    global _REGIME_SERIES
    if _REGIME_SERIES is None:
        path = Path(__file__).resolve().parent/REGIME_GATE["data_file"]
        pairs = []
        for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
            if not row or row[0] == "date":
                continue
            try:
                pairs.append((row[0], float(row[1])))
            except ValueError:
                continue
        pairs.sort()
        if not pairs:
            raise ValueError("Stored NFCI series is empty")
        _REGIME_SERIES = ([p[0] for p in pairs], [p[1] for p in pairs])
    return _REGIME_SERIES


def regime_state(day):
    """NFCI regime visible on a decision date, after the publication lag.

    Returns the latest weekly print dated on or before day - lag, its value,
    and how many consecutive weeks (ending at that print) were at or above
    the tightening threshold. blocked is True only for an entrenched
    tightening, never for a brief spike above zero.
    """
    dates, values = _regime_series()
    target = (date.fromisoformat(day)-timedelta(days=REGIME_GATE["publication_lag_days"])).isoformat()
    i = bisect.bisect_right(dates, target)-1
    if i < 0:
        return dict(value=None, consecutive_weeks=0, print_date=None, blocked=False)
    weeks = 0
    while i-weeks >= 0 and values[i-weeks] >= REGIME_GATE["threshold"]:
        weeks += 1
    return dict(value=values[i], consecutive_weeks=weeks, print_date=dates[i],
                blocked=weeks >= REGIME_GATE["min_consecutive_weeks"])


def validate_strategy(metadata):
    rules = metadata["legacy_rules"]
    if (rules["buy_reference_radius"], rules["emergency_buffer_atr"], rules["major_decline_bars"]) != (1, .10, 15):
        raise ValueError("v0.13 requires radius=1, emergency buffer=.10, decline bars=15")
    if metadata["campaign_config"]["enable_continuation_buy"]:
        raise ValueError("Ordinary continuation BUY must remain disabled")


def stamp_metadata(metadata):
    validate_strategy(metadata)
    dates, _ = _regime_series()
    gate = dict(REGIME_GATE, data_rows=len(dates), data_start=dates[0], data_end=dates[-1])
    metadata.update(strategy_version=STRATEGY_VERSION, strategy_base_stage=BASE_STAGE,
                    strategy_change={"legacy_rules.buy_reference_radius": {"from": 2, "to": 1},
                                     "regime_gate": {"from": None, "to": REGIME_GATE["id"]}},
                    extra_supplied_pivot_radii=[1], regime_gate=gate)
    return metadata


def install_pivot_supply():
    """Supply radius-one strict extrema at origin+1, identical to tested alpha.

    Existing radius-two/five confirmations remain untouched. No pivot is supplied
    before its right-hand bar is observed; ties are rejected, as in the study.
    Install once in the market-isolated interpreter before creating the source.
    """
    import numpy as np
    from nq_ict_campaign_examples_20260921.source import CampaignSource
    from nq_ict_retest_trade_state_machine_20260915 import Pivot, number
    if getattr(CampaignSource, "_v012_pivot_supply", False):
        return
    native = CampaignSource.confirm_pivots

    def confirm(self):
        new = native(self)
        if self.rules.buy_reference_radius != 1:
            raise ValueError("Radius-one adapter used with a different strategy")
        origin = self.i - 1
        if origin < 1:
            return new
        window = self.d.iloc[origin-1:self.i+1]
        row = self.d.iloc[origin]
        for kind, column, extremum in [("high", "high", max), ("low", "low", min)]:
            values = window[column].to_numpy(float)
            value = float(row[column])
            if value != extremum(values) or np.count_nonzero(values == value) != 1:
                continue
            pivot = Pivot(f"pivot:1:{kind}:{row.datetime.date()}", kind, 1, origin,
                          row.datetime, self.t, value, number(row.atr_prior))
            self.pivots.append(pivot)
            new.append(pivot)
            self.log("pivot_confirmed", pivot.id, pivot=asdict(pivot))
        return new

    CampaignSource.confirm_pivots = confirm
    CampaignSource._v012_pivot_supply = True


def install_regime_gate():
    """Add the NFCI persistence gate to BUY offer assessment (v0.13).

    Wraps CampaignLedger.assess_offer without touching vendor code. The
    native assessment runs unchanged; the wrapper records a first-class
    "regime_gate" check next to the vendor's own gates and, only when the
    regime is blocked and the offer was otherwise allowed, flips the
    assessment to rejected with reason "nfci_tightening_persistent" in both
    the selection row and the entry audit, exactly as a native gate failure
    is recorded. Install once in the market-isolated interpreter.
    """
    from copy import deepcopy
    from nq_ict_campaign_examples_20260921.campaign import CampaignLedger
    if getattr(CampaignLedger, "_v013_regime_gate", False):
        return
    native = CampaignLedger.assess_offer

    def assess(self, offer, packet, previous, sells):
        offer, allowed, why, row, obs = native(self, offer, packet, previous, sells)
        at = packet.base.signal.bar.at
        day = at.date().isoformat() if hasattr(at, "date") else str(at)[:10]
        state = regime_state(day)
        if state["value"] is None:
            check = dict(passed=True, reason="nfci_no_print_available", status="evaluated",
                         decision_date=day)
        else:
            check = dict(passed=not state["blocked"],
                         reason=REGIME_GATE["reason_blocked"] if state["blocked"] else REGIME_GATE["reason_clear"],
                         status="evaluated", decision_date=day,
                         nfci=state["value"], nfci_print_date=state["print_date"],
                         consecutive_weeks_at_or_above_threshold=state["consecutive_weeks"],
                         threshold=REGIME_GATE["threshold"],
                         min_consecutive_weeks=REGIME_GATE["min_consecutive_weeks"],
                         publication_lag_days=REGIME_GATE["publication_lag_days"])
        row["checks"]["regime_gate"] = check
        audit = self.entry_audit[-1] if self.entry_audit else None
        audit_matches = (audit is not None and audit.get("action") == "entry_assessed"
                         and audit.get("index") == row["index"] and audit.get("key") == row["key"])
        if audit_matches:
            audit["checks"]["regime_gate"] = deepcopy(check)
        if state["blocked"] and allowed:
            failure = dict(gate="regime_gate", reason=REGIME_GATE["reason_blocked"])
            row["failed_conditions"] = list(row["failed_conditions"])+[failure]
            row["selected_eligibility"] = False
            row["reason"] = REGIME_GATE["reason_blocked"]
            if audit_matches:
                audit["failed_conditions"] = list(audit["failed_conditions"])+[deepcopy(failure)]
                audit["allowed"] = False
                audit["reason"] = REGIME_GATE["reason_blocked"]
            return offer, False, REGIME_GATE["reason_blocked"], row, obs
        return offer, allowed, why, row, obs

    CampaignLedger.assess_offer = assess
    CampaignLedger._v013_regime_gate = True
