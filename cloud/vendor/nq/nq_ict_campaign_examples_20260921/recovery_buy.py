"""Revision 0.11 post-exit recovery proposal, NOT EXECUTED OR VALIDATED.

No fabricated OB/FVG or contact: zone/touch_index are explicitly null. Reuse
the price-only continuation structure book, but do not require a zone retest.
Actual exit -> frozen level reclaim -> permitted support/new accepted break ->
next open. Reusing a pre-exit low is an explicit, local-stop-only alternative.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nq_ict_macro_reversal_examples_20260916.common import down_tick, positive
from nq_ict_priority_entry_examples_20260916.common import EntryOffer


@dataclass(frozen=True)
class RecoveryBuyEvidence:
    # Execution-compatible shape, intentionally NOT an OB BuyEvidence.
    key: str
    episode_id: str
    at: Any
    index: int
    low: float
    low_index: int
    low_atr: float
    recovery_index: int
    reference: Any
    breakout_index: int
    emergency_stop: float
    htf: Any
    setup_id: str
    exit_context_id: str
    structure_known_index: int
    next_open_min_price: float
    structure_route: str = "post_exit_low"
    preserved_support_id: str | None = None
    zone: None = None
    touch_index: None = None
    confirmation: str = "uptrend_recovery"
    linked_fvg_ids: tuple = ()
    htf_overlap_ids: tuple = ()
    sweep_depth_atr: None = None
    ltf_conditions: str = "unknown_data"


class RecoveryBuyBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.emitted = set()
        self.details = {}
        self.events, self.candidate_rows = [], []
        self.snapshot = dict(enabled=cfg.enable_recovery_buy, phase="disabled" if not cfg.enable_recovery_buy
                             else "waiting_actual_exit")

    @staticmethod
    def is_offer(offer):
        return offer.kind == "uptrend_recovery"

    def on_close(self, packet, continuation, reentry):
        bar = packet.base.signal.bar
        if not self.cfg.enable_recovery_buy:
            return ()
        context, s = reentry.context, continuation.structure
        cutoff = max(context["index"], reentry.last_top_index if reentry.last_top_index is not None else -1) if context else None
        threshold = reentry.snapshot.get("reclaim_threshold")
        qualification = reentry.recovery_structure_status(s, bar)
        checks = dict(
            actual_exit=bool(context and context.get("executed")),
            after_exit=bool(cutoff is not None and bar.index > cutoff),
            exit_level_reclaimed=bool(reentry.snapshot.get("reclaimed")),
            structure_matches_exit_scope=qualification["eligible"],
            structure_available=bool(s and s["known_index"] < bar.index
                                     and s["effective_index"] <= bar.index),
            pivots_preknown_at_break=bool(s and s["low"].available_index < s["breakout_index"]
                                          and s["high"].available_index < s["breakout_index"]),
            structure_still_valid=bool(s and bar.close > s["boundary"]
                                       and bar.close > s["low"].price),
            no_same_close_top=not bool(packet.base.signal.sells),
            prior_atr=positive(bar.atr_prior))
        self.snapshot = dict(enabled=True, index=bar.index, phase="waiting_recovery",
                             context_id=context["id"] if context else None,
                             structure_id=s["id"] if s else None, cutoff=cutoff,
                             checks=checks, structure_qualification=deepcopy(qualification),
                             contact_required=False, probability=None)
        if not all(checks.values()):
            return ()
        token = (context["id"], s["id"])
        if token in self.emitted:
            self.snapshot["phase"] = "already_offered_for_this_exit_and_structure"
            return ()
        stop = down_tick(s["low"].price - self.rules.emergency_buffer_atr * bar.atr_prior,
                         self.rules.tick_size)
        if not 0 < stop < bar.close:
            self.snapshot["phase"] = "invalid_stop_geometry"
            return ()
        sid = f"recovery:{context['id']}:{s['id']}"
        c = RecoveryBuyEvidence(
            key=f"{sid}:{bar.index}", episode_id=s["id"], at=bar.at, index=bar.index,
            low=s["low"].price, low_index=s["low"].origin_index,
            low_atr=s["low"].atr_at_origin, recovery_index=bar.index,
            reference=s["high"], breakout_index=s["breakout_index"],
            emergency_stop=stop, htf=packet.base.signal.htf, setup_id=sid,
            exit_context_id=context["id"], structure_known_index=s["known_index"],
            next_open_min_price=max(threshold, s["boundary"]),
            structure_route=qualification["path"],
            preserved_support_id=(qualification["preserved_support"]["structure_id"]
                                  if qualification["path"] == "preserved_support_after_local_stop" else None))
        role = dict(
            kind="recovery_structure", grade=-1, structure_id=s["id"], zone_id=None,
            known_index=s["known_index"], effective_index=s["effective_index"],
            source_timeframe="1D", high_id=s["high"].id, low_id=s["low"].id,
            low_index=s["low"].origin_index, breakout_index=s["breakout_index"],
            scale=dict(tier="major_daily_management_after_recovery",
                       pivot_radius=self.rules.pivot_radius,
                       min_leg_bars=self.cfg.structure_min_leg_bars,
                       min_leg_atr=self.cfg.structure_min_leg_atr,
                       min_pullback_bars=self.rules.management_pullback_bars,
                       min_pullback_atr=self.rules.management_pullback_atr),
            recovery_structure_route=c.structure_route,
            preserved_support_id=c.preserved_support_id,
            definition="post_exit_price_structure_not_ob_contact_or_probability")
        self.details[c.key] = dict(
            role=role, structure=deepcopy(s), exit_context=deepcopy(context),
            structure_qualification=deepcopy(qualification),
            checks=deepcopy(checks), cutoff=cutoff, reclaim_threshold=threshold,
            risk=dict(decision_index=bar.index, decision_atr=bar.atr_prior,
                      stop=stop, mode="recovery_structural_low",
                      risk_at_decision_atr=(bar.close - stop) / bar.atr_prior,
                      next_open_min_price=c.next_open_min_price, maximum_risk_atr=None),
            contact_index=None, source_zone=None, intrabar_order="not_inferred")
        self.emitted.add(token)
        self.candidate_rows.append(c)
        self.events.append(dict(index=bar.index, action="uptrend_recovery_offered",
                                candidate=c, detail=deepcopy(self.details[c.key]),
                                contact_required=False, probability=None))
        self.snapshot["phase"] = "recovery_offered"
        return (EntryOffer("confirmed", "uptrend_recovery", c, c),)

    def assess(self, offer, bar, reentry):
        c, d = offer.decision, self.details.get(offer.decision.key)
        current = reentry.context
        qualification = reentry.recovery_structure_status(d["structure"] if d else None, bar)
        valid = bool(d and current and current["id"] == c.exit_context_id
                     and c.index == bar.index and c.at == bar.at
                     and qualification["eligible"] and qualification["path"] == c.structure_route
                     and qualification["cutoff"] == d["cutoff"]
                     and c.breakout_index > d["cutoff"]
                     and c.structure_known_index < bar.index
                     and d["structure"]["effective_index"] <= bar.index
                     and reentry.snapshot.get("reclaimed")
                     and bar.close > c.next_open_min_price)
        checks = dict(recovery_structure=(valid, "actual_exit_reclaim_and_new_available_structure_required"),
                      source_zone=(c.zone is None and c.touch_index is None,
                                   "not_applicable_recovery_does_not_claim_zone_contact"))
        return checks, deepcopy(d["role"]) if d else None, deepcopy(d) if d else {}

    def describe(self, offer):
        c = offer.decision
        return dict(setup_id=c.setup_id, family="recovery", structure_id=c.episode_id,
                    exit_context_id=c.exit_context_id, source_zone=None, contact_index=None,
                    structure_route=c.structure_route, preserved_support_id=c.preserved_support_id,
                    required_evidence="exit_level_reclaim_and_new_accepted_structure",
                    probability=None, probability_status="not_estimated")
