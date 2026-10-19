"""Revision 0.8: freeze the reason for leaving and observe its resolution.

AUTHORED_NOT_EXECUTED_OR_VALIDATED. No monetary score or retry count affects
permission. Assessment never consumes a candidate or mutates the market book.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import positive


class ExitReasonReentry:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = None
        self.reclaim_count = 0
        self.events = []
        self.snapshot = dict(index=None, context=None, reclaimed=False)

    def _replace(self, context, action):
        self.context = deepcopy(context)
        self.reclaim_count = 0
        self.events.append(dict(index=context["index"], action=action, context=deepcopy(context)))

    def register_exit(self, position, index, at, reason, price, frozen=None, thesis=None):
        c = position.candidate
        frozen, thesis = frozen or {}, thesis or {}
        planned_reason = frozen.get("reason", "")
        top = reason.startswith(("campaign_major_top_", "campaign_double_top_"))
        top = top or planned_reason.startswith(("campaign_major_top_", "campaign_double_top_"))
        active = thesis.get("active_protected_low")
        structure = (reason == "major_protected_low_close_failure"
                     or (thesis.get("management_phase") == "wave" and active is not None))
        reference = frozen.get("reference") or c.reference
        if top:
            kind, level = "top", reference.price
        elif structure:
            kind = "structure"
            level = frozen.get("invalidation_level", active.pivot.price if active else reference.price)
        else:
            kind = "reaction"
            level = frozen.get("reclaim_level", thesis.get("failure_reclaim_level", c.zone.body_high))
        atr = frozen.get("atr") or thesis.get("decision_atr") or c.low_atr
        self._replace(dict(
            id=f"exit-thesis:{position.id}:{index}", index=index, at=at,
            kind=kind, executed=True, executed_reason=reason, planned_reason=planned_reason,
            position_id=position.id, fill_price=price,
            reclaim_level=level, atr=atr, reference=reference,
            signal_index=frozen.get("signal_index", index),
            signal_low=frozen.get("signal_low", c.low),
            reaction_low=c.low, reaction_low_index=c.low_index, zone_id=c.zone.id,
            source_setup_id=c.setup_id, source_contact_index=c.touch_index,
            failure_clock="actual_exit_after_frozen_close_plan",
        ), "executed_exit_recorded")

    def on_close(self, bar, top_sells=(), top_details=()):
        # A later market top remains relevant even if it occurs while flat.
        # It is evidence, never a fictitious executed SELL or PnL record.
        if top_sells and (self.context is None or bar.index > self.context["index"]):
            sell = top_sells[0]
            detail = next((d for d in top_details if d.get("top_id") == sell.source_id), {})
            self._replace(dict(
                id=f"market-exit-thesis:{sell.source_id}:{bar.index}", index=bar.index, at=bar.at,
                kind="top", executed=False, executed_reason=None, planned_reason=sell.reason,
                position_id=None, fill_price=None, reclaim_level=sell.reference.price,
                atr=bar.atr_prior, reference=sell.reference, signal_index=bar.index,
                signal_low=bar.low, reaction_low=bar.low, reaction_low_index=bar.index,
                zone_id=None, source_setup_id=None, source_contact_index=None,
                peak=detail.get("peak"), failure_clock="market_close_evidence_only",
            ), "market_top_thesis_recorded")
        context = self.context
        threshold = None
        if context and positive(context["reclaim_level"]) and positive(context["atr"]):
            threshold = context["reclaim_level"] + self.rules.structure_buffer_atr * context["atr"]
            self.reclaim_count = (self.reclaim_count + 1 if bar.index > context["index"]
                                  and bar.close > threshold else 0)
        else:
            self.reclaim_count = 0
        reclaimed = self.reclaim_count >= self.cfg.exit_reclaim_closes
        if reclaimed != self.snapshot.get("reclaimed", False):
            self.events.append(dict(index=bar.index, action="exit_reclaim_state_changed",
                                    context_id=context["id"] if context else None,
                                    reclaimed=reclaimed, threshold=threshold,
                                    close_count=self.reclaim_count))
        self.snapshot = dict(index=bar.index, policy=self.cfg.exit_reentry_policy,
                             context=deepcopy(context), reclaim_threshold=threshold,
                             reclaim_closes=self.reclaim_count, reclaimed=reclaimed,
                             probability=None, purchase_quota=None)

    def assess(self, offer, obs, role, bar, wave_snapshot):
        context = self.context
        detail = dict(path="initial_no_exit", context_id=context["id"] if context else None,
                      kind=context["kind"] if context else None,
                      reclaimed=bool(self.snapshot.get("reclaimed")), policy=self.cfg.exit_reentry_policy)
        if self.cfg.exit_reentry_policy == "off":
            detail["path"] = "exit_reason_policy_off"
            return True, "exit_reason_policy_off", detail
        if context is None:
            return True, "no_previous_exit_thesis", detail
        c = offer.decision
        cutoff = context["index"]
        fresh = (obs is not None and obs.recovery_index is not None
                 and cutoff < obs.contact_index <= obs.low_index <= obs.recovery_index <= bar.index
                 and c.index == bar.index and c.index > cutoff)
        detail.update(cutoff=cutoff, fresh_contact=fresh,
                      reclaim_level=context["reclaim_level"], reclaim_closes=self.reclaim_count)
        if not fresh:
            detail["path"] = "waiting_new_post_exit_reaction"
            return False, detail["path"], detail
        if detail["reclaimed"]:
            detail["path"] = "exit_reclaim"
            return True, "frozen_exit_level_reclaimed_with_new_reaction", detail

        wave = wave_snapshot.get("active")
        fresh_wave = bool(wave_snapshot.get("valid") and wave
                          and wave.pivot.origin_index > cutoff and wave.breakout_index > cutoff
                          and wave.effective_from_index <= obs.contact_index)
        detail["new_wave_before_contact"] = fresh_wave
        if fresh_wave:
            detail["path"] = "fresh_structure_after_exit"
            return True, "new_daily_wave_and_later_reaction", detail

        # A genuinely new low at a qualified, pre-existing OB can initiate an
        # EARLY reversal without waiting for weekly agreement or old-top reclaim.
        # Major-decline/provenance eligibility is still checked by qualify_support.
        continuation = c.confirmation == "continuation_zone_reaction"
        main_role = bool(role and role.get("kind") in {
            "major_origin", "established_major_support", "upward_leg_support"}
            and role["known_index"] < obs.contact_index
            and role["effective_index"] <= obs.contact_index)
        low_boundary = (context["reaction_low"] if context["kind"] == "reaction"
                        else context["signal_low"])
        new_lower_support = (not continuation and main_role and positive(context["atr"])
                             and obs.low < low_boundary - self.rules.structure_buffer_atr * context["atr"])
        detail["new_lower_support"] = new_lower_support
        if new_lower_support:
            detail["path"] = "fresh_major_support_after_exit"
            return True, "new_lower_ob_reaction_requires_separate_major_context", detail
        detail["path"] = "exit_thesis_unresolved"
        return False, "new_bounce_does_not_resolve_prior_failure", detail
