"""Revision 0.9 proposal: failed-reaction memory and reason-specific reentry.

AUTHORED_NOT_EXECUTED_OR_VALIDATED. This module neither detects OBs nor places
orders. It gates already-observed offers, without a quota or a future cohort.
All evidence is recorded at daily close; actual exits precede that day's close.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import positive


class SelectiveReentry:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = None
        self.last_exit_index = None
        self.reclaim_count = 0
        self.reclaimed_index = None
        self.entries = {}
        self.failed_zones = {}
        self.bars = {}
        self.structures = ()
        self.invalid_structures = set()
        self.seen_top_sources = set()
        self.events = []
        self.snapshot = dict(index=None, context=None, reclaimed=False)

    def register_entry(self, order_id, candidate, role, obs, previous, bar):
        """Freeze a plan; a cancelled/unfilled plan does not create a failure."""
        contact = obs.contact_index if obs else candidate.touch_index
        highs = [b.high for i, b in self.bars.items() if contact <= i <= bar.index]
        # The decision candle is completed. Its high is not a future fill-day high.
        reaction_high = max(highs + [bar.high, candidate.zone.body_high])
        self.entries[order_id] = dict(
            decision_index=bar.index, contact_index=contact,
            zone_id=candidate.zone.id, zone_low=candidate.zone.wick_low,
            zone_high=candidate.zone.wick_high, recovery_boundary=candidate.zone.body_high,
            reaction_low=candidate.low, reaction_low_index=candidate.low_index,
            reaction_high_at_decision=reaction_high, known_reaction_high=reaction_high,
            high_known_through=bar.index, atr=bar.atr_prior,
            role=deepcopy(role), source_setup_id=candidate.setup_id,
        )
        self.events.append(dict(index=bar.index, action="reentry_entry_plan_frozen",
                                order_id=order_id, entry=deepcopy(self.entries[order_id])))

    def supply_structures(self, structures):
        self.structures = structures

    def _replace(self, context, action):
        self.context = deepcopy(context)
        self.reclaim_count, self.reclaimed_index = 0, None
        self.events.append(dict(index=context["index"], action=action,
                                context=deepcopy(context)))

    def register_exit(self, position, index, at, reason, price, frozen=None, thesis=None):
        """Use the ACTUAL exit reason, not a SELL plan superseded by a gap stop."""
        c = position.candidate
        frozen, thesis = frozen or {}, thesis or {}
        entry = self.entries.get(position.id)
        if entry is None:
            raise ValueError("selective reentry requires the original frozen entry plan")
        if entry["high_known_through"] >= index:
            raise ValueError("exit cannot use its own later daily high")
        top = reason.startswith(("campaign_major_top_", "campaign_double_top_"))
        active = thesis.get("active_protected_low")
        structural = (reason == "major_protected_low_close_failure"
                      or (reason == "trailing_protection_stop" and active is not None))
        reference = frozen.get("reference") or c.reference
        if top:
            kind, level = "top", reference.price
        elif structural:
            kind = "structure"
            level = frozen.get("invalidation_level", active.pivot.price if active else reference.price)
        else:
            kind, level = "reaction", entry["known_reaction_high"]
        # A pending SELL must not donate its ATR/reference to an unrelated gap
        # stop. The reaction failure always uses its original decision ATR.
        atr = frozen.get("atr", entry["atr"]) if kind != "reaction" else entry["atr"]
        context = dict(
            id=f"selective-exit:{position.id}:{index}", index=index, at=at,
            kind=kind, executed=True, executed_reason=reason,
            planned_reason=frozen.get("reason"), position_id=position.id,
            fill_price=price, reclaim_level=level, atr=atr, reference=reference,
            signal_low=frozen.get("signal_low", c.low) if kind != "reaction" else c.low,
            signal_index=frozen.get("signal_index", index) if kind != "reaction" else index,
            reaction_low=c.low, reaction_low_index=c.low_index, zone_id=c.zone.id,
            zone_low=c.zone.wick_low, zone_high=c.zone.wick_high,
            source_setup_id=c.setup_id, source_contact_index=entry["contact_index"],
            entry_role=deepcopy(entry["role"]), high_known_through=entry["high_known_through"],
            failure_clock="actual_exit_using_prior_completed_bars",
        )
        self.last_exit_index = index
        self._replace(context, "actual_exit_reason_frozen")
        if reason in {"initial_risk_stop", "early_reaction_close_failure",
                      "continuation_reaction_close_failure"}:
            failed = dict(
                id=f"failed-reaction:{position.id}:{index}", index=index,
                position_id=position.id, zone_id=c.zone.id,
                zone_low=c.zone.wick_low, zone_high=c.zone.wick_high,
                structure_id=(entry["role"] or {}).get("structure_id"),
                reaction_low=c.low, reaction_high=entry["known_reaction_high"],
                decision_reaction_high=entry["reaction_high_at_decision"],
                high_known_through=entry["high_known_through"],
                atr=entry["atr"], contact_index=entry["contact_index"],
                closes=0, reclaimed_index=None,
            )
            # Only a new actual failure replaces this zone's checkpoint. A new
            # market scope or another profitable trade cannot erase it.
            self.failed_zones[c.zone.id] = failed
            self.events.append(dict(index=index, action="failed_reaction_registered",
                                    failure=deepcopy(failed), pnl_used=False))

    def on_close(self, bar, top_sells=(), top_details=(), position=None):
        self.bars[bar.index] = bar
        if position is not None:
            entry = self.entries.get(position.id)
            if entry is None:
                raise ValueError("missing entry plan for an actual holding")
            entry["known_reaction_high"] = max(entry["known_reaction_high"], bar.high)
            entry["high_known_through"] = bar.index
        for r in self.structures:
            if r.effective_index <= bar.index and bar.close < r.low.price:
                self.invalid_structures.add(r.id)
        for sell in top_sells:
            if sell.source_id in self.seen_top_sources:
                continue
            self.seen_top_sources.add(sell.source_id)
            detail = next((d for d in top_details if d.get("top_id") == sell.source_id), {})
            self._replace(dict(
                id=f"selective-market-top:{sell.source_id}:{bar.index}", index=bar.index,
                at=bar.at, kind="top", executed=False, executed_reason=None,
                planned_reason=sell.reason, position_id=None, fill_price=None,
                reclaim_level=sell.reference.price, atr=bar.atr_prior,
                reference=sell.reference, signal_index=bar.index, signal_low=bar.low,
                reaction_low=bar.low, zone_id=None, zone_low=None, zone_high=None,
                entry_role=None, peak=detail.get("peak"),
                failure_clock="market_close_evidence_not_a_trade",
            ), "market_top_observed")

        context = self.context
        threshold = None
        if context and positive(context["reclaim_level"]) and positive(context["atr"]):
            threshold = context["reclaim_level"] + self.rules.structure_buffer_atr * context["atr"]
            self.reclaim_count = (self.reclaim_count + 1 if bar.index > context["index"]
                                  and bar.close > threshold else 0)
            if self.reclaim_count >= self.cfg.exit_reclaim_closes and self.reclaimed_index is None:
                self.reclaimed_index = bar.index
                self.events.append(dict(index=bar.index, action="exit_level_reclaim_observed",
                                        context_id=context["id"], threshold=threshold,
                                        usable_for_contacts_after=bar.index))
        for failed in self.failed_zones.values():
            if failed["reclaimed_index"] is not None:
                continue
            threshold_failed = failed["reaction_high"] + self.rules.structure_buffer_atr * failed["atr"]
            failed["closes"] = (failed["closes"] + 1 if bar.index > failed["index"]
                                and bar.close > threshold_failed else 0)
            if failed["closes"] >= self.cfg.exit_reclaim_closes:
                failed["reclaimed_index"] = bar.index
                self.events.append(dict(index=bar.index, action="failed_reaction_high_reclaimed",
                                        failure_id=failed["id"], threshold=threshold_failed,
                                        new_contact_must_follow_this_close=True))
        self.snapshot = dict(
            index=bar.index, policy=self.cfg.exit_reentry_policy,
            failed_reaction_policy=self.cfg.failed_reaction_policy,
            context=deepcopy(context), last_exit_index=self.last_exit_index,
            reclaim_threshold=threshold, reclaim_closes=self.reclaim_count,
            reclaimed=self.reclaim_count >= self.cfg.exit_reclaim_closes,
            reclaimed_index=self.reclaimed_index,
            failed_zones=deepcopy(self.failed_zones), probability=None, purchase_quota=None,
        )

    def _failed_reaction_check(self, c, obs):
        if self.cfg.failed_reaction_policy == "off":
            return True, dict(path="failed_reaction_policy_off")
        # Overlapping price intervals are conservatively one failed support
        # area, even when a second detector assigned another ID. A different ID
        # alone cannot make a failed band independent. Disjoint zones are free
        # of THIS gate, but still face the structural/exit/candidate gates.
        matches = [f for f in self.failed_zones.values()
                   if f["zone_id"] == c.zone.id
                   or (c.zone.wick_low <= f["zone_high"] and c.zone.wick_high >= f["zone_low"])]
        details = []
        for failed in sorted(matches, key=lambda f: (f["index"], f["id"])):
            recovered = failed["reclaimed_index"]
            fresh = (obs is not None and recovered is not None
                     and obs.contact_index > max(failed["index"], recovered)
                     and c.touch_index == obs.contact_index)
            details.append(dict(failure_id=failed["id"], source_zone_id=failed["zone_id"],
                                same_zone_id=failed["zone_id"] == c.zone.id,
                                reaction_high=failed["reaction_high"], reclaimed_index=recovered,
                                contact_index=obs.contact_index if obs else None, passed=fresh))
        allowed = all(d["passed"] for d in details)
        return allowed, dict(path="failed_area_reclaimed_then_new_visit" if matches and allowed else
                             "waiting_failed_reaction_high_then_new_visit" if matches else
                             "no_prior_failure_in_this_area", failures=details)

    def _new_major_structure(self, obs, cutoff):
        if obs is None:
            return None
        choices = [r for r in self.structures if r.id not in self.invalid_structures
                   and r.low.origin_index > cutoff and r.breakout_index > cutoff
                   and r.known_index < obs.contact_index and r.effective_index <= obs.contact_index]
        return max(choices, key=lambda r: (r.known_index, r.id)) if choices else None

    def assess(self, offer, obs, role, bar, wave_snapshot=None):
        c, context = offer.decision, self.context
        failed_ok, failed_detail = self._failed_reaction_check(c, obs)
        detail = dict(path="initial_no_exit", policy=self.cfg.exit_reentry_policy,
                      context_id=context["id"] if context else None,
                      kind=context["kind"] if context else None,
                      failed_reaction=failed_detail, reclaimed=bool(self.snapshot.get("reclaimed")))
        if not failed_ok:
            detail["path"] = failed_detail["path"]
            return False, detail["path"], detail
        if self.cfg.exit_reentry_policy == "off":
            detail["path"] = "selective_exit_policy_off"
            return True, detail["path"], detail
        if context is None:
            return True, "no_previous_exit_thesis", detail
        cutoff = max(context["index"], self.last_exit_index if self.last_exit_index is not None else -1)
        fresh = bool(obs and obs.recovery_index is not None
                     and cutoff < obs.contact_index <= obs.low_index <= obs.recovery_index <= bar.index
                     and c.touch_index == obs.contact_index and c.index == bar.index)
        detail.update(cutoff=cutoff, fresh_contact=fresh, reclaim_level=context["reclaim_level"],
                      reclaimed_index=self.reclaimed_index)
        if not fresh:
            detail["path"] = "waiting_new_post_exit_reaction"
            return False, detail["path"], detail

        continuation = c.confirmation == "continuation_zone_reaction"
        structure = obs.features.get("continuation_structure") if continuation else None
        accepted = bool(structure and structure["known_index"] < obs.contact_index
                        and structure["effective_index"] <= obs.contact_index
                        and bar.close >= structure["low"].price)
        reclaimed_before_contact = (self.reclaimed_index is not None
                                    and self.reclaimed_index < obs.contact_index)
        if continuation:
            detail["accepted_continuation_before_contact"] = accepted
            if not accepted:
                detail["path"] = "waiting_accepted_structure_before_retest"
                return False, detail["path"], detail
            if context["kind"] == "reaction":
                # A still-valid accepted rise can survive a local trade failure.
                # No old peak reclaim or large new decline is imposed here.
                detail["path"] = "continuation_after_local_failure"
                return True, "valid_rise_and_distinct_reaction_after_local_exit", detail
            newly_accepted = (structure["breakout_index"] > cutoff
                              and reclaimed_before_contact
                              and structure["known_index"] >= self.reclaimed_index)
            new_major = self._new_major_structure(obs, cutoff)
            detail["new_major_structure_id"] = new_major.id if new_major else None
            if newly_accepted or new_major is not None:
                detail["path"] = "exit_reclaim" if newly_accepted else "fresh_structure_after_exit"
                return True, "resolved_exit_or_new_major_rise_then_later_retest", detail
            detail["path"] = "waiting_exit_resolution_before_continuation"
            return False, detail["path"], detail

        # Reversal offers still need a qualified, pre-contact OB. Returning a
        # permission here never bypasses qualify_support's major-decline check.
        main_role = bool(role and role.get("kind") in {
            "major_origin", "established_major_support", "upward_leg_support"}
            and role["known_index"] < obs.contact_index
            and role["effective_index"] <= obs.contact_index
            and c.zone.kind == "ob_candidate" and c.zone.available_index < obs.contact_index)
        if not main_role:
            detail["path"] = "waiting_precontact_structural_ob"
            return False, detail["path"], detail
        if reclaimed_before_contact:
            detail["path"] = "exit_reclaim"
            return True, "exit_level_reclaimed_then_new_structural_ob_reaction", detail
        new_major = self._new_major_structure(obs, cutoff)
        if new_major is not None:
            detail.update(path="fresh_structure_after_exit", new_major_structure_id=new_major.id)
            return True, "new_major_structure_before_ob_contact", detail
        if context["kind"] == "reaction":
            old_role = context.get("entry_role") or {}
            independent = (c.zone.id != context["zone_id"]
                           and c.zone.wick_high < context["zone_low"]
                           and role.get("structure_id") != old_role.get("structure_id")
                           and obs.low < context["reaction_low"])
        else:
            independent = (positive(context["atr"])
                           and obs.low < context["signal_low"]
                           - self.rules.structure_buffer_atr * context["atr"])
        detail["independent_lower_support"] = independent
        if independent:
            detail["path"] = "fresh_major_support_after_exit"
            return True, "lower_structural_ob_still_requires_major_correction", detail
        detail["path"] = "exit_thesis_unresolved"
        return False, "bounce_has_not_changed_the_prior_failure", detail
