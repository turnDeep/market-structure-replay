"""Revision 0.11 proposal; AUTHORED_NOT_EXECUTED_OR_VALIDATED.

An actual failed trade is evidence about its own reaction/structure, not every
overlapping price interval forever. No PnL, purchase quota or future cohort is
used. Existing reversal/continuation paths keep their v0.7 eligibility rules.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import positive


class EpisodeScopedReentry:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.entries, self.failures, self.bars = {}, {}, {}
        self.structures, self.continuation_structure = (), None
        self.invalid_structures = set()
        self.context = None
        self.preserved_support_failure = None
        self.last_exit_index = self.last_top_index = None
        self.reclaim_count = 0
        self.events = []
        self.snapshot = dict(index=None, context=None, reclaimed=False)

    def supply_structures(self, structures):
        self.structures = structures

    def supply_continuation(self, structure):
        self.continuation_structure = deepcopy(structure)

    @staticmethod
    def _zone_identity(zone):
        if zone is None:
            return None
        # Same underlying source candle under another ID is not new evidence.
        return (zone.kind, zone.direction, zone.timeframe, zone.origin_index,
                zone.available_index, zone.wick_low, zone.wick_high)

    def register_entry(self, order_id, candidate, role, obs, previous, bar):
        zone = candidate.zone
        contact = obs.contact_index if obs else candidate.touch_index
        start = contact if contact is not None else candidate.low_index
        highs = [b.high for i, b in self.bars.items() if start <= i <= bar.index]
        high = max(highs + [bar.high])
        self.entries[order_id] = dict(
            decision_index=bar.index, source_setup_id=candidate.setup_id,
            family=("recovery" if candidate.confirmation == "uptrend_recovery" else
                    "continuation" if candidate.confirmation == "continuation_zone_reaction" else "reversal"),
            contact_index=contact, group_id=obs.group_id if obs else None,
            zone_id=zone.id if zone else None, zone_identity=self._zone_identity(zone),
            zone_low=zone.wick_low if zone else None, zone_high=zone.wick_high if zone else None,
            structure_id=(role or {}).get("structure_id"), role=deepcopy(role),
            reaction_low=candidate.low, reaction_low_index=candidate.low_index,
            initial_stop=candidate.emergency_stop,
            # Immutable. Do NOT update this to the eventual holding-period high.
            entry_reaction_high=high if zone else None,
            structure_recovery_level=candidate.reference.price,
            atr=bar.atr_prior, information_known_through=bar.index)
        self.events.append(dict(index=bar.index, action="episode_entry_plan_frozen",
                                order_id=order_id, entry=deepcopy(self.entries[order_id])))

    def _freeze_preserved_support(self, entry, index, price, reason, kind):
        """Select once, at an actual local exit, using prior completed data.

        A major-daily structure below the local stop is required. A new top
        after that structure, a known close failure or a gap through its low
        disqualifies it. Never switch to a lower support after it later fails.
        """
        if (self.cfg.recovery_structure_policy != "preserved_support_after_local_stop"
                or kind != "reaction"
                or reason not in {"initial_risk_stop", "early_reaction_close_failure",
                                  "continuation_reaction_close_failure"}):
            return None
        choices = [r for r in self.structures
                   if r.id not in self.invalid_structures
                   and r.known_index < index and r.effective_index <= index
                   and r.low.available_index < index
                   and positive(r.low.price)
                   and r.low.price < entry["initial_stop"]
                   and price > r.low.price
                   and (self.last_top_index is None or r.known_index > self.last_top_index)]
        if not choices:
            return None
        r = max(choices, key=lambda item: (item.low.price, item.known_index, item.id))
        return dict(
            structure_id=r.id, high=deepcopy(r.high), low=deepcopy(r.low),
            known_index=r.known_index, effective_index=r.effective_index,
            breakout_index=r.breakout_index, scale=deepcopy(r.scale),
            frozen_at_exit_index=index, observation_known_through=max(self.bars, default=-1),
            local_initial_stop=entry["initial_stop"], actual_exit_price=price,
            selection="highest_valid_preknown_major_support_below_local_stop",
            invalidation="later_close_below_low_or_new_major_top",
            replacement_after_failure=False)

    def register_exit(self, position, index, at, reason, price, frozen=None, thesis=None):
        entry = self.entries.get(position.id)
        if entry is None or entry["information_known_through"] >= index:
            raise ValueError("exit requires a prior-close entry plan, not exit-day evidence")
        frozen, thesis = frozen or {}, thesis or {}
        top = reason.startswith(("campaign_major_top_", "campaign_double_top_"))
        active = thesis.get("active_protected_low")
        structural = (reason == "major_protected_low_close_failure"
                      or (reason == "trailing_protection_stop" and active is not None))
        if top:
            reference = frozen.get("reference")
            if reference is None or frozen.get("signal_index", index) >= index:
                raise ValueError("filled top SELL needs its frozen close-time reference")
            kind, level, atr = "top", reference.price, frozen["atr"]
        elif structural:
            reference = frozen.get("reference")
            if active is None and reference is None:
                raise ValueError("structural exit without known protection")
            kind = "structure"
            level = active.pivot.price if active is not None else reference.price
            atr = frozen.get("atr", entry["atr"])
        else:
            # A gap stop superseding a planned top SELL must NOT borrow the
            # abandoned top's reference or ATR.
            kind = "recovery_failure" if entry["family"] == "recovery" else "reaction"
            level = (entry["structure_recovery_level"] if entry["family"] == "recovery"
                     else entry["entry_reaction_high"])
            atr = entry["atr"]
        if not positive(level) or not positive(atr):
            raise ValueError("invalid frozen exit checkpoint")
        preserved = self._freeze_preserved_support(entry, index, price, reason, kind)
        self.preserved_support_failure = None
        self.context = dict(
            id=f"episode-exit:{position.id}:{index}", index=index, at=at,
            kind=kind, executed=True, executed_reason=reason,
            planned_reason=frozen.get("reason"), position_id=position.id,
            fill_price=price, reclaim_level=level, atr=atr,
            source_setup_id=entry["source_setup_id"], structure_id=entry["structure_id"],
            zone_id=entry["zone_id"], entry_reaction_high=entry["entry_reaction_high"],
            exit_structure_level=level if kind in {"top", "structure"} else None,
            recovery_structure_policy=self.cfg.recovery_structure_policy,
            preserved_major_support=preserved,
            information_known_through=entry["information_known_through"],
            failure_clock="actual_exit_not_planned_sell_or_later_high")
        self.last_exit_index, self.reclaim_count = index, 0
        self.events.append(dict(index=index, action="episode_actual_exit_frozen",
                                context=deepcopy(self.context)))
        if preserved is not None:
            self.events.append(dict(index=index, action="local_exit_major_support_frozen",
                                    context_id=self.context["id"], support=deepcopy(preserved)))
        if entry["zone_id"] is not None and reason in {
                "initial_risk_stop", "early_reaction_close_failure", "continuation_reaction_close_failure"}:
            fid = f"episode-failure:{position.id}:{index}"
            self.failures[fid] = dict(
                id=fid, exit_index=index, position_id=position.id, **deepcopy(entry),
                reclaim_closes=0, reclaimed_index=None)
            self.events.append(dict(index=index, action="reaction_episode_failed",
                                    failure=deepcopy(self.failures[fid]), pnl_used=False))

    def on_close(self, bar, top_sells=(), top_details=(), position=None):
        self.bars[bar.index] = bar
        for r in self.structures:
            if r.effective_index <= bar.index and bar.close < r.low.price:
                self.invalid_structures.add(r.id)
        if top_sells:
            # Flat-state warnings do not invent an exit or rewrite a filled
            # exit. They only require NEW recovery structure after this warning.
            self.last_top_index = bar.index
            self.events.append(dict(index=bar.index, action="recovery_top_warning",
                                    executed=False, actual_exit_unchanged=True))
        context, threshold = self.context, None
        support = context.get("preserved_major_support") if context else None
        if support is not None and self.preserved_support_failure is None:
            new_top = self.last_top_index is not None and self.last_top_index >= context["index"]
            if bar.close < support["low"].price or new_top:
                self.preserved_support_failure = dict(
                    index=bar.index, structure_id=support["structure_id"],
                    reason="new_major_top" if new_top else "close_below_preserved_support")
                self.events.append(dict(index=bar.index, action="preserved_major_support_invalidated",
                                        context_id=context["id"],
                                        failure=deepcopy(self.preserved_support_failure)))
        if context:
            threshold = context["reclaim_level"] + self.rules.structure_buffer_atr * context["atr"]
            cutoff = max(context["index"], self.last_top_index if self.last_top_index is not None else -1)
            self.reclaim_count = (self.reclaim_count + 1
                                  if bar.index > cutoff and bar.close > threshold else 0)
        for f in self.failures.values():
            threshold_f = f["entry_reaction_high"] + self.rules.structure_buffer_atr * f["atr"]
            f["reclaim_closes"] = (f["reclaim_closes"] + 1
                                     if bar.index > f["exit_index"] and bar.close > threshold_f else 0)
            if f["reclaimed_index"] is None and f["reclaim_closes"] >= self.cfg.exit_reclaim_closes:
                f["reclaimed_index"] = bar.index
                self.events.append(dict(index=bar.index, action="failed_episode_high_reclaimed",
                                        failure_id=f["id"], fixed_threshold=threshold_f))
        self.snapshot = dict(
            index=bar.index, policy="episode_scoped", context=deepcopy(context),
            reclaim_threshold=threshold, reclaim_closes=self.reclaim_count,
            reclaimed=bool(context and self.reclaim_count >= self.cfg.exit_reclaim_closes),
            last_top_index=self.last_top_index, last_exit_index=self.last_exit_index,
            failure_count=len(self.failures), probability=None, purchase_quota=None,
            preserved_support_failure=deepcopy(self.preserved_support_failure),
            ordinary_entries_require_exit_reclaim=False)

    def recovery_structure_status(self, structure, bar):
        """Qualify a structure for this exit without relaxing the fill clock.

        The v0.10 path is retained. The optional v0.11 path relaxes only the
        local low's origin time after a local stop. Reclaim, a NEW accepted
        breakout, preknown pivots and next-bar availability remain mandatory
        in RecoveryBuyBook. A current top/major-support failure cannot use it.
        """
        context = self.context
        cutoff = (max(context["index"], self.last_top_index if self.last_top_index is not None else -1)
                  if context else None)
        strict = bool(structure and cutoff is not None
                      and structure["low"].origin_index > cutoff
                      and structure["breakout_index"] > cutoff)
        result = dict(policy=self.cfg.recovery_structure_policy, cutoff=cutoff,
                      eligible=strict, strict_post_exit_low=strict,
                      path="post_exit_low" if strict else "waiting_post_exit_low",
                      preserved_support=None, local_checks={})
        if strict or self.cfg.recovery_structure_policy == "post_exit_low":
            return result

        support = context.get("preserved_major_support") if context else None
        low = structure["low"] if structure else None
        checks = dict(
            actual_local_exit=bool(context and context.get("executed")
                                   and context["kind"] == "reaction"
                                   and context["executed_reason"] in {
                                       "initial_risk_stop", "early_reaction_close_failure",
                                       "continuation_reaction_close_failure"}),
            major_support_frozen_before_exit=bool(support and context
                                                   and support["known_index"] < context["index"]
                                                   and support["effective_index"] <= context["index"]),
            major_support_survived=bool(support and self.preserved_support_failure is None
                                        and support["structure_id"] not in self.invalid_structures
                                        and bar.close > support["low"].price),
            no_top_since_exit=bool(context and (self.last_top_index is None
                                                or self.last_top_index < context["index"])),
            low_predates_exit=bool(low and context and low.origin_index < context["index"]),
            low_inside_preserved_structure=bool(low and support and low.price >= support["low"].price),
            breakout_after_exit=bool(structure and cutoff is not None
                                     and structure["breakout_index"] > cutoff),
        )
        # Do not resurrect an old, broken local low simply because the broader
        # support still holds. All inspected bars end at this decision close.
        history = ([b for i, b in self.bars.items() if low.origin_index <= i <= bar.index]
                   if low is not None and all(checks.values()) else [])
        checks["local_low_has_no_close_failure"] = bool(
            history and all(b.close >= low.price for b in history))
        allowed = all(checks.values())
        result.update(
            eligible=allowed,
            path="preserved_support_after_local_stop" if allowed else "waiting_qualified_recovery_structure",
            preserved_support=deepcopy(support), local_checks=checks,
            preserved_support_failure=deepcopy(self.preserved_support_failure))
        return result

    def _new_structure_before_contact(self, role, contact, failure, bar):
        if not role or contact is None or contact <= failure["exit_index"]:
            return False
        sid, exit_index = role.get("structure_id"), failure["exit_index"]
        if sid == failure["structure_id"]:
            return False
        for r in self.structures:
            if (r.id == sid and r.id not in self.invalid_structures
                    and r.low.origin_index > exit_index and r.breakout_index > exit_index
                    and r.known_index < contact and r.effective_index <= contact
                    and bar.close > r.low.price):
                return True
        if role.get("kind") == "upward_leg_support":
            anchors = role.get("anchors", {})
            low, high = anchors.get("low"), anchors.get("reference")
            breakout = role.get("breakout_index", -1)
            if (low is not None and high is not None
                    and low.origin_index > exit_index and breakout > exit_index
                    and low.available_index < breakout and high.available_index < breakout
                    and role["known_index"] < contact and role["effective_index"] <= contact
                    and bar.close > low.price):
                return True
        s = self.continuation_structure
        return bool(s and s["id"] == sid and s["low"].origin_index > exit_index
                    and s["breakout_index"] > exit_index and s["known_index"] < contact
                    and s["effective_index"] <= contact and bar.close > s["low"].price)

    def assess(self, offer, obs, role, bar, wave_snapshot):
        c = offer.decision
        detail = dict(path="v07_paths_plus_episode_memory", failures=[],
                      policy="episode_scoped", broad_overlap_used=False,
                      holding_period_high_used=False)
        if c.confirmation == "uptrend_recovery":
            # The separate book enforces its explicit post-exit or preserved-
            # support policy and a newly accepted post-exit breakout.
            # It is not an OB contact and cannot recycle the failed observation.
            detail["path"] = "recovery_book_must_verify_fresh_structure"
            return True, detail["path"], detail
        if self.cfg.failed_reaction_policy == "off":
            return True, "episode_failure_memory_off", detail
        if obs is None or c.zone is None:
            return False, "episode_gate_requires_actual_zone_observation", detail
        for f in self.failures.values():
            same_zone = (c.zone.id == f["zone_id"]
                         or self._zone_identity(c.zone) == f["zone_identity"])
            overlap = c.zone.wick_low <= f["zone_high"] and c.zone.wick_high >= f["zone_low"]
            same_structure = bool(role and role.get("structure_id") == f["structure_id"])
            same_observation = (obs.group_id == f["group_id"] or obs.contact_index == f["contact_index"])
            # Price overlap alone (including a wide weekly zone) is not enough.
            matches = same_zone or (overlap and (same_structure or same_observation))
            if not matches:
                continue
            fresh_contact = obs.contact_index > f["exit_index"]
            recovered = (f["reclaimed_index"] is not None
                         and obs.contact_index > f["reclaimed_index"])
            new_structure = self._new_structure_before_contact(role, obs.contact_index, f, bar)
            allowed = fresh_contact and (recovered or new_structure)
            detail["failures"].append(dict(
                failure_id=f["id"], same_zone=same_zone, same_structure=same_structure,
                same_observation=same_observation, fresh_contact=fresh_contact,
                fixed_entry_high=f["entry_reaction_high"], reclaimed_then_new_contact=recovered,
                new_confirmed_structure_before_contact=new_structure, passed=allowed))
        allowed = all(d["passed"] for d in detail["failures"])
        return allowed, ("episode_memory_allows_fresh_evidence" if allowed else
                         "same_failed_reaction_needs_reclaim_or_new_structure"), detail
