"""Revision 0.8 proposal: causal H -> L -> close above H and staged management.

AUTHORED_NOT_EXECUTED_OR_VALIDATED.  Numeric choices are design values.
The market book observes every close even while flat; position management uses
its records prospectively and never selects an entry by later success.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import ProtectedLow, SellEvidence, check_bar, positive
from nq_ict_macro_reversal_examples_20260916.proposal_02_macro_position_management import ManagementResult
from .structural_management import EntryScalePositionManager


class CausalWaveBook:
    """One daily scale: confirmed radius-R pivots and one ATR depth threshold.

    H is known before its close breakout. L is the actual interval minimum,
    not whichever nearby pivot happens to be convenient. L may be confirmed
    later; the record becomes usable only after that confirmation, never at L.
    No simultaneous 20-bar leg and 20-bar pullback requirements are applied.
    """

    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.bars = []
        self.waiting = {}
        self.consumed = set()
        self.records = {}
        self.invalid = set()
        self.active_id = None
        self.pending_id = None
        self.last_top_index = -1
        self.events = []
        self.snapshot = dict(phase="unestablished", valid=False, active=None)

    def _invalidate(self, rid, bar, reason):
        if rid not in self.invalid:
            self.invalid.add(rid)
            self.events.append(dict(index=bar.index, action="wave_record_invalidated",
                                    record_id=rid, reason=reason))

    def on_close(self, bar, pivots, top_sells=()):
        previous = self.bars[-1] if self.bars else None
        check_bar(bar, previous)
        known = [p for p in pivots if p.available_index <= bar.index and p.available_at <= bar.at
                 and p.radius >= self.cfg.wave_pivot_radius]
        if top_sells:
            self.last_top_index = bar.index
            for rid in self.records:
                self._invalidate(rid, bar, "canonical_top_sell")
            self.waiting.clear()
            self.active_id = self.pending_id = None
        else:
            # A close below support removes it from NEW entry qualification.
            # A held position separately uses its frozen two-close exit rule.
            for rid, record in self.records.items():
                if record.breakout_index <= bar.index and bar.close < record.pivot.price:
                    self._invalidate(rid, bar, "support_close_failure")
            if self.active_id in self.invalid:
                self.active_id = None
            if self.pending_id is not None:
                pending = self.records[self.pending_id]
                if pending.effective_from_index <= bar.index:
                    if self.pending_id not in self.invalid:
                        self.active_id = self.pending_id
                        self.events.append(dict(index=bar.index, action="wave_effective",
                                                record_id=self.active_id))
                    self.pending_id = None

            if previous is not None:
                for high in known:
                    if (high.kind != "high" or high.available_index >= bar.index
                            or not positive(high.atr_at_origin)):
                        continue
                    boundary = high.price + self.rules.structure_buffer_atr * high.atr_at_origin
                    if not previous.close <= boundary < bar.close:
                        continue
                    interval = [b for b in self.bars if high.origin_index < b.index]
                    if not interval:
                        continue
                    low_bar = min(interval, key=lambda b: (b.low, b.index))
                    if bar.low < low_bar.low:
                        self.events.append(dict(index=bar.index, action="wave_break_order_unknown",
                                                high_id=high.id, prior_low_index=low_bar.index,
                                                reason="deeper_low_and_breakout_on_same_daily_bar"))
                        continue
                    key = (high.id, low_bar.index)
                    if key in self.consumed:
                        continue
                    if high.price - low_bar.low < self.cfg.wave_min_pullback_atr * high.atr_at_origin:
                        continue
                    self.consumed.add(key)
                    rid = f"wave:{high.id}:{low_bar.index}:{bar.index}"
                    self.waiting[rid] = dict(high=high, low_bar=low_bar, breakout_index=bar.index)
                    self.events.append(dict(index=bar.index, action="wave_break_observed",
                                            record_id=rid, high=high, interval_low=low_bar,
                                            low_confirmation="pending", intrabar_order="unknown_data"))

            for rid, waiting in list(self.waiting.items()):
                high, low_bar = waiting["high"], waiting["low_bar"]
                if bar.close < low_bar.low:
                    self.events.append(dict(index=bar.index, action="wave_wait_cancelled",
                                            record_id=rid, reason="low_failed_before_confirmation"))
                    del self.waiting[rid]
                    continue
                lows = [p for p in known if p.kind == "low" and p.origin_index == low_bar.index
                        and p.price == low_bar.low]
                if not lows or bar.close <= high.price:
                    continue
                low = min(lows, key=lambda p: (p.available_index, p.id))
                record = ProtectedLow(low, high, waiting["breakout_index"], bar.index, bar.index + 1)
                self.records[rid] = record
                del self.waiting[rid]
                self.events.append(dict(index=bar.index, action="wave_confirmed", record_id=rid,
                                        record=record, effective_from_index=bar.index + 1))

            active = self.records.get(self.active_id)
            choices = [(rid, r) for rid, r in self.records.items() if rid not in self.invalid
                       and r.breakout_index > self.last_top_index
                       and bar.close > r.supporting_high.price
                       and (active is None or (r.pivot.origin_index > active.pivot.origin_index
                                              and r.pivot.price > active.pivot.price))]
            if choices:
                rid, _ = max(choices, key=lambda item: (item[1].pivot.origin_index,
                                                      item[1].breakout_index, item[0]))
                self.pending_id = rid

        self.bars.append(bar)
        active = self.records.get(self.active_id)
        self.snapshot = dict(index=bar.index, phase="advancing" if active else "unestablished_or_broken",
                             valid=active is not None, active=active, active_id=self.active_id,
                             pending_id=self.pending_id, last_top_index=self.last_top_index,
                             scale=dict(pivot_radius=self.cfg.wave_pivot_radius,
                                        minimum_depth_atr=self.cfg.wave_min_pullback_atr),
                             weekly_or_elliott_degree="not_asserted")


class ThesisWaveManager(EntryScalePositionManager):
    """Frozen entry-recovery thesis, then a monotonically raised wave support."""

    def __init__(self, rules, cfg, waves):
        super().__init__(rules)
        self.cfg, self.waves = cfg, waves
        self.entry_theses = {}
        self.phase = "flat"
        self.reaction_below_count = 0
        self.snapshot = dict(phase="flat")
        self.exit_detail = None

    def register_entry(self, order_id, candidate, role, observation=None, previous=None, bar=None):
        super().register_entry(order_id, candidate, role)
        boundary = candidate.zone.body_high
        atr = bar.atr_prior if bar is not None else candidate.low_atr
        self.entry_theses[order_id] = dict(
            decision_index=candidate.index, decision_at=candidate.at,
            family="continuation" if candidate.confirmation == "continuation_zone_reaction" else "reversal",
            recovery_boundary=boundary, reaction_low=candidate.low,
            reaction_low_index=candidate.low_index, source_zone_id=candidate.zone.id,
            failure_reclaim_level=max(boundary, previous.high if previous else boundary),
            initial_stop=candidate.emergency_stop, decision_atr=atr,
            failure_closes=self.cfg.reaction_failure_closes,
            management_mode="thesis_wave",
            wave_scale=dict(pivot_radius=self.cfg.wave_pivot_radius,
                            minimum_pullback_atr=self.cfg.wave_min_pullback_atr,
                            exit_closes=self.cfg.wave_exit_closes),
            reference=candidate.reference, phase="reaction", later_success_required_for_entry=False)

    def thesis_for(self, position_id):
        return deepcopy(self.entry_theses.get(position_id, {}))

    def exit_context_for(self, position_id):
        detail = self.thesis_for(position_id)
        detail.update(management_phase=self.phase if self.position_id == position_id else "reaction",
                      active_protected_low=(self.active if self.position_id == position_id else None))
        return detail

    def _result(self, bar, position, signal=None, proposal=None, reason="hold_thesis_or_wave"):
        thesis = self.entry_theses.get(position.id) if position else None
        self.snapshot = dict(index=bar.index, position_id=position.id if position else None,
                             phase=self.phase, thesis=deepcopy(thesis), active=self.active,
                             pending=self.pending, reaction_below_closes=self.reaction_below_count,
                             wave_below_closes=self.below_count, exit_detail=deepcopy(self.exit_detail))
        return ManagementResult(self.active, proposal, signal, reason)

    def on_close(self, bar, pivots, position):
        pid = position.id if position else None
        self.exit_detail = None
        if pid != self.position_id:
            self.active = self.pending = None
            self.below_count = self.reaction_below_count = 0
            self.position_id = pid
            self.phase = "reaction" if position else "flat"
            if position:
                self.events.append(dict(index=bar.index, action="reaction_management_started",
                                        position_id=pid, thesis=self.thesis_for(pid)))
        if position is None:
            return self._result(bar, None, reason="wave_observed_while_flat")
        thesis = self.entry_theses.get(pid)
        if thesis is None:
            raise ValueError("missing frozen entry thesis")
        if self.pending and self.pending.effective_from_index <= bar.index:
            self.active, self.pending = self.pending, None
            self.below_count = 0
            self.phase = "wave"
            self.events.append(dict(index=bar.index, action="wave_management_effective",
                                    position_id=pid, active=self.active,
                                    entry_rewritten=False, position_size_changed=False))

        if self.phase == "reaction":
            level = thesis["recovery_boundary"]
            self.reaction_below_count = self.reaction_below_count + 1 if bar.close < level else 0
            if self.reaction_below_count >= self.cfg.reaction_failure_closes:
                reason = ("continuation_reaction_close_failure" if thesis["family"] == "continuation"
                          else "early_reaction_close_failure")
                signal = SellEvidence(f"reaction-failure:{pid}:{bar.index}", bar.at, bar.index,
                                      reason, thesis["source_zone_id"], position.candidate.reference)
                self.exit_detail = dict(kind="reaction", invalidation_level=level,
                                        reclaim_level=thesis["failure_reclaim_level"],
                                        reference_role="entry_reference_not_invalidation_level",
                                        signal_index=bar.index, reason=reason)
                self.events.append(dict(index=bar.index, action="reaction_exit_proposed",
                                        position_id=pid, **deepcopy(self.exit_detail)))
                return self._result(bar, position, signal, reason=reason)
            # Do not promote a currently failing early reaction into a wider hold.
            if self.reaction_below_count:
                return self._result(bar, position, reason="entry_recovery_wait")
        elif self.active is not None:
            level = self.active.pivot.price
            self.below_count = self.below_count + 1 if bar.close < level else 0
            if self.below_count >= self.cfg.wave_exit_closes:
                reason = "major_protected_low_close_failure"
                signal = SellEvidence(f"thesis-wave-low:{pid}:{self.active.pivot.id}:{bar.index}",
                                      bar.at, bar.index, reason, self.active.pivot.id, self.active.pivot)
                self.exit_detail = dict(kind="structure", invalidation_level=level, reclaim_level=level,
                                        reference_role="protected_low", signal_index=bar.index, reason=reason)
                self.events.append(dict(index=bar.index, action="wave_exit_proposed",
                                        position_id=pid, **deepcopy(self.exit_detail)))
                return self._result(bar, position, signal, reason=reason)
            if self.below_count:
                return self._result(bar, position, reason="protected_low_recovery_wait")

        choices = []
        for rid, record in self.waves.records.items():
            if (rid in self.waves.invalid or record.decided_index > bar.index
                    or record.breakout_index < position.entry_index
                    or record.pivot.origin_index < position.candidate.low_index
                    or not position.candidate.emergency_stop < record.pivot.price < bar.close
                    or bar.close <= record.supporting_high.price):
                continue
            anchor = self.pending or self.active
            if anchor and (record.pivot.origin_index <= anchor.pivot.origin_index
                           or record.pivot.price <= anchor.pivot.price):
                continue
            choices.append(record)
        proposal = None
        if choices:
            chosen = max(choices, key=lambda r: (r.pivot.origin_index, r.breakout_index, r.pivot.id))
            proposal = ProtectedLow(chosen.pivot, chosen.supporting_high, chosen.breakout_index,
                                    bar.index, bar.index + 1)
            self.pending = proposal
            self.events.append(dict(index=bar.index, action="thesis_wave_protected_low_proposed",
                                    position_id=pid, proposal=proposal,
                                    effective_from_next_bar=True, early_entry_retained=True))
        return self._result(bar, position, proposal=proposal)
