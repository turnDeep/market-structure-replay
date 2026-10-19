"""Market-only phase observation and selectable risk-scope policies.

CycleRecoveryBook retains the old boundary/structure recovery policy.
IndependentAdvanceBook retains the revision 0.5 policy and supplies the phase
observer reused by revision 0.6. Its budget hook may be overridden without
changing BUY reactions or upward-phase confirmation. Fills never set a scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import positive


@dataclass(frozen=True)
class CycleContext:
    scope_id: str
    phase: str
    effective_index: int
    contact_after_index: int | None
    boundary: object | None
    completed_from: str | None = None
    structure_id: str | None = None
    advance_id: str | None = None
    advance_known_index: int | None = None
    advance_low_index: int | None = None


class CycleRecoveryBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = CycleContext("market-cycle:initial", "initial", 0, None, None)
        self.pending = None
        self.seen_reversal = None
        self.recovery = None
        self.recovery_index = None
        self.above = 0
        self.events = []
        self.completed_today = False
        self.invalid_structures = set()

    def on_close(self, bar, reversal, sells, structures, top_details=()):
        self.completed_today = False
        for record in structures:
            if (record.known_index <= bar.index and bar.close < record.low.price
                    and record.id not in self.invalid_structures):
                self.invalid_structures.add(record.id)
                self.events.append(dict(index=bar.index, action="cycle_structure_invalidated",
                                        structure_id=record.id, reason="close_below_support"))
        if self.pending is not None and self.pending.effective_index <= bar.index:
            self.context, self.pending = self.pending, None
        if reversal is not None and reversal.id != self.seen_reversal:
            self.seen_reversal = reversal.id
            self.context = CycleContext(reversal.id, "declining", bar.index,
                                        reversal.index, reversal)
            self.pending, self.recovery, self.recovery_index, self.above = None, None, None, 0
            self.events.append(dict(index=bar.index, action="market_cycle_started",
                                    context=self.context, depends_on_fills=False))
        context = self.context
        b = context.boundary
        if context.phase != "declining" or b is None or sells:
            return context
        if b.peak is None or b.atr is None or not positive(b.atr):
            return context
        boundary = b.peak + self.rules.structure_buffer_atr * b.atr
        self.above = self.above + 1 if bar.close > boundary else 0
        if self.recovery is not None and bar.close < self.recovery.low.price:
            self.events.append(dict(index=bar.index, action="cycle_recovery_failed",
                                    scope_id=context.scope_id, structure_id=self.recovery.id))
            self.recovery, self.recovery_index = None, None
        if self.above < self.cfg.cycle_reclaim_closes:
            return context
        ready = [r for r in structures if r.known_index <= bar.index
                 and r.id not in self.invalid_structures
                 and r.low.origin_index > b.index and r.breakout_index > b.index
                 and bar.close > r.high.price]
        if self.recovery is None:
            if ready:
                self.recovery = max(ready, key=lambda r: (r.known_index, r.id))
                self.recovery_index = bar.index
                self.events.append(dict(index=bar.index, action="cycle_boundary_and_structure_recovered",
                                        scope_id=context.scope_id, structure=self.recovery,
                                        recovered_level=boundary, new_budget=False))
            return context
        following = [r for r in ready if r.id != self.recovery.id
                     and r.low.origin_index > self.recovery_index
                     and r.low.price > self.recovery.low.price
                     and r.high.origin_index > self.recovery.low.origin_index
                     and r.known_index > self.recovery_index]
        if following:
            chosen = max(following, key=lambda r: (r.known_index, r.id))
            self.pending = CycleContext(
                f"market-cycle:advance:{context.scope_id}:{chosen.id}", "advancing",
                bar.index + 1, bar.index, None, context.scope_id, chosen.id)
            self.completed_today = True
            self.events.append(dict(index=bar.index, action="market_cycle_completed",
                                    previous_scope_id=context.scope_id, next_context=self.pending,
                                    first_recovery=self.recovery.id, higher_support=chosen,
                                    effective_from_next_packet=True, depends_on_fills=False))
        return context


class IndependentAdvanceBook:
    """Latest reversal context != budget identity; market-only revision 0.5.

Recovering a confirmed lower high can establish an upward phase; reclaiming
the old ultimate peak and then a second complete structure is not required.
The budget is NOT renewed at that recovery. Only a distinct major top formed
after the recovery low, with H2 after advance confirmation, renews it once.
"""
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = CycleContext("market-cycle:initial", "initial", 0, None, None)
        self.pending = None
        self.events = []
        self.invalid_structures = set()
        self.completed_today = False
        self.seen_reversal = None
        self.decline_start = None
        self.acceptance = {}
        self.advance = None
        self.used_advances = set()
        self.seen_parent_tops = set()
        self.latest_reversal = None

    def _budget_reset_decision(self, bar, reversal, sells, top_details):
        """Revision 0.5 policy, retained as an explicit comparison mode."""
        parent_id = reversal.parent_top_id or reversal.source_ids[0]
        a = self.advance
        distinct = bool(
            a is not None and a["id"] not in self.used_advances
            and parent_id not in self.seen_parent_tops
            and reversal.parent_h1_index is not None
            and reversal.parent_h1_index > a["low_index"]
            and reversal.peak_index is not None
            and reversal.peak_index > a["known_index"]
            and bar.index >= a["effective_index"])
        first = self.decline_start is None
        return dict(renew=first or distinct, uses_independent_advance=distinct,
                    reason="first_decline" if first else "new_top_after_independent_advance" if distinct
                    else "same_budget_new_reference_only", policy="independent_advance")

    def on_close(self, bar, reversal, sells, structures, top_details=()):
        self.completed_today = False
        if self.pending is not None and self.pending.effective_index <= bar.index:
            self.context, self.pending = self.pending, None
        for r in structures:
            if r.known_index <= bar.index and bar.close < r.low.price:
                self.invalid_structures.add(r.id)
        if self.advance and self.advance["structure_id"] in self.invalid_structures:
            self.events.append(dict(index=bar.index, action="independent_advance_failed",
                                    advance=deepcopy(self.advance), budget_renewed=False))
            self.advance, self.pending, self.acceptance = None, None, {}
            self.context = CycleContext(self.context.scope_id, "declining", bar.index,
                                        bar.index, self.latest_reversal)
        fresh = reversal is not None and reversal.id != self.seen_reversal
        if fresh:
            self.seen_reversal = reversal.id
            self.latest_reversal = reversal
            parent_id = reversal.parent_top_id or reversal.source_ids[0]
            a = self.advance
            first = self.decline_start is None
            decision = self._budget_reset_decision(bar, reversal, sells, top_details)
            renew = decision["renew"]
            distinct = renew and not first
            old_scope = self.context.scope_id
            scope = f"market-cycle:decline:{reversal.id}" if renew else old_scope
            if renew:
                self.decline_start = reversal.index
            if decision["uses_independent_advance"]:
                self.used_advances.add(a["id"])
            self.seen_parent_tops.add(parent_id)
            self.context = CycleContext(
                scope, "declining", bar.index, reversal.index, reversal,
                old_scope if distinct else self.context.completed_from,
                a["structure_id"] if a else None, a["id"] if a else None,
                a["known_index"] if a else None, a["low_index"] if a else None)
            # Refresh the observation boundary independently of portfolio fills.
            # Only the explicit policy above may create a NEW budget scope;
            # the old filled-position budget is never cleared or rewritten.
            self.pending, self.acceptance = None, {}
            self.events.append(dict(
                index=bar.index, action="latest_major_reversal_updated", context=self.context,
                previous_scope_id=old_scope, budget_renewed=renew,
                qualifying_advance=deepcopy(a), parent_top_id=parent_id,
                reason=decision["reason"], budget_decision=deepcopy(decision),
                depends_on_fills=False))
            self.advance = None
            return self.context
        if self.decline_start is None or self.context.phase != "declining" or sells:
            return self.context
        boundary = self.context.boundary
        after = boundary.index if boundary else self.decline_start
        ready = []
        for r in structures:
            if (r.id in self.invalid_structures or r.known_index > bar.index
                    or r.low.origin_index <= self.decline_start
                    or r.high.origin_index <= self.decline_start
                    or r.breakout_index <= after
                    or r.leg_bars < self.cfg.structure_min_leg_bars
                    or r.leg_atr < self.cfg.structure_min_leg_atr):
                continue
            level = r.high.price + self.rules.structure_buffer_atr * r.high.atr_at_origin
            above = bar.close > level and bar.low >= r.low.price
            self.acceptance[r.id] = self.acceptance.get(r.id, 0) + 1 if above else 0
            if self.acceptance[r.id] >= self.cfg.cycle_reclaim_closes:
                ready.append(r)
        if ready:
            r = max(ready, key=lambda v: (v.low.origin_index, v.known_index, v.id))
            a = dict(id=f"independent-advance:{r.id}:{bar.index}", structure_id=r.id,
                     low_index=r.low.origin_index, known_index=bar.index,
                     effective_index=bar.index + 1, reference=r.high, support=r.low)
            self.advance = a
            self.pending = CycleContext(
                self.context.scope_id, "advancing", bar.index + 1, bar.index, None,
                self.context.completed_from, r.id, a["id"], bar.index, r.low.origin_index)
            self.completed_today = True
            self.events.append(dict(
                index=bar.index, action="independent_upward_phase_confirmed", advance=deepcopy(a),
                next_context=self.pending, budget_renewed=False, depends_on_fills=False,
                boundary_policy="confirmed_rebound_high_not_old_ultimate_peak"))
        return self.context
