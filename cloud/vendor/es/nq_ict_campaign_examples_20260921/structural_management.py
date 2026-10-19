"""Close-based management in the structural scale frozen at BUY planning.

Uses the same major-daily structure records as OB-role registration. Neither
weekly trend labels nor a later position's scale rewrite an earlier trade.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import ProtectedLow, SellEvidence
from nq_ict_macro_reversal_examples_20260916.proposal_02_macro_position_management import ManagementResult


class EntryScalePositionManager:
    def __init__(self, rules):
        self.rules = rules
        self.position_id = None
        self.active = None
        self.pending = None
        self.below_count = 0
        self.events = []
        self.closed_below = set()
        self.entry_profiles = {}
        self.structures = ()

    def register_entry(self, order_id, candidate, role):
        if not role or not role.get("scale") or role["known_index"] >= candidate.touch_index:
            raise ValueError("entry-scale management needs a precontact structural role")
        if (role.get("kind") not in {"major_origin", "established_major_support", "upward_leg_support",
                                    "continuation_support"}
                or role["effective_index"] > candidate.touch_index):
            raise ValueError("unknown or not yet effective entry management profile")
        self.entry_profiles[order_id] = deepcopy(role)

    def supply_structures(self, structures):
        self.structures = structures

    def register_recovery_entry(self, order_id, candidate, role):
        """Same holding scale; validate a structure clock, NOT a fictitious touch."""
        if (candidate.confirmation != "uptrend_recovery" or candidate.zone is not None
                or candidate.touch_index is not None or not role or not role.get("scale")
                or role.get("kind") != "recovery_structure"
                or role["structure_id"] != candidate.episode_id
                or role["known_index"] >= candidate.index
                or role["effective_index"] > candidate.index):
            raise ValueError("recovery entry requires a predecision effective structure")
        self.entry_profiles[order_id] = deepcopy(role)

    def on_close(self, bar, pivots, position):
        for r in self.structures:
            if r.effective_index <= bar.index and bar.close < r.low.price:
                self.closed_below.add(r.id)
        pid = position.id if position else None
        if pid != self.position_id:
            self.active, self.pending, self.below_count = None, None, 0
            self.position_id = pid
        if position is None:
            return ManagementResult(None, None, None, "major_structure_observed_without_position")
        role = self.entry_profiles.get(pid)
        if role is None:
            raise ValueError("missing immutable entry structural scale")
        scale = role["scale"]
        if self.pending and self.pending.effective_from_index <= bar.index:
            self.active, self.pending, self.below_count = self.pending, None, 0
        if self.active:
            self.below_count = self.below_count + 1 if bar.close < self.active.pivot.price else 0
            if self.below_count >= self.rules.structural_exit_closes:
                signal = SellEvidence(f"entry-scale-low:{pid}:{self.active.pivot.id}:{bar.index}",
                                      bar.at, bar.index, "major_protected_low_close_failure",
                                      self.active.pivot.id, self.active.pivot)
                return ManagementResult(self.active, None, signal, "entry_scale_structure_failed")
        choices = []
        for r in self.structures:
            if (r.known_index > bar.index or r.id in self.closed_below
                    or r.low.origin_index <= position.entry_index
                    or r.high.origin_index <= position.candidate.low_index
                    or r.high.price < position.candidate.reference.price
                    or r.high.radius < scale["pivot_radius"] or r.low.radius < scale["pivot_radius"]
                    or r.leg_bars < scale["min_leg_bars"] or r.leg_atr < scale["min_leg_atr"]
                    or r.pullback_bars < scale["min_pullback_bars"]
                    or r.pullback_atr < scale["min_pullback_atr"]
                    or not position.candidate.emergency_stop < r.low.price < bar.close
                    or bar.close <= r.high.price):
                continue
            if self.active and (r.low.origin_index <= self.active.pivot.origin_index
                                or r.low.price <= self.active.pivot.price):
                continue
            choices.append(r)
        chosen = max(choices, key=lambda r: (r.low.origin_index, r.id)) if choices else None
        proposal = None
        if chosen:
            proposal = ProtectedLow(chosen.low, chosen.high, chosen.breakout_index, bar.index, bar.index + 1)
            self.pending = proposal
            self.events.append(dict(index=bar.index, action="entry_scale_protected_low_proposed",
                                    position_id=pid, entry_structure_id=role["structure_id"],
                                    protecting_structure_id=chosen.id, scale=deepcopy(scale),
                                    proposal=proposal, effective_from_next_bar=True))
        return ManagementResult(self.active, proposal, None, "hold_entry_scale_wave")
