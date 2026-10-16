"""Revision 0.4 proposal: causal major-daily structures and precontact OB roles.

No runner, data loading or fitted parameters. A weekly OB is still a weekly
OB; a daily radius-5 pivot is explicitly a daily proxy, never a weekly pivot.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from nq_ict_macro_reversal_examples_20260916.common import check_bar, positive


@dataclass(frozen=True)
class MajorStructure:
    id: str
    high: object
    low: object
    breakout_index: int
    known_index: int
    effective_index: int
    leg_bars: int
    leg_atr: float
    pullback_bars: int
    pullback_atr: float
    scale: dict


class MajorStructureBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.bars = []
        self.pending = {}
        self.records = []
        self.finished = set()
        self.roles = {}
        self.events = []
        self.role_events = []

    def before_close(self, zones, bar):
        """Called BEFORE processing this close; contact cannot promote itself."""
        result = {}
        for view in zones:
            z = view.zone
            if z.kind != "ob_candidate" or z.direction != "bullish":
                continue
            role = self.roles.get(z.id)
            result[z.id] = dict(
                classification="known_major_role" if role else "no_prior_major_role",
                known_through_index=bar.index - 1,
                role=deepcopy(role) if role and role["known_index"] < bar.index else None,
            )
        return result

    def on_close(self, bar, pivots, zones):
        previous = self.bars[-1] if self.bars else None
        check_bar(bar, previous)
        if any(p.available_index > bar.index for p in pivots):
            raise ValueError("future structural pivot")
        # One previous major high, a later major low and an actual upward close
        # crossing. Size is measured using ATR at the old high/low, not future ATR.
        if previous is not None:
            for high in pivots:
                if (high.kind != "high" or high.radius < self.rules.pivot_radius
                        or high.available_index >= bar.index or not positive(high.atr_at_origin)):
                    continue
                boundary = high.price + self.rules.structure_buffer_atr * high.atr_at_origin
                if not previous.close <= boundary < bar.close:
                    continue
                interval = [b for b in self.bars if high.origin_index < b.index < bar.index]
                if not interval:
                    continue
                low_bar = min(interval, key=lambda b: (b.low, b.index))
                identity = (high.id, low_bar.index)
                if identity in self.finished or identity in self.pending or not positive(low_bar.atr_prior):
                    continue
                leg_bars = bar.index - low_bar.index
                leg_atr = (bar.close - low_bar.low) / low_bar.atr_prior
                pullback_bars = low_bar.index - high.origin_index
                pullback_atr = (high.price - low_bar.low) / high.atr_at_origin
                if (leg_bars < self.cfg.structure_min_leg_bars
                        or leg_atr < self.cfg.structure_min_leg_atr
                        or pullback_bars < self.rules.management_pullback_bars
                        or pullback_atr < self.rules.management_pullback_atr):
                    continue
                self.pending[identity] = dict(
                    high=high, low_bar=low_bar, index=bar.index, boundary=boundary, closes=0,
                    leg_bars=leg_bars, leg_atr=leg_atr,
                    pullback_bars=pullback_bars, pullback_atr=pullback_atr)
                self.events.append(dict(index=bar.index, action="major_break_observed",
                                        high=high, low_index=low_bar.index,
                                        low_confirmation="pending", order_clock="daily_close"))
        self.bars.append(bar)
        new_records = []
        for identity, attempt in list(self.pending.items()):
            low_bar, high = attempt["low_bar"], attempt["high"]
            if (bar.close <= attempt["boundary"] or bar.low < low_bar.low
                    or bar.index - attempt["index"] > self.rules.reaction_lifetime):
                self.events.append(dict(index=bar.index, action="major_break_attempt_retired",
                                        high_id=high.id, first_break_index=attempt["index"]))
                del self.pending[identity]
                continue
            attempt["closes"] += 1
            lows = [p for p in pivots if p.kind == "low" and p.radius >= self.rules.pivot_radius
                    and p.origin_index == low_bar.index and p.price == low_bar.low
                    and p.available_index <= bar.index]
            if not lows or attempt["closes"] < self.rules.breakout_accept_closes:
                continue
            low = min(lows, key=lambda p: (p.available_index, p.id))
            scale = dict(tier="major_daily", pivot_radius=self.rules.pivot_radius,
                         min_leg_bars=self.cfg.structure_min_leg_bars,
                         min_leg_atr=self.cfg.structure_min_leg_atr,
                         min_pullback_bars=self.rules.management_pullback_bars,
                         min_pullback_atr=self.rules.management_pullback_atr)
            record = MajorStructure(
                f"major-structure:{high.id}:{low.id}:{attempt['index']}", high, low,
                attempt["index"], bar.index, bar.index + 1,
                attempt["leg_bars"], attempt["leg_atr"],
                attempt["pullback_bars"], attempt["pullback_atr"], scale)
            self.records.append(record)
            new_records.append(record)
            self.finished.add(identity)
            del self.pending[identity]
            self.events.append(dict(index=bar.index, action="major_structure_confirmed", structure=record))
            self._link_zones(record, zones, bar)
        return tuple(new_records)

    def _link_zones(self, record, zones, bar):
        low = record.low
        margin = self.cfg.structure_zone_tolerance_atr * low.atr_at_origin
        for view in zones:
            z = view.zone
            if (not view.observable or z.kind != "ob_candidate" or z.direction != "bullish"
                    or z.available_index > record.breakout_index
                    or not z.wick_low - margin <= low.price <= z.wick_high + margin):
                continue
            origin = abs(z.origin_index - low.origin_index) <= self.cfg.structure_origin_window
            touched = any(
                z.available_index < b.index <= record.breakout_index
                and abs(b.index - low.origin_index) <= self.cfg.structure_origin_window
                and b.low <= z.wick_high and b.high >= z.wick_low
                for b in self.bars if b.index <= record.breakout_index)
            if not origin and not touched:
                continue
            role = dict(zone_id=z.id, structure_id=record.id,
                        kind="major_origin" if origin else "established_major_support",
                        grade=2 if origin else 1, known_index=bar.index,
                        effective_index=bar.index + 1, source_timeframe=z.timeframe,
                        high_id=record.high.id, low_id=low.id,
                        low_index=low.origin_index, breakout_index=record.breakout_index,
                        scale=deepcopy(record.scale), definition="historical_structural_role_not_win_probability")
            prior = self.roles.get(z.id)
            # Never prefer an OB simply because its maximum historical departure grew.
            if prior is None or (role["grade"], role["known_index"]) > (prior["grade"], prior["known_index"]):
                self.roles[z.id] = role
                self.role_events.append(dict(index=bar.index, action="ob_role_registered", role=deepcopy(role),
                                             prior_structure_id=prior["structure_id"] if prior else None))


def selected_support_role(observation, cfg):
    """Choose a frozen precontact role, never promote one using this rebound."""
    if observation is None:
        return None
    structural = observation.features.get("structural_support", {})
    roles = [structural.get("role")]
    if cfg.support_selection == "structural_or_leg":
        roles.append(structural.get("upward_leg_role"))
    eligible = [r for r in roles if r is not None and r["known_index"] < observation.contact_index
                and r["effective_index"] <= observation.contact_index
                and structural.get("known_through_index", observation.contact_index) < observation.contact_index]
    return deepcopy(max(eligible, key=lambda r: (r["grade"], r["known_index"], r["structure_id"]))) if eligible else None


def reaction_description(offer, observation, bar, previous=None):
    if observation is None or not positive(observation.low_atr):
        return dict(status="unknown_data")
    c, obs = offer.decision, observation
    span = bar.high - bar.low
    reference_drop = c.reference.price - obs.low
    reclaim = ("touch_recovery" if obs.low >= c.zone.wick_low else
               "same_bar_sweep_reclaim" if obs.low_index == obs.recovery_index else "multi_bar_reclaim")
    return dict(
        status="observed", known_index=bar.index, atr_at_reaction_low=obs.low_atr,
        rebound_atr=(bar.close - obs.low) / obs.low_atr,
        recovery_fraction=(bar.close - obs.low) / reference_drop if reference_drop > 0 else None,
        close_location=(bar.close - bar.low) / span if span > 0 else None,
        contact_to_recovery_bars=obs.recovery_index - obs.contact_index,
        low_to_decision_bars=bar.index - obs.low_index,
        recovery_type=reclaim, recovered_boundary=c.zone.body_high,
        distance_above_recovered_boundary_atr=(bar.close - c.zone.body_high) / obs.low_atr,
        structural_reference=c.reference.price,
        structural_reference_crossed=bar.close > c.reference.price,
        prior_high=previous.high if previous else None,
        prior_high_close_break=bar.close > previous.high if previous else None,
        zone_recovered_now=bar.close >= c.zone.body_high and bar.close > bar.open,
        intrabar_order="unknown_data", route=offer.route,
    )


def reaction_allowed(description, cfg, role=None):
    if role and role["kind"] == "upward_leg_support":
        if not (description.get("zone_recovered_now") and description.get("prior_high_close_break")):
            return False, "leg_ob_requires_current_reclaim_and_prior_high_close_break"
    if cfg.reaction_filter == "observe_only":
        return True, "reaction_metrics_observed_only"
    if description.get("status") != "observed" or description.get("close_location") is None:
        return False, "reaction_metrics_missing"
    if (description["rebound_atr"] < cfg.reaction_min_rebound_atr
            or description["close_location"] < cfg.reaction_min_close_location):
        return False, "reaction_below_provisional_thresholds"
    return True, "reaction_thresholds_met"
