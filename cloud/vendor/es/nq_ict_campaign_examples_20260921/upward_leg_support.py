"""Bounded OB formation attribution; revision 0.5, AUTHORING ONLY.

An upward-leg role is weaker/different evidence than a major reversal origin.
Everything is registered on its actual knowledge date. No current reaction,
future-confirmed cohort, lifetime maximum departure or calendar date qualifies it.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import check_bar, positive


class UpwardLegSupportBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.bars = []
        self.seen = set()
        self.pending = {}
        self.roles = {}
        self.events = []

    def before_close(self, zones, bar):
        result = {}
        for view in zones:
            z = view.zone
            if z.kind != "ob_candidate" or z.direction != "bullish":
                continue
            role = self.roles.get(z.id)
            result[z.id] = dict(
                classification="known_upward_leg_role" if role else "no_prior_upward_leg_role",
                known_through_index=bar.index - 1,
                role=deepcopy(role) if role and role["known_index"] < bar.index else None)
        return result

    def on_close(self, bar, pivots, zones):
        check_bar(bar, self.bars[-1] if self.bars else None)
        self.bars.append(bar)
        views = {v.zone.id: v for v in zones}
        windows = {"1D": self.cfg.leg_ob_window_daily,
                   "1W": self.cfg.leg_ob_window_weekly,
                   "1M": self.cfg.leg_ob_window_monthly}
        for view in zones:
            z = view.zone
            if (z.id in self.seen or z.kind != "ob_candidate" or z.direction != "bullish"
                    or z.available_index > bar.index):
                continue
            self.seen.add(z.id)
            # Freeze anchors when the source OB first becomes available.
            # They must have been known BEFORE its origin, not supplied later.
            highs = [p for p in pivots if p.kind == "high"
                     and p.radius >= self.rules.buy_reference_radius
                     and p.available_index < z.origin_index and p.price > z.wick_low]
            ref = max(highs, key=lambda p: (p.origin_index, p.id)) if highs else None
            lows = [p for p in pivots if p.kind == "low"
                    and p.radius >= self.rules.pivot_radius
                    and p.available_index < z.origin_index
                    and 0 < z.origin_index - p.origin_index <= self.cfg.top_leg_lookback
                    and p.price <= z.wick_low and positive(p.atr_at_origin)
                    and (ref is not None and p.origin_index < ref.origin_index)
                    and not any(b.low < p.price for b in self.bars
                                if p.origin_index < b.index <= bar.index)]
            low = max(lows, key=lambda p: (p.origin_index, p.id)) if lows else None
            window = windows.get(z.timeframe)
            origin_bar = next((b for b in self.bars if b.index == z.origin_index), None)
            atr = origin_bar.atr_prior if origin_bar else None
            if (not view.observable or ref is None or low is None or window is None
                    or not positive(atr) or bar.index != z.available_index):
                self.events.append(dict(index=bar.index, action="leg_ob_not_registered", zone_id=z.id,
                                        reason="missing_preorigin_anchors_or_source_or_supported_timeframe"))
                continue
            self.pending[z.id] = dict(zone=z, reference=ref, low=low, atr=atr,
                                      deadline=z.available_index + window, registered_index=bar.index,
                                      closes=0, first_break=None, first_departure=None)
            self.events.append(dict(index=bar.index, action="leg_ob_formation_watch", zone_id=z.id,
                                    reference=ref, low=low, deadline=z.available_index + window,
                                    anchors_frozen=True, source_timeframe=z.timeframe,
                                    timeframe_of_leg="daily_proxy"))
        for zid, watch in list(self.pending.items()):
            z, ref, low = watch["zone"], watch["reference"], watch["low"]
            view = views.get(zid)
            if (bar.index > watch["deadline"] or view is None or not view.observable
                    or bar.low < low.price or bar.close < z.wick_low):
                self.events.append(dict(index=bar.index, action="leg_ob_watch_retired", zone_id=zid,
                                        reason="bounded_window_or_support_or_source_failed"))
                del self.pending[zid]
                continue
            # Inspect already observed formation bars once; do not backdate
            # availability even if the actual price crossing preceded activation.
            recent = ([b for b in self.bars if z.origin_index <= b.index <= bar.index]
                      if bar.index == watch["registered_index"] else [bar])
            boundary = ref.price + self.rules.structure_buffer_atr * watch["atr"]
            departure = z.wick_high + self.cfg.leg_ob_departure_atr * watch["atr"]
            for b in recent:
                prior = self.bars[b.index - 1] if b.index > 0 else None
                if prior is not None and prior.close <= boundary < b.close:
                    watch["first_break"] = b.index
                if b.close >= departure and watch["first_departure"] is None:
                    watch["first_departure"] = b.index
                above = b.close > boundary and b.close >= departure
                watch["closes"] = watch["closes"] + 1 if above else 0
                if not above:
                    watch["first_break"] = None
            leg_bars = bar.index - low.origin_index
            leg_atr = (bar.close - low.price) / low.atr_at_origin
            if (watch["first_break"] is None or watch["first_departure"] is None
                    or watch["closes"] < self.cfg.leg_ob_accept_closes
                    or leg_bars < self.cfg.structure_min_leg_bars
                    or leg_atr < self.cfg.structure_min_leg_atr or not view.tradeable):
                continue
            role = dict(
                zone_id=zid, structure_id=f"upward-leg-ob:{zid}:{bar.index}",
                kind="upward_leg_support", grade=0, known_index=bar.index,
                effective_index=bar.index + 1, source_timeframe=z.timeframe,
                high_id=ref.id, low_id=low.id, low_index=low.origin_index,
                breakout_index=watch["first_break"], departure_index=watch["first_departure"],
                attribution_deadline=watch["deadline"], formation_atr=watch["atr"],
                leg_bars=leg_bars, leg_atr=leg_atr, acceptance_closes=watch["closes"],
                anchors=dict(low=low, reference=ref),
                scale=dict(tier="major_daily_leg", pivot_radius=self.rules.pivot_radius,
                           min_leg_bars=self.cfg.structure_min_leg_bars,
                           min_leg_atr=self.cfg.structure_min_leg_atr,
                           min_pullback_bars=self.rules.management_pullback_bars,
                           min_pullback_atr=self.rules.management_pullback_atr),
                definition="bounded_upward_leg_attribution_not_major_bottom_or_probability")
            self.roles[zid] = role
            self.events.append(dict(index=bar.index, action="upward_leg_ob_registered", role=deepcopy(role)))
            del self.pending[zid]
