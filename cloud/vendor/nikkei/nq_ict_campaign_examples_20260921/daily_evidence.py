"""Streaming supplementary daily detectors. NOT run or validated.

Conceptual adaptation of the supplied daily swing specification, not an import
or claim of identical results. Main OB geometry remains the existing v09 OB.
FVG uses three observed consecutive sessions and prior-day ATR; its local
close-based lifecycle is separate from v09's wick-fill lifecycle.
"""
from __future__ import annotations

from nq_ict_macro_reversal_examples_20260916.common import (
    Bar, Rules, check_bar, check_inputs, positive,
)
from .common import CampaignConfig, Evidence, TacticalZone


class DailyEvidenceBook:
    def __init__(self, rules: Rules, cfg: CampaignConfig):
        self.rules, self.cfg = rules, cfg
        self.bars: list[Bar] = []
        self.events: list[Evidence] = []
        self.zone_events: list[dict] = []
        self.zones: dict[str, TacticalZone] = {}
        self.all_zones: dict[str, TacticalZone] = {}
        self.pending_sweeps: dict[str, dict] = {}
        self.broken_pivots: set[str] = set()
        self.direction = "unknown"
        self.counter = 0

    def emit(self, bar, kind, direction, low, high, source_id, origin_index, **details):
        self.counter += 1
        event = Evidence(f"daily-evidence:{self.counter}", bar.index, bar.at, kind,
                         direction, low, high, source_id, origin_index, details)
        self.events.append(event)
        return event

    def add_zone(self, bar, kind, direction, lo, hi, origin, parent=None):
        zid = f"campaign:{kind}:{direction}:{bar.index}:{parent or origin}"
        if zid in self.all_zones:
            raise ValueError("duplicate supplementary zone")
        z = TacticalZone(zid, kind, direction, lo, hi, origin, bar.index, parent)
        self.zones[zid] = self.all_zones[zid] = z
        self.zone_events.append(dict(index=bar.index, action="formed", zone=z))
        self.emit(bar, "zone_formed", direction, lo, hi, zid, origin,
                  zone_kind=kind, parent_id=parent, available_index=bar.index,
                  definition="campaign_daily_3bar_prior_atr_close_lifecycle")

    def on_close(self, bar, pivots, zones, htf, previous_week_levels=()):
        previous = self.bars[-1] if self.bars else None
        check_bar(bar, previous)
        check_inputs(bar, pivots, zones, htf)
        offset = len(self.events)
        # Only levels known BEFORE today's bar can be swept or broken.
        refs = list(previous_week_levels)
        prior = self.bars[-self.cfg.reference_lookback:]
        if len(prior) == self.cfg.reference_lookback:
            for side in ("high", "low"):
                price = (max(b.high for b in prior) if side == "high"
                         else min(b.low for b in prior))
                refs.append(dict(id=f"rolling:{side}:{prior[0].index}:{prior[-1].index}",
                                 side=side, price=price, known_index=prior[-1].index))
        for side in ("high", "low"):
            known = [p for p in pivots if p.kind == side and p.radius >= self.rules.pivot_radius
                     and p.available_index < bar.index]
            if known:
                p = max(known, key=lambda x: (x.origin_index, x.id))
                refs.append(dict(id=p.id, side=side, price=p.price, known_index=p.available_index))
        for ref in refs:
            if ref["known_index"] >= bar.index:
                raise ValueError("sweep reference is not pre-session information")
            crossed = bar.low < ref["price"] if ref["side"] == "low" else bar.high > ref["price"]
            if crossed and ref["id"] not in self.pending_sweeps:
                self.pending_sweeps[ref["id"]] = dict(ref=ref, crossed_index=bar.index)
                self.emit(bar, "level_crossed", "bullish" if ref["side"] == "low" else "bearish",
                          ref["price"], ref["price"], ref["id"], bar.index,
                          reference_known_index=ref["known_index"], recovery="not_yet_assessed")
        for rid, attempt in list(self.pending_sweeps.items()):
            ref, start = attempt["ref"], attempt["crossed_index"]
            if bar.index - start > self.cfg.sweep_reclaim_wait:
                self.emit(bar, "sweep_wait_expired", "bullish" if ref["side"] == "low" else "bearish",
                          ref["price"], ref["price"], rid, start)
                del self.pending_sweeps[rid]
                continue
            recovered = bar.close >= ref["price"] if ref["side"] == "low" else bar.close <= ref["price"]
            if recovered:
                self.emit(bar, "sweep_reclaim", "bullish" if ref["side"] == "low" else "bearish",
                          ref["price"], ref["price"], rid, start,
                          recovery_type="same_bar" if start == bar.index else "multi_bar",
                          reference_known_index=ref["known_index"], crossed_index=start,
                          intrabar_order="unknown_data")
                del self.pending_sweeps[rid]

        if previous is not None and positive(bar.atr_prior):
            breaks = []
            for side, direction in (("high", "bullish"), ("low", "bearish")):
                known = [p for p in pivots if p.kind == side
                         and p.radius >= self.rules.buy_reference_radius
                         and p.available_index < bar.index and positive(p.atr_at_origin)]
                if not known:
                    continue
                p = max(known, key=lambda x: (x.origin_index, -x.radius, x.id))
                buffer = self.rules.structure_buffer_atr * p.atr_at_origin
                boundary = p.price + buffer if side == "high" else p.price - buffer
                crossed = (previous.close <= boundary < bar.close if side == "high"
                           else previous.close >= boundary > bar.close)
                if crossed and p.id not in self.broken_pivots:
                    breaks.append((p, direction))
            # Conflicting supplied references are not assigned a made-up order.
            if len(breaks) > 1:
                self.emit(bar, "ambiguous_structure", "unknown", bar.low, bar.high,
                          "local_structure", bar.index, order="unknown_data")
            elif breaks:
                p, direction = breaks[0]
                role = ("INITIAL_BREAK" if self.direction == "unknown" else
                        "BOS" if self.direction == direction else "CHoCH")
                move = (bar.close - previous.close) * (1 if direction == "bullish" else -1)
                displacement = move / bar.atr_prior
                mss = role == "CHoCH" and displacement >= self.cfg.mss_min_displacement_atr
                self.emit(bar, "structure_break", direction, p.price, p.price, p.id, p.origin_index,
                          role=role, mss=mss, displacement_atr=displacement,
                          previous_direction=self.direction, reference_known_index=p.available_index,
                          definition="daily_close_choch_with_displacement_not_intraday_mss")
                self.broken_pivots.add(p.id)
                self.direction = direction

        inversions = []
        for zid, z in list(self.zones.items()):
            if bar.index - z.available_index >= self.cfg.tactical_zone_lifetime:
                self.zone_events.append(dict(index=bar.index, action="expired", zone_id=zid))
                del self.zones[zid]
                continue
            failed = bar.close < z.low if z.direction == "bullish" else bar.close > z.high
            if failed:
                self.zone_events.append(dict(index=bar.index, action="distal_close_break", zone_id=zid))
                del self.zones[zid]
                if z.kind == "fvg":
                    inversions.append(z)
                continue
            hit = z.available_index < bar.index and bar.low <= z.high and bar.high >= z.low
            rejected = (bar.close > z.high and bar.close > bar.open if z.direction == "bullish"
                        else bar.close < z.low and bar.close < bar.open)
            if hit and rejected:
                self.emit(bar, "zone_rejection", z.direction, z.low, z.high, zid, z.origin_index,
                          zone_kind=z.kind, available_index=z.available_index, parent_id=z.parent_id)
        for z in inversions:
            self.add_zone(bar, "ifvg", "bearish" if z.direction == "bullish" else "bullish",
                          z.low, z.high, z.origin_index, z.id)
        if len(self.bars) >= 2 and positive(bar.atr_prior):
            first = self.bars[-2]
            minimum = max(self.cfg.fvg_min_ticks * self.rules.tick_size,
                          self.cfg.fvg_min_atr * bar.atr_prior)
            if bar.low - first.high >= minimum:
                self.add_zone(bar, "fvg", "bullish", first.high, bar.low, self.bars[-1].index)
            if first.low - bar.high >= minimum:
                self.add_zone(bar, "fvg", "bearish", bar.high, first.low, self.bars[-1].index)

        # Do not silently replace v09's OB definition with the other script's OB.
        for v in zones:
            z = v.zone
            if not v.tradeable or z.kind != "ob_candidate" or z.available_index >= bar.index:
                continue
            hit = bar.low <= z.wick_high and bar.high >= z.wick_low
            rejected = (bar.close >= z.body_high and bar.close > bar.open if z.direction == "bullish"
                        else bar.close <= z.body_low and bar.close < bar.open)
            if hit and rejected:
                self.emit(bar, "zone_rejection", z.direction, z.wick_low, z.wick_high,
                          z.id, z.origin_index, zone_kind="ob_candidate", timeframe=z.timeframe,
                          available_index=z.available_index, definition="legacy_v09_ob")
        self.bars.append(bar)
        return tuple(self.events[offset:])
