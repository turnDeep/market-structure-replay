"""Explicit H1 -> pullback -> H2 test -> rejection. UNEXECUTED proposal.

H2 is an observed high-area test unless a right-confirmed pivot is actually
available. Levels and ATR margins are frozen before counting close breaks.
One H1/H2 attempt emits at most once; a revised high cannot reset its budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nq_ict_macro_reversal_examples_20260916.common import SellEvidence, check_bar, positive


@dataclass
class DoubleTop:
    id: str
    h1: object
    h2_index: int
    h2_price: float
    touch_index: int
    neckline: object
    early_support: object | None
    levels_known_index: int
    atr: float
    upper: float
    qualification: dict = field(default_factory=dict)
    cluster_id: str | None = None
    suppressed: bool = False
    h2_pivot_id: str | None = None
    outside_closes: int = 0
    below_counts: dict = field(default_factory=lambda: {"major": 0, "early": 0})
    first_breaks: dict = field(default_factory=lambda: {"major": None, "early": None})
    emitted: bool = False
    evidence_ids: list = field(default_factory=list)


class DoubleTopDetector:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.bars = []
        self.active = {}
        self.consumed_h1 = set()
        self.events = []
        self.major_signals = []
        self.early_signals = []
        self.leg_cache = {}
        self.clusters = {}
        self.emitted_clusters = set()

    def _major_leg(self, h1, pivots, bar):
        if self.cfg.double_top_qualification == "legacy":
            return dict(qualified=True, mode="legacy")
        if h1.id in self.leg_cache:
            return self.leg_cache[h1.id]
        choices = []
        for p in pivots:
            age = h1.origin_index - p.origin_index
            if (p.kind != "low" or p.radius < self.cfg.top_support_radius
                    or p.available_index >= h1.origin_index
                    or not self.cfg.top_leg_min_bars <= age <= self.cfg.top_leg_lookback):
                continue
            rise = h1.price - p.price
            if rise < self.cfg.top_leg_min_atr * h1.atr_at_origin:
                continue
            # The leg starts at a known low that was not already undercut on
            # the way to H1. No future post-H1 rally is used for qualification.
            if any(b.low < p.price for b in self.bars
                   if p.origin_index < b.index <= h1.origin_index):
                continue
            choices.append(p)
        low = max(choices, key=lambda p: (p.origin_index, p.id)) if choices else None
        result = dict(qualified=low is not None, mode="major_leg", known_index=bar.index,
                      reason="preceding_major_rally" if low else "preceding_major_rally_not_available",
                      impulse_low=low, impulse_bars=h1.origin_index - low.origin_index if low else None,
                      impulse_points=h1.price - low.price if low else None,
                      impulse_atr=(h1.price - low.price) / h1.atr_at_origin if low else None)
        self.leg_cache[h1.id] = result
        return result

    def _cluster(self, h1, bar, pair_id):
        if self.cfg.double_top_qualification == "legacy":
            return pair_id
        matches = [g for g in self.clusters.values()
                   if abs(h1.origin_index - g["origin_index"]) <= self.cfg.top_cluster_bars
                   and abs(h1.price - g["price"]) <= self.cfg.top_cluster_atr * g["atr"]]
        if matches:
            return min(matches, key=lambda g: (g["known_index"], g["id"]))["id"]
        cid = f"major-top-cluster:{h1.origin_index}:{bar.index}"
        self.clusters[cid] = dict(id=cid, origin_index=h1.origin_index, price=h1.price,
                                  atr=h1.atr_at_origin, known_index=bar.index)
        return cid

    def on_close(self, bar, pivots, evidence):
        previous = self.bars[-1] if self.bars else None
        check_bar(bar, previous)
        if any(p.available_index > bar.index or p.available_at > bar.at for p in pivots):
            raise ValueError("future pivot in double-top detector")
        if previous is not None:
            for h1 in sorted(pivots, key=lambda p: (p.origin_index, -p.radius, p.id)):
                age = bar.index - h1.origin_index
                if (h1.kind != "high" or h1.radius < self.rules.pivot_radius
                        or h1.available_index >= bar.index or not positive(h1.atr_at_origin)
                        or not self.cfg.double_top_min_separation <= age <= self.cfg.double_top_max_separation
                        or h1.origin_index in self.consumed_h1 or h1.origin_index in self.active):
                    continue
                lower = h1.price - self.cfg.double_top_tolerance_atr * h1.atr_at_origin
                upper = h1.price + self.cfg.double_top_max_sweep_atr * h1.atr_at_origin
                if not (previous.close < lower and bar.high >= lower and bar.low <= upper
                        and bar.high <= upper):
                    continue
                interval = [b for b in self.bars if h1.origin_index < b.index < bar.index]
                outside = 0
                accepted = False
                for b in interval:
                    outside = outside + 1 if b.close > upper else 0
                    if outside >= self.rules.breakout_accept_closes:
                        accepted = True
                        break
                if accepted:
                    continue
                qualification = self._major_leg(h1, pivots, bar)
                if not qualification["qualified"]:
                    self.events.append(dict(index=bar.index, action="pair_not_qualified",
                                            h1_id=h1.id, qualification=dict(qualification)))
                    continue
                trough = min(interval, key=lambda b: (b.low, b.index)) if interval else None
                if (trough is None or h1.price - trough.low < self.cfg.double_top_min_depth_atr * h1.atr_at_origin
                        or bar.high - trough.low < self.cfg.top_support_rise_atr * h1.atr_at_origin
                        or bar.index - trough.index < self.cfg.top_support_bars):
                    continue
                if (self.cfg.double_top_qualification == "major_leg"
                        and (h1.price - trough.low) / qualification["impulse_points"]
                        < self.cfg.top_pullback_min_fraction):
                    self.events.append(dict(index=bar.index, action="pair_not_qualified", h1_id=h1.id,
                                            reason="pullback_too_small_relative_to_prior_leg"))
                    continue
                lows = [p for p in pivots if p.kind == "low" and p.origin_index == trough.index
                        and p.price == trough.low and p.radius >= self.cfg.top_support_radius
                        and p.available_index < bar.index]
                if not lows:
                    self.events.append(dict(index=bar.index, action="pair_not_ready", h1_id=h1.id,
                                            reason="intervening_low_pivot_not_yet_available"))
                    continue
                neck = min(lows, key=lambda p: (p.available_index, p.id))
                local = [p for p in pivots if p.kind == "low"
                         and p.radius >= self.rules.sell_support_radius
                         and neck.origin_index < p.origin_index < bar.index
                         and p.available_index < bar.index and p.price > neck.price
                         and not any(b.close < p.price for b in interval if b.index >= p.available_index)]
                support = max(local, key=lambda p: (p.origin_index, p.id)) if local else None
                pair = DoubleTop(f"double-top:{h1.id}:h2-test:{bar.index}", h1,
                                 bar.index, bar.high, bar.index, neck, support,
                                 bar.index, h1.atr_at_origin, upper)
                pair.qualification = dict(qualification)
                pair.cluster_id = self._cluster(h1, bar, pair.id)
                self.active[h1.origin_index] = pair
                self.events.append(dict(index=bar.index, action="double_top_pair_observed",
                                        top_id=pair.id, h1=h1, neckline=neck,
                                        h2_status="observed_retest_not_confirmed_pivot",
                                        early_support=support, qualification=pair.qualification,
                                        cluster_id=pair.cluster_id))

        major, early, details = [], [], []
        for origin, t in list(self.active.items()):
            # Emitted pairs are terminal; later data cannot un-emit or rewrite them.
            if t.emitted:
                continue
            if bar.index - t.touch_index > self.cfg.double_top_reaction_window:
                self._retire(origin, bar, "pair_expired")
                continue
            t.outside_closes = t.outside_closes + 1 if bar.close > t.upper else 0
            if (bar.high > t.upper or t.outside_closes >= self.rules.breakout_accept_closes):
                self._retire(origin, bar, "maximum_h2_sweep_exceeded_or_breakout_accepted")
                continue
            if bar.high > t.h2_price:
                t.h2_price, t.h2_index, t.h2_pivot_id = bar.high, bar.index, None
                t.below_counts = {"major": 0, "early": 0}
                t.first_breaks = {"major": None, "early": None}
            h2s = [p for p in pivots if p.kind == "high" and p.radius >= self.rules.buy_reference_radius
                   and p.origin_index == t.h2_index and p.price == t.h2_price]
            if h2s:
                t.h2_pivot_id = min(h2s, key=lambda p: (p.available_index, p.id)).id
            # A short support may become known after H2. Adopt it NOW and count
            # future closes only, never reinterpret earlier failures as signals.
            if t.early_support is None:
                local = [p for p in pivots if p.kind == "low" and p.radius >= self.rules.sell_support_radius
                         and t.neckline.origin_index < p.origin_index < t.h2_index
                         and p.available_index < bar.index and p.price > t.neckline.price
                         and not any(b.close < p.price for b in self.bars
                                     if p.available_index <= b.index < bar.index)]
                if local:
                    t.early_support = max(local, key=lambda p: (p.origin_index, p.id))
                    t.levels_known_index = bar.index
            for e in evidence:
                if (e.direction == "bearish" and t.touch_index <= e.index <= bar.index
                        and e.low <= t.upper and e.high >= t.neckline.price
                        and e.id not in t.evidence_ids):
                    t.evidence_ids.append(e.id)
            ready = {}
            for route, ref in (("major", t.neckline), ("early", t.early_support)):
                known = t.touch_index if route == "major" else t.levels_known_index
                if ref is None or bar.index <= max(t.h2_index, known):
                    ready[route] = False
                    continue
                boundary = ref.price - self.rules.structure_buffer_atr * t.atr
                below = bar.close < boundary
                if below:
                    t.below_counts[route] += 1
                    if t.first_breaks[route] is None:
                        t.first_breaks[route] = bar.index
                else:
                    t.below_counts[route], t.first_breaks[route] = 0, None
                if self.cfg.top_exit_mode == "close_sequence":
                    confirmed = t.below_counts[route] >= self.rules.sell_confirm_closes
                else:
                    confirmed = (below and t.first_breaks[route] is not None
                                 and t.first_breaks[route] < bar.index and bar.close < bar.open
                                 and bar.high >= ref.price - self.rules.retest_tolerance_atr * t.atr)
                ready[route] = (confirmed and bar.close < t.h1.price
                                and t.h2_price - bar.close >= self.rules.sell_departure_atr * t.atr
                                and (not self.cfg.double_top_require_confirmed_h2 or t.h2_pivot_id is not None))
            selected_route = "early" if self.cfg.sell_detector == "double_top_early" else "major"
            if ready.get(selected_route):
                ref = t.early_support if selected_route == "early" else t.neckline
                if t.cluster_id in self.emitted_clusters:
                    t.suppressed = True
                    self.events.append(dict(index=bar.index, action="same_top_cluster_sell_suppressed",
                                            top_id=t.id, cluster_id=t.cluster_id))
                else:
                    signal = SellEvidence(f"{t.id}:{selected_route}:{bar.index}", bar.at, bar.index,
                                          f"campaign_double_top_{selected_route}", t.id, ref)
                    (early if selected_route == "early" else major).append(signal)
                    self.emitted_clusters.add(t.cluster_id)
                t.emitted = True
                self.consumed_h1.add(origin)
            row = dict(index=bar.index, top_id=t.id, h1=t.h1, neckline=t.neckline,
                       peak=t.h2_price, peak_index=t.h2_index, touch_index=t.touch_index,
                       h2_pivot_id=t.h2_pivot_id, h2_status=("confirmed_pivot" if t.h2_pivot_id else "observed_retest"),
                       support=t.early_support if selected_route == "early" else t.neckline,
                       early_support=t.early_support, levels_known_index=t.levels_known_index,
                       below_counts=dict(t.below_counts), first_breaks=dict(t.first_breaks),
                       route_ready=ready, selected_route=selected_route, emitted=t.emitted,
                       cluster_id=t.cluster_id, qualification=t.qualification, suppressed=t.suppressed,
                       reason=("same_top_cluster_already_emitted" if t.suppressed else
                               "double_top_reversal_confirmed" if t.emitted else "await_reversal_confirmation"),
                       evidence_ids=tuple(t.evidence_ids))
            # Stable top_id refers to the FIRST test; it is never a substitute
            # for the current second peak. Keep these clocks separate in exports.
            def observed_at(index):
                return bar.at if index == bar.index else self.bars[index].at
            h2_pivot = next((p for p in h2s if p.id == t.h2_pivot_id), None)
            row.update(first_test_index=t.touch_index, first_test_at=observed_at(t.touch_index),
                       second_peak_index=t.h2_index, second_peak_at=observed_at(t.h2_index),
                       second_peak_price=t.h2_price,
                       second_peak_pivot_available_index=h2_pivot.available_index if h2_pivot else None,
                       second_peak_pivot_available_at=h2_pivot.available_at if h2_pivot else None,
                       reversal_confirmed_at=bar.at if ready.get(selected_route) else None,
                       sell_decided_at=bar.at if t.emitted and not t.suppressed else None,
                       fill_at=None, fill_status="execution_log_only")
            details.append(row)
            self.events.append(dict(action="double_top_state", **row))
        self.bars.append(bar)
        self.major_signals.extend(major)
        self.early_signals.extend(early)
        return tuple(early if self.cfg.sell_detector == "double_top_early" else major), tuple(details)

    def _retire(self, origin, bar, reason):
        t = self.active.pop(origin)
        self.consumed_h1.add(origin)
        self.events.append(dict(index=bar.index, action="double_top_retired", top_id=t.id, reason=reason))
