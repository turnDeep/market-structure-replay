"""Major high-area retest -> support-of-that-rise failure, not every local MSS."""
from __future__ import annotations

from nq_ict_macro_reversal_examples_20260916.common import SellEvidence, check_bar, positive
from .common import CampaignConfig, TopAttempt


class MajorTopDetector:
    def __init__(self, rules, cfg: CampaignConfig):
        self.rules, self.cfg = rules, cfg
        self.bars = []
        self.active: dict[str, TopAttempt] = {}
        self.retired = set()
        self.rearm_after = {}
        self.events = []

    def select_support(self, top, bar, pivots):
        choices = []
        for p in pivots:
            if not (p.kind == "low" and p.radius >= self.cfg.top_support_radius
                    and p.available_index < bar.index
                    and top.area.anchor.origin_index < p.origin_index < top.peak_index
                    and top.peak_index - p.origin_index >= self.cfg.top_support_bars
                    and positive(p.atr_at_origin)
                    and top.peak - p.price >= self.cfg.top_support_rise_atr * p.atr_at_origin):
                continue
            # The rise must not already have lost this support before its peak.
            if any(b.close < p.price for b in self.bars
                   if p.available_index <= b.index < top.peak_index):
                continue
            choices.append(p)
        return max(choices, key=lambda p: (p.origin_index, p.id)) if choices else None

    def on_close(self, bar, market, pivots, evidence):
        previous = self.bars[-1] if self.bars else None
        check_bar(bar, previous)
        live = {a.id: a for a in market.areas if a.active}
        for a in live.values():
            if a.id in self.rearm_after:
                if previous and previous.index > self.rearm_after[a.id] and previous.close < a.low:
                    del self.rearm_after[a.id]
                else:
                    continue
            if (a.id in self.active or a.id in self.retired or a.qualified_index is None
                    or a.qualified_index >= bar.index or a.known_at >= bar.at
                    or bar.index - a.anchor.origin_index < self.rules.top_min_separation):
                continue
            if previous and previous.close < a.high and bar.high >= a.low and bar.low <= a.high:
                t = TopAttempt(f"major-top:{a.id}:{bar.index}", a, bar.index, bar.index, bar.high)
                t.support = self.select_support(t, bar, pivots)
                t.support_selected_index = bar.index if t.support else None
                self.active[a.id] = t
                self.events.append(dict(index=bar.index, action="top_watch_started", top_id=t.id,
                                        anchor_id=a.anchor.id, support_id=t.support.id if t.support else None,
                                        definition="major_area_retest_not_two_confirmed_peak_pivots"))
        signals, details = [], []
        for aid, t in list(self.active.items()):
            if aid not in live or bar.index - t.touch_index > self.rules.top_lifetime:
                self.events.append(dict(index=bar.index, action="top_watch_retired", top_id=t.id,
                                        reason="area_inactive" if aid not in live else "expired"))
                if aid in live:
                    self.rearm_after[aid] = bar.index
                else:
                    self.retired.add(aid)
                del self.active[aid]
                continue
            if bar.high > t.peak:
                t.peak, t.peak_index = bar.high, bar.index
                t.revision += 1
                t.first_break_index, t.below_count, t.emitted = None, 0, False
                t.support = self.select_support(t, bar, pivots)
                t.support_selected_index = bar.index if t.support else None
            elif t.support is None:
                # Late confirmation may become useful NOW, never on the old peak day.
                t.support = self.select_support(t, bar, pivots)
                t.support_selected_index = bar.index if t.support else None
            floor = t.support.price if t.support else t.area.low
            margin = self.cfg.evidence_margin_atr * t.area.anchor.atr_at_origin
            for e in evidence:
                if (e.direction == "bearish" and e.index >= t.touch_index
                        and e.high >= floor - margin and e.low <= t.area.high + margin
                        and e.kind in {"sweep_reclaim", "structure_break", "zone_formed", "zone_rejection"}):
                    # A zone/level overlapping this path is descriptive, not an independent vote.
                    if e.id not in t.linked_evidence:
                        t.linked_evidence.append(e.id)
            reason = "waiting_major_rise_support"
            ref = t.support
            if (ref is not None and t.support_selected_index < bar.index and bar.index > t.peak_index):
                boundary = ref.price - self.rules.structure_buffer_atr * ref.atr_at_origin
                below = bar.close < boundary
                if below:
                    if t.first_break_index is None:
                        t.first_break_index = bar.index
                    t.below_count += 1
                else:
                    t.first_break_index, t.below_count = None, 0
                if self.cfg.top_exit_mode == "close_sequence":
                    confirmed = t.below_count >= self.rules.sell_confirm_closes
                else:
                    confirmed = (t.first_break_index is not None and t.first_break_index < bar.index
                                 and bar.high >= ref.price - self.rules.retest_tolerance_atr * ref.atr_at_origin
                                 and below and bar.close < bar.open)
                departure = t.peak - bar.close >= self.rules.sell_departure_atr * ref.atr_at_origin
                if confirmed and departure and not t.emitted and bar.close < t.area.high:
                    signals.append(SellEvidence(f"{t.id}:v{t.revision}:{bar.index}", bar.at, bar.index,
                                                f"campaign_major_top_{self.cfg.top_exit_mode}", t.id, ref))
                    t.emitted = True
                    reason = "major_top_reversal_confirmed"
                else:
                    reason = "await_major_support_failure" if not t.emitted else "sell_already_emitted"
            row = dict(index=bar.index, top_id=t.id, area_id=aid, peak=t.peak, peak_index=t.peak_index,
                       touch_index=t.touch_index, support=ref, support_selected_index=t.support_selected_index,
                       below_count=t.below_count, first_break_index=t.first_break_index, reason=reason,
                       evidence_ids=tuple(t.linked_evidence), revision=t.revision)
            details.append(row)
            self.events.append(dict(action="top_watch_state", **row))
        self.bars.append(bar)
        return tuple(signals), tuple(details)
