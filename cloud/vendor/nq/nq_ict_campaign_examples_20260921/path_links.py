"""Keep place/order links separate from independent evidence or probabilities."""
from __future__ import annotations

from dataclasses import replace

from nq_ict_priority_entry_examples_20260916.common import EntryOffer
from .common import BuyPath, context_label


class BuyPathBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.paths: dict[str, BuyPath] = {}
        self.history = []
        self.links = []
        self.events = []

    def on_close(self, packet):
        s = packet.base.signal
        bar, live = s.bar, packet.base.reactions.live_setup_ids
        self.history.extend(packet.evidence)
        for c in s.early_buys:
            if c.setup_id in live and c.setup_id not in self.paths:
                self.paths[c.setup_id] = BuyPath(c)
                recovery_type = ("touch_and_recovery" if c.low >= c.zone.wick_low else
                                 "same_bar_ob_sweep_recovery" if c.low_index == c.recovery_index
                                 else "multi_bar_ob_reclaim")
                self.events.append(dict(index=bar.index, action="path_registered", setup_id=c.setup_id,
                                        source_ob_id=c.zone.id, source_ob_available=c.zone.available_index,
                                        touch_index=c.touch_index, recovery_index=c.recovery_index,
                                        recovery_type=recovery_type, context=context_label(c.htf),
                                        weekly_trend=c.htf.weekly_trend, monthly_trend=c.htf.monthly_trend,
                                        weekly_bias=c.htf.weekly_bias, monthly_bias=c.htf.monthly_bias))
        for c in s.confirmed_buys:
            path = self.paths.get(c.setup_id)
            if path is not None and c.setup_id in live:
                path.confirmed = c
                self.events.append(dict(index=bar.index, action="parent_structure_confirmed",
                                        setup_id=c.setup_id, reference_id=c.reference.id,
                                        breakout_index=c.breakout_index))
        for sid, path in self.paths.items():
            c = path.seed
            if path.retired:
                continue
            if sid not in live:
                path.retired = True
                self.events.append(dict(index=bar.index, action="path_retired", setup_id=sid,
                                        reason="source_reaction_not_live"))
                continue
            margin = self.cfg.evidence_margin_atr * c.low_atr
            # Freeze this corridor from the original, then-known setup.
            lo, hi = min(c.low, c.zone.wick_low) - margin, max(c.reference.price, c.zone.wick_high) + margin
            candidates = self.history if not path.seen_events else packet.evidence
            for e in candidates:
                if not (c.touch_index <= e.index <= bar.index and e.direction == "bullish"
                        and e.id not in path.seen_events and e.high >= lo and e.low <= hi):
                    continue
                if e.kind not in {"sweep_reclaim", "structure_break", "zone_formed", "zone_rejection"}:
                    continue
                if e.details.get("zone_kind") == "ob_candidate" and e.source_id != c.zone.id:
                    # Do not append an unrelated OB merely because it is nearby.
                    continue
                if e.details.get("zone_kind") in {"fvg", "ifvg"} and e.index < c.recovery_index:
                    continue
                if e.kind == "structure_break" and not (c.touch_index < e.index and
                                                         e.origin_index < c.low_index):
                    continue
                path.events.append(e)
                path.seen_events.add(e.id)
                self.links.append(dict(setup_id=sid, event_id=e.id, event_index=e.index,
                                       linked_at_index=bar.index, source_ob_id=c.zone.id,
                                       link_type="same_direction_spatial_temporal_association",
                                       independent_evidence=False,
                                       same_bar_order="unknown" if e.index == c.touch_index else "different_bars"))

    def early_allowed(self, offer):
        c = offer.decision
        if offer.route != "early":
            return True, "confirmed_route_not_filtered_as_early"
        if self.cfg.require_early_support_sweep and not c.low < c.zone.wick_low:
            return False, "optional_ob_sweep_not_observed"
        if self.cfg.require_early_htf_ob_overlap and not c.htf_overlap_ids:
            return False, "optional_htf_ob_overlap_not_observed"
        return True, "early_recovery_preserved_without_mss_or_fvg_requirement"

    def retest_offers(self, packet):
        if not self.cfg.enable_post_structure_retest:
            return ()
        s, bar = packet.base.signal, packet.base.signal.bar
        views = {v.zone.id: v for v in packet.base.zones}
        extra = []
        for sid, path in self.paths.items():
            c = path.confirmed
            if path.retired or c is None or c.index >= bar.index:
                continue
            if sid not in packet.base.reactions.live_setup_ids:
                continue
            parent = views.get(c.zone.id)
            if parent is None or not parent.tradeable or bar.low < c.low:
                continue
            if (s.market.episode is None or s.market.episode.id != c.episode_id
                    or bar.index - c.touch_index > self.rules.reaction_lifetime):
                continue
            boundary = c.reference.price + self.rules.structure_buffer_atr * c.reference.atr_at_origin
            if not (bar.close > boundary and bar.close > bar.open and c.emergency_stop < bar.close):
                continue
            possible = [(c.zone.id, "ob_candidate", c.zone.wick_low, c.zone.wick_high,
                         c.zone.available_index, c.zone.body_high)]
            for z in packet.tactical_zones:
                if (z.direction == "bullish" and z.available_index < bar.index
                        and z.available_index >= c.recovery_index
                        and z.origin_index >= c.touch_index
                        and z.low <= boundary + self.cfg.evidence_margin_atr * c.low_atr
                        and z.high >= c.low):
                    possible.append((z.id, z.kind, z.low, z.high, z.available_index, z.high))
            for zid, kind, low, high, known, recovery in sorted(possible, key=lambda z: (-z[4], z[0])):
                if known >= bar.index or not (bar.low <= high and bar.high >= low and bar.close > recovery):
                    continue
                token = f"{sid}:{zid}"
                if token in path.retest_keys:
                    continue
                # The previous structural break is kept at its original clock.
                # A retest is a new observation, NOT a newly invented MSS.
                current = replace(c, key=f"{c.key}:campaign_retest:{bar.index}:{zid}",
                                  at=bar.at, index=bar.index, recovery_index=bar.index,
                                  confirmation="breakout_retest", htf=s.htf)
                extra.append(EntryOffer("confirmed", "post_structure_zone_retest", path.seed, current))
                path.retest_keys.add(token)
                self.events.append(dict(index=bar.index, action="retest_offer", setup_id=sid,
                                        retest_zone_id=zid, retest_zone_kind=kind,
                                        original_confirmation_index=c.index,
                                        original_breakout_index=c.breakout_index,
                                        order_relation="confirmation_before_retest",
                                        intraday_mss="unknown_data"))
                break
        return tuple(extra)

    def describe(self, setup_id):
        path = self.paths.get(setup_id)
        if path is None:
            return dict(setup_id=setup_id, status="not_registered")
        return dict(setup_id=setup_id, source_ob=path.seed.zone.id, retired=path.retired,
                    structure_confirmed=path.confirmed is not None,
                    event_ids=[e.id for e in path.events],
                    daily_mss_observed=any(e.details.get("mss") is True for e in path.events),
                    probability=None, probability_status="not_estimated")
