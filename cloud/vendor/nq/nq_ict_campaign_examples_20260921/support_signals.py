"""Candidates from today's OB observations, without inherited BUY seeds.

No candidate is retained because of later success. Primary OB validity and
mechanically available references remain required. Daily sequencing only.
Authored revision 0.3, not executed or validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from nq_ict_macro_reversal_examples_20260916.common import BuyEvidence, down_tick, positive
from nq_ict_priority_entry_examples_20260916.common import EntryOffer


class ObHistoryBook:
    """Return history BEFORE today's observation; then append today's facts."""
    def __init__(self):
        self.history = {}

    def on_close(self, bar, zones):
        before = {}
        for view in zones:
            z = view.zone
            if (z.kind != "ob_candidate" or z.direction != "bullish"
                    or z.available_index > bar.index):
                continue
            h = self.history.setdefault(z.id, dict(
                zone=z, atr=bar.atr_prior, peak=z.wick_high, touch_episodes=0,
                touching=False, below_closes=0, awaiting_reclaim=False, reclaims=0))
            if h["zone"] != z:
                raise ValueError("OB geometry changed after first availability")
            before[z.id] = dict(
                history_known_through_index=bar.index - 1,
                origin_index=z.origin_index, available_index=z.available_index,
                age_bars=bar.index - z.available_index,
                prior_departure_atr=((h["peak"] - z.wick_high) / h["atr"]
                                     if positive(h["atr"]) else None),
                prior_touch_episodes=h["touch_episodes"],
                prior_below_closes=h["below_closes"], prior_reclaims=h["reclaims"],
                source_timeframe=z.timeframe, source_state=view.state)
            if not view.observable:
                continue
            h["peak"] = max(h["peak"], bar.high)
            # The formation/availability candle is not a retest.
            if z.available_index >= bar.index:
                continue
            touching = bar.low <= z.wick_high and bar.high >= z.wick_low
            if touching and not h["touching"]:
                h["touch_episodes"] += 1
            h["touching"] = touching
            if bar.close < z.wick_low:
                h["below_closes"] += 1
                h["awaiting_reclaim"] = True
            elif h["awaiting_reclaim"] and bar.close >= z.body_high:
                h["reclaims"] += 1
                h["awaiting_reclaim"] = False
        return before


@dataclass
class ObservedCandidate:
    token: tuple
    seed: BuyEvidence
    early_emitted: bool = False
    confirmed_emitted: bool = False
    above_count: int = 0
    first_break: int | None = None
    structural: BuyEvidence | None = None
    retested_zones: set[str] = field(default_factory=set)
    seen_evidence: set[str] = field(default_factory=set)
    ordered_evidence: list = field(default_factory=list)


class IndependentSupportSignals:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.active = {}
        self.confirmed_setup_ids = set()
        self.events = []
        self.early_rows = []
        self.confirmed_rows = []

    def _reference(self, obs, episode, pivots, bar):
        highs = sorted((p for p in pivots if p.kind == "high"
                        and p.radius >= self.rules.buy_reference_radius
                        and p.available_index < bar.index and p.available_at < bar.at
                        and episode.anchor.origin_index < p.origin_index < obs.low_index),
                       key=lambda p: (p.origin_index, p.id))
        choices = []
        for i, p in enumerate(highs):
            prior_price = highs[i - 1].price if i else episode.anchor.price
            if (positive(p.atr_at_origin) and p.price < prior_price
                    and obs.low_index - p.origin_index >= self.rules.buy_reference_bars
                    and p.price - obs.low >= self.rules.buy_reference_drop_atr * p.atr_at_origin):
                choices.append(p)
        return max(choices, key=lambda p: (p.origin_index, p.id)) if choices else None

    def on_close(self, bar, previous, market, htf, zones, supports, pivots, tactical_zones=(), evidence=()):
        if previous is None or supports.context is None:
            return ()
        views = {v.zone.id: v for v in zones}
        ep = market.episode
        for zid, state in list(self.active.items()):
            obs, view = supports.active.get(zid), views.get(zid)
            if (obs is None or view is None or not view.tradeable or obs.recovery_index is None
                    or ep is None or state.token != self._token(supports, obs, ep)):
                self.events.append(dict(index=bar.index, action="independent_candidate_retired",
                                        setup_id=state.seed.setup_id,
                                        reason="source_reaction_revised_or_unavailable"))
                del self.active[zid]
        if ep is None or not positive(bar.atr_prior):
            return ()
        offers = []
        for zid, obs in sorted(supports.active.items()):
            view = views.get(zid)
            if view is None or not view.tradeable or obs.recovery_index is None:
                continue
            z = obs.zone
            if not (z.available_index < obs.contact_index <= obs.low_index <= obs.recovery_index <= bar.index):
                continue
            state = self.active.get(zid)
            if state is None:
                ref = self._reference(obs, ep, pivots, bar)
                if ref is None:
                    self.events.append(dict(index=bar.index, action="independent_candidate_waiting",
                                            zone_id=zid, reason="mechanical_reference_not_yet_available"))
                    continue
                stop = down_tick(min(z.wick_low, obs.low)
                                 - self.rules.emergency_buffer_atr * bar.atr_prior, self.rules.tick_size)
                if not 0 < stop < bar.close:
                    continue
                linked = tuple(sorted(v.zone.id for v in zones if v.tradeable
                    and v.zone.kind == "fvg" and v.zone.direction == "bullish"
                    and obs.recovery_index <= v.zone.origin_index <= v.zone.available_index <= bar.index
                    and v.zone.wick_low <= bar.close and v.zone.wick_high >= obs.low))
                overlaps = tuple(sorted(v.zone.id for v in zones if v.tradeable
                    and v.zone.kind == "ob_candidate" and v.zone.direction == "bullish"
                    and v.zone.timeframe in {"1W", "1M"}
                    and v.zone.wick_low <= z.wick_high and v.zone.wick_high >= z.wick_low))
                if self.rules.require_linked_fvg and not linked:
                    continue
                sid = (f"independent-ob:{supports.context[0]}:{ep.id}:{zid}:"
                       f"contact:{obs.contact_index}:low:{obs.low_index}:recovery:{obs.recovery_index}")
                seed = BuyEvidence(
                    key=f"{sid}:seed:{bar.index}", episode_id=ep.id, at=bar.at, index=bar.index,
                    zone=z, touch_index=obs.contact_index, low=obs.low, low_index=obs.low_index,
                    low_atr=obs.low_atr, recovery_index=obs.recovery_index, reference=ref,
                    breakout_index=obs.recovery_index, confirmation="early_recovery",
                    emergency_stop=stop, linked_fvg_ids=linked, htf_overlap_ids=overlaps,
                    sweep_depth_atr=max(0.0, z.wick_low - obs.low) / obs.low_atr,
                    htf=htf, setup_id=sid)
                state = ObservedCandidate(self._token(supports, obs, ep), seed)
                self.active[zid] = state
                self.events.append(dict(index=bar.index, action="independent_candidate_registered",
                                        seed=seed, precontact_history=dict(obs.features),
                                        source="current_ob_observation_not_a_legacy_watch"))
            c = state.seed
            for e in evidence:
                if (e.id not in state.seen_evidence and e.direction == "bullish"
                        and c.recovery_index <= e.index <= bar.index
                        and e.high >= c.low and e.low <= c.reference.price
                        and e.kind in {"structure_break", "zone_formed", "sweep_reclaim", "zone_rejection"}):
                    state.seen_evidence.add(e.id)
                    linked = dict(index=bar.index, action="independent_path_evidence", setup_id=c.setup_id,
                                  evidence=e, seed_index=c.index, recovery_index=c.recovery_index,
                                  order_relation=("after_seed" if e.index > c.index else
                                                  "same_close_order_unknown" if e.index == c.index else
                                                  "after_recovery_before_seed"),
                                  prior_buy_eligibility_rewritten=False)
                    state.ordered_evidence.append(linked)
                    self.events.append(linked)
            boundary = c.reference.price + self.rules.structure_buffer_atr * c.reference.atr_at_origin
            above = bar.close > boundary
            if not above:
                state.above_count, state.first_break = 0, None
            elif state.first_break is not None:
                state.above_count += 1
            elif previous.close <= boundary:
                state.first_break, state.above_count = bar.index, 1
            if self.rules.buy_mode == "close_sequence":
                confirmed = state.above_count >= self.rules.buy_confirm_closes
            else:
                confirmed = (above and state.first_break is not None and state.first_break < bar.index
                             and bar.close > bar.open and bar.low > c.low
                             and bar.low <= c.reference.price + self.rules.retest_tolerance_atr * c.reference.atr_at_origin)
            confirmed = confirmed and bar.close - c.low >= self.rules.buy_departure_atr * c.low_atr
            # Both models observe the same immutable candidate. A later
            # confirmation is a separate event at its actual observation time.
            if confirmed and not state.confirmed_emitted:
                current = replace(c, key=f"{c.setup_id}:confirmed:{bar.index}", at=bar.at,
                                  index=bar.index, breakout_index=state.first_break,
                                  confirmation=self.rules.buy_mode, htf=htf)
                state.confirmed_emitted = True
                state.structural = current
                state.early_emitted = True
                self.confirmed_setup_ids.add(c.setup_id)
                self.confirmed_rows.append(current)
                if self.cfg.confirmed_entry_mode != "retest_only":
                    offers.append(EntryOffer("confirmed", "independent_support_confirmation", c, current))
            elif (not state.early_emitted and not state.confirmed_emitted
                  and bar.close >= z.body_high and bar.close > previous.high
                  and c.emergency_stop < bar.close):
                current = replace(c, key=f"{c.setup_id}:local:{bar.index}", at=bar.at,
                                  index=bar.index, breakout_index=bar.index,
                                  confirmation="independent_support_local_break", htf=htf)
                offers.append(EntryOffer("early", "independent_support_local_break", c, current))
                state.early_emitted = True
                self.early_rows.append(current)
            if (state.structural is not None and state.structural.index < bar.index
                    and self.cfg.enable_post_structure_retest
                    and self.cfg.confirmed_entry_mode != "structure_only"
                    and bar.close > boundary and bar.close > bar.open and bar.low >= c.low):
                possible = [(zid, "ob_candidate", z.wick_low, z.wick_high,
                             z.available_index, z.body_high)]
                for tz in tactical_zones:
                    if (tz.direction == "bullish" and tz.available_index < bar.index
                            and tz.available_index >= c.recovery_index and tz.origin_index >= c.touch_index
                            and tz.low <= boundary + self.cfg.evidence_margin_atr * c.low_atr
                            and tz.high >= c.low):
                        possible.append((tz.id, tz.kind, tz.low, tz.high, tz.available_index, tz.high))
                for target_id, kind, low, high, known, recovery in sorted(possible, key=lambda v: (-v[4], v[0])):
                    if (target_id in state.retested_zones or known >= bar.index
                            or not (bar.low <= high and bar.high >= low and bar.close > recovery)):
                        continue
                    current = replace(state.structural,
                                      key=f"{c.setup_id}:retest:{bar.index}:{target_id}",
                                      at=bar.at, index=bar.index, recovery_index=bar.index,
                                      confirmation="breakout_retest", htf=htf)
                    offers.append(EntryOffer("confirmed", "post_structure_zone_retest", c, current))
                    state.retested_zones.add(target_id)
                    self.events.append(dict(index=bar.index, action="independent_retest_observed",
                                            setup_id=c.setup_id, retest_zone_id=target_id,
                                            retest_zone_kind=kind, original_confirmation=state.structural.index,
                                            original_breakout=state.structural.breakout_index))
                    break
        for offer in offers:
            self.events.append(dict(index=bar.index, action="independent_support_offer",
                                    offer=offer, probability=None))
        return tuple(offers)

    def describe(self, setup_id):
        state = next((s for s in self.active.values() if s.seed.setup_id == setup_id), None)
        return dict(setup_id=setup_id, source="independent_ob_observation",
                    source_ob=state.seed.zone.id if state else None,
                    retired=state is None,
                    structure_confirmed=setup_id in self.confirmed_setup_ids,
                    evidence_ids=sorted(state.seen_evidence) if state else [],
                    probability=None, probability_status="not_estimated")

    @staticmethod
    def _token(supports, obs, ep):
        return (supports.context, ep.id, obs.group_id, obs.contact_index,
                obs.low_index, obs.recovery_index)
