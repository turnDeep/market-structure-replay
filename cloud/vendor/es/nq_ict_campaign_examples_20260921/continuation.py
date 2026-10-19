"""Revision 0.11 structure observation and optional continuation BUY proposal.

AUTHORED, NOT EXECUTED/VALIDATED. Observation does not depend on whether the
continuation order route is enabled. Recovery BUY consumes the same structure.

Reuse observed daily structure-break events, available pivots, v09 OBs and
the existing supplementary daily FVG lifecycle. No major-decline prerequisite,
future cohort, count allowance, fixed dates, or breakout-only chase entry.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import BuyEvidence, Zone, down_tick, positive
from nq_ict_priority_entry_examples_20260916.common import EntryOffer
from .reentry import SupportObservation


class ContinuationBuyBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.bars = {}
        self.pending = None
        self.structure = None
        self.linked = {}
        self.active = {}
        self.blocked = set()
        self.emitted = set()
        self.events = []
        self.candidate_rows = []
        self.snapshot = dict(phase="unestablished", enabled=cfg.observe_continuation_structure,
                             orders_enabled=cfg.enable_continuation_buy)

    @staticmethod
    def is_offer(offer):
        return offer.kind == "continuation_zone_reaction"

    def observation(self, offer):
        return self.active.get(offer.decision.zone.id)

    def _zone_views(self, packet):
        # OB source validity remains upstream-owned. FVGs use ONLY the existing
        # daily close-based lifecycle; do not silently mix v09 wick-fill FVGs.
        result = {v.zone.id: (v.zone, v.observable, v.tradeable)
                  for v in packet.base.zones
                  if v.zone.kind == "ob_candidate" and v.zone.direction == "bullish"}
        for z in packet.tactical_zones:
            if z.kind != "fvg" or z.direction != "bullish":
                continue
            origin, available = self.bars.get(z.origin_index), self.bars.get(z.available_index)
            if origin is None or available is None:
                continue
            zone = Zone(z.id, "fvg", "bullish", "1D", z.origin_index, origin.at,
                        z.available_index, available.at, z.low, z.high, z.low, z.high)
            result[z.id] = (zone, True, True)
        return result

    def _retire(self, bar, reason):
        self.events.append(dict(index=bar.index, action="continuation_structure_retired",
                                reason=reason,
                                structure_id=self.structure["id"] if self.structure else None))
        self.structure, self.pending = None, None
        self.linked, self.active, self.blocked = {}, {}, set()

    def _observe_breaks(self, packet, bar):
        pivots = {p.id: p for p in packet.base.signal.pivots}
        for event in packet.evidence:
            if not (event.index == bar.index and event.kind == "structure_break"
                    and event.direction == "bullish"):
                continue
            high = pivots.get(event.source_id)
            if high is None or high.available_index >= bar.index:
                continue
            lows = [p for p in pivots.values() if p.kind == "low"
                    and p.radius >= self.rules.buy_reference_radius
                    and high.origin_index < p.origin_index < bar.index
                    and p.available_index < bar.index and positive(p.atr_at_origin)]
            if not lows:
                self.events.append(dict(index=bar.index, action="continuation_break_waiting",
                                        event_id=event.id, reason="no_preknown_post_high_support_pivot"))
                continue
            low = max(lows, key=lambda p: (p.origin_index, p.radius, p.id))
            if not positive(high.atr_at_origin) or bar.close <= low.price:
                continue
            if self.structure and low.price < self.structure["low"].price:
                # Do not silently weaken a still-valid protected structure.
                continue
            self.pending = dict(high=high, low=low, event_id=event.id,
                                observed_index=bar.index, breakout_index=bar.index,
                                closes=0, peak=bar.high)
            self.events.append(dict(index=bar.index, action="continuation_break_observed",
                                    reference=high, support=low, event_id=event.id,
                                    weekly_trend=packet.base.signal.htf.weekly_trend,
                                    monthly_trend=packet.base.signal.htf.monthly_trend,
                                    higher_timeframes="context_only_not_unanimous_direction_gate"))

    def _accept_structure(self, bar, views):
        w = self.pending
        if w is None:
            return
        if (bar.close < w["low"].price
                or bar.index - w["observed_index"] > self.rules.reaction_lifetime):
            self.events.append(dict(index=bar.index, action="continuation_break_retired",
                                    event_id=w["event_id"], reason="support_close_failure_or_watch_expiry"))
            self.pending = None
            return
        boundary = w["high"].price + self.rules.structure_buffer_atr * w["high"].atr_at_origin
        if bar.close <= boundary:
            w["closes"] = 0
            return
        if w["closes"] == 0:
            w["breakout_index"], w["peak"] = bar.index, bar.high
        w["closes"] += 1
        w["peak"] = max(w["peak"], bar.high)
        if w["closes"] < self.cfg.continuation_accept_closes:
            return
        structure = dict(id=f"continuation:{w['high'].id}:{w['low'].id}:{w['breakout_index']}",
                         high=w["high"], low=w["low"], event_id=w["event_id"],
                         breakout_index=w["breakout_index"], known_index=bar.index,
                         effective_index=bar.index + 1, boundary=boundary,
                         leg_peak=w["peak"], acceptance_closes=w["closes"])
        self.structure, self.pending = structure, None
        self.linked, self.active, self.blocked = {}, {}, set()
        self.events.append(dict(index=bar.index, action="continuation_structure_confirmed",
                                structure=deepcopy(structure), usable_from_next_bar=True))
        low = structure["low"]
        # Fixed geometric/temporal attribution, known before the later contact.
        # No lifetime best departure or outcome-based promotion of a zone.
        for zid, (zone, observable, _) in sorted(views.items()):
            if not (observable and low.origin_index - self.cfg.structure_origin_window
                    <= zone.origin_index <= structure["breakout_index"]
                    and low.origin_index <= zone.available_index <= bar.index
                    and zone.wick_high >= low.price and zone.wick_low <= structure["leg_peak"]):
                continue
            role = dict(zone_id=zid, structure_id=structure["id"], kind="continuation_support",
                        grade=0, known_index=bar.index, effective_index=bar.index + 1,
                        source_timeframe=zone.timeframe, high_id=structure["high"].id,
                        low_id=low.id, low_index=low.origin_index,
                        breakout_index=structure["breakout_index"],
                        # Origin role stays local. In thesis_wave mode the
                        # manager starts with this reaction, then promotes only
                        # after a subsequently confirmed larger daily wave.
                        scale=dict(tier="major_daily_management_after_local_continuation",
                                   pivot_radius=self.rules.pivot_radius,
                                   min_leg_bars=self.cfg.structure_min_leg_bars,
                                   min_leg_atr=self.cfg.structure_min_leg_atr,
                                   min_pullback_bars=self.rules.management_pullback_bars,
                                   min_pullback_atr=self.rules.management_pullback_atr),
                        definition="precontact_link_to_accepted_break_not_major_bottom_or_probability")
            self.linked[zid] = dict(zone=zone, role=role)
            self.events.append(dict(index=bar.index, action="continuation_zone_linked",
                                    zone=zone, role=deepcopy(role),
                                    source_definition="v09_ob" if zone.kind == "ob_candidate"
                                    else "campaign_daily_fvg_close_lifecycle"))

    def _update_snapshot(self, bar, phase=None):
        structure = self.structure
        self.snapshot = dict(
            enabled=self.cfg.observe_continuation_structure,
            orders_enabled=self.cfg.enable_continuation_buy, index=bar.index,
            phase=phase or ("advancing" if structure else "unestablished"),
            effective_index=structure["effective_index"] if structure else None,
            structure=deepcopy(structure), pending=deepcopy(self.pending),
            linked_zone_ids=sorted(self.linked), active_zone_ids=sorted(self.active),
            buy_quota=None, htf_direction_gate=False,
            observation_depends_on_position=False)

    def observe_structure(self, packet):
        """Advance the shared, price-only feed even when no orders are enabled."""
        bar, signal = packet.base.signal.bar, packet.base.signal
        self.bars[bar.index] = bar
        if not self.cfg.observe_continuation_structure:
            self._update_snapshot(bar, "structure_observation_disabled")
            return None
        views = self._zone_views(packet)
        if signal.sells:
            self._retire(bar, "canonical_major_sell")
            self._update_snapshot(bar, "major_sell_wait_new_structure")
            return None
        if self.structure and bar.close < self.structure["low"].price:
            self._retire(bar, "close_below_structural_support")
        self._observe_breaks(packet, bar)
        self._accept_structure(bar, views)
        self._update_snapshot(bar)
        return views

    def on_close(self, packet, previous):
        views = self.observe_structure(packet)
        if views is None:
            return ()
        bar, signal = packet.base.signal.bar, packet.base.signal
        # Retain the same candidate observations when orders are disabled.
        # assess() supplies the explicit order gate, so candidate loss cannot
        # be mistaken for a change in the price-structure feed.
        offers = []
        structure = self.structure
        if (structure is not None and structure["effective_index"] <= bar.index
                and previous is not None and positive(bar.atr_prior)):
            for zid, linked in sorted(self.linked.items()):
                view = views.get(zid)
                if view is None or not view[1]:
                    self.active.pop(zid, None)
                    continue
                zone, _, tradeable = view
                if zone != linked["zone"]:
                    raise ValueError("continuation source geometry changed")
                if zone.available_index >= bar.index or zone.available_at >= bar.at:
                    continue
                touch = bar.low <= zone.wick_high and bar.high >= zone.wick_low
                if zid in self.blocked:
                    if tradeable and bar.low > zone.wick_high:
                        self.blocked.remove(zid)
                    continue
                obs = self.active.get(zid)
                if obs is not None and touch and obs.away_bars >= self.cfg.support_join_bars:
                    self.events.append(dict(index=bar.index, action="continuation_new_visit",
                                            zone_id=zid, prior_contact=obs.contact_index))
                    self.active.pop(zid)
                    obs = None
                if obs is None:
                    if not touch:
                        continue
                    obs = SupportObservation(
                        zone, f"{structure['id']}:{zid}:reaction:{bar.index}", bar.index,
                        bar.low, bar.index, bar.atr_prior, latest_contact_index=bar.index,
                        features=dict(structural_support=dict(
                            role=deepcopy(linked["role"]), known_through_index=bar.index - 1),
                            continuation_structure=deepcopy(structure),
                            leg_peak_before_contact=max(b.high for i, b in self.bars.items()
                                                        if structure["low"].origin_index <= i < bar.index)))
                    self.active[zid] = obs
                    self.events.append(dict(index=bar.index, action="continuation_contact",
                                            zone_id=zid, structure_id=structure["id"],
                                            source_available_index=zone.available_index,
                                            role_known_index=linked["role"]["known_index"]))
                if bar.index - obs.contact_index > self.rules.reaction_lifetime:
                    self.active.pop(zid)
                    self.blocked.add(zid)
                    self.events.append(dict(index=bar.index, action="continuation_reaction_expired",
                                            zone_id=zid, contact_index=obs.contact_index))
                    continue
                obs.away_bars = obs.away_bars + 1 if bar.low > zone.wick_high else 0
                if touch:
                    obs.latest_contact_index = bar.index
                if bar.low < obs.low:
                    obs.low, obs.low_index, obs.low_atr = bar.low, bar.index, bar.atr_prior
                    obs.recovery_index = None
                if bar.close < zone.body_high:
                    obs.recovery_index = None
                if bar.close < zone.wick_low and obs.first_below_index is None:
                    obs.first_below_index = bar.index
                if (obs.first_below_index is not None
                        and bar.index - obs.first_below_index > self.cfg.sweep_reclaim_wait):
                    self.active.pop(zid)
                    self.blocked.add(zid)
                    self.events.append(dict(index=bar.index, action="continuation_reclaim_expired",
                                            zone_id=zid))
                    continue
                recovered = tradeable and bar.close >= zone.body_high and bar.close > bar.open
                if recovered:
                    if obs.recovery_index is None:
                        obs.recovery_index = bar.index
                        self.events.append(dict(index=bar.index, action="continuation_recovery",
                                                zone_id=zid, contact_index=obs.contact_index,
                                                intrabar_order="unknown_data"))
                    obs.first_below_index = None
                if not (recovered and bar.close > previous.high and bar.close > structure["boundary"]):
                    continue
                token = (structure["id"], zid, obs.contact_index, obs.low_index, obs.recovery_index)
                if token in self.emitted:
                    continue
                stop_low = min(zone.wick_low, obs.low)
                if self.cfg.continuation_risk_mode == "legacy_structure":
                    stop_low = min(stop_low, structure["low"].price)
                stop = down_tick(stop_low
                                 - self.rules.emergency_buffer_atr * bar.atr_prior, self.rules.tick_size)
                if not 0 < stop < bar.close:
                    continue
                obs.features["entry_risk"] = dict(
                    mode=self.cfg.continuation_risk_mode, decision_index=bar.index,
                    decision_atr=bar.atr_prior, reaction_stop=stop,
                    risk_at_decision_atr=(bar.close - stop) / bar.atr_prior,
                    maximum_risk_atr=(self.cfg.continuation_max_risk_atr
                                      if self.cfg.continuation_risk_mode == "reaction" else None),
                    structural_low=structure["low"].price, structural_low_added_to_stop=
                    self.cfg.continuation_risk_mode == "legacy_structure")
                sid = (f"{structure['id']}:{zid}:contact:{obs.contact_index}:"
                       f"low:{obs.low_index}:recovery:{obs.recovery_index}")
                c = BuyEvidence(
                    key=f"{sid}:continuation:{bar.index}", episode_id=structure["id"],
                    at=bar.at, index=bar.index, zone=zone, touch_index=obs.contact_index,
                    low=obs.low, low_index=obs.low_index, low_atr=obs.low_atr,
                    recovery_index=obs.recovery_index, reference=structure["high"],
                    breakout_index=structure["breakout_index"], confirmation="continuation_zone_reaction",
                    emergency_stop=stop, linked_fvg_ids=(zid,) if zone.kind == "fvg" else (),
                    htf_overlap_ids=(), sweep_depth_atr=max(0.0, zone.wick_low - obs.low) / obs.low_atr,
                    htf=signal.htf, setup_id=sid)
                offer = EntryOffer("confirmed", "continuation_zone_reaction", c, c)
                self.emitted.add(token)
                self.candidate_rows.append(c)
                self.events.append(dict(index=bar.index, action="continuation_offer",
                                        candidate=c, structure_id=structure["id"],
                                        orders_enabled=self.cfg.enable_continuation_buy, probability=None))
                offers.append(offer)
        self._update_snapshot(bar)
        return tuple(offers)

    def assess(self, offer, packet, previous):
        c, bar = offer.decision, packet.base.signal.bar
        obs = self.observation(offer)
        structure = self.structure
        view = self._zone_views(packet).get(c.zone.id)
        role = obs.features["structural_support"]["role"] if obs else None
        valid_structure = bool(structure and structure["id"] == c.episode_id
                               and structure["effective_index"] <= bar.index
                               and bar.close >= structure["low"].price)
        linked_before_contact = bool(obs and role and role["known_index"] < obs.contact_index
                                     and role["effective_index"] <= obs.contact_index
                                     and c.zone.available_index < obs.contact_index)
        checks = dict(
            continuation_orders_enabled=(self.cfg.enable_continuation_buy,
                                         "ordinary_continuation_orders_are_an_independent_switch"),
            continuation_structure=(valid_structure, "accepted_structure_must_still_be_valid"),
            support_provenance=(linked_before_contact, "continuation_link_must_precede_contact"),
            current_support_reaction=(bool(obs and obs.contact_index <= obs.low_index
                                          <= obs.recovery_index <= bar.index)
                                      if obs and obs.recovery_index is not None else False,
                                      "ordered_contact_recovery_required"),
            source_zone=(bool(view and view[2] and view[0] == c.zone
                              and c.zone.available_at < bar.at), "continuation_source_must_be_tradeable"),
            reaction_trigger=(bool(previous and structure and bar.close > previous.high
                                   and bar.close > bar.open and bar.close >= c.zone.body_high
                                   and bar.close > structure["boundary"]),
                              "zone_reclaim_prior_high_break_and_breakout_level_required"))
        return checks, deepcopy(role), obs

    def describe(self, offer):
        obs = self.observation(offer)
        return dict(setup_id=offer.decision.setup_id, family="continuation",
                    structure_id=offer.decision.episode_id, source_zone=offer.decision.zone.id,
                    source_kind=offer.decision.zone.kind,
                    contact_index=obs.contact_index if obs else None,
                    probability=None, probability_status="not_estimated")
