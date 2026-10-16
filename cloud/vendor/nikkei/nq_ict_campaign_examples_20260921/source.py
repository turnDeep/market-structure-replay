"""Daily point-in-time adapter; never constructs a future-confirmed cohort.

The main BUY detector, OB objects, pivot adapter and major-decline episodes are
reused. Market SELL notifications use THIS proposal's major-top detector.
Consequently this is not advertised as an unchanged P0 packet comparison.
"""
from __future__ import annotations

import pandas as pd

from run_nq_ict_early_buy_comparison_20260916 import SharedCandidateReplay, SharedSignal
from nq_ict_retest_trade_state_machine_20260915 import number
from nq_ict_macro_reversal_examples_20260916.common import Zone, ZoneView
from nq_ict_priority_entry_examples_20260916.integration_example import Packet, snapshot_reactions
from .common import CampaignPacket
from .daily_evidence import DailyEvidenceBook
from .major_top import MajorTopDetector
from .double_top import DoubleTopDetector
from .reentry import ReversalBook
from .sell_selection import SellRepresentativeBook
from .structural_support import MajorStructureBook
from .upward_leg_support import UpwardLegSupportBook
from .cycle_recovery import CycleRecoveryBook, IndependentAdvanceBook
from .major_top_cycle import MajorTopBoundaryBook


class SessionSchedule:
    """Explicit trading-date labels supplied by the caller, not downloaded here."""
    def __init__(self, specification):
        self.start = pd.Timestamp(specification["coverage_start"]).normalize()
        self.end = pd.Timestamp(specification["coverage_end"]).normalize()
        raw = [pd.Timestamp(x) for x in specification["session_dates"]]
        if any(x.tzinfo is not None or x != x.normalize() for x in raw):
            raise ValueError("calendar must use naive session-date labels, not intraday timestamps")
        if not raw or raw != sorted(set(raw)):
            raise ValueError("calendar sessions must be sorted, unique and nonempty")
        if self.start > raw[0] or self.end < raw[-1] or self.start > self.end:
            raise ValueError("invalid calendar coverage")
        self.dates = tuple(raw)

    def validate_input(self, daily, cutoff):
        first = pd.Timestamp(daily.datetime.iloc[0])
        monday = first - pd.Timedelta(days=first.weekday())
        if self.start > monday or self.end < cutoff:
            raise ValueError("calendar coverage must include first input week through cutoff")
        expected = [d for d in self.dates if first <= d <= cutoff]
        observed = list(daily.datetime)
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            unexpected = sorted(set(observed) - set(expected))
            raise ValueError(f"session mismatch; missing={missing[:12]}, unexpected={unexpected[:12]}")

    def previous_week_levels(self, prior_bars, bar):
        monday = pd.Timestamp(bar.at).normalize() - pd.Timedelta(days=pd.Timestamp(bar.at).weekday())
        start = monday - pd.Timedelta(days=7)
        expected = [d for d in self.dates if start <= d < monday]
        observed = [b for b in prior_bars if start <= pd.Timestamp(b.at) < monday]
        if not expected or [pd.Timestamp(b.at) for b in observed] != expected:
            return ()  # Partial input-start week, no inferred weekly high/low.
        known = observed[-1].index
        return tuple(dict(id=f"previous-week:{start.date()}:{side}", side=side,
                          price=max(b.high for b in observed) if side == "high" else min(b.low for b in observed),
                          known_index=known) for side in ("high", "low"))


class CampaignSource(SharedCandidateReplay):
    def __init__(self, daily, objects, contexts, cfg, rules, campaign_cfg, schedule):
        super().__init__(daily, objects, contexts, cfg, rules)
        self.schedule = schedule
        self.evidence_book = DailyEvidenceBook(rules, campaign_cfg)
        self.top_detector = MajorTopDetector(rules, campaign_cfg)
        self.double_top_detector = DoubleTopDetector(rules, campaign_cfg)
        self.reversals = ReversalBook()
        self.sell_selector = SellRepresentativeBook(campaign_cfg)
        self.structures = MajorStructureBook(rules, campaign_cfg)
        self.leg_supports = UpwardLegSupportBook(rules, campaign_cfg)
        cycle_book = {"major_top_boundary": MajorTopBoundaryBook,
                      "independent_advance": IndependentAdvanceBook}.get(
                          campaign_cfg.cycle_mode, CycleRecoveryBook)
        self.cycles = cycle_book(rules, campaign_cfg)
        self.campaign_cfg = campaign_cfg
        self._dispatched_sells = set()
        self.stream_rows = []
        self.zone_state_events = []
        self.pivot_events = []
        self._zone_states = {}
        self._pivot_ids = set()
        self._zone_cache = {}
        self._last_packet = None
        self._independent_structure_keys = set()
        self.independent_structure_events = []

    def observe_independent_structure(self, candidates):
        """Called after the ledger observes this close, before the next packet.

        Uses market evidence even when flat, never whether an order was filled.
        Previously yielded packets and today's actions are not rewritten.
        """
        if not candidates:
            return
        packet = self._last_packet
        if packet is None:
            raise ValueError("structure observation requires a yielded packet")
        s, bar = packet.base.signal, packet.base.signal.bar
        fresh = []
        for c in candidates:
            if c.index != bar.index or c.at != bar.at or c.reference.available_index >= bar.index:
                raise ValueError("independent structure observation must be causal and current")
            if c.key not in self._independent_structure_keys:
                fresh.append(c)
                self._independent_structure_keys.add(c.key)
        if not fresh:
            return
        if s.sells:
            self.independent_structure_events.append(dict(index=bar.index, action="structure_sell_conflict",
                                                         candidate_keys=[c.key for c in fresh],
                                                         policy="SELL_has_priority_no_retroactive_rearm"))
            return
        selected = max(fresh, key=lambda c: (c.reference.origin_index, c.key))
        self.market.confirm_upward_structure(selected.episode_id, bar, selected.reference)
        if (self.campaign_cfg.cycle_mode == "legacy" and self.reversals.current
                and bar.index > self.reversals.current.index):
            if not self.reversals.armed:
                self.reversals.events.append(dict(index=bar.index, action="new_reversal_can_be_armed",
                                                 after_reversal_id=self.reversals.current.id,
                                                 source="independent_ob_structure"))
            self.reversals.armed = True
        self.independent_structure_events.append(dict(index=bar.index, action="independent_structure_observed",
                                                     candidate_key=selected.key,
                                                     effective_from_next_packet=True,
                                                     depends_on_fills=False))

    def _zones(self):
        """Return causal zone views without resorting/rebuilding old geometry.

        RevisionEngine adds a zone only on its available session, so every
        member of ``self.zones`` is already available to the current bar.
        Geometry is immutable; only the state string changes. Caching the
        converted Zone removes a repeated dataframe search and full sort from
        every daily step without changing the buyer's own deterministic ID
        ordering.
        """
        result = []
        for old in self.zones.values():
            converted = self._zone_cache.get(old.id)
            if converted is None:
                converted = Zone(
                    old.id, old.kind, old.direction, old.timeframe,
                    int(self.d.datetime.searchsorted(old.origin_at, side="left")),
                    pd.Timestamp(old.origin_at),
                    int(self.d.datetime.searchsorted(old.available_at)),
                    pd.Timestamp(old.available_at), float(old.low), float(old.high),
                    float(old.wick_low), float(old.wick_high),
                )
                self._zone_cache[old.id] = converted
            result.append(ZoneView(converted, self.t, str(old.state)))
        return tuple(result)

    def packets(self):
        for self.i in range(len(self.d)):
            row = self.d.iloc[self.i]
            self.t = pd.Timestamp(row.datetime)
            self.c = self.ctx.loc[self.t].to_dict()
            bar = self._bar(row)
            self.activate()
            for zone in list(self.zones.values()):
                self.update_zone(zone, row, number(row.atr_prior))
            self.confirm_pivots()
            pivots = self.adapter.convert(self.pivots)
            zones, htf = self._zones(), self._htf()
            # Snapshot precedes today's processing: a new major break cannot
            # qualify a zone for today's already occurring contact.
            support_roles = self.structures.before_close(zones, bar)
            if self.campaign_cfg.support_selection == "structural_or_leg":
                for zid, info in self.leg_supports.before_close(zones, bar).items():
                    structural = support_roles.setdefault(zid, dict(
                        classification="no_prior_major_role", known_through_index=bar.index - 1,
                        role=None))
                    structural["upward_leg_role"] = info["role"]
                self.leg_supports.on_close(bar, pivots, zones)
            self.structures.on_close(bar, pivots, zones)
            for v in zones:
                if v.zone.id not in self._zone_states or self._zone_states[v.zone.id] != v.state:
                    self.zone_state_events.append(dict(index=bar.index, datetime=bar.at,
                                                       zone=v.zone, prior_state=self._zone_states.get(v.zone.id),
                                                       state=v.state))
                    self._zone_states[v.zone.id] = v.state
            for p in pivots:
                if p.id not in self._pivot_ids:
                    self.pivot_events.append(dict(index=bar.index, datetime=bar.at, pivot=p))
                    self._pivot_ids.add(p.id)
            market = self.market.on_close(bar, pivots)
            prior_week = self.schedule.previous_week_levels(self.evidence_book.bars, bar)
            evidence = self.evidence_book.on_close(bar, pivots, zones, htf, prior_week)
            area_sells, area_details = self.top_detector.on_close(bar, market, pivots, evidence)
            # The selected detector owns the source-side SELL clock. Running a
            # non-selected full-history detector here would add a large amount
            # of work while its diagnostic rows cannot affect this replay.
            if self.campaign_cfg.sell_detector == "major_area":
                pair_sells, pair_details = (), ()
            else:
                pair_sells, pair_details = self.double_top_detector.on_close(bar, pivots, evidence)
            if self.campaign_cfg.sell_detector == "major_area":
                raw_sells, top_details = area_sells, area_details
            else:
                raw_sells, top_details = pair_sells, pair_details
            # A revised high must not replay the same structural SELL after a
            # later BUY. Non-selected detectors remain diagnostic only.
            fresh_sells = tuple(s for s in raw_sells if s.source_id not in self._dispatched_sells)
            self._dispatched_sells.update(s.source_id for s in fresh_sells)
            sells = self.sell_selector.select(bar, fresh_sells, top_details)
            top_details = self.sell_selector.selected_details
            confirmed = self.buyer.on_close(bar, market, pivots, zones, htf)
            early = self.buyer.early_result
            catalog = snapshot_reactions(self.buyer, bar)
            reversal = self.reversals.on_close(bar, sells, confirmed, market, top_details,
                                               allow_local_rearm=self.campaign_cfg.cycle_mode == "legacy",
                                               refresh_context=self.campaign_cfg.cycle_mode in {
                                                   "independent_advance", "major_top_boundary"},
                                               budget_authority=type(self.cycles).__name__)
            cycle = None
            if self.campaign_cfg.cycle_mode in {"structure_recovery", "independent_advance", "major_top_boundary"}:
                cycle = self.cycles.on_close(bar, reversal, sells, tuple(self.structures.records),
                                            top_details=top_details)
                if self.cycles.completed_today and self.campaign_cfg.cycle_mode == "structure_recovery":
                    # Takes effect on the next source step; no extra BUY is created.
                    self.reversals.armed = True
            # Market state is independent of holdings/fills. Changes become part
            # of subsequent snapshots, not revisions of already-issued packets.
            if sells:
                for sell in sells:
                    self.market.note_failed_reversal(bar, sell.reference)
            elif confirmed:
                chosen = max(confirmed, key=lambda c: (c.reference.origin_index, c.key))
                self.market.confirm_upward_structure(chosen.episode_id, bar, chosen.reference)
            base = Packet(SharedSignal(bar, pivots, market, htf, tuple(early),
                                       tuple(confirmed), tuple(sells)), zones, catalog)
            packet = CampaignPacket(base, evidence, tuple(self.evidence_book.zones.values()),
                                    tuple(d["top_id"] for d in top_details), top_details, reversal,
                                    support_roles=support_roles, major_structures=tuple(self.structures.records),
                                    market_cycle=cycle, sell_selection=dict(self.sell_selector.snapshot))
            self.stream_rows.append(dict(index=bar.index, datetime=bar.at,
                                         early=len(early), confirmed=len(confirmed), major_sell=len(sells),
                                         episode=market.episode.id if market.episode else None,
                                         phase=market.episode.phase if market.episode else None,
                                         sell_detector=self.campaign_cfg.sell_detector,
                                         major_reversal_id=reversal.id if reversal else None,
                                         market_cycle=cycle,
                                         representative_sell_source_id=self.sell_selector.snapshot["selected_source_id"],
                                         duplicate_sell_sources_suppressed=len(raw_sells) - len(sells),
                                         evidence_ids=[e.id for e in evidence],
                                         previous_week_reference_status="available" if prior_week else "unknown_or_partial"))
            self.early_rows.extend(early)
            self.confirmed_rows.extend(confirmed)
            self.sell_rows.extend(sells)
            self.market.history.clear()
            self._last_packet = packet
            yield packet

    def run(self):
        raise RuntimeError("use packets(); the old matched-cohort runner is deliberately disabled")
