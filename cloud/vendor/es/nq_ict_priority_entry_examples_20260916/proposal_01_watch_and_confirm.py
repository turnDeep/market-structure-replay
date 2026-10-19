"""①強い回復で参加し、見送り後も同じ反応の構造確認まで追跡する例。

未来の確認は使わない。confirmed_buysにはその日に実際に発行されたものだけ。
追跡中の局所高値突破は、新しい判断日時を持つ派生シグナルとして発行する。
"""
from __future__ import annotations

from dataclasses import replace

from nq_ict_macro_reversal_examples_20260916.common import (
    Bar, BuyEvidence, HigherTimeframe, MarketSnapshot, Pivot, Rules, ZoneView,
    check_bar, check_inputs, positive,
)
from .common import EntryOffer, PriorityRules, ReactionCatalog, Watch


class WatchedEntrySignals:
    def __init__(self, legacy_rules: Rules, priority: PriorityRules):
        self.rules = legacy_rules
        self.priority = priority
        self.previous: Bar | None = None
        self.watches: dict[str, Watch] = {}
        self.seen_seeds: set[str] = set()
        self.events: list[dict] = []

    def _record(self, bar: Bar, setup_id: str, action: str, **extra) -> None:
        self.events.append(dict(index=bar.index, at=bar.at, setup_id=setup_id,
                                action=action, **extra))

    @staticmethod
    def _check_evidence(c: BuyEvidence, bar: Bar, htf: HigherTimeframe) -> None:
        if c.index != bar.index or c.at != bar.at or c.htf != htf:
            raise ValueError("only today's point-in-time detector evidence is accepted")
        if not c.setup_id or c.reference is None:
            raise ValueError("setup id and available mechanical reference are required")
        if c.zone.kind != "ob_candidate" or c.zone.direction != "bullish":
            raise ValueError("this proposal handles bullish OB long entries")
        if not (c.zone.available_index < c.touch_index <= c.low_index <= c.index
                and c.touch_index <= c.recovery_index <= c.index
                and c.reference.available_index < c.index):
            raise ValueError("invalid candidate chronology")
        check_inputs(bar, (c.reference,), (ZoneView(c.zone, bar.at, "active"),), htf)

    def on_close(
        self, bar: Bar, market: MarketSnapshot, pivots: tuple[Pivot, ...],
        zones: tuple[ZoneView, ...], htf: HigherTimeframe,
        early_buys: tuple[BuyEvidence, ...], confirmed_buys: tuple[BuyEvidence, ...],
        catalog: ReactionCatalog,
    ) -> tuple[EntryOffer, ...]:
        check_bar(bar, self.previous)
        check_inputs(bar, pivots, zones, htf)
        if (market.at != bar.at or catalog.at != bar.at or catalog.index != bar.index):
            raise ValueError("current market and complete reaction catalog are required")
        views = {v.zone.id: v for v in zones}
        ep = market.episode

        # 検知器のrevision/寿命を尊重する。古い反応の復活や予算リセットはしない。
        for sid in tuple(self.watches):
            c = self.watches[sid].source
            reason = None
            if ep is None or ep.id != c.episode_id:
                reason = "episode_finished_or_changed"
            elif sid not in catalog.live_setup_ids:
                reason = "source_revision_replaced_or_reaction_removed"
            elif bar.index - c.touch_index > self.rules.reaction_lifetime:
                reason = "original_reaction_lifetime_expired"
            elif c.zone.id not in views or not views[c.zone.id].observable:
                reason = "source_zone_unavailable"
            elif bar.low < c.low:
                reason = "new_low_requires_a_new_detector_revision"
            if reason:
                self._record(bar, sid, "watch_retired", reason=reason)
                del self.watches[sid]

        for c in early_buys:
            self._check_evidence(c, bar, htf)
            if c.confirmation != "early_recovery":
                raise ValueError("expected unchanged legacy early evidence")
            sid = c.setup_id
            if sid not in catalog.live_setup_ids or ep is None or ep.id != c.episode_id:
                # 同じ検知器更新内で発行後に旧versionが取り消された場合も残す。
                self._record(bar, sid, "seed_not_live_after_detector_update")
                continue
            if sid in self.seen_seeds:
                raise ValueError("early evidence must be emitted once per setup version")
            self.seen_seeds.add(sid)
            self.watches[sid] = Watch(c)
            self._record(bar, sid, "watch_registered", source_key=c.key,
                         source_decision_index=c.index)

        offers: list[EntryOffer] = []
        confirmed_today: set[str] = set()
        for c in confirmed_buys:
            self._check_evidence(c, bar, htf)
            if c.confirmation not in {"close_sequence", "breakout_retest"}:
                raise ValueError("expected today's legacy confirmed-structure event")
            sid = c.setup_id
            w = self.watches.get(sid)
            if w is None or sid not in catalog.live_setup_ids:
                self._record(bar, sid, "confirmation_without_live_registered_seed")
                continue
            if w.confirmed_trigger_emitted:
                raise ValueError("duplicate confirmation event in the same setup version")
            if c.zone != w.source.zone or c.episode_id != w.source.episode_id:
                raise ValueError("confirmation must refer to the same source setup")
            if not (c.recovery_index <= c.breakout_index <= c.index
                    and c.reference.available_index < c.breakout_index):
                raise ValueError("reference must be known before the structural breakout")
            if not views[c.zone.id].tradeable:
                self._record(bar, sid, "confirmed_source_not_tradeable")
                continue
            if not self.priority.enable_confirmed_entry:
                self._record(bar, sid, "structure_confirmation_observed_entry_disabled")
                continue
            confirmed_today.add(sid)
            w.confirmed_trigger_emitted = True
            # 同じ日に同じ候補を早期と確認後で二重発行しない。
            w.local_trigger_emitted = True
            offers.append(EntryOffer("confirmed", "observed_confirmation", w.source, c))
            self._record(bar, sid, "structure_confirmation_observed",
                         enabled=self.priority.enable_confirmed_entry,
                         breakout_index=c.breakout_index, decision_index=c.index)

        for sid, w in self.watches.items():
            c = w.source
            if sid in confirmed_today or w.local_trigger_emitted or w.confirmed_trigger_emitted:
                continue
            if ep is None or ep.phase != "declining":
                continue  # 早期入口の市場状態は旧版どおり
            if c.zone.id not in views or not views[c.zone.id].tradeable:
                continue
            if views[c.zone.id].zone != c.zone:
                raise ValueError("source OB geometry changed after registration")
            if (bar.index > c.index and not self.priority.track_after_early_rejection):
                continue
            body_recovered = bar.close >= c.zone.body_high
            local_break = self.previous is not None and bar.close > self.previous.high
            valid_stop = positive(c.emergency_stop) and c.emergency_stop < bar.close
            if not (body_recovered and local_break and valid_stop):
                self._record(bar, sid, "watch_waiting",
                             body_recovered=body_recovered, local_high_break=local_break,
                             valid_frozen_stop=valid_stop)
                continue

            # source_earlyは一切書き換えない。注文用の派生証拠を別に作る。
            # ストップと反応安値は登録時に固定。新安値は上のretireで除外済み。
            kind = "original_early" if bar.index == c.index else "deferred_local_break"
            current = replace(
                c, key=f"{c.key}:priority_local:{bar.index}", at=bar.at, index=bar.index,
                confirmation="priority_local_break", breakout_index=bar.index,
                htf=htf,
            )
            # ここでのbreakout_indexは局所高値突破。戻り高値の構造突破ではない。
            offers.append(EntryOffer("early", kind, c, current))
            w.local_trigger_emitted = True
            self._record(bar, sid, "local_trigger_emitted", kind=kind,
                         source_decision_index=c.index, decision_index=bar.index)

        self.previous = bar
        return tuple(offers)
