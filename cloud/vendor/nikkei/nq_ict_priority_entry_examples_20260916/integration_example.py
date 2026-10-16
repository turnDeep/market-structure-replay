"""旧版売買台帳へ接続する例。ファイル入出力やバックテストmainは持たない。

旧on_openと決済処理を継承し、入口選択・ゲートだけを差し替える。
生データからPacketを作る実行ランナーは別モジュールで実装する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nq_ict_macro_reversal_examples_20260916.common import (
    Bar, OrderIntent, Rules, SellEvidence, ZoneView, check_bar, down_tick,
)
from nq_ict_macro_reversal_examples_20260916.proposal_04_confirmed_ob_buy import ConfirmedObBuy
from run_nq_ict_early_buy_comparison_20260916 import SharedSignal, VariantLedger
from .common import PriorityRules, ReactionCatalog
from .proposal_01_watch_and_confirm import WatchedEntrySignals
from .proposal_02_reserved_attempts import ReservedConfirmationGate


@dataclass(frozen=True)
class Packet:
    signal: SharedSignal
    zones: tuple[ZoneView, ...]
    reactions: ReactionCatalog


def snapshot_reactions(buyer: ConfirmedObBuy, bar: Bar) -> ReactionCatalog:
    """buyer.on_closeを当日分だけ実行した直後に呼び出す。"""
    if buyer.previous is None or buyer.previous.index != bar.index or buyer.previous.at != bar.at:
        raise ValueError("snapshot must follow today's detector update")
    return ReactionCatalog(bar.index, bar.at, frozenset(
        f"{r.episode_id}:{r.id}:v{r.revision}" for r in buyer.active.values()
    ))


class PriorityLedgerExample(VariantLedger):
    def __init__(self, rules: Rules, cfg: Any,
                 priority: PriorityRules = PriorityRules(),
                 name: str = "priority_entry_example"):
        # 既存台帳の会計・ストップ・約定処理を再利用。matched集合は使わない。
        super().__init__(name, rules, cfg, set())
        self.priority = priority
        self.gate = ReservedConfirmationGate(rules, priority)
        self.tracker = WatchedEntrySignals(rules, priority)
        self.entry_audit: list[dict] = []

    def on_close(self, signal: SharedSignal) -> None:
        raise RuntimeError("use feed(Packet); the ordinary matched-cohort path is disabled")

    def feed(self, packet: Packet) -> None:
        """観測日足順に1回ずつ供給。別途on_openを呼ぶと二重処理になる。"""
        s = packet.signal
        bar = s.bar
        check_bar(bar, self.tracker.previous)
        # 前日の注文・ストップを先に処理。旧版の始値/場中stop時計をそのまま使用。
        super().on_open(bar)
        offers = self.tracker.on_close(
            bar, s.market, s.pivots, packet.zones, s.htf,
            s.early_buys, s.confirmed_buys, packet.reactions,
        )
        self.eligible_setup_ids.update(c.setup_id for c in s.early_buys if c.setup_id)
        for offer in offers:
            self.gate.register_offer(offer)

        # ここから出口・保護安値の流れは旧VariantLedger.on_closeと同じ。
        management = self.manager.on_close(bar, s.pivots, self.position)
        all_sells = tuple(s.sells) + ((management.exit_signal,) if management.exit_signal else ())
        intent: OrderIntent | None = None
        reason = "hold_major_wave"
        chosen_sells: list[SellEvidence] = []
        if self.position is not None:
            chosen_sells = [x for x in all_sells if x.index >= self.position.entry_index]
            if chosen_sells:
                chosen = management.exit_signal or chosen_sells[0]
                intent = OrderIntent(self._new_id(), "SELL", bar.at, bar.index,
                                     chosen.reason, position_id=self.position.id)
                reason = "confirmed_exit_for_next_open"
        elif all_sells:
            reason = "bearish_structure_observed_flat_no_short_order"
        elif self.pending is not None:
            raise ValueError("previous order must be filled or cancelled before this close")
        else:
            # 同日競合時は確認後を先に審査。その中は旧版と同じ参照高値の新しい順。
            ordered = sorted(offers, key=lambda o: (
                int(o.route == "confirmed"), o.decision.reference.origin_index, o.decision.key,
            ), reverse=True)
            for offer in ordered:
                c = offer.decision
                allowed, why = self.gate.assess(c, bar, s.pivots, False, False)
                self.entry_audit.append(dict(
                    index=bar.index, source_key=offer.source_early.key, decision_key=c.key,
                    route=offer.route, kind=offer.kind, allowed=allowed, reason=why,
                ))
                if allowed:
                    intent = OrderIntent(self._new_id(), "BUY", bar.at, bar.index, why, c)
                    self.gate.reserve(intent.id, c)
                    reason = f"{offer.kind}_entry_for_next_open"
                    break

        selected_key = intent.candidate.key if intent is not None and intent.candidate is not None else None
        self.entry_audit.append(dict(
            index=bar.index, action="daily_entry_selection",
            offered_keys=[o.decision.key for o in offers], selected_key=selected_key,
            position_exists=self.position is not None, sell_priority=bool(all_sells),
        ))
        if management.proposed_protected_low is not None and self.position is not None:
            low = management.proposed_protected_low.pivot
            stop = down_tick(low.price - self.rules.emergency_buffer_atr * bar.atr_prior,
                             self.rules.tick_size)
            self.pending_structural_stop = (self.position.id, stop)
        self.pending = intent
        self.plan_rows.append(dict(datetime=bar.at, variant=self.name, reason=reason,
                                   intent=intent, management=management,
                                   shared_early_count=len(s.early_buys),
                                   shared_confirmed_count=len(s.confirmed_buys),
                                   selected_sell_count=len(chosen_sells),
                                   offered_count=len(offers)))
        self.state_rows.append(dict(datetime=bar.at, variant=self.name,
                                    position_id=self.position.id if self.position else None,
                                    pending_order_id=self.pending.id if self.pending else None,
                                    live_stop=self.live_stop,
                                    episode_id=s.market.episode.id if s.market.episode else None))
        self.previous_at = bar.at

    def final_state(self) -> dict[str, Any]:
        result = super().final_state()
        result.update(route_fills=dict(self.gate.route_fills),
                      tracked_setup_ids=sorted(self.tracker.watches),
                      verification_status="runner_executed_validator_pending")
        return result
