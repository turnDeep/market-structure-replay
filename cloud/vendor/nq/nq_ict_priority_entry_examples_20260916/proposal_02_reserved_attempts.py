"""局面内の試行枠を早期1回・確認後1回に分ける例。

旧版の2回/累積損失2R制限も共有する。回数は注文数ではなく実約定数。
確認後再BUYに限り、旧回復日が決済前でも、決済後の新しい構造突破を認める。
"""
from __future__ import annotations

from collections import defaultdict

from nq_ict_macro_reversal_examples_20260916.common import Bar, BuyEvidence, Pivot, Rules
from nq_ict_macro_reversal_examples_20260916.proposal_05_episode_attempts import EpisodeAttemptGate
from .common import EntryOffer, PriorityRules


class ReservedConfirmationGate(EpisodeAttemptGate):
    def __init__(self, legacy_rules: Rules, priority: PriorityRules):
        super().__init__(legacy_rules)
        if (priority.max_early_fills_per_episode < 0
                or priority.max_confirmed_fills_per_episode < 0):
            raise ValueError("route budgets must be nonnegative")
        self.priority = priority
        self.route_fills: dict[str, dict[str, int]] = defaultdict(lambda: {"early": 0, "confirmed": 0})
        self.offers: dict[str, EntryOffer] = {}
        self.route_reservations: dict[str, tuple[str, str]] = {}

    def register_offer(self, offer: EntryOffer) -> None:
        c = offer.decision
        if c.key in self.offers and self.offers[c.key] != offer:
            raise ValueError("entry evidence cannot be rewritten")
        if c.setup_id != offer.source_early.setup_id or c.episode_id != offer.source_early.episode_id:
            raise ValueError("source and decision must share setup/episode identity")
        if offer.route not in {"early", "confirmed"}:
            raise ValueError("unsupported entry route")
        self.offers[c.key] = offer

    def assess(self, c: BuyEvidence, bar: Bar, pivots: tuple[Pivot, ...],
               has_position: bool, pending: bool) -> tuple[bool, str]:
        offer = self.offers.get(c.key)
        if offer is None or offer.decision != c:
            return False, "unregistered_or_changed_entry_offer"
        caps = {"early": self.priority.max_early_fills_per_episode,
                "confirmed": self.priority.max_confirmed_fills_per_episode}
        if self.route_fills[c.episode_id][offer.route] >= caps[offer.route]:
            return False, f"{offer.route}_route_attempt_limit"

        allowed, reason = super().assess(c, bar, pivots, has_position, pending)
        if allowed:
            return True, f"{offer.route}:{reason}"
        # 総回数・損失・保有・同日再BUY禁止などの共通制限は緩めない。
        retry_reasons = {"recovery_and_breakout_must_follow_previous_exit",
                         "no_material_new_structure_since_exit"}
        if offer.route != "confirmed" or reason not in retry_reasons:
            return False, reason
        b = self.budgets.get(c.episode_id)
        if b is None or b.last_exit_index is None:
            return False, reason
        new_confirmation = (
            c.confirmation in {"close_sequence", "breakout_retest"}
            and b.last_exit_index < c.breakout_index <= bar.index
            and c.reference.available_index < c.breakout_index
        )
        if not new_confirmation:
            return False, "confirmed_retry_requires_new_breakout_after_exit"
        return True, "confirmed_retry:new_structure_breakout_after_exit"

    def reserve(self, intent_id: str, candidate: BuyEvidence) -> None:
        offer = self.offers.get(candidate.key)
        if offer is None:
            raise ValueError("register and assess an offer before reservation")
        super().reserve(intent_id, candidate)
        self.route_reservations[intent_id] = (candidate.episode_id, offer.route)

    def cancel(self, intent_id: str) -> None:
        super().cancel(intent_id)
        self.route_reservations.pop(intent_id, None)

    def filled(self, intent_id: str, index: int) -> None:
        episode_id, route = self.route_reservations[intent_id]
        super().filled(intent_id, index)
        del self.route_reservations[intent_id]
        self.route_fills[episode_id][route] += 1
        self.events.append(dict(index=index, action="route_attempt_filled",
                                episode=episode_id, route=route,
                                route_attempts=self.route_fills[episode_id][route]))

    # closed()は継承: max(0, -net_pnl / initial_risk)を同じ局面へ累積。
    # confirmedになっても損失履歴を消さず、別OB IDでも同じepisode予算を使う。
