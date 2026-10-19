"""提案5: 同一下落局面内の全OBで試行回数と損失予算を共有する。

新しいOB IDに移っても予算は増やさない。再エントリーは前回決済後の回復、
新しい突破に加え、新安値/別支持帯/確認された切り上げ安値のいずれかを要求。
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import Bar, BuyEvidence, Pivot, Rules


@dataclass
class EpisodeBudget:
    attempts: int = 0
    cumulative_loss_r: float = 0.0
    last_exit_index: int | None = None
    last_candidate: BuyEvidence | None = None


class EpisodeAttemptGate:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.budgets: dict[str, EpisodeBudget] = {}
        self.reserved: dict[str, BuyEvidence] = {}
        self.used_keys: set[str] = set()
        self.position_episodes: dict[str, tuple[str, BuyEvidence]] = {}
        self.last_exit_index: int | None = None
        self.events: list[dict] = []

    def assess(self, c: BuyEvidence, bar: Bar, pivots: tuple[Pivot, ...],
               has_position: bool, pending: bool) -> tuple[bool, str]:
        if has_position or pending:
            return False, "position_or_order_exists"
        if c.index != bar.index or c.at != bar.at:
            return False, "fresh_today_confirmation_required"
        if c.key in self.used_keys:
            return False, "evidence_already_reserved"
        if self.last_exit_index is not None and bar.index <= self.last_exit_index:
            return False, "same_day_reentry_blocked"
        b = self.budgets.get(c.episode_id, EpisodeBudget())
        if b.attempts >= self.rules.max_attempts_per_episode:
            return False, "episode_attempt_limit"
        if b.cumulative_loss_r >= self.rules.max_loss_r_per_episode:
            return False, "episode_loss_limit"
        if b.last_exit_index is None or b.last_candidate is None:
            return True, "first_confirmed_reversal_in_episode"
        old, exit_index = b.last_candidate, b.last_exit_index
        if c.recovery_index <= exit_index or c.breakout_index <= exit_index:
            return False, "recovery_and_breakout_must_follow_previous_exit"
        material_new_low = (c.low_index > exit_index
                            and c.low <= old.low - self.rules.retry_new_low_atr * old.low_atr)
        different_lower_support = (c.touch_index > exit_index
                                   and c.zone.wick_high < old.zone.wick_low)
        higher_low = any(p.kind == "low" and p.radius >= self.rules.sell_support_radius
                         and exit_index < p.origin_index < c.breakout_index
                         and p.available_index < c.breakout_index
                         and p.price >= old.low + self.rules.retry_higher_low_atr * old.low_atr
                         and p.price < c.reference.price for p in pivots)
        if not (material_new_low or different_lower_support or higher_low):
            return False, "no_material_new_structure_since_exit"
        reason = "new_lower_extreme" if material_new_low else (
            "new_lower_support" if different_lower_support else "confirmed_higher_low")
        return True, f"episode_retry:{reason}"

    def reserve(self, intent_id: str, candidate: BuyEvidence) -> None:
        if self.reserved or candidate.key in self.used_keys:
            raise ValueError("duplicate reservation")
        self.used_keys.add(candidate.key)
        self.reserved[intent_id] = candidate

    def cancel(self, intent_id: str) -> None:
        # 取消しは約定回数に数えない。同じ古い確認を無限に再発注はしない。
        self.reserved.pop(intent_id, None)

    def filled(self, intent_id: str, index: int) -> None:
        candidate = self.reserved.pop(intent_id)
        b = self.budgets.setdefault(candidate.episode_id, EpisodeBudget())
        b.attempts += 1
        b.last_candidate = candidate
        self.position_episodes[intent_id] = (candidate.episode_id, candidate)
        self.events.append(dict(index=index, action="attempt_filled", episode=candidate.episode_id,
                                attempts=b.attempts))

    def closed(self, position_id: str, index: int, net_pnl: float, initial_risk: float) -> None:
        if initial_risk <= 0:
            raise ValueError("positive initial risk required")
        episode_id, candidate = self.position_episodes.pop(position_id)
        b = self.budgets[episode_id]
        b.cumulative_loss_r += max(0.0, -net_pnl / initial_risk)
        b.last_exit_index = self.last_exit_index = index
        b.last_candidate = candidate
        self.events.append(dict(index=index, action="attempt_closed", episode=episode_id,
                                cumulative_loss_r=b.cumulative_loss_r))
