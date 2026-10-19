"""提案2: 大きな押し安値を終値で管理する。初期の緊急stopは別に保つ。

高値H→十分な幅/期間の調整L→Hを終値で突破、を満たすLを採用。
Lがまだ未確認なら後日の確認を待つ。有効化は判定の次バーから。
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import Bar, Pivot, Position, ProtectedLow, Rules, SellEvidence, check_bar, positive


@dataclass(frozen=True)
class Breakout:
    high: Pivot
    low_bar: Bar
    breakout_index: int


@dataclass(frozen=True)
class ManagementResult:
    active_protected_low: ProtectedLow | None
    proposed_protected_low: ProtectedLow | None
    exit_signal: SellEvidence | None
    reason: str


class MacroPositionManager:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.bars: list[Bar] = []
        self.breakouts: list[Breakout] = []
        self.position_id: str | None = None
        self.active: ProtectedLow | None = None
        self.pending: ProtectedLow | None = None
        self.below_count = 0
        self.events: list[dict] = []

    def on_close(self, bar: Bar, pivots: tuple[Pivot, ...],
                 position: Position | None) -> ManagementResult:
        prev = self.bars[-1] if self.bars else None
        check_bar(bar, prev)
        if any(p.available_index > bar.index or p.available_at > bar.at for p in pivots):
            raise ValueError("future pivot")
        pid = position.id if position else None
        if pid != self.position_id:
            self.active = self.pending = None
            self.below_count = 0
            self.position_id = pid
        if self.pending and self.pending.effective_from_index <= bar.index:
            self.active, self.pending = self.pending, None
            self.below_count = 0

        if prev and positive(bar.atr_prior):
            for high in pivots:
                if (high.kind != "high" or high.radius < self.rules.pivot_radius
                        or high.available_index >= bar.index or not positive(high.atr_at_origin)
                        or not prev.close <= high.price < bar.close):
                    continue
                interval = [b for b in self.bars if high.origin_index < b.index < bar.index]
                if len(interval) < self.rules.management_pullback_bars:
                    continue
                low = min(interval, key=lambda b: (b.low, b.index))
                if high.price - low.low >= self.rules.management_pullback_atr * high.atr_at_origin:
                    self.breakouts.append(Breakout(high, low, bar.index))
        self.bars.append(bar)
        if position is None:
            return ManagementResult(None, None, None, "market_breakouts_observed_without_position")

        signal = None
        if self.active:
            self.below_count = self.below_count + 1 if bar.close < self.active.pivot.price else 0
            if self.below_count == self.rules.structural_exit_closes:
                signal = SellEvidence(f"macro_low:{pid}:{self.active.pivot.id}:{bar.index}",
                                      bar.at, bar.index, "major_protected_low_close_failure",
                                      self.active.pivot.id, self.active.pivot)

        choices: list[ProtectedLow] = []
        if signal is None:
            for record in self.breakouts:
                if (record.high.origin_index <= position.candidate.low_index
                        or bar.close <= record.high.price):
                    continue
                lows = [p for p in pivots if p.kind == "low"
                        and p.radius >= self.rules.pivot_radius
                        and p.origin_index == record.low_bar.index
                        and p.price == record.low_bar.low and p.available_index <= bar.index]
                if not lows:
                    continue
                low = min(lows, key=lambda p: (p.available_index, p.id))
                if any(b.close < low.price for b in self.bars if b.index >= record.breakout_index):
                    continue
                if low.price <= position.candidate.emergency_stop or low.price >= bar.close:
                    continue
                if self.active and (low.origin_index <= self.active.pivot.origin_index
                                    or low.price <= self.active.pivot.price):
                    continue
                choices.append(ProtectedLow(low, record.high, record.breakout_index,
                                            bar.index, bar.index + 1))
        proposal = max(choices, key=lambda p: (p.pivot.origin_index, p.pivot.id)) if choices else None
        if proposal:
            self.pending = proposal
            self.events.append(dict(at=bar.at, kind="macro_protected_low_proposed", position_id=pid,
                                    low_id=proposal.pivot.id, effective_index=bar.index + 1))
        reason = "major_structure_exit" if signal else "hold_with_macro_structure"
        return ManagementResult(self.active, proposal, signal, reason)

    @staticmethod
    def emergency_stop(position: Position) -> float:
        # 高値更新のたびにこのstopを引き上げない。場中執行は接続する執行層の責務。
        return position.candidate.emergency_stop
