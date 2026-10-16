"""提案3: 主要高値圏の再試験後、上昇を支えた安値の失敗を確認する。

建玉の波の起点より古い第一高値も使用する。市場シグナルは建玉なしでも出す。
close_sequenceは連続終値、breakdown_retestは別バーでの戻り失敗を要求する。
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import Bar, HighArea, MarketSnapshot, Pivot, Rules, SellEvidence, check_bar, positive


@dataclass
class TopAttempt:
    id: str
    area: HighArea
    touch_index: int
    second_high: float
    second_index: int
    support: Pivot | None
    outside_closes: int = 0
    below_count: int = 0
    break_index: int | None = None
    emitted: bool = False
    revision: int = 0


class ConfirmedTopSell:
    def __init__(self, rules: Rules):
        if rules.sell_mode not in {"close_sequence", "breakdown_retest"}:
            raise ValueError("unsupported sell_mode")
        self.rules = rules
        self.previous: Bar | None = None
        self.active: dict[str, TopAttempt] = {}
        self.retired: set[str] = set()
        self.rearm_after: dict[str, int] = {}
        self.trace: list[dict] = []

    def select_support(self, area: HighArea, bar: Bar, pivots: tuple[Pivot, ...]) -> Pivot | None:
        lows = [p for p in pivots if p.kind == "low" and p.radius >= self.rules.sell_support_radius
                and p.available_index < bar.index and p.available_at < bar.at
                and area.anchor.origin_index < p.origin_index < bar.index
                and bar.index - p.origin_index >= self.rules.sell_support_bars
                and positive(p.atr_at_origin)
                and bar.high - p.price >= self.rules.sell_support_rise_atr * p.atr_at_origin]
        return max(lows, key=lambda p: (p.origin_index, p.id)) if lows else None

    def on_close(self, bar: Bar, market: MarketSnapshot,
                 pivots: tuple[Pivot, ...]) -> tuple[SellEvidence, ...]:
        check_bar(bar, self.previous)
        if market.at != bar.at or any(p.available_index > bar.index for p in pivots):
            raise ValueError("current point-in-time inputs required")
        live = {a.id: a for a in market.areas if a.active}
        for area in live.values():
            if area.id in self.rearm_after:
                if (self.previous is not None and self.previous.close < area.low
                        and self.previous.index > self.rearm_after[area.id]):
                    del self.rearm_after[area.id]
                else:
                    continue
            if (area.id in self.active or area.id in self.retired
                    or area.qualified_index is None or area.qualified_index >= bar.index
                    or area.known_at >= bar.at
                    or bar.index - area.anchor.origin_index < self.rules.top_min_separation):
                continue
            approached = self.previous is not None and self.previous.close < area.high
            touched = bar.high >= area.low and bar.low <= area.high
            if approached and touched:
                self.active[area.id] = TopAttempt(f"top:{area.id}:{bar.index}", area,
                                                   bar.index, bar.high, bar.index,
                                                   self.select_support(area, bar, pivots))

        result: list[SellEvidence] = []
        for aid, top in list(self.active.items()):
            reason = "await_reversal"
            expired = bar.index - top.touch_index > self.rules.top_lifetime
            top.outside_closes = top.outside_closes + 1 if bar.close > top.area.high else 0
            accepted = top.outside_closes >= self.rules.breakout_accept_closes
            if aid not in live or expired or accepted:
                if expired and aid in live and not accepted:
                    self.rearm_after[aid] = bar.index
                else:
                    self.retired.add(aid)
                del self.active[aid]
                self.trace.append(dict(at=bar.at, top_id=top.id, reason="expired_or_breakout_accepted"))
                continue
            if bar.high > top.second_high:
                top.second_high, top.second_index = bar.high, bar.index
                top.break_index, top.below_count = None, 0
                top.emitted = False
                top.revision += 1
                # 新高値を観測した日からのみ、当時既知の支持Pivotを再評価。
                top.support = self.select_support(top.area, bar, pivots)
            ref = top.support
            if ref is None:
                reason = "no_preconfirmed_support_of_sufficient_scale"
            elif bar.index > top.second_index and positive(ref.atr_at_origin):
                boundary = ref.price - self.rules.structure_buffer_atr * ref.atr_at_origin
                below = bar.close < boundary
                if below:
                    if top.break_index is None:
                        top.break_index = bar.index
                    top.below_count += 1
                else:
                    top.break_index, top.below_count = None, 0
                departure = top.second_high - bar.close >= self.rules.sell_departure_atr * ref.atr_at_origin
                if self.rules.sell_mode == "close_sequence":
                    confirmed = top.below_count >= self.rules.sell_confirm_closes
                else:
                    confirmed = (top.break_index is not None and bar.index > top.break_index
                                 and bar.high >= ref.price - self.rules.retest_tolerance_atr * ref.atr_at_origin
                                 and below and bar.close < bar.open)
                if confirmed and departure and bar.close < top.area.high and not top.emitted:
                    result.append(SellEvidence(f"{top.id}:v{top.revision}:{bar.index}", bar.at,
                                               bar.index, f"major_top_{self.rules.sell_mode}", top.id, ref))
                    top.emitted = True
                    reason = "confirmed_sell_structure"
                else:
                    reason = "await_support_failure_confirmation"
            self.trace.append(dict(at=bar.at, top_id=top.id, reason=reason,
                                   reference_id=ref.id if ref else None,
                                   reference_available_at=ref.available_at if ref else None,
                                   second_index=top.second_index, below_count=top.below_count))
        self.previous = bar
        return tuple(result)
